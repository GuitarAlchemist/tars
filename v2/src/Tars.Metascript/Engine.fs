namespace Tars.Metascript

open System
open System.Threading.Tasks
open System.Text.RegularExpressions
open Tars.Core
open Tars.Cortex
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Tools
open Domain

module Engine =

    /// Metadata filter for retrieval
    type MetadataFilter =
        { Field: string
          Operator: string // "eq", "ne", "contains", "gt", "lt", "gte", "lte"
          Value: string }

    /// Query type for routing
    type QueryType =
        | Factual // Direct fact lookup
        | Analytical // Requires reasoning/analysis
        | Conversational // Follow-up or contextual
        | Keyword // Keyword-heavy search
        | Unknown

    /// Retrieval metrics for observability
    type RetrievalMetrics =
        { mutable TotalQueries: int64
          mutable CacheHits: int64
          mutable CacheMisses: int64
          mutable TotalLatencyMs: int64
          mutable AvgResultCount: float
          mutable FallbackTriggered: int64
          mutable CompressionApplied: int64 }

        static member Create() =
            { TotalQueries = 0L
              CacheHits = 0L
              CacheMisses = 0L
              TotalLatencyMs = 0L
              AvgResultCount = 0.0
              FallbackTriggered = 0L
              CompressionApplied = 0L }

    /// Configuration for RAG (Retrieval Augmented Generation)
    type RagConfig =
        {
            /// Collection name for storing/retrieving embeddings
            CollectionName: string
            /// Number of results to retrieve for context
            TopK: int
            /// Minimum similarity score (0.0-1.0) for including results
            MinScore: float32
            /// Whether to auto-index agent outputs
            AutoIndex: bool
            /// Max characters per stored chunk
            MaxChunkChars: int
            /// Max chunks per document to index
            MaxChunks: int
            /// Max characters to include in assembled context
            MaxContextChars: int
            /// Enable hybrid search (combine keyword + semantic)
            EnableHybridSearch: bool
            /// Weight for semantic score in hybrid search (0.0-1.0)
            SemanticWeight: float32
            /// Rerank results using LLM (slower but more accurate)
            EnableReranking: bool
            /// Use LLM to expand query with related terms
            EnableQueryExpansion: bool
            /// Number of expanded queries to generate
            QueryExpansionCount: int
            /// Enable multi-hop retrieval via knowledge graph
            EnableMultiHop: bool
            /// Max hops in knowledge graph traversal
            MaxHops: int
            /// Metadata filters to apply before scoring
            MetadataFilters: MetadataFilter list
            /// Cache embeddings to avoid recomputation
            EnableEmbeddingCache: bool
            /// Max cache entries
            EmbeddingCacheSize: int
            /// Enable async batching of embeddings
            EnableAsyncBatching: bool
            /// Batch size for async embedding
            BatchSize: int
            /// Use Reciprocal Rank Fusion to combine retrieval methods
            EnableRRF: bool
            /// RRF constant k (typically 60)
            RRFConstant: int
            // ===== NEW BATCH 2 OPTIONS =====
            /// Use LLM to extract only relevant portions from retrieved docs
            EnableContextualCompression: bool
            /// Max chars to keep per doc after compression
            CompressionMaxChars: int
            /// Enable parent document retrieval (store small, retrieve large)
            EnableParentDocRetrieval: bool
            /// Collection name for parent documents
            ParentCollectionName: string
            /// Enable sentence window expansion
            EnableSentenceWindow: bool
            /// Number of sentences to expand on each side
            SentenceWindowSize: int
            /// Apply time decay to fresher documents
            EnableTimeDecay: bool
            /// Half-life for time decay in days (after this, score halves)
            TimeDecayHalfLifeDays: float
            /// Enable semantic chunking (vs fixed-size)
            EnableSemanticChunking: bool
            /// Min chars per semantic chunk
            SemanticChunkMinChars: int
            /// Max chars per semantic chunk
            SemanticChunkMaxChars: int
            /// Enable cross-encoder reranking (lighter than full LLM)
            EnableCrossEncoder: bool
            /// Cross-encoder model hint
            CrossEncoderModel: string
            /// Enable automatic query routing
            EnableQueryRouting: bool
            /// Enable answer attribution tracking
            EnableAnswerAttribution: bool
            /// Enable retrieval metrics collection
            EnableMetrics: bool
            /// Shared metrics instance
            Metrics: RetrievalMetrics option
            /// Enable fallback chain when results are insufficient
            EnableFallbackChain: bool
            /// Minimum results before triggering fallback
            FallbackMinResults: int
        }

        static member Default =
            { CollectionName = "tars_context"
              TopK = 5
              MinScore = 0.3f
              AutoIndex = true
              MaxChunkChars = 800
              MaxChunks = 8
              MaxContextChars = 4000
              EnableHybridSearch = true
              SemanticWeight = 0.7f
              EnableReranking = false
              EnableQueryExpansion = false
              QueryExpansionCount = 3
              EnableMultiHop = false
              MaxHops = 2
              MetadataFilters = []
              EnableEmbeddingCache = true
              EmbeddingCacheSize = 1000
              EnableAsyncBatching = true
              BatchSize = 10
              EnableRRF = false
              RRFConstant = 60
              // Batch 2 defaults
              EnableContextualCompression = false
              CompressionMaxChars = 500
              EnableParentDocRetrieval = false
              ParentCollectionName = "tars_parents"
              EnableSentenceWindow = false
              SentenceWindowSize = 2
              EnableTimeDecay = false
              TimeDecayHalfLifeDays = 30.0
              EnableSemanticChunking = false
              SemanticChunkMinChars = 200
              SemanticChunkMaxChars = 1000
              EnableCrossEncoder = false
              CrossEncoderModel = "fast"
              EnableQueryRouting = false
              EnableAnswerAttribution = false
              EnableMetrics = false
              Metrics = None
              EnableFallbackChain = false
              FallbackMinResults = 2 }

    type MetascriptContext =
        {
            Llm: ILlmService
            Kernel: KernelContext
            Tools: ToolRegistry
            Budget: BudgetGovernor option
            /// Vector store for RAG - optional for backward compatibility
            VectorStore: IVectorStore option
            /// Knowledge graph for relationship-based context
            KnowledgeGraph: KnowledgeGraph option
            /// RAG configuration
            RagConfig: RagConfig
        }

    let private resolveVariables (text: string) (state: WorkflowState) =
        let pattern = "\{\{([^}]+)\}\}"

        Regex.Replace(
            text,
            pattern,
            fun m ->
                let key = m.Groups.[1].Value.Trim()

                if key.Contains(".") then
                    let parts = key.Split('.')
                    let stepId = parts.[0]
                    let outputName = parts.[1]

                    match state.StepOutputs.TryFind stepId with
                    | Some outputs ->
                        match outputs.TryFind outputName with
                        | Some value -> string value
                        | None -> m.Value
                    | None -> m.Value
                else
                    match state.Variables.TryFind key with
                    | Some value -> string value
                    | None -> m.Value
        )

    let private tryGetValue (path: string) (state: WorkflowState) =
        if path.Contains "." then
            let parts = path.Split('.')
            let stepId = parts.[0]
            let outputName = parts.[1]

            state.StepOutputs
            |> Map.tryFind stepId
            |> Option.bind (fun outputs -> outputs |> Map.tryFind outputName)
        else
            state.Variables |> Map.tryFind path

    let private recordBudget (budget: BudgetGovernor option) (tokens: int option) =
        match budget with
        | Some governor ->
            let tokenCost = tokens |> Option.filter (fun t -> t > 0) |> Option.defaultValue 0

            match
                governor.TryConsume
                    { Cost.Zero with
                        Tokens = tokenCost * 1<token>
                        CallCount = 1<requests> }
            with
            | Result.Ok _ -> ()
            | Result.Error e -> raise (InvalidOperationException($"Budget exceeded: {e}"))
        | None -> ()

    /// Auto-indexes content into the vector store for future retrieval
    let private chunkContent (text: string) (maxChunkChars: int) (maxChunks: int) =
        let paragraphs =
            text.Split([| "\n\n"; "\r\n\r\n" |], StringSplitOptions.RemoveEmptyEntries)

        let mutable current = System.Text.StringBuilder()
        let chunks = System.Collections.Generic.List<string>()

        for p in paragraphs do
            if current.Length + p.Length + 2 > maxChunkChars then
                if current.Length > 0 then
                    chunks.Add(current.ToString())
                    current <- System.Text.StringBuilder()

            if p.Length > maxChunkChars then
                // hard split long paragraph
                let mutable idx = 0

                while idx < p.Length && chunks.Count < maxChunks do
                    let len = min maxChunkChars (p.Length - idx)
                    chunks.Add(p.Substring(idx, len))
                    idx <- idx + len
            else if chunks.Count < maxChunks then
                if current.Length > 0 then
                    current.AppendLine() |> ignore

                current.Append(p) |> ignore

        if current.Length > 0 && chunks.Count < maxChunks then
            chunks.Add(current.ToString())

        chunks |> Seq.take maxChunks |> Seq.toList

    let private autoIndexContent
        (ctx: MetascriptContext)
        (stepId: string)
        (outputName: string)
        (content: string)
        (metadata: Map<string, string>)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if ctx.RagConfig.AutoIndex && not (String.IsNullOrWhiteSpace content) then
                match ctx.VectorStore with
                | Some vectorStore ->
                    try
                        let chunks =
                            chunkContent content ctx.RagConfig.MaxChunkChars ctx.RagConfig.MaxChunks

                        for idx, chunk in chunks |> List.indexed do
                            let! embedding = ctx.Llm.EmbedAsync(chunk)

                            let id =
                                sprintf "%s_%s_%s_%d" stepId outputName (Guid.NewGuid().ToString("N").[..6]) idx

                            let payload =
                                metadata
                                |> Map.add "content" chunk
                                |> Map.add "stepId" stepId
                                |> Map.add "outputName" outputName
                                |> Map.add "chunkIndex" (string idx)
                                |> Map.add "chunkTotal" (string chunks.Length)
                                |> Map.add "timestamp" (DateTime.UtcNow.ToString("o"))

                            do! vectorStore.SaveAsync(ctx.RagConfig.CollectionName, id, embedding, payload)

                        notes.Add($"Auto-indexed {chunks.Length} chunk(s) from {stepId}/{outputName}")
                    with ex ->
                        notes.Add($"Auto-index failed: {ex.Message}")
                | None -> ()
        }

    /// Enriches context using the knowledge graph by finding related concepts
    let private enrichWithKnowledgeGraph
        (kg: KnowledgeGraph option)
        (conceptHints: string list)
        (notes: System.Collections.Generic.List<string>)
        =
        match kg with
        | Some graph ->
            let related =
                conceptHints
                |> List.collect (fun hint ->
                    let node = GraphNode.Concept hint
                    let neighbors = graph.GetNeighbors(node)

                    neighbors
                    |> List.choose (fun (neighborNode, edge) ->
                        match neighborNode with
                        | GraphNode.Concept name ->
                            let weight =
                                match edge with
                                | GraphEdge.RelatesTo w -> w
                                | _ -> 0.5

                            Some(name, weight)
                        | _ -> None)
                    |> List.filter (fun (_, w) -> w > 0.3))
                |> List.distinctBy fst
                |> List.sortByDescending snd
                |> List.truncate 10

            if not (List.isEmpty related) then
                notes.Add($"Knowledge graph: found {List.length related} related concepts")

                let conceptStr =
                    related
                    |> List.map (fun (name, weight) -> sprintf "- %s (relevance: %.2f)" name weight)
                    |> String.concat "\n"

                sprintf "\nRelated concepts from knowledge graph:\n%s\n" conceptStr
            else
                ""
        | None -> ""

    /// Simple BM25-like keyword scoring for hybrid search
    let private computeKeywordScore (query: string) (content: string) =
        if String.IsNullOrWhiteSpace query || String.IsNullOrWhiteSpace content then
            0.0f
        else
            let queryTerms =
                query
                    .ToLowerInvariant()
                    .Split([| ' '; ','; '.'; '!'; '?'; ';'; ':' |], StringSplitOptions.RemoveEmptyEntries)
                |> Array.distinct

            let contentLower = content.ToLowerInvariant()
            let contentLen = float32 content.Length
            let avgDocLen = 500.0f // assumed average document length
            let k1 = 1.2f
            let b = 0.75f

            let mutable score = 0.0f

            for term in queryTerms do
                // Count term frequency
                let mutable tf = 0
                let mutable idx = 0

                while idx >= 0 do
                    idx <- contentLower.IndexOf(term, idx)

                    if idx >= 0 then
                        tf <- tf + 1
                        idx <- idx + 1

                if tf > 0 then
                    // Simplified BM25 scoring (without IDF since we don't have corpus stats)
                    let tfNorm =
                        (float32 tf * (k1 + 1.0f))
                        / (float32 tf + k1 * (1.0f - b + b * contentLen / avgDocLen))

                    score <- score + tfNorm

            // Normalize by number of query terms
            if queryTerms.Length > 0 then
                score / float32 queryTerms.Length
            else
                0.0f

    /// Combine semantic and keyword scores for hybrid ranking
    let private hybridScore (semanticScore: float32) (keywordScore: float32) (semanticWeight: float32) =
        // Normalize keyword score to 0-1 range (cap at reasonable max)
        let normalizedKeyword = min 1.0f (keywordScore / 2.0f)
        semanticWeight * semanticScore + (1.0f - semanticWeight) * normalizedKeyword

    /// Rerank results using LLM for better relevance (optional, slower)
    let private rerankWithLlm
        (llm: ILlmService)
        (query: string)
        (results: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if List.isEmpty results then
                return results
            else
                // Build a prompt asking LLM to rank the documents
                let docsText =
                    results
                    |> List.mapi (fun i (id, _, payload) ->
                        let content = payload |> Map.tryFind "content" |> Option.defaultValue ""

                        let truncated =
                            if content.Length > 300 then
                                content.[..297] + "..."
                            else
                                content

                        sprintf "[%d] %s" (i + 1) truncated)
                    |> String.concat "\n\n"

                let prompt =
                    sprintf
                        "Given the query: \"%s\"\n\nRank these documents by relevance (most relevant first). Return ONLY a comma-separated list of document numbers.\nExample: 3,1,4,2\n\nDocuments:\n%s\n\nRanking:"
                        query
                        docsText

                let req =
                    { ModelHint = Some "fast"
                      MaxTokens = Some 50
                      Temperature = Some 0.0
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                try
                    let! response = llm.CompleteAsync req
                    let rankStr = response.Text.Trim()

                    // Parse ranking like "3,1,4,2"
                    let indices =
                        rankStr.Split(',')
                        |> Array.choose (fun s ->
                            match Int32.TryParse(s.Trim()) with
                            | true, v when v >= 1 && v <= results.Length -> Some(v - 1)
                            | _ -> None)
                        |> Array.distinct
                        |> Array.toList

                    notes.Add($"Reranked {results.Length} results via LLM")

                    // Reorder results based on LLM ranking, append any not mentioned
                    let reranked = indices |> List.choose (fun i -> results |> List.tryItem i)

                    let remaining =
                        results
                        |> List.indexed
                        |> List.filter (fun (i, _) -> not (List.contains i indices))
                        |> List.map snd

                    return reranked @ remaining
                with ex ->
                    notes.Add($"Reranking failed: {ex.Message}")
                    return results
        }

    // ========== EMBEDDING CACHE ==========
    /// Thread-safe LRU cache for embeddings
    module private EmbeddingCache =
        open System.Collections.Concurrent

        let private cache = ConcurrentDictionary<string, float32[]>()
        let private accessOrder = ConcurrentDictionary<string, int64>()
        let mutable private accessCounter = 0L

        let tryGet (key: string) =
            match cache.TryGetValue(key) with
            | true, embedding ->
                accessOrder.[key] <- System.Threading.Interlocked.Increment(&accessCounter)
                Some embedding
            | false, _ -> None

        let set (key: string) (embedding: float32[]) (maxSize: int) =
            // Evict oldest if at capacity
            if cache.Count >= maxSize then
                let oldest = accessOrder |> Seq.sortBy (fun kv -> kv.Value) |> Seq.tryHead

                match oldest with
                | Some kv ->
                    cache.TryRemove(kv.Key) |> ignore
                    accessOrder.TryRemove(kv.Key) |> ignore
                | None -> ()

            cache.[key] <- embedding
            accessOrder.[key] <- System.Threading.Interlocked.Increment(&accessCounter)

        let clear () =
            cache.Clear()
            accessOrder.Clear()

    /// Get embedding with caching
    let private getEmbeddingCached (llm: ILlmService) (config: RagConfig) (text: string) =
        task {
            if config.EnableEmbeddingCache then
                let cacheKey = text.GetHashCode().ToString()

                match EmbeddingCache.tryGet cacheKey with
                | Some cached -> return cached
                | None ->
                    let! embedding = llm.EmbedAsync(text)
                    EmbeddingCache.set cacheKey embedding config.EmbeddingCacheSize
                    return embedding
            else
                return! llm.EmbedAsync(text)
        }

    // ========== ASYNC BATCHING ==========
    /// Embed multiple texts in parallel batches
    let private embedBatch (llm: ILlmService) (config: RagConfig) (texts: string list) =
        task {
            if config.EnableAsyncBatching && texts.Length > 1 then
                let batches = texts |> List.chunkBySize config.BatchSize

                let! results =
                    batches
                    |> List.map (fun batch ->
                        task {
                            let! embeddings =
                                batch |> List.map (fun t -> getEmbeddingCached llm config t) |> Task.WhenAll

                            return embeddings |> Array.toList
                        })
                    |> Task.WhenAll

                return results |> Array.toList |> List.concat
            else
                let! embeddings = texts |> List.map (fun t -> getEmbeddingCached llm config t) |> Task.WhenAll
                return embeddings |> Array.toList
        }

    // ========== QUERY EXPANSION ==========
    /// Use LLM to generate expanded/related queries
    let private expandQuery
        (llm: ILlmService)
        (config: RagConfig)
        (query: string)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if not config.EnableQueryExpansion then
                return [ query ]
            else
                let prompt =
                    sprintf
                        "Generate %d alternative search queries for: \"%s\"\n\nReturn ONLY the queries, one per line, no numbering or explanations."
                        config.QueryExpansionCount
                        query

                let req =
                    { ModelHint = Some "fast"
                      MaxTokens = Some 150
                      Temperature = Some 0.7
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                try
                    let! response = llm.CompleteAsync req

                    let expanded =
                        response.Text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.map (fun s -> s.Trim().TrimStart([| '-'; '*'; '1'; '2'; '3'; '4'; '5'; '.'; ' ' |]))
                        |> Array.filter (fun s -> not (String.IsNullOrWhiteSpace s))
                        |> Array.truncate config.QueryExpansionCount
                        |> Array.toList

                    notes.Add($"Query expanded to {expanded.Length + 1} variants")
                    return query :: expanded
                with ex ->
                    notes.Add($"Query expansion failed: {ex.Message}")
                    return [ query ]
        }

    // ========== METADATA FILTERING ==========
    /// Apply metadata filters to results
    let private applyMetadataFilters
        (filters: MetadataFilter list)
        (results: (string * float32 * Map<string, string>) list)
        =
        if List.isEmpty filters then
            results
        else
            results
            |> List.filter (fun (_, _, payload) ->
                filters
                |> List.forall (fun filter ->
                    match payload |> Map.tryFind filter.Field with
                    | None -> filter.Operator = "ne" // missing field passes "not equal" check
                    | Some value ->
                        match filter.Operator.ToLower() with
                        | "eq" -> value = filter.Value
                        | "ne" -> value <> filter.Value
                        | "contains" -> value.ToLower().Contains(filter.Value.ToLower())
                        | "gt" ->
                            match Double.TryParse value, Double.TryParse filter.Value with
                            | (true, v1), (true, v2) -> v1 > v2
                            | _ -> value > filter.Value
                        | "lt" ->
                            match Double.TryParse value, Double.TryParse filter.Value with
                            | (true, v1), (true, v2) -> v1 < v2
                            | _ -> value < filter.Value
                        | "gte" ->
                            match Double.TryParse value, Double.TryParse filter.Value with
                            | (true, v1), (true, v2) -> v1 >= v2
                            | _ -> value >= filter.Value
                        | "lte" ->
                            match Double.TryParse value, Double.TryParse filter.Value with
                            | (true, v1), (true, v2) -> v1 <= v2
                            | _ -> value <= filter.Value
                        | _ -> true))

    // ========== MULTI-HOP RETRIEVAL ==========
    /// Perform multi-hop retrieval using knowledge graph
    let private multiHopRetrieval
        (vectorStore: IVectorStore)
        (llm: ILlmService)
        (kg: KnowledgeGraph option)
        (config: RagConfig)
        (query: string)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            match kg with
            | None -> return []
            | Some graph when not config.EnableMultiHop -> return []
            | Some graph ->
                // Extract key concepts from query
                let queryTerms =
                    query.ToLowerInvariant().Split([| ' '; ','; '.'; '?'; '!' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.filter (fun t -> t.Length > 3) // Skip short words
                    |> Array.map (fun t -> GraphNode.Concept t)
                    |> Array.toList

                // Helper to extract name from GraphNode for deduplication
                let nodeKey (node: GraphNode) =
                    match node with
                    | GraphNode.Concept name -> name
                    | GraphNode.AgentNode id -> id.ToString()
                    | GraphNode.FileNode path -> path
                    | GraphNode.TaskNode id -> id.ToString()

                // Find related concepts via BFS up to MaxHops
                let mutable visited = Set.empty<string>
                let mutable frontier = queryTerms |> List.map nodeKey |> Set.ofList
                let mutable frontierNodes = queryTerms
                let mutable relatedConcepts = []

                for hop in 1 .. config.MaxHops do
                    let mutable nextFrontier = Set.empty<string>
                    let mutable nextFrontierNodes = []

                    for concept in frontierNodes do
                        let key = nodeKey concept

                        if not (visited.Contains key) then
                            visited <- visited.Add key
                            let neighbors = graph.GetNeighbors concept

                            for (neighbor, edge) in neighbors do
                                let neighborKey = nodeKey neighbor

                                let weight =
                                    match edge with
                                    | GraphEdge.RelatesTo w -> w
                                    | _ -> 0.5

                                if weight > 0.3 && not (visited.Contains neighborKey) then
                                    relatedConcepts <- (neighbor, weight, hop) :: relatedConcepts

                                    if not (nextFrontier.Contains neighborKey) then
                                        nextFrontier <- nextFrontier.Add neighborKey
                                        nextFrontierNodes <- neighbor :: nextFrontierNodes

                    frontierNodes <- nextFrontierNodes

                // Retrieve documents for top related concepts
                let topConcepts =
                    relatedConcepts
                    |> List.sortByDescending (fun (_, w, _) -> w)
                    |> List.truncate 5
                    |> List.map (fun (c, _, _) -> nodeKey c)

                if List.isEmpty topConcepts then
                    return []
                else
                    notes.Add($"Multi-hop: exploring {topConcepts.Length} related concepts")

                    // Embed and search for each concept
                    let! allResults =
                        topConcepts
                        |> List.map (fun concept ->
                            task {
                                let! embedding = getEmbeddingCached llm config concept

                                let! results =
                                    vectorStore.SearchAsync(config.CollectionName, embedding, config.TopK / 2)

                                return results |> List.map (fun (id, d, p) -> (id, d, p, concept))
                            })
                        |> Task.WhenAll

                    // Flatten and dedupe by id
                    let hopResults =
                        allResults
                        |> Array.toList
                        |> List.concat
                        |> List.distinctBy (fun (id, _, _, _) -> id)
                        |> List.map (fun (id, d, p, _) -> (id, d, p))

                    notes.Add($"Multi-hop: found {hopResults.Length} additional results")
                    return hopResults
        }

    // ========== RECIPROCAL RANK FUSION ==========
    /// Combine multiple result lists using RRF
    let private reciprocalRankFusion (resultLists: (string * float32 * Map<string, string>) list list) (k: int) =
        // RRF score = sum(1 / (k + rank)) for each list
        let scores = System.Collections.Generic.Dictionary<string, float32>()
        let payloads = System.Collections.Generic.Dictionary<string, Map<string, string>>()

        for results in resultLists do
            results
            |> List.iteri (fun rank (id, _, payload) ->
                let rrfScore = 1.0f / (float32 k + float32 (rank + 1))

                match scores.TryGetValue(id) with
                | true, existing -> scores.[id] <- existing + rrfScore
                | false, _ -> scores.[id] <- rrfScore

                payloads.[id] <- payload)

        // Sort by combined score descending
        scores
        |> Seq.map (fun kv -> (kv.Key, 1.0f - kv.Value, payloads.[kv.Key])) // Convert to distance format
        |> Seq.sortBy (fun (_, dist, _) -> dist)
        |> Seq.toList

    // ========== QUERY ROUTING ==========
    /// Classify query type for routing to appropriate retrieval strategy
    let private classifyQuery (query: string) : QueryType =
        let q = query.ToLowerInvariant()

        let words =
            q.Split([| ' '; ','; '.'; '?'; '!' |], StringSplitOptions.RemoveEmptyEntries)

        // Keyword-heavy: mostly nouns/proper nouns, no question words
        let questionWords =
            [| "what"
               "why"
               "how"
               "when"
               "where"
               "who"
               "which"
               "explain"
               "describe" |]

        let hasQuestionWord = questionWords |> Array.exists (fun w -> q.Contains(w))

        // Analytical: contains reasoning indicators
        let analyticalWords =
            [| "analyze"
               "compare"
               "evaluate"
               "explain why"
               "reason"
               "implications"
               "impact" |]

        let isAnalytical = analyticalWords |> Array.exists (fun w -> q.Contains(w))

        // Conversational: short, pronouns, follow-up indicators
        let conversationalWords =
            [| "it"; "this"; "that"; "those"; "more"; "also"; "another" |]

        let isConversational =
            words.Length < 5
            && conversationalWords |> Array.exists (fun w -> words |> Array.contains w)

        if isConversational then
            QueryType.Conversational
        elif isAnalytical then
            QueryType.Analytical
        elif hasQuestionWord then
            QueryType.Factual
        elif words.Length >= 3 && not hasQuestionWord then
            QueryType.Keyword
        else
            QueryType.Unknown

    /// Get retrieval strategy based on query type
    let private getStrategyForQueryType (queryType: QueryType) (config: RagConfig) =
        match queryType with
        | QueryType.Factual ->
            // Factual: prioritize semantic search, lower TopK for precision
            { config with
                SemanticWeight = 0.8f
                TopK = min config.TopK 5 }
        | QueryType.Analytical ->
            // Analytical: broader search, enable multi-hop if available
            { config with
                TopK = config.TopK + 3
                EnableMultiHop = true }
        | QueryType.Conversational ->
            // Conversational: smaller context, faster
            { config with
                TopK = min config.TopK 3
                EnableReranking = false }
        | QueryType.Keyword ->
            // Keyword: boost keyword weight in hybrid search
            { config with
                SemanticWeight = 0.4f
                EnableHybridSearch = true }
        | QueryType.Unknown -> config

    // ========== TIME DECAY SCORING ==========
    /// Apply time decay to score based on document age
    let private applyTimeDecay (config: RagConfig) (results: (string * float32 * Map<string, string>) list) =
        if not config.EnableTimeDecay then
            results
        else
            let now = DateTime.UtcNow
            let halfLifeMs = config.TimeDecayHalfLifeDays * 24.0 * 60.0 * 60.0 * 1000.0

            results
            |> List.map (fun (id, distance, payload) ->
                let ageMs =
                    payload
                    |> Map.tryFind "timestamp"
                    |> Option.bind (fun ts ->
                        match DateTime.TryParse(ts) with
                        | true, dt -> Some (now - dt).TotalMilliseconds
                        | _ -> None)
                    |> Option.defaultValue 0.0

                // Exponential decay: score * 0.5^(age/halfLife)
                let decayFactor = Math.Pow(0.5, ageMs / halfLifeMs) |> float32
                let similarity = 1.0f - distance
                let decayedSimilarity = similarity * (0.5f + 0.5f * decayFactor) // Blend to avoid too harsh decay
                (id, 1.0f - decayedSimilarity, payload))
            |> List.sortBy (fun (_, dist, _) -> dist)

    // ========== CONTEXTUAL COMPRESSION ==========
    /// Use LLM to extract only relevant portions from retrieved content
    let private compressContext
        (llm: ILlmService)
        (config: RagConfig)
        (query: string)
        (results: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if not config.EnableContextualCompression || List.isEmpty results then
                return results
            else
                notes.Add($"Compression: processing {results.Length} results")

                let! compressed =
                    results
                    |> List.map (fun (id, distance, payload) ->
                        task {
                            let content = payload |> Map.tryFind "content" |> Option.defaultValue ""

                            if content.Length <= config.CompressionMaxChars then
                                return (id, distance, payload)
                            else
                                let prompt =
                                    sprintf
                                        "Extract ONLY the parts relevant to this query: \"%s\"\n\nDocument:\n%s\n\nReturn only the relevant excerpts, nothing else. Max %d characters."
                                        query
                                        content
                                        config.CompressionMaxChars

                                let req =
                                    { ModelHint = Some "fast"
                                      MaxTokens = Some(config.CompressionMaxChars / 3)
                                      Temperature = Some 0.0
                                      Messages = [ { Role = Role.User; Content = prompt } ] }

                                try
                                    let! response = llm.CompleteAsync req
                                    let compressed = response.Text.Trim()

                                    let newPayload =
                                        payload |> Map.add "content" compressed |> Map.add "compressed" "true"

                                    return (id, distance, newPayload)
                                with _ ->
                                    return (id, distance, payload)
                        })
                    |> Task.WhenAll

                notes.Add($"Compression: completed")
                return compressed |> Array.toList
        }

    // ========== SEMANTIC CHUNKING ==========
    /// Split text into semantic chunks based on paragraph/section boundaries
    let private semanticChunk (config: RagConfig) (text: string) : string list =
        if not config.EnableSemanticChunking then
            // Fall back to fixed-size chunking
            let rec chunk (s: string) acc =
                if s.Length <= config.MaxChunkChars then
                    List.rev (s :: acc)
                else
                    let breakPoint =
                        // Try to break at sentence boundary
                        let sentenceEnd =
                            s.LastIndexOfAny([| '.'; '!'; '?' |], min (config.MaxChunkChars - 1) (s.Length - 1))

                        if sentenceEnd > config.MaxChunkChars / 2 then
                            sentenceEnd + 1
                        else
                            config.MaxChunkChars

                    chunk (s.Substring(breakPoint).TrimStart()) (s.Substring(0, breakPoint) :: acc)

            chunk text []
        else
            // Split by double newlines (paragraphs) first
            let paragraphs =
                text.Split([| "\n\n"; "\r\n\r\n" |], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun s -> s.Trim())
                |> Array.filter (fun s -> s.Length > 0)

            // Merge small paragraphs, split large ones
            let mutable chunks = []
            let mutable current = ""

            for para in paragraphs do
                if current.Length + para.Length + 2 <= config.SemanticChunkMaxChars then
                    current <- if current.Length = 0 then para else current + "\n\n" + para
                else
                    if current.Length >= config.SemanticChunkMinChars then
                        chunks <- current :: chunks

                    current <- para

                // Split if paragraph itself is too large
                while current.Length > config.SemanticChunkMaxChars do
                    let breakPoint = current.LastIndexOf('.', config.SemanticChunkMaxChars - 1)

                    let bp =
                        if breakPoint > config.SemanticChunkMinChars then
                            breakPoint + 1
                        else
                            config.SemanticChunkMaxChars

                    chunks <- current.Substring(0, bp) :: chunks
                    current <- current.Substring(bp).TrimStart()

            if current.Length > 0 then
                chunks <- current :: chunks

            chunks |> List.rev

    // ========== SENTENCE WINDOW RETRIEVAL ==========
    /// Expand retrieved content to include surrounding sentences
    let private expandSentenceWindow
        (config: RagConfig)
        (results: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        =
        if not config.EnableSentenceWindow then
            results
        else
            results
            |> List.map (fun (id, distance, payload) ->
                let content = payload |> Map.tryFind "content" |> Option.defaultValue ""
                let fullText = payload |> Map.tryFind "fullText" |> Option.defaultValue content

                if fullText = content || String.IsNullOrEmpty fullText then
                    (id, distance, payload)
                else
                    // Find content position in fullText and expand
                    let idx = fullText.IndexOf(content)

                    if idx < 0 then
                        (id, distance, payload)
                    else
                        // Find sentence boundaries
                        let sentences =
                            fullText.Split([| '.'; '!'; '?' |], StringSplitOptions.RemoveEmptyEntries)
                            |> Array.map (fun s -> s.Trim() + ".")

                        // Find which sentence contains our content
                        let mutable pos = 0
                        let mutable sentenceIdx = 0

                        for i, s in sentences |> Array.indexed do
                            if pos <= idx && idx < pos + s.Length then
                                sentenceIdx <- i

                            pos <- pos + s.Length

                        // Expand window
                        let startIdx = max 0 (sentenceIdx - config.SentenceWindowSize)
                        let endIdx = min (sentences.Length - 1) (sentenceIdx + config.SentenceWindowSize)
                        let expanded = sentences.[startIdx..endIdx] |> String.concat " "

                        notes.Add($"SentenceWindow: expanded from {content.Length} to {expanded.Length} chars")

                        let newPayload =
                            payload |> Map.add "content" expanded |> Map.add "windowExpanded" "true"

                        (id, distance, newPayload))

    // ========== PARENT DOCUMENT RETRIEVAL ==========
    /// Retrieve parent documents for small chunks
    let private retrieveParentDocs
        (vectorStore: IVectorStore)
        (config: RagConfig)
        (results: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if not config.EnableParentDocRetrieval || List.isEmpty results then
                return results
            else
                let! enriched =
                    results
                    |> List.map (fun (id, distance, payload) ->
                        task {
                            match payload |> Map.tryFind "parentId" with
                            | Some parentId ->
                                // Search for parent in parent collection
                                let! parentResults =
                                    vectorStore.SearchAsync(config.ParentCollectionName, [| 0.0f |], 1)
                                // Note: This is a simplified lookup - in practice you'd want a direct ID lookup
                                let parent = parentResults |> List.tryFind (fun (pid, _, _) -> pid = parentId)

                                match parent with
                                | Some(_, _, parentPayload) ->
                                    let parentContent =
                                        parentPayload |> Map.tryFind "content" |> Option.defaultValue ""

                                    notes.Add($"ParentDoc: enriched {id} with parent {parentId}")

                                    let newPayload =
                                        payload |> Map.add "content" parentContent |> Map.add "parentRetrieved" "true"

                                    return (id, distance, newPayload)
                                | None -> return (id, distance, payload)
                            | None -> return (id, distance, payload)
                        })
                    |> Task.WhenAll

                return enriched |> Array.toList
        }

    // ========== CROSS-ENCODER RERANKING ==========
    /// Rerank using cross-encoder (lighter than full LLM reranking)
    let private crossEncoderRerank
        (llm: ILlmService)
        (config: RagConfig)
        (query: string)
        (results: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if not config.EnableCrossEncoder || List.isEmpty results then
                return results
            else
                notes.Add($"CrossEncoder: reranking {results.Length} results")
                // Use a simpler prompt for cross-encoder-style scoring
                let! scored =
                    results
                    |> List.map (fun (id, distance, payload) ->
                        task {
                            let content = payload |> Map.tryFind "content" |> Option.defaultValue ""

                            let prompt =
                                sprintf
                                    "Rate relevance 0-10 of this document to query: \"%s\"\nDocument: %s\nScore (just the number):"
                                    query
                                    (content.Substring(0, min 500 content.Length))

                            let req =
                                { ModelHint = Some config.CrossEncoderModel
                                  MaxTokens = Some 5
                                  Temperature = Some 0.0
                                  Messages = [ { Role = Role.User; Content = prompt } ] }

                            try
                                let! response = llm.CompleteAsync req
                                let scoreText = response.Text.Trim()

                                match Double.TryParse(scoreText.Replace(",", ".")) with
                                | true, score -> return (id, 1.0f - (float32 score / 10.0f), payload)
                                | _ -> return (id, distance, payload)
                            with _ ->
                                return (id, distance, payload)
                        })
                    |> Task.WhenAll

                let reranked = scored |> Array.sortBy (fun (_, dist, _) -> dist) |> Array.toList
                notes.Add($"CrossEncoder: completed")
                return reranked
        }

    // ========== ANSWER ATTRIBUTION ==========
    /// Track which chunks were used in the answer
    let private attributeSources
        (results: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        : Map<string, obj> =
        let attributions =
            results
            |> List.mapi (fun i (id, distance, payload) ->
                let source = payload |> Map.tryFind "source" |> Option.defaultValue id
                let chunk = payload |> Map.tryFind "chunkIndex" |> Option.defaultValue ""

                {| Index = i + 1
                   Id = id
                   Source = source
                   ChunkIndex = chunk
                   Score = 1.0f - distance |}
                :> obj)

        notes.Add($"Attribution: tracking {attributions.Length} sources")
        Map [ "attributions", box attributions ]

    // ========== FALLBACK CHAIN ==========
    /// Execute fallback retrieval when primary results are insufficient
    let private executeFallback
        (vectorStore: IVectorStore)
        (llm: ILlmService)
        (kg: KnowledgeGraph option)
        (config: RagConfig)
        (query: string)
        (currentResults: (string * float32 * Map<string, string>) list)
        (notes: System.Collections.Generic.List<string>)
        =
        task {
            if
                not config.EnableFallbackChain
                || currentResults.Length >= config.FallbackMinResults
            then
                return currentResults
            else
                notes.Add($"Fallback: triggered (only {currentResults.Length} results)")

                // Fallback 1: Try with lower min score
                let! fallback1 =
                    task {
                        let! embedding = llm.EmbedAsync(query)
                        let! results = vectorStore.SearchAsync(config.CollectionName, embedding, config.TopK * 2)
                        return results |> List.filter (fun (_, d, _) -> (1.0f - d) >= config.MinScore * 0.5f)
                    }

                if fallback1.Length >= config.FallbackMinResults then
                    notes.Add($"Fallback: lower threshold yielded {fallback1.Length} results")
                    return fallback1
                else
                    // Fallback 2: Try knowledge graph exploration
                    let! kgResults =
                        multiHopRetrieval
                            vectorStore
                            llm
                            kg
                            { config with
                                EnableMultiHop = true
                                MaxHops = 3 }
                            query
                            notes

                    let combined =
                        (currentResults @ fallback1 @ kgResults)
                        |> List.distinctBy (fun (id, _, _) -> id)
                        |> List.sortBy (fun (_, d, _) -> d)

                    notes.Add($"Fallback: final combined count = {combined.Length}")

                    config.Metrics
                    |> Option.iter (fun m -> System.Threading.Interlocked.Increment(&m.FallbackTriggered) |> ignore)

                    return combined
        }

    // ========== METRICS COLLECTION ==========
    /// Record retrieval metrics
    let private recordMetrics (config: RagConfig) (startTime: DateTime) (resultCount: int) (cacheHit: bool) =
        config.Metrics
        |> Option.iter (fun m ->
            System.Threading.Interlocked.Increment(&m.TotalQueries) |> ignore
            let elapsed = (DateTime.UtcNow - startTime).TotalMilliseconds |> int64
            System.Threading.Interlocked.Add(&m.TotalLatencyMs, elapsed) |> ignore

            if cacheHit then
                System.Threading.Interlocked.Increment(&m.CacheHits) |> ignore
            else
                System.Threading.Interlocked.Increment(&m.CacheMisses) |> ignore
            // Update average (approximate)
            let total = m.TotalQueries

            if total > 0L then
                m.AvgResultCount <- (m.AvgResultCount * float (total - 1L) + float resultCount) / float total)

    let executeStep
        (ctx: MetascriptContext)
        (step: WorkflowStep)
        (state: WorkflowState)
        : Task<Map<string, obj> * string list> =
        task {
            let notes = System.Collections.Generic.List<string>()

            match step.Type.ToLower() with
            | "agent" ->
                let instruction = resolveVariables (defaultArg step.Instruction "") state
                notes.Add($"Instruction resolved length={instruction.Length}")

                // Gather context from previous steps
                let contextStr =
                    defaultArg step.Context []
                    |> List.map (fun c ->
                        match state.StepOutputs.TryFind c.StepId with
                        | Some outputs ->
                            match outputs.TryFind c.OutputName with
                            | Some value -> sprintf "Context from %s (%s):\n%s" c.StepId c.OutputName (string value)
                            | None -> ""
                        | None -> "")
                    |> String.concat "\n\n"

                // Extract concept hints from params for knowledge graph lookup
                let conceptHints =
                    step.Params
                    |> Option.defaultValue Map.empty
                    |> Map.tryFind "concepts"
                    |> Option.map (fun s -> s.Split(',') |> Array.map (fun c -> c.Trim()) |> Array.toList)
                    |> Option.defaultValue []

                // Enrich with knowledge graph if concepts are specified
                let kgContext = enrichWithKnowledgeGraph ctx.KnowledgeGraph conceptHints notes

                let prompt =
                    sprintf
                        """You are %s.
Instruction: %s

%s%s

Output only the requested result."""
                        (defaultArg step.Agent "Assistant")
                        instruction
                        contextStr
                        kgContext

                let req =
                    { ModelHint = Some "reasoning"
                      MaxTokens = None
                      Temperature = None
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = ctx.Llm.CompleteAsync req

                response.Usage
                |> Option.iter (fun u -> notes.Add($"Tokens prompt={u.PromptTokens} completion={u.CompletionTokens}"))

                recordBudget ctx.Budget (response.Usage |> Option.map (fun u -> u.TotalTokens))

                // Assume the first output is the main text response
                let outputName =
                    match defaultArg step.Outputs [] with
                    | head :: _ -> head
                    | [] -> "output"

                // Auto-index the output for future retrieval
                let metadata =
                    Map
                        [ "agent", defaultArg step.Agent "Assistant"
                          "instruction", instruction.[.. min 200 (instruction.Length - 1)]
                          "source", "agent_output" ]

                do! autoIndexContent ctx step.Id outputName response.Text metadata notes

                return (Map [ outputName, box response.Text ], List.ofSeq notes)

            | "tool" ->
                match step.Tool with
                | Some toolName ->
                    match ctx.Tools.Get(toolName) with
                    | Some tool ->
                        let args =
                            step.Params
                            |> Option.defaultValue Map.empty
                            |> Map.map (fun _ v -> resolveVariables v state)

                        let input =
                            if args.ContainsKey("input") then
                                args["input"]
                            elif args.ContainsKey("command") then
                                args["command"]
                            else
                                // Fallback: Serialize to JSON
                                System.Text.Json.JsonSerializer.Serialize(args)

                        let! result = tool.Execute(input)

                        match result with
                        | Result.Ok s -> return (Map [ "stdout", box s ], List.ofSeq notes)
                        | Result.Error e -> return (Map [ "error", box e ], List.ofSeq notes)
                    | None -> return (Map [ "error", box (sprintf "Tool '%s' not found" toolName) ], List.ofSeq notes)
                | None -> return (Map.empty, List.ofSeq notes)

            | "loop" ->
                let stepParams = step.Params |> Option.defaultValue Map.empty
                let listKey = stepParams |> Map.tryFind "list" |> Option.defaultValue ""
                let itemVar = stepParams |> Map.tryFind "itemVar" |> Option.defaultValue "item"

                let maxIterations =
                    stepParams
                    |> Map.tryFind "maxIterations"
                    |> Option.bind (fun s ->
                        match System.Int32.TryParse s with
                        | true, v -> Some v
                        | _ -> None)
                    |> Option.defaultValue 100

                let collection =
                    if listKey <> "" then
                        match tryGetValue listKey state with
                        | Some(:? System.Collections.IEnumerable as e) -> e |> Seq.cast<obj> |> Seq.toList
                        | Some value -> [ value ]
                        | None ->
                            try
                                let doc = System.Text.Json.JsonDocument.Parse(listKey)

                                if doc.RootElement.ValueKind = System.Text.Json.JsonValueKind.Array then
                                    doc.RootElement.EnumerateArray()
                                    |> Seq.map (fun el -> el.ToString() :> obj)
                                    |> Seq.toList
                                else
                                    []
                            with _ ->
                                if listKey.Contains(",") then
                                    listKey.Split(',') |> Array.map (fun s -> s.Trim() :> obj) |> Array.toList
                                else
                                    []
                    else
                        []

                let mutable outputs = []
                let mutable idx = 0

                for item in collection do
                    if idx < maxIterations then
                        let mutable itemState =
                            { state with
                                Variables = state.Variables.Add(itemVar, item) }

                        let instruction = resolveVariables (defaultArg step.Instruction "") itemState

                        let prompt =
                            sprintf
                                """You are %s.
Instruction: %s"""
                                (defaultArg step.Agent "Assistant")
                                instruction

                        let req =
                            { ModelHint = Some "reasoning"
                              MaxTokens = None
                              Temperature = None
                              Messages = [ { Role = Role.User; Content = prompt } ] }

                        let! response = ctx.Llm.CompleteAsync req
                        recordBudget ctx.Budget (response.Usage |> Option.map (fun u -> u.TotalTokens))
                        outputs <- outputs @ [ response.Text :> obj ]
                        idx <- idx + 1
                    else
                        notes.Add($"Loop truncated at {maxIterations} iterations")

                let outputName =
                    match defaultArg step.Outputs [] with
                    | head :: _ -> head
                    | [] -> "items"

                return (Map [ outputName, box outputs ], List.ofSeq notes)

            | "decision" ->
                let stepParams = step.Params |> Option.defaultValue Map.empty
                let condition = stepParams |> Map.tryFind "condition" |> Option.defaultValue ""

                let conditionValue =
                    if condition.Contains("==") then
                        let parts = condition.Split("==", System.StringSplitOptions.RemoveEmptyEntries)

                        if parts.Length = 2 then
                            let left = resolveVariables (parts[0].Trim()) state
                            let right = resolveVariables (parts[1].Trim()) state
                            left.Trim().Equals(right.Trim(), StringComparison.OrdinalIgnoreCase)
                        else
                            false
                    else
                        let value = resolveVariables condition state

                        match Boolean.TryParse value with
                        | true, v -> v
                        | _ -> not (String.IsNullOrWhiteSpace value)

                let outputName =
                    match defaultArg step.Outputs [] with
                    | head :: _ -> head
                    | [] -> "decision"

                let trueOut = stepParams |> Map.tryFind "trueOutput" |> Option.defaultValue "true"

                let falseOut =
                    stepParams |> Map.tryFind "falseOutput" |> Option.defaultValue "false"

                return (Map [ outputName, if conditionValue then box trueOut else box falseOut ], List.ofSeq notes)

            | "retrieval" ->
                // RAG step: Advanced retrieval with all features
                match ctx.VectorStore with
                | Some vectorStore ->
                    let startTime = DateTime.UtcNow
                    let stepParams = step.Params |> Option.defaultValue Map.empty
                    let query = stepParams |> Map.tryFind "query" |> Option.defaultValue ""
                    let resolvedQuery = resolveVariables query state

                    if String.IsNullOrWhiteSpace resolvedQuery then
                        notes.Add("Retrieval: empty query, skipping")
                        return (Map [ "context", box ""; "results", box []; "attributions", box [] ], List.ofSeq notes)
                    else
                        notes.Add($"Retrieval: query length={resolvedQuery.Length}")

                        // Step 0: Query routing (adjust config based on query type)
                        let effectiveConfig =
                            if ctx.RagConfig.EnableQueryRouting then
                                let queryType = classifyQuery resolvedQuery
                                notes.Add($"Routing: classified as {queryType}")
                                getStrategyForQueryType queryType ctx.RagConfig
                            else
                                ctx.RagConfig

                        // Step 1: Query expansion (generate related queries)
                        let! expandedQueries = expandQuery ctx.Llm effectiveConfig resolvedQuery notes

                        // Step 2: Embed all queries (with caching and batching)
                        let! queryEmbeddings = embedBatch ctx.Llm effectiveConfig expandedQueries

                        let topK =
                            stepParams
                            |> Map.tryFind "topK"
                            |> Option.bind (fun s ->
                                match Int32.TryParse s with
                                | true, v -> Some v
                                | _ -> None)
                            |> Option.defaultValue effectiveConfig.TopK

                        let fetchK =
                            if effectiveConfig.EnableHybridSearch || effectiveConfig.EnableRRF then
                                topK * 3
                            else
                                topK * 2

                        let collectionName =
                            stepParams
                            |> Map.tryFind "collection"
                            |> Option.defaultValue effectiveConfig.CollectionName

                        // Step 3: Search with each query embedding
                        let! allSearchResults =
                            queryEmbeddings
                            |> List.map (fun emb -> vectorStore.SearchAsync(collectionName, emb, fetchK))
                            |> Task.WhenAll

                        // Step 4: Multi-hop retrieval (explore knowledge graph)
                        let! multiHopResults =
                            multiHopRetrieval vectorStore ctx.Llm ctx.KnowledgeGraph effectiveConfig resolvedQuery notes

                        // Step 5: Combine results using RRF or simple merge
                        let allResultLists =
                            (allSearchResults |> Array.toList) @ [ multiHopResults ]
                            |> List.filter (not << List.isEmpty)

                        let combinedResults =
                            if effectiveConfig.EnableRRF && allResultLists.Length > 1 then
                                notes.Add($"Retrieval: applying RRF across {allResultLists.Length} result sets")
                                reciprocalRankFusion allResultLists effectiveConfig.RRFConstant
                            else
                                allResultLists
                                |> List.concat
                                |> List.distinctBy (fun (id, _, _) -> id)
                                |> List.sortBy (fun (_, dist, _) -> dist)

                        // Step 6: Apply metadata filters
                        let filteredByMeta =
                            applyMetadataFilters effectiveConfig.MetadataFilters combinedResults

                        // Step 7: Apply time decay scoring
                        let timeDecayed = applyTimeDecay effectiveConfig filteredByMeta

                        // Step 8: Apply hybrid scoring (semantic + keyword)
                        let scoredResults =
                            if effectiveConfig.EnableHybridSearch then
                                notes.Add("Retrieval: applying hybrid search (semantic + keyword)")

                                timeDecayed
                                |> List.map (fun (id, distance, payload) ->
                                    let semanticScore = 1.0f - distance
                                    let content = payload |> Map.tryFind "content" |> Option.defaultValue ""
                                    let keywordScore = computeKeywordScore resolvedQuery content

                                    let combined =
                                        hybridScore semanticScore keywordScore effectiveConfig.SemanticWeight

                                    (id, 1.0f - combined, payload))
                                |> List.sortBy (fun (_, distance, _) -> distance)
                            else
                                timeDecayed

                        // Step 9: Filter by minimum score
                        let minScore =
                            stepParams
                            |> Map.tryFind "minScore"
                            |> Option.bind (fun s ->
                                match Single.TryParse s with
                                | true, v -> Some v
                                | _ -> None)
                            |> Option.defaultValue effectiveConfig.MinScore

                        let filteredResults =
                            scoredResults
                            |> List.filter (fun (_, distance, _) -> (1.0f - distance) >= minScore)
                            |> List.truncate (topK * 2) // Keep extra for reranking

                        // Step 10: Fallback chain if insufficient results
                        let! afterFallback =
                            executeFallback
                                vectorStore
                                ctx.Llm
                                ctx.KnowledgeGraph
                                effectiveConfig
                                resolvedQuery
                                filteredResults
                                notes

                        // Step 11: Parent document retrieval
                        let! withParents = retrieveParentDocs vectorStore effectiveConfig afterFallback notes

                        // Step 12: Sentence window expansion
                        let withSentenceWindow = expandSentenceWindow effectiveConfig withParents notes

                        // Step 13: Cross-encoder reranking (lighter option)
                        let! crossEncoderReranked =
                            if effectiveConfig.EnableCrossEncoder && not effectiveConfig.EnableReranking then
                                crossEncoderRerank ctx.Llm effectiveConfig resolvedQuery withSentenceWindow notes
                            else
                                Task.FromResult withSentenceWindow

                        // Step 14: Optional full LLM reranking
                        let! rerankedResults =
                            if effectiveConfig.EnableReranking && not (List.isEmpty crossEncoderReranked) then
                                rerankWithLlm ctx.Llm resolvedQuery crossEncoderReranked notes
                            else
                                Task.FromResult crossEncoderReranked

                        // Step 15: Contextual compression
                        let! compressedResults =
                            compressContext
                                ctx.Llm
                                effectiveConfig
                                resolvedQuery
                                (rerankedResults |> List.truncate topK)
                                notes

                        // Step 16: Knowledge graph enrichment for explicit concepts
                        let conceptHints =
                            stepParams
                            |> Map.tryFind "concepts"
                            |> Option.map (fun s -> s.Split(',') |> Array.map (fun c -> c.Trim()) |> Array.toList)
                            |> Option.defaultValue []

                        let kgContext = enrichWithKnowledgeGraph ctx.KnowledgeGraph conceptHints notes

                        // Step 17: Record metrics
                        if effectiveConfig.EnableMetrics then
                            recordMetrics effectiveConfig startTime compressedResults.Length false

                        notes.Add($"Retrieval: final result count = {List.length compressedResults}")

                        // Step 18: Answer attribution
                        let attributionMap =
                            if effectiveConfig.EnableAnswerAttribution then
                                attributeSources compressedResults notes
                            else
                                Map.empty

                        // Format results as context string with provenance
                        let contextParts =
                            compressedResults
                            |> List.mapi (fun i (id, distance, payload) ->
                                let similarity = 1.0f - distance
                                let content = payload |> Map.tryFind "content" |> Option.defaultValue ""
                                let source = payload |> Map.tryFind "source" |> Option.defaultValue id
                                let chunk = payload |> Map.tryFind "chunkIndex" |> Option.defaultValue ""

                                let compressed =
                                    if payload |> Map.containsKey "compressed" then
                                        " [compressed]"
                                    else
                                        ""

                                let windowed =
                                    if payload |> Map.containsKey "windowExpanded" then
                                        " [expanded]"
                                    else
                                        ""

                                sprintf
                                    "[%d] (score: %.2f, source: %s%s%s%s)\n%s"
                                    (i + 1)
                                    similarity
                                    source
                                    (if chunk <> "" then $" chunk={chunk}" else "")
                                    compressed
                                    windowed
                                    content)
                            |> String.concat "\n\n"
                            |> fun s ->
                                if s.Length > effectiveConfig.MaxContextChars then
                                    notes.Add(
                                        $"Retrieval: context truncated from {s.Length} to {effectiveConfig.MaxContextChars} chars"
                                    )

                                    s.Substring(0, effectiveConfig.MaxContextChars)
                                else
                                    s

                        let outputName =
                            match defaultArg step.Outputs [] with
                            | head :: _ -> head
                            | [] -> "context"

                        // Structured results for programmatic access
                        let structuredResults =
                            compressedResults
                            |> List.map (fun (id, distance, payload) ->
                                {| Id = id
                                   Score = 1.0f - distance
                                   Payload = payload |}
                                :> obj)

                        let baseOutputs =
                            Map [ outputName, box (contextParts + kgContext); "results", box structuredResults ]

                        let finalOutputs =
                            attributionMap |> Map.fold (fun acc k v -> Map.add k v acc) baseOutputs

                        return (finalOutputs, List.ofSeq notes)

                | None ->
                    notes.Add("Retrieval: no vector store configured, skipping")
                    return (Map [ "context", box ""; "results", box []; "attributions", box [] ], List.ofSeq notes)

            | _ -> return (Map.empty, List.ofSeq notes)
        }

    let run (ctx: MetascriptContext) (workflow: Workflow) (inputs: Map<string, obj>) =
        task {
            let validated: Workflow =
                match Tars.Metascript.Validation.validateWorkflow workflow with
                | Result.Ok wf -> wf
                | Result.Error errs ->
                    let msg = String.concat "; " errs
                    raise (ArgumentException(msg))

            let mutable state =
                { Workflow = validated
                  CurrentStepIndex = 0
                  Variables = inputs
                  StepOutputs = Map.empty
                  ExecutionTrace = [] }

            for step in workflow.Steps do
                let started = DateTime.UtcNow
                let sw = System.Diagnostics.Stopwatch.StartNew()
                let! outputs, notes = executeStep ctx step state
                sw.Stop()

                let trace =
                    { StepId = step.Id
                      StartedAt = started
                      Duration = sw.Elapsed
                      Outputs = outputs
                      Notes = notes }

                state <-
                    { state with
                        StepOutputs = state.StepOutputs.Add(step.Id, outputs) }

                state <-
                    { state with
                        ExecutionTrace = state.ExecutionTrace @ [ trace ] }

            return state
        }
