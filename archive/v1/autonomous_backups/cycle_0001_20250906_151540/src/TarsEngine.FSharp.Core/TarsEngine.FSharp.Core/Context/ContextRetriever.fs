namespace TarsEngine.FSharp.Core.Context

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Configuration for context retrieval
type RetrievalConfig = {
    DefaultMaxSpans: int
    DefaultSalienceThreshold: float
    RetrievalProfiles: Map<Intent, RetrievalProfile>
    EnableCaching: bool
    CacheExpiryMinutes: int
}

/// Intent-aware context retriever
type IntentAwareContextRetriever(config: RetrievalConfig, memory: IContextMemory, logger: ILogger<IntentAwareContextRetriever>) =
    
    /// Cache for retrieval results
    let mutable retrievalCache = Map.empty<string, DateTime * ContextSpan list>
    
    /// Get retrieval profile for intent
    let getRetrievalProfile (intent: Intent) =
        config.RetrievalProfiles.TryFind intent
        |> Option.defaultValue {
            Intent = intent
            Retrievers = ["default"]
            ChunkStrategy = "semantic-para"
            MaxSpans = config.DefaultMaxSpans
            SalienceThreshold = config.DefaultSalienceThreshold
        }
    
    /// Extract query features for better retrieval
    let extractQueryFeatures (query: string) =
        let features = ResizeArray<string>()
        
        // Technical terms
        if query.Contains("CUDA") || query.Contains("GPU") then
            features.Add("cuda_performance")
        if query.Contains("F#") then
            features.Add("fsharp_code")
        if query.Contains("C#") then
            features.Add("csharp_code")
        if query.Contains("test") then
            features.Add("testing")
        if query.Contains("autonomous") then
            features.Add("autonomous_behavior")
        if query.Contains("Agent OS") then
            features.Add("agent_os")
        if query.Contains("metascript") then
            features.Add("metascript")
        
        // Performance indicators
        if query.Contains("184M") || query.Contains("searches/second") then
            features.Add("performance_metrics")
        if query.Contains("coverage") then
            features.Add("test_coverage")
        
        features |> List.ofSeq
    
    /// Score span relevance to query
    let scoreSpanRelevance (span: ContextSpan) (query: string) (features: string list) =
        let mutable score = span.Salience
        
        // Text similarity (simple keyword matching)
        let queryWords = query.ToLower().Split([|' '; '\t'; '\n'|], StringSplitOptions.RemoveEmptyEntries)
        let spanWords = span.Text.ToLower().Split([|' '; '\t'; '\n'|], StringSplitOptions.RemoveEmptyEntries)
        
        let matchingWords = 
            queryWords 
            |> Array.filter (fun qw -> spanWords |> Array.contains qw)
            |> Array.length
        
        let textSimilarity = float matchingWords / float queryWords.Length
        score <- score + (0.3 * textSimilarity)
        
        // Feature matching
        let featureMatches = 
            features
            |> List.filter (fun feature -> 
                span.Text.ToLower().Contains(feature.Replace("_", " ")) ||
                span.Source.ToLower().Contains(feature))
            |> List.length
        
        if features.Length > 0 then
            let featureScore = float featureMatches / float features.Length
            score <- score + (0.2 * featureScore)
        
        // Source relevance
        let sourceBoost = 
            match span.Source.ToLower() with
            | s when s.Contains("code") -> 0.1
            | s when s.Contains("test") -> 0.1
            | s when s.Contains("doc") -> 0.05
            | s when s.Contains("consolidated") -> 0.15
            | _ -> 0.0
        
        score <- score + sourceBoost
        
        // Recency boost
        let ageHours = (DateTime.UtcNow - span.Timestamp).TotalHours
        let recencyBoost = Math.Max(0.0, 0.1 * Math.Exp(-ageHours / 24.0))
        score <- score + recencyBoost
        
        Math.Min(1.0, score)
    
    /// Retrieve spans using BM25-style approach
    let retrieveBM25Style (query: string) (spans: ContextSpan list) (features: string list) =
        spans
        |> List.map (fun span -> 
            let relevanceScore = scoreSpanRelevance span query features
            (span, relevanceScore))
        |> List.sortByDescending snd
        |> List.map fst
    
    /// Retrieve spans with recency boost
    let retrieveWithRecencyBoost (query: string) (spans: ContextSpan list) (features: string list) =
        let recentSpans = 
            spans
            |> List.filter (fun span -> 
                (DateTime.UtcNow - span.Timestamp).TotalHours < 24.0)
        
        let olderSpans = 
            spans
            |> List.filter (fun span -> 
                (DateTime.UtcNow - span.Timestamp).TotalHours >= 24.0)
        
        // Score recent spans higher
        let scoredRecent = 
            recentSpans
            |> List.map (fun span -> 
                let score = scoreSpanRelevance span query features
                (span, score + 0.2)) // Recency boost
        
        let scoredOlder = 
            olderSpans
            |> List.map (fun span -> 
                let score = scoreSpanRelevance span query features
                (span, score))
        
        (scoredRecent @ scoredOlder)
        |> List.sortByDescending snd
        |> List.map fst
    
    // TODO: Implement real functionality
    let retrieveFromBeliefGraph (query: string) (spans: ContextSpan list) (features: string list) =
        // Prioritize spans that contain belief updates or contradictions
        spans
        |> List.filter (fun span -> 
            span.Source.Contains("belief") || 
            span.Text.Contains("contradiction") ||
            span.Text.Contains("belief_update"))
        |> List.map (fun span -> 
            let score = scoreSpanRelevance span query features
            (span, score + 0.1)) // Belief graph boost
        |> List.sortByDescending snd
        |> List.map fst
    
    /// Execute retrieval strategy
    let executeRetrieval (retriever: string) (query: string) (spans: ContextSpan list) (features: string list) =
        match retriever.ToLower() with
        | "bm25+vec" -> retrieveBM25Style query spans features
        | "recent_boost" -> retrieveWithRecencyBoost query spans features
        | "belief_graph" -> retrieveFromBeliefGraph query spans features
        | "code_ast+vec" -> 
            spans 
            |> List.filter (fun s -> s.Source.Contains("code") || s.Text.Contains("```"))
            |> retrieveBM25Style query features
        | "symbol_xref" ->
            spans
            |> List.filter (fun s -> s.Text.Contains("::") || s.Text.Contains("->"))
            |> retrieveBM25Style query features
        | "trace-index" ->
            spans
            |> List.filter (fun s -> s.Source.Contains("trace") || s.Source.Contains("log"))
            |> retrieveBM25Style query features
        | "test_results" ->
            spans
            |> List.filter (fun s -> s.Source.Contains("test") || s.Text.Contains("test"))
            |> retrieveBM25Style query features
        | "performance_metrics" ->
            spans
            |> List.filter (fun s -> s.Text.Contains("performance") || s.Text.Contains("metric"))
            |> retrieveBM25Style query features
        | _ -> retrieveBM25Style query spans features
    
    /// Check cache for recent results
    let checkCache (cacheKey: string) =
        if config.EnableCaching then
            match retrievalCache.TryFind cacheKey with
            | Some (timestamp, results) ->
                let ageMinutes = (DateTime.UtcNow - timestamp).TotalMinutes
                if ageMinutes < float config.CacheExpiryMinutes then
                    Some results
                else
                    None
            | None -> None
        else
            None
    
    /// Update cache with results
    let updateCache (cacheKey: string) (results: ContextSpan list) =
        if config.EnableCaching then
            retrievalCache <- retrievalCache.Add(cacheKey, (DateTime.UtcNow, results))
    
    interface IContextRetriever with
        
        member _.RetrieveAsync(intent, query) =
            task {
                logger.LogDebug("Retrieving context for intent {Intent} with query: {Query}", intent, query)
                
                let profile = getRetrievalProfile intent
                let features = extractQueryFeatures query
                let cacheKey = $"{intent}:{query.GetHashCode()}"
                
                // Check cache first
                match checkCache cacheKey with
                | Some cachedResults ->
                    logger.LogDebug("Retrieved {SpanCount} spans from cache", cachedResults.Length)
                    return cachedResults
                | None ->
                    // Load all available spans
                    let! ephemeralSpans = memory.LoadEphemeralAsync()
                    let! workingSpans = memory.LoadWorkingSetAsync()
                    let! longTermSpans = memory.LoadLongTermAsync()
                    
                    let allSpans = ephemeralSpans @ workingSpans @ longTermSpans
                    
                    logger.LogDebug("Loaded {TotalSpans} spans from memory (ephemeral: {Ephemeral}, working: {Working}, long-term: {LongTerm})",
                        allSpans.Length, ephemeralSpans.Length, workingSpans.Length, longTermSpans.Length)
                    
                    // Execute retrieval strategies
                    let retrievedSpans = 
                        profile.Retrievers
                        |> List.collect (fun retriever -> 
                            executeRetrieval retriever query allSpans features)
                        |> List.distinctBy (fun span -> span.Id)
                        |> List.filter (fun span -> span.Salience >= profile.SalienceThreshold)
                        |> List.sortByDescending (fun span -> span.Salience)
                        |> List.truncate profile.MaxSpans
                    
                    logger.LogInformation("Retrieved {RetrievedCount} spans for intent {Intent} using retrievers: {Retrievers}",
                        retrievedSpans.Length, intent, String.concat ", " profile.Retrievers)
                    
                    // Update cache
                    updateCache cacheKey retrievedSpans
                    
                    return retrievedSpans
            }
        
        member _.GetProfile(intent) =
            Some (getRetrievalProfile intent)
