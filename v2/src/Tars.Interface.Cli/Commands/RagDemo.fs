/// <summary>
/// TARS RAG Demo - Demonstrates Retrieval Augmented Generation capabilities
/// </summary>
/// <remarks>
/// This module provides an interactive demo showcasing 14 different RAG features
/// including hybrid search, query expansion, reranking, and more.
///
/// Usage:
///   tars demo-rag                    Interactive mode
///   tars demo-rag --quick            Non-interactive mode (for CI)
///   tars demo-rag --scenario 1,5,14  Run specific scenarios
///   tars demo-rag --verbose          Show detailed internal state
///   tars demo-rag --output json      Output results as JSON
///   tars demo-rag --benchmark 5      Run each scenario 5 times with stats
/// </remarks>
module Tars.Interface.Cli.Commands.RagDemo

open System
open System.IO
open System.Net.Http
open System.Text.Json
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Llm
open Tars.Metascript.Domain
open Tars.Metascript.Config
open Tars.Cortex
open Tars.Tools

// ============================================================
// CONFIGURATION CONSTANTS
// ============================================================

/// <summary>Configuration constants for the RAG demo</summary>
module DemoConfig =
    /// Embedding vector dimension (must match LLM provider)
    [<Literal>]
    let EmbeddingDimension = 384

    /// Number of semantic keyword groups for demo embeddings
    [<Literal>]
    let SemanticKeywordGroups = 10

    /// Multiplier for keyword dimension spread
    [<Literal>]
    let KeywordDimensionSpread = 7

    /// Preview length for document content display
    [<Literal>]
    let ContentPreviewLength = 100

    /// Progress bar width in characters
    [<Literal>]
    let ProgressBarWidth = 30

    /// Default collection name for demo documents
    [<Literal>]
    let CollectionName = "tars_docs"

    /// Maximum results to display per scenario
    [<Literal>]
    let MaxDisplayResults = 8

    /// Maximum attributions to display
    [<Literal>]
    let MaxDisplayAttributions = 3

    /// Default benchmark iterations
    [<Literal>]
    let DefaultBenchmarkRuns = 3

    /// High score threshold (green)
    let HighScoreThreshold = 0.7f
    /// Medium score threshold (yellow)
    let MediumScoreThreshold = 0.4f

// ============================================================
// COMMAND-LINE OPTIONS
// ============================================================

/// <summary>Output format for demo results</summary>
type OutputFormat =
    | Text
    | Json

/// <summary>Command-line options for the RAG demo</summary>
type DemoOptions =
    {
        /// Skip interactive prompts
        Quick: bool
        /// Show detailed internal state
        Verbose: bool
        /// Output format (text or json)
        OutputFormat: OutputFormat
        /// Specific scenarios to run (empty = all)
        Scenarios: int list
        /// Number of benchmark iterations (0 = no benchmark)
        BenchmarkRuns: int
        /// Path to export results (None = no export)
        ExportPath: string option
        /// Path to custom documents folder (None = use sample docs)
        DocsPath: string option
        /// Use live LLM instead of stubs
        UseLiveLlm: bool
        /// Compare mode - show before/after with different configs
        CompareMode: bool
        /// Run diagnostics before demo
        ShowDiagnostics: bool
    }

// ============================================================
// STRUCTURED RESULT TYPES
// ============================================================

/// <summary>Represents a single retrieval result with typed fields</summary>
type RetrievalResult =
    { Rank: int
      Score: float32
      Source: string
      Content: string
      IsCompressed: bool }

/// <summary>Benchmark statistics for a scenario</summary>
type BenchmarkStats =
    { Runs: int
      MinLatencyMs: int64
      MaxLatencyMs: int64
      AvgLatencyMs: float
      StdDevMs: float }

/// <summary>Represents the outcome of a scenario execution</summary>
type ScenarioOutcome =
    | Success of results: RetrievalResult list * latencyMs: int64
    | Failure of error: string

/// <summary>JSON-serializable scenario result</summary>
type ScenarioJsonResult =
    { ScenarioNumber: int
      Name: string
      Query: string
      Success: bool
      LatencyMs: int64
      ResultCount: int
      TopScore: float32
      Results: RetrievalResult list
      Error: string option
      BenchmarkStats: BenchmarkStats option }

/// <summary>JSON-serializable demo result</summary>
type DemoJsonResult =
    { Timestamp: DateTime
      TotalScenarios: int
      SuccessfulScenarios: int
      FailedScenarios: int
      TotalRetrievalTimeMs: int64
      TotalDocumentsRetrieved: int
      Scenarios: ScenarioJsonResult list }

/// <summary>Aggregate metrics across all scenarios</summary>
type AggregateMetrics =
    { mutable TotalScenarios: int
      mutable SuccessfulScenarios: int
      mutable FailedScenarios: int
      mutable TotalRetrievalTimeMs: int64
      mutable TotalDocumentsRetrieved: int
      mutable TotalCacheHits: int64
      mutable TotalCacheMisses: int64 }

/// Sample knowledge base for the demo - a mini "TARS Documentation"
let private sampleDocuments =
    [
      // Architecture documents
      ("doc-arch-001",
       "TARS Architecture Overview",
       "TARS (Transformative Autonomous Reasoning System) is a multi-agent AI framework built in F#. \
      The core architecture consists of several layers: the Kernel layer handles agent lifecycle and messaging, \
      the Cortex layer provides memory and knowledge management with vector stores and knowledge graphs, \
      the Metascript layer enables declarative workflow definitions, and the Evolution layer supports \
      self-improvement through genetic algorithms and A/B testing.",
       "architecture",
       DateTime.UtcNow.AddDays(-5.0))

      ("doc-arch-002",
       "Agent Communication Protocol",
       "Agents in TARS communicate through a semantic messaging system based on FIPA-ACL standards. \
      Each message contains a performative (request, inform, query, etc.), semantic constraints, \
      and typed content. The EventBus handles pub/sub messaging with correlation IDs for tracking \
      conversation threads. Agents can be supervisor or worker types with hierarchical relationships.",
       "architecture",
       DateTime.UtcNow.AddDays(-10.0))

      // RAG documents
      ("doc-rag-001",
       "RAG Pipeline Overview",
       "The Retrieval Augmented Generation (RAG) pipeline in TARS uses a sophisticated 18-step process. \
      It starts with query routing to classify queries as factual, analytical, or conversational. \
      Then query expansion generates related queries for broader recall. The system supports hybrid search \
      combining semantic similarity with BM25 keyword matching. Results can be reranked using cross-encoders \
      or full LLM reranking for maximum precision.",
       "rag",
       DateTime.UtcNow.AddDays(-1.0))

      ("doc-rag-002",
       "Vector Store Implementation",
       "TARS includes an in-memory vector store with SIMD-optimized cosine similarity calculations. \
      Documents are chunked using semantic boundaries (paragraphs and sections) rather than fixed sizes. \
      The store supports metadata filtering with operators like equals, contains, greater-than, and less-than. \
      Embeddings are cached using an LRU strategy to avoid redundant API calls.",
       "rag",
       DateTime.UtcNow)

      ("doc-rag-003",
       "Knowledge Graph Integration",
       "The Knowledge Graph in TARS stores relationships between concepts, agents, files, and tasks. \
      Multi-hop retrieval traverses the graph using BFS to find related documents up to N hops away. \
      This enables finding contextually relevant information that might not match the query directly \
      but is semantically connected through the knowledge graph structure.",
       "rag",
       DateTime.UtcNow.AddDays(-3.0))

      // Metascript documents
      ("doc-meta-001",
       "Metascript Workflow Language",
       "Metascript is TARS's declarative workflow language for defining multi-agent pipelines. \
      Workflows consist of steps that can be: agent (LLM-powered), tool (function calls), \
      decision (branching), or retrieval (RAG search). Steps can reference outputs from previous \
      steps using {{stepId.outputName}} syntax. Workflows support parallel execution and loops.",
       "metascript",
       DateTime.UtcNow.AddDays(-7.0))

      ("doc-meta-002",
       "Metascript Step Types",
       "TARS Metascript supports four step types: 1) Agent steps invoke LLM with instructions and context, \
      2) Tool steps call registered functions like file operations or web searches, \
      3) Decision steps evaluate conditions for branching logic, \
      4) Retrieval steps search the vector store with configurable RAG options including \
      hybrid search, query expansion, reranking, and metadata filtering.",
       "metascript",
       DateTime.UtcNow.AddDays(-2.0))

      // Evolution documents
      ("doc-evol-001",
       "Self-Improvement Engine",
       "The Evolution module enables TARS to improve itself through experimentation. \
      It uses genetic algorithms to evolve prompts, agent configurations, and workflow structures. \
      A/B testing compares variants with statistical significance testing. The fitness function \
      measures task completion, response quality, and efficiency metrics.",
       "evolution",
       DateTime.UtcNow.AddDays(-15.0))

      // Security documents
      ("doc-sec-001",
       "Security and Credentials",
       "TARS uses a CredentialVault for secure secret management with encryption at rest. \
      API keys for LLM providers (OpenAI, Anthropic, Ollama) are stored securely. \
      The system supports capability-based access control for agent permissions. \
      All external API calls use HTTPS with certificate validation.",
       "security",
       DateTime.UtcNow.AddDays(-20.0))

      // Recent update
      ("doc-update-001",
       "Latest RAG Improvements",
       "Recent updates to TARS RAG include: contextual compression to extract relevant portions, \
      parent document retrieval for better context, sentence window expansion, time decay scoring \
      to favor recent documents, answer attribution for explainability, and comprehensive metrics \
      collection for observability. Query routing automatically adjusts retrieval strategy based \
      on query classification.",
       "rag",
       DateTime.UtcNow) ]

// ============================================================
// CUSTOM DOCUMENTS LOADING
// ============================================================

/// <summary>Load documents from a custom folder</summary>
/// <param name="folderPath">Path to folder containing documents</param>
/// <returns>List of (id, title, content, category, timestamp) tuples</returns>
let private loadCustomDocuments (folderPath: string) =
    if not (Directory.Exists(folderPath)) then
        failwith $"Documents folder not found: {folderPath}"

    let files =
        [ "*.txt"; "*.md"; "*.text"; "*.markdown" ]
        |> List.collect (fun pattern ->
            Directory.GetFiles(folderPath, pattern, SearchOption.AllDirectories)
            |> Array.toList)
        |> List.distinct

    if files.IsEmpty then
        failwith $"No documents found in: {folderPath} (supported: .txt, .md, .text, .markdown)"

    files
    |> List.mapi (fun idx filePath ->
        let fileName = Path.GetFileNameWithoutExtension(filePath)
        let relativePath = Path.GetRelativePath(folderPath, filePath)

        let category =
            let dir = Path.GetDirectoryName(relativePath)

            if String.IsNullOrEmpty(dir) then
                "custom"
            else
                dir.Replace(Path.DirectorySeparatorChar, '/')

        let content = File.ReadAllText(filePath)
        let lastWriteTime = File.GetLastWriteTimeUtc(filePath)

        // Try to extract title from first line if markdown
        let title =
            let lines = content.Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)

            if lines.Length > 0 && lines.[0].StartsWith("# ") then
                lines.[0].Substring(2).Trim()
            else
                fileName.Replace("-", " ").Replace("_", " ")

        let id = $"custom-{idx:D4}"
        (id, title, content, category, lastWriteTime))

/// <summary>Get documents to index (either custom or sample)</summary>
let private getDocuments (docsPath: string option) =
    match docsPath with
    | Some path -> loadCustomDocuments path
    | None -> sampleDocuments

// ============================================================
// CONSOLE OUTPUT UTILITIES
// ============================================================

/// <summary>Console output utilities with colors and formatting</summary>
module Console =
    /// Write colored text without newline
    let private write (color: ConsoleColor) (text: string) =
        let prev = Console.ForegroundColor
        Console.ForegroundColor <- color
        Console.Write(text)
        Console.ForegroundColor <- prev

    /// Write colored text with newline
    let private writeLine (color: ConsoleColor) (text: string) =
        write color text
        Console.WriteLine()

    /// Display a major section header
    let header text =
        let line = String.replicate 60 "="
        writeLine ConsoleColor.Cyan $"\n{line}\n  {text}\n{line}"

    /// Display a sub-section header
    let subHeader text =
        writeLine ConsoleColor.Yellow $"\n▶ {text}"

    /// Display informational text
    let info text =
        writeLine ConsoleColor.White $"  {text}"

    /// Display success message
    let success text =
        writeLine ConsoleColor.Green $"  ✓ {text}"

    /// Display warning message
    let warning text =
        writeLine ConsoleColor.Yellow $"  ⚠ {text}"

    /// Display highlighted text
    let highlight text =
        writeLine ConsoleColor.Magenta $"  ★ {text}"

    /// Display dimmed/secondary text
    let dim text =
        writeLine ConsoleColor.DarkGray $"    {text}"

    /// Display verbose output (only when verbose mode is enabled)
    let verbose (isVerbose: bool) text =
        if isVerbose then
            writeLine ConsoleColor.DarkCyan $"    [v] {text}"

    /// Display a bullet point
    let bullet text =
        writeLine ConsoleColor.White $"    • {text}"

    /// Display a result line
    let result text =
        writeLine ConsoleColor.Cyan $"  → {text}"

    /// Display an error message
    let error text =
        writeLine ConsoleColor.Red $"  ✗ {text}"

    /// Display a score with color based on value (green=high, yellow=medium, red=low)
    let coloredScore (score: float32) =
        let color =
            if score >= DemoConfig.HighScoreThreshold then
                ConsoleColor.Green
            elif score >= DemoConfig.MediumScoreThreshold then
                ConsoleColor.Yellow
            else
                ConsoleColor.Red

        let scoreStr = $"%.2f{score}"
        write color scoreStr

    /// Display a result with colored score
    let resultWithScore (rank: int) (score: float32) (source: string) (extra: string) =
        write ConsoleColor.Cyan $"  → [{rank}] (score: "
        coloredScore score
        writeLine ConsoleColor.Cyan $", source: {source}{extra})"

    /// Progress bar with ETA
    let mutable private progressStartTime = DateTime.MinValue

    /// Display progress bar with percentage and ETA
    let progressBarWithEta (current: int) (total: int) (label: string) =
        if current = 1 then
            progressStartTime <- DateTime.UtcNow

        let pct = float current / float total
        let filled = int (pct * float DemoConfig.ProgressBarWidth)

        let bar =
            String.replicate filled "█"
            + String.replicate (DemoConfig.ProgressBarWidth - filled) "░"

        let pctStr = sprintf "%.0f%%" (pct * 100.0)

        // Calculate ETA
        let elapsed = DateTime.UtcNow - progressStartTime

        let eta =
            if current > 0 && current < total then
                let avgPerItem = elapsed.TotalSeconds / float current
                let remaining = float (total - current) * avgPerItem

                if remaining < 60.0 then
                    $" ETA: %.0f{remaining}s"
                else
                    $" ETA: %.1f{remaining / 60.0}m"
            else
                ""

        Console.Write($"\r  [{bar}] {current}/{total} {label} ({pctStr}){eta}    ")

        if current = total then
            Console.WriteLine()

    /// Legacy progress bar (no ETA)
    let progressBar (current: int) (total: int) (label: string) =
        let pct = float current / float total
        let filled = int (pct * float DemoConfig.ProgressBarWidth)

        let bar =
            String.replicate filled "█"
            + String.replicate (DemoConfig.ProgressBarWidth - filled) "░"

        Console.Write($"\r  [{bar}] {current}/{total} {label}  ")

        if current = total then
            Console.WriteLine()

    /// Wait for user input (respects quick mode)
    let waitForEnter (quick: bool) (message: string) =
        if not quick then
            info ""
            info message
            Console.ReadLine() |> ignore

// ============================================================
// DEMO LLM IMPLEMENTATION
// ============================================================

/// Keywords that define semantic dimensions for the demo embeddings
let private semanticKeywords =
    [| "agent"
       "communication"
       "message"
       "protocol"
       "async"
       "vector"
       "embedding"
       "similarity"
       "search"
       "retrieval"
       "rag"
       "augmented"
       "generation"
       "context"
       "knowledge"
       "metascript"
       "workflow"
       "step"
       "pipeline"
       "execution"
       "evolution"
       "genetic"
       "mutation"
       "fitness"
       "optimization"
       "security"
       "sandbox"
       "permission"
       "isolation"
       "trust"
       "graph"
       "node"
       "edge"
       "traversal"
       "hop"
       "simd"
       "performance"
       "cache"
       "batch"
       "parallel"
       "llm"
       "model"
       "prompt"
       "completion"
       "token"
       "architecture"
       "module"
       "component"
       "system"
       "design" |]

/// Create embedding based on keyword presence (simulates semantic meaning)
let private createSemanticEmbedding (text: string) =
    let lowerText = text.ToLowerInvariant()
    let embedding = Array.zeroCreate<float32> DemoConfig.EmbeddingDimension
    let spread = DemoConfig.KeywordDimensionSpread

    // Set dimensions based on keyword presence using fold for immutability
    semanticKeywords
    |> Array.iteri (fun i keyword ->
        if lowerText.Contains(keyword) then
            // Strong signal for exact match
            embedding.[i * spread] <- 0.8f
            embedding.[i * spread + 1] <- 0.6f
            embedding.[i * spread + 2] <- 0.4f
        elif keyword.Length >= 4 && lowerText.Contains(keyword.Substring(0, 4)) then
            // Weaker signal for partial match
            embedding.[i * spread] <- 0.3f)

    // Add deterministic noise based on text hash for variety
    let hash = text.GetHashCode()
    let rng = Random(hash)

    for i in 0 .. embedding.Length - 1 do
        embedding.[i] <- embedding.[i] + float32 (rng.NextDouble() * 0.1 - 0.05)

    // Normalize the embedding vector
    let magnitude = sqrt (embedding |> Array.sumBy (fun x -> x * x))

    if magnitude > 0.0f then
        embedding |> Array.map (fun x -> x / magnitude)
    else
        // Fallback: random normalized vector
        let rng2 = Random(hash)

        let e =
            Array.init DemoConfig.EmbeddingDimension (fun _ -> float32 (rng2.NextDouble() * 2.0 - 1.0))

        let m = sqrt (e |> Array.sumBy (fun x -> x * x))
        e |> Array.map (fun x -> x / m)

/// Stub LLM for demo (simulates responses without API calls)
type DemoLlm() =
    let mutable embedCallCount = 0

    let makeResponse text =
        { Text = text
          FinishReason = Some "stop"
          Usage =
            Some
                { PromptTokens = 10
                  CompletionTokens = 20
                  TotalTokens = 30 }
          Raw = None }

    interface ILlmService with
        member _.CompleteAsync(req) =
            task {
                let content =
                    if req.Messages.Length > 0 then
                        req.Messages.[0].Content
                    else
                        ""

                // Simulate query expansion
                if content.Contains("Generate") && content.Contains("related") then
                    return makeResponse "semantic search vector\nembedding similarity\nretrieval augmented generation"

                // Simulate reranking - shuffle order based on content
                elif content.Contains("Rerank") || content.Contains("rank") then
                    return makeResponse "2,1,4,3,5"

                // Simulate compression - extract a relevant sentence
                elif content.Contains("Extract") && content.Contains("relevant") then
                    let docIdx = content.IndexOf("Document:")

                    if docIdx >= 0 then
                        let afterDoc = content.Substring(docIdx + 9).Trim()
                        let periodIdx = afterDoc.IndexOf('.')

                        let compressed =
                            if periodIdx > 0 && periodIdx < 200 then
                                afterDoc.Substring(0, periodIdx + 1)
                            elif afterDoc.Length > 150 then
                                afterDoc.Substring(0, 150) + "..."
                            else
                                afterDoc

                        return makeResponse compressed
                    else
                        return makeResponse "Relevant content extracted from document."

                // Simulate cross-encoder scoring - vary scores based on document position
                elif content.Contains("Rate relevance") then
                    let hash = abs (content.GetHashCode())
                    let scores = [| 9; 7; 5; 4; 3 |]
                    return makeResponse (string scores.[hash % scores.Length])

                // Default response
                else
                    return makeResponse "This is a simulated LLM response for the demo."
            }

        member _.RouteAsync(req) =
            task {
                return
                    { Backend = Ollama "demo"
                      Endpoint = Uri("http://localhost:11434")
                      ApiKey = None }
            }

        member _.EmbedAsync(text) =
            task {

                embedCallCount <- embedCallCount + 1
                return createSemanticEmbedding text
            }

        member this.CompleteStreamAsync(req, onToken) =
            task {
                let! response = (this :> ILlmService).CompleteAsync(req)
                onToken response.Text
                return response
            }

    member _.EmbedCallCount = embedCallCount

/// <summary>Live LLM that uses actual Ollama API</summary>
type LiveLlm(chatModel: string, embedModel: string) =
    let http = new HttpClient(Timeout = TimeSpan.FromSeconds(120.0))

    let baseUri =
        let url = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL")

        if String.IsNullOrEmpty(url) then
            Uri("http://localhost:11434")
        else
            Uri(url)

    let mutable embedCallCount = 0

    interface ILlmService with
        member _.CompleteAsync(req) =
            OllamaClient.sendChatAsync http baseUri chatModel None req

        member _.EmbedAsync(text) =
            task {
                embedCallCount <- embedCallCount + 1
                let! embedding = OllamaClient.getEmbeddingsAsync http baseUri embedModel text
                // Normalize if embedding dimension differs from expected
                if embedding.Length > 0 then
                    // Normalize to unit vector
                    let mag = sqrt (embedding |> Array.sumBy (fun x -> x * x))

                    if mag > 0.0f then
                        return embedding |> Array.map (fun x -> x / mag)
                    else
                        return embedding
                else
                    // Fallback to random if no embedding returned
                    return createSemanticEmbedding text
            }

        member _.CompleteStreamAsync(req, onToken) =
            OllamaClient.sendChatStreamAsync http baseUri chatModel None req onToken

        member _.RouteAsync(req) =
            task {
                return
                    { Backend = Ollama chatModel
                      Endpoint = baseUri
                      ApiKey = None }
            }

    member _.EmbedCallCount = embedCallCount

    interface IDisposable with
        member _.Dispose() = http.Dispose()

// ============================================================
// DEMO SCENARIOS
// ============================================================

/// <summary>Demo scenario definition</summary>
type DemoScenario =
    { Name: string
      Description: string
      Query: string
      Config: RagConfig
      ExpectedFeatures: string list }

// ============================================================
// CONFIG BUILDERS (Reduce Duplication)
// ============================================================

/// <summary>Base config with correct collection name</summary>
let private baseConfig =
    { RagConfig.Default with
        CollectionName = DemoConfig.CollectionName
        MinScore = 0.0f }

/// <summary>Config builders for common RAG configurations</summary>
module ConfigBuilders =
    /// Basic config with specified top-k
    let basic topK = { baseConfig with TopK = topK }

    /// Hybrid search config
    let hybrid topK weight =
        { baseConfig with
            TopK = topK
            EnableHybridSearch = true
            SemanticWeight = weight }

    /// Query expansion config
    let withExpansion topK count =
        { baseConfig with
            TopK = topK
            EnableQueryExpansion = true
            QueryExpansionCount = count }

    /// Query routing config
    let withRouting topK =
        { baseConfig with
            TopK = topK
            EnableQueryRouting = true }

    /// Time decay config
    let withTimeDecay topK halfLifeDays =
        { baseConfig with
            TopK = topK
            EnableTimeDecay = true
            TimeDecayHalfLifeDays = halfLifeDays }

    /// Metadata filter config
    let withFilter topK field op value =
        { baseConfig with
            TopK = topK
            MetadataFilters =
                [ { Field = field
                    Operator = op
                    Value = value } ] }

    /// RRF config
    let withRRF topK =
        { baseConfig with
            TopK = topK
            EnableHybridSearch = true
            EnableRRF = true
            RRFConstant = 60 }

    /// Cache config with metrics
    let withCache topK cacheSize =
        { baseConfig with
            TopK = topK
            EnableEmbeddingCache = true
            EmbeddingCacheSize = cacheSize
            EnableMetrics = true
            Metrics = Some(RetrievalMetrics.Create()) }

    /// Cross-encoder config
    let withCrossEncoder topK =
        { baseConfig with
            TopK = topK
            EnableCrossEncoder = true
            CrossEncoderModel = "cross-encoder-demo" }

    /// LLM reranking config
    let withLlmRerank topK =
        { baseConfig with
            TopK = topK
            EnableReranking = true }

    /// Compression config
    let withCompression topK maxChars =
        { baseConfig with
            TopK = topK
            EnableContextualCompression = true
            CompressionMaxChars = maxChars }

    /// Sentence window config
    let withSentenceWindow topK windowSize =
        { baseConfig with
            TopK = topK
            EnableSentenceWindow = true
            SentenceWindowSize = windowSize }

    /// Fallback chain config
    let withFallback topK minScore minResults =
        { baseConfig with
            TopK = topK
            MinScore = minScore
            EnableFallbackChain = true
            FallbackMinResults = minResults
            EnableHybridSearch = true }

    /// Full pipeline config
    let fullPipeline topK =
        { baseConfig with
            TopK = topK
            EnableHybridSearch = true
            EnableQueryExpansion = true
            EnableQueryRouting = true
            EnableTimeDecay = true
            TimeDecayHalfLifeDays = 14.0
            EnableRRF = true
            EnableEmbeddingCache = true
            EnableCrossEncoder = true
            EnableAnswerAttribution = true
            EnableMetrics = true
            Metrics = Some(RetrievalMetrics.Create()) }

// ============================================================
// COMPARE MODE TYPES AND HELPERS
// ============================================================

/// <summary>Comparison result between two configurations</summary>
type ComparisonResult =
    { Query: string
      BaselineConfig: string
      EnhancedConfig: string
      BaselineLatencyMs: int64
      EnhancedLatencyMs: int64
      BaselineTopScore: float32
      EnhancedTopScore: float32
      BaselineResultCount: int
      EnhancedResultCount: int
      ScoreImprovement: float32
      LatencyDifference: int64 }

/// <summary>Helper to run a single retrieval with a given config</summary>
let private runRetrievalWithConfig (llm: ILlmService) (vectorStore: IVectorStore) (config: RagConfig) (query: string) =
    task {
        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vectorStore
              KnowledgeGraph = None
              SemanticMemory = None
              EpisodeService = None
              RagConfig = config
              MacroRegistry = None
              MetascriptRegistry = None }

        let workflow =
            { Name = "compare-retrieval"
              Description = "Compare mode retrieval"
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", query) ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        let sw = System.Diagnostics.Stopwatch.StartNew()
        let! state = Tars.Metascript.Engine.run ctx workflow Map.empty
        sw.Stop()

        let rawResults = state.StepOutputs.["retrieve"].["results"] :?> obj list
        return (rawResults.Length, sw.ElapsedMilliseconds)
    }

/// <summary>Run compare mode displaying before/after for each scenario</summary>
let private runCompareMode
    (llm: ILlmService)
    (vectorStore: IVectorStore)
    (options: DemoOptions)
    (scenariosToRun: (int * DemoScenario) list)
    =
    task {
        Console.header "Compare Mode: Basic vs Enhanced"
        Console.info "Running each scenario with two configurations"
        Console.info ""

        let mutable comparisons: ComparisonResult list = []

        for (num, scenario) in scenariosToRun do
            Console.subHeader $"Scenario {num}: {scenario.Name}"

            // Run with basic config
            let basicConfig = ConfigBuilders.basic 3
            let! (basicCount, basicMs) = runRetrievalWithConfig llm vectorStore basicConfig scenario.Query

            // Run with enhanced config (full pipeline)
            let enhancedConfig = ConfigBuilders.fullPipeline 5
            let! (enhancedCount, enhancedMs) = runRetrievalWithConfig llm vectorStore enhancedConfig scenario.Query

            // Display comparison table
            Console.info $"  Query: {scenario.Query}"
            Console.info ""
            Console.dim "  ┌─────────────────────┬────────────────┬────────────────┐"
            Console.dim "  │ Metric              │ Basic          │ Full Pipeline  │"
            Console.dim "  ├─────────────────────┼────────────────┼────────────────┤"
            Console.info $"  │ Results             │ {basicCount, -14} │ {enhancedCount, -14} │"
            Console.info $"  │ Latency (ms)        │ {basicMs, -14} │ {enhancedMs, -14} │"
            Console.dim "  └─────────────────────┴────────────────┴────────────────┘"

            let latencyDiff = enhancedMs - basicMs

            if latencyDiff > 0L then
                Console.dim $"  ⏱ Enhanced took {latencyDiff}ms longer"
            else
                Console.dim $"  ⏱ Enhanced was {abs latencyDiff}ms faster"

            Console.info ""

            comparisons <-
                { Query = scenario.Query
                  BaselineConfig = "Basic"
                  EnhancedConfig = "Full Pipeline"
                  BaselineLatencyMs = basicMs
                  EnhancedLatencyMs = enhancedMs
                  BaselineTopScore = 0.0f
                  EnhancedTopScore = 0.0f
                  BaselineResultCount = basicCount
                  EnhancedResultCount = enhancedCount
                  ScoreImprovement = 0.0f
                  LatencyDifference = latencyDiff }
                :: comparisons

            if not options.Quick then
                Console.waitForEnter options.Quick "Press ENTER for next comparison..."

        // Summary
        Console.header "Compare Mode Summary"

        let avgLatencyDiff =
            comparisons |> List.averageBy (fun c -> float c.LatencyDifference)

        let totalBasic = comparisons |> List.sumBy (fun c -> c.BaselineResultCount)
        let totalEnhanced = comparisons |> List.sumBy (fun c -> c.EnhancedResultCount)

        Console.info $"  Scenarios analyzed: {comparisons.Length}"
        Console.info $"  Total results (Basic): {totalBasic}"
        Console.info $"  Total results (Enhanced): {totalEnhanced}"
        Console.info $"  Average latency difference: {avgLatencyDiff:F0}ms"
        Console.info ""

        return 0
    }

/// <summary>All demo scenarios using config builders</summary>
let private scenarios =
    [
      // ===== PART 1: Basic Features =====
      { Name = "Basic Semantic Search"
        Description = "Simple vector similarity search without advanced features"
        Query = "How does TARS handle agent communication?"
        Config = ConfigBuilders.basic 3
        ExpectedFeatures = [ "Semantic similarity"; "Top-K retrieval" ] }

      { Name = "Hybrid Search (Semantic + Keyword)"
        Description = "Combines embedding similarity with BM25 keyword matching"
        Query = "vector store SIMD optimization cosine"
        Config = ConfigBuilders.hybrid 3 0.6f
        ExpectedFeatures = [ "Semantic similarity"; "BM25 keyword scoring"; "Weighted combination" ] }

      { Name = "Query Expansion"
        Description = "Generates related queries for broader recall"
        Query = "How does retrieval work?"
        Config = ConfigBuilders.withExpansion 3 2
        ExpectedFeatures = [ "Original query"; "Expanded queries"; "Multi-query search" ] }

      { Name = "Query Routing"
        Description = "Automatically adjusts strategy based on query type"
        Query = "Compare the architecture of Metascript with the Evolution module"
        Config = ConfigBuilders.withRouting 5
        ExpectedFeatures = [ "Query classification"; "Strategy adjustment"; "Analytical query handling" ] }

      { Name = "Time Decay Scoring"
        Description = "Favors more recent documents in results"
        Query = "What are the latest RAG improvements?"
        Config = ConfigBuilders.withTimeDecay 5 7.0
        ExpectedFeatures = [ "Recency weighting"; "Exponential decay"; "Fresh content priority" ] }

      // ===== PART 2: Advanced Features =====
      { Name = "Metadata Filtering"
        Description = "Filter documents by metadata before scoring"
        Query = "What is the architecture of TARS?"
        Config = ConfigBuilders.withFilter 5 "category" "eq" "architecture"
        ExpectedFeatures = [ "Category filter"; "Pre-scoring filter"; "Precise targeting" ] }

      { Name = "Reciprocal Rank Fusion (RRF)"
        Description = "Combines multiple retrieval methods with rank-based fusion"
        Query = "How does the evolution module improve agents?"
        Config = ConfigBuilders.withRRF 5
        ExpectedFeatures = [ "Rank fusion"; "Multi-signal combination"; "Robust ranking" ] }

      { Name = "Embedding Cache"
        Description = "Caches embeddings to avoid recomputation for repeated queries"
        Query = "What is the RAG pipeline?"
        Config = ConfigBuilders.withCache 3 100
        ExpectedFeatures = [ "LRU cache"; "Embedding reuse"; "Latency reduction" ] }

      { Name = "Cross-Encoder Reranking"
        Description = "Uses cross-encoder model to rerank results (lighter than full LLM)"
        Query = "How do agents communicate in TARS?"
        Config = ConfigBuilders.withCrossEncoder 5
        ExpectedFeatures = [ "Cross-encoder scoring"; "Pairwise relevance"; "Fast reranking" ] }

      { Name = "LLM Reranking"
        Description = "Uses full LLM to rerank results for maximum precision"
        Query = "Explain the metascript workflow language"
        Config = ConfigBuilders.withLlmRerank 5
        ExpectedFeatures = [ "LLM reranking"; "Semantic reordering"; "Maximum precision" ] }

      { Name = "Contextual Compression"
        Description = "Uses LLM to extract only relevant portions from retrieved docs"
        Query = "What is the purpose of the knowledge graph?"
        Config = ConfigBuilders.withCompression 3 200
        ExpectedFeatures = [ "Content extraction"; "Noise reduction"; "Focused context" ] }

      { Name = "Sentence Window Expansion"
        Description = "Expands retrieved sentences to include surrounding context"
        Query = "How does TARS handle security?"
        Config = ConfigBuilders.withSentenceWindow 3 2
        ExpectedFeatures = [ "Sentence expansion"; "Surrounding context"; "Fuller understanding" ] }

      { Name = "Fallback Chain"
        Description = "Activates fallback strategies when initial results are insufficient"
        Query = "vector embedding search optimization"
        Config = ConfigBuilders.withFallback 3 0.5f 2
        ExpectedFeatures = [ "Fallback detection"; "Lower threshold recovery"; "Result recovery" ] }

      // ===== PART 3: Full Pipeline =====
      { Name = "Full Pipeline (All Features)"
        Description = "All features enabled for maximum quality"
        Query = "Explain how TARS uses knowledge graphs for multi-hop retrieval"
        Config = ConfigBuilders.fullPipeline 5
        ExpectedFeatures =
          [ "Hybrid search"
            "Query expansion"
            "Query routing"
            "Time decay"
            "RRF"
            "Cross-encoder"
            "Attribution"
            "Metrics" ] } ]

// ============================================================
// DOCUMENT INDEXING
// ============================================================

/// Index sample documents into the vector store
let private indexDocuments (llm: ILlmService) (vectorStore: IVectorStore) =
    task {
        Console.subHeader "Indexing Knowledge Base"
        Console.info $"Loading {sampleDocuments.Length} documents..."

        // Use fold pattern to track progress immutably
        let! _ =
            sampleDocuments
            |> List.indexed
            |> List.fold
                (fun (acc: Task<int>) (idx, (id, title, content, category, timestamp)) ->
                    task {
                        let! count = acc
                        let! embedding = llm.EmbedAsync(content)

                        let metadata =
                            Map
                                [ ("title", title)
                                  ("content", content)
                                  ("category", category)
                                  ("timestamp", timestamp.ToString("o"))
                                  ("source", $"{category}/{id}.md") ]

                        do! vectorStore.SaveAsync(DemoConfig.CollectionName, id, embedding, metadata)
                        let newCount = count + 1
                        Console.progressBar newCount sampleDocuments.Length "documents"
                        return newCount
                    })
                (Task.FromResult 0)

        Console.success $"Indexed {sampleDocuments.Length} documents with embeddings"
    }

// ============================================================
// RESULT PARSING
// ============================================================

/// Parse structured results from the context string
let private parseResults (context: string) (results: obj list) : RetrievalResult list =
    let lines = context.Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)

    lines
    |> Array.indexed
    |> Array.fold
        (fun (acc, currentRank) (idx, line) ->
            if line.StartsWith("[") && line.Contains("score:") then
                // Parse score and source from line like "[1] (score: 0.45, source: rag/doc.md)"
                let scoreStart = line.IndexOf("score:") + 6
                let scoreEnd = line.IndexOf(",", scoreStart)

                let score =
                    if scoreStart > 5 && scoreEnd > scoreStart then
                        match Single.TryParse(line.Substring(scoreStart, scoreEnd - scoreStart).Trim()) with
                        | true, s -> s
                        | false, _ -> 0.0f
                    else
                        0.0f

                let sourceStart = line.IndexOf("source:") + 7
                let sourceEnd = line.IndexOf(")", sourceStart)

                let source =
                    if sourceStart > 6 && sourceEnd > sourceStart then
                        line.Substring(sourceStart, sourceEnd - sourceStart).Trim()
                    else
                        "unknown"

                let isCompressed = line.Contains("[compressed]")
                let newRank = currentRank + 1

                // Get content from next line if available
                let content =
                    if idx + 1 < lines.Length then
                        let nextLine = lines.[idx + 1].Trim()

                        if nextLine.Length > DemoConfig.ContentPreviewLength then
                            nextLine.Substring(0, DemoConfig.ContentPreviewLength) + "..."
                        else
                            nextLine
                    else
                        ""

                let result =
                    { Rank = newRank
                      Score = score
                      Source = source
                      Content = content
                      IsCompressed = isCompressed }

                (result :: acc, newRank)
            else
                (acc, currentRank))
        ([], 0)
    |> fst
    |> List.rev
    |> List.truncate DemoConfig.MaxDisplayResults

// ============================================================
// SCENARIO EXECUTION
// ============================================================

/// <summary>Run a single demo scenario with error handling</summary>
/// <param name="llm">LLM service for embeddings and completions</param>
/// <param name="vectorStore">Vector store containing indexed documents</param>
/// <param name="scenario">Scenario definition to execute</param>
/// <param name="scenarioNum">Scenario number for display</param>
/// <param name="options">Demo options (verbose, output format, etc.)</param>
/// <param name="aggregateMetrics">Aggregate metrics to update</param>
/// <returns>Scenario outcome (success with results or failure with error)</returns>
let private runScenario
    (llm: ILlmService)
    (vectorStore: IVectorStore)
    (scenario: DemoScenario)
    (scenarioNum: int)
    (options: DemoOptions)
    (aggregateMetrics: AggregateMetrics)
    : Task<ScenarioOutcome> =
    task {
        // Skip console output for JSON mode
        let isTextMode = options.OutputFormat = Text

        if isTextMode then
            Console.header $"Scenario {scenarioNum}: {scenario.Name}"
            Console.info scenario.Description
            Console.subHeader "Query"
            Console.highlight $"\"{scenario.Query}\""

            Console.subHeader "Enabled Features"
            scenario.ExpectedFeatures |> List.iter (fun f -> Console.dim $"• {f}")

            // Verbose: show config details
            Console.verbose options.Verbose $"TopK: {scenario.Config.TopK}, MinScore: {scenario.Config.MinScore}"

            if scenario.Config.EnableHybridSearch then
                Console.verbose options.Verbose $"Hybrid: weight={scenario.Config.SemanticWeight}"

            if scenario.Config.EnableQueryExpansion then
                Console.verbose options.Verbose $"Expansion: count={scenario.Config.QueryExpansionCount}"

        try
            // Build context and workflow
            let ctx =
                { Llm = llm
                  Tools = ToolRegistry()
                  Budget = None
                  VectorStore = Some vectorStore
                  KnowledgeGraph = None
                  SemanticMemory = None
                  EpisodeService = None
                  RagConfig = scenario.Config
                  MacroRegistry = None
                  MetascriptRegistry = None }

            let workflow =
                { Name = $"demo-{scenarioNum}"
                  Description = scenario.Description
                  Version = "1.0"
                  Inputs = []
                  Steps =
                    [ { Id = "retrieve"
                        Type = "retrieval"
                        Agent = None
                        Tool = None
                        Instruction = None
                        Params = Some(Map [ ("query", scenario.Query) ])
                        Context = None
                        DependsOn = None
                        Outputs = Some [ "context" ]
                        Tools = None } ] }

            if isTextMode then
                Console.subHeader "Executing Retrieval..."

            let sw = System.Diagnostics.Stopwatch.StartNew()
            let! state = Tars.Metascript.Engine.run ctx workflow Map.empty
            sw.Stop()

            if isTextMode then
                Console.subHeader "Results"

            let context = state.StepOutputs.["retrieve"].["context"] :?> string
            let rawResults = state.StepOutputs.["retrieve"].["results"] :?> obj list

            // Parse structured results
            let results = parseResults context rawResults

            if isTextMode then
                Console.info $"Retrieved {rawResults.Length} documents in {sw.ElapsedMilliseconds}ms"

                // Display results using colored scores
                results
                |> List.iter (fun r ->
                    let compressedTag = if r.IsCompressed then " [compressed]" else ""
                    Console.resultWithScore r.Rank r.Score r.Source compressedTag

                    if r.Content.Length > 0 then
                        Console.dim r.Content)

            // Update aggregate metrics
            aggregateMetrics.TotalDocumentsRetrieved <- aggregateMetrics.TotalDocumentsRetrieved + rawResults.Length
            aggregateMetrics.TotalRetrievalTimeMs <- aggregateMetrics.TotalRetrievalTimeMs + sw.ElapsedMilliseconds

            // Show attribution if enabled
            if isTextMode && scenario.Config.EnableAnswerAttribution then
                Console.subHeader "Answer Attribution"
                Console.dim "Sources used in answer (ranked by relevance):"

                match state.StepOutputs.["retrieve"].TryFind("attributions") with
                | Some attrs ->
                    let attrList = attrs :?> obj list

                    attrList
                    |> List.truncate DemoConfig.MaxDisplayAttributions
                    |> List.iter (fun attr ->
                        let attrType = attr.GetType()
                        let idx = attrType.GetProperty("Index").GetValue(attr)
                        let source = attrType.GetProperty("Source").GetValue(attr)
                        let score = attrType.GetProperty("Score").GetValue(attr) :?> float32
                        Console.result $"[{idx}] {source} (relevance: {score:F2})")
                | None -> Console.dim "No attributions recorded"

            // Show and aggregate metrics if enabled
            if isTextMode then
                scenario.Config.Metrics
                |> Option.iter (fun m ->
                    Console.subHeader "Retrieval Metrics"
                    Console.dim $"Total Queries: {m.TotalQueries}"
                    Console.dim $"Total Latency: {m.TotalLatencyMs}ms"
                    Console.dim $"Avg Results: {m.AvgResultCount:F1}"
                    Console.dim $"Cache Hits: {m.CacheHits} / Misses: {m.CacheMisses}"
                    Console.verbose options.Verbose $"Cache key: {scenario.Query.GetHashCode()}"
                    aggregateMetrics.TotalCacheHits <- aggregateMetrics.TotalCacheHits + m.CacheHits
                    aggregateMetrics.TotalCacheMisses <- aggregateMetrics.TotalCacheMisses + m.CacheMisses)

            if isTextMode then
                Console.success "Scenario complete"

            aggregateMetrics.SuccessfulScenarios <- aggregateMetrics.SuccessfulScenarios + 1
            return Success(results, sw.ElapsedMilliseconds)

        with ex ->
            if isTextMode then
                Console.error $"Scenario failed: {ex.Message}"

                let stackPreview =
                    if isNull ex.StackTrace then
                        "(no stack trace)"
                    elif ex.StackTrace.Length > 200 then
                        ex.StackTrace.Substring(0, 200) + "..."
                    else
                        ex.StackTrace

                Console.dim $"Stack: {stackPreview}"

            aggregateMetrics.FailedScenarios <- aggregateMetrics.FailedScenarios + 1
            return Failure ex.Message
    }

// ============================================================
// MAIN ENTRY POINT
// ============================================================

/// <summary>Display aggregate metrics summary</summary>
let private displayAggregateSummary (metrics: AggregateMetrics) =
    Console.subHeader "Aggregate Metrics"
    Console.dim $"Scenarios: {metrics.SuccessfulScenarios} passed, {metrics.FailedScenarios} failed"
    Console.dim $"Total Documents Retrieved: {metrics.TotalDocumentsRetrieved}"
    Console.dim $"Total Retrieval Time: {metrics.TotalRetrievalTimeMs}ms"

    if metrics.SuccessfulScenarios > 0 then
        Console.dim $"Avg Time per Scenario: {metrics.TotalRetrievalTimeMs / int64 metrics.SuccessfulScenarios}ms"

    if metrics.TotalCacheHits + metrics.TotalCacheMisses > 0L then
        let hitRate =
            float metrics.TotalCacheHits
            / float (metrics.TotalCacheHits + metrics.TotalCacheMisses)
            * 100.0

        let hitRateStr = $"%.1f{hitRate}"
        Console.dim $"Cache Hit Rate: {hitRateStr}%% ({metrics.TotalCacheHits} hits, {metrics.TotalCacheMisses} misses)"

/// <summary>Calculate benchmark statistics from latency samples</summary>
let private calculateBenchmarkStats (latencies: int64 list) : BenchmarkStats =
    let n = latencies.Length
    let avg = latencies |> List.averageBy float
    let variance = latencies |> List.averageBy (fun x -> (float x - avg) ** 2.0)
    let stdDev = sqrt variance

    { Runs = n
      MinLatencyMs = latencies |> List.min
      MaxLatencyMs = latencies |> List.max
      AvgLatencyMs = avg
      StdDevMs = stdDev }

/// <summary>Output results as JSON (to console and/or file)</summary>
let private outputJson (result: DemoJsonResult) (exportPath: string option) =
    let options = JsonSerializerOptions(WriteIndented = true)
    let json = JsonSerializer.Serialize(result, options)
    Console.WriteLine(json)
    // Export to file if path specified
    match exportPath with
    | Some path ->
        File.WriteAllText(path, json)
        Console.info $"Results exported to: {path}"
    | None -> ()

/// <summary>Default demo options</summary>
let defaultOptions =
    { Quick = false
      Verbose = false
      OutputFormat = Text
      Scenarios = []
      BenchmarkRuns = 0
      ExportPath = None
      DocsPath = None
      UseLiveLlm = false
      CompareMode = false
      ShowDiagnostics = false }

/// <summary>Main demo entry point with full options</summary>
/// <param name="logger">Serilog logger</param>
/// <param name="options">Demo options (quick, verbose, scenarios, output, benchmark)</param>
/// <returns>Exit code (0 = success, 1 = failure)</returns>
let runWithOptions (logger: ILogger) (options: DemoOptions) =
    task {
        let isTextMode = options.OutputFormat = Text

        // Run diagnostics if requested
        if options.ShowDiagnostics then
            let! _ = Diagnostics.run logger
            ()

        if isTextMode then
            Console.header "TARS RAG Demo"
            Console.info "Demonstrating Retrieval Augmented Generation capabilities"
            Console.info "This demo uses simulated LLM responses for reproducibility"

            if options.Quick then
                Console.info "(Running in quick mode - no prompts)"

            if options.Verbose then
                Console.info "(Verbose mode enabled)"

            if options.BenchmarkRuns > 0 then
                Console.info $"(Benchmark mode: {options.BenchmarkRuns} runs per scenario)"

            if options.Scenarios.Length > 0 then
                let scenarioList = String.Join(", ", options.Scenarios)
                Console.info $"(Running scenarios: {scenarioList})"

            Console.WriteLine()

        // Initialize components
        let llm =
            if options.UseLiveLlm then
                if isTextMode then
                    Console.info "(Using LIVE Ollama LLM - responses may take longer)"

                new LiveLlm("llama3.2:3b", "nomic-embed-text:latest") :> ILlmService
            else
                DemoLlm() :> ILlmService

        let vectorStore = InMemoryVectorStore() :> IVectorStore

        // Filter scenarios if specific ones requested
        let scenariosToRun =
            if options.Scenarios.Length > 0 then
                scenarios
                |> List.mapi (fun idx s -> (idx + 1, s))
                |> List.filter (fun (num, _) -> List.contains num options.Scenarios)
            else
                scenarios |> List.mapi (fun idx s -> (idx + 1, s))

        // Initialize aggregate metrics
        let aggregateMetrics =
            { TotalScenarios = scenariosToRun.Length
              SuccessfulScenarios = 0
              FailedScenarios = 0
              TotalRetrievalTimeMs = 0L
              TotalDocumentsRetrieved = 0
              TotalCacheHits = 0L
              TotalCacheMisses = 0L }

        // Load documents (either custom or sample)
        let documents = getDocuments options.DocsPath

        // Index documents (use ETA progress bar)
        if isTextMode then
            match options.DocsPath with
            | Some path -> Console.subHeader $"Indexing Custom Documents from: {path}"
            | None -> Console.subHeader "Indexing Documents..."

            Console.info $"Found {documents.Length} documents to index"

        let! _ =
            documents
            |> List.mapi (fun idx doc -> (idx, doc))
            |> List.fold
                (fun (acc: Task<int>) (idx, (id, title, content, category, timestamp)) ->
                    task {
                        let! _ = acc
                        let! embedding = llm.EmbedAsync(content)

                        let metadata =
                            Map
                                [ ("title", title)
                                  ("content", content)
                                  ("category", category)
                                  ("timestamp", timestamp.ToString("o"))
                                  ("source", $"{category}/{id}.md") ]

                        do! vectorStore.SaveAsync(DemoConfig.CollectionName, id, embedding, metadata)

                        if isTextMode then
                            Console.progressBarWithEta (idx + 1) documents.Length "documents"

                        return idx + 1
                    })
                (Task.FromResult 0)

        if isTextMode then
            Console.waitForEnter options.Quick "Press ENTER to start the demo scenarios..."

        // Handle compare mode separately
        if options.CompareMode then
            return! runCompareMode llm vectorStore options scenariosToRun
        else

            // Collect JSON results if needed
            let jsonResults = ResizeArray<ScenarioJsonResult>()

            // Run each scenario
            let! outcomes =
                scenariosToRun
                |> List.fold
                    (fun (acc: Task<ScenarioOutcome list>) (scenarioNum, scenario) ->
                        task {
                            let! prevOutcomes = acc

                            // Benchmark mode: run multiple times
                            if options.BenchmarkRuns > 0 then
                                let! latencies =
                                    [ 1 .. options.BenchmarkRuns ]
                                    |> List.fold
                                        (fun (latAcc: Task<int64 list>) runNum ->
                                            task {
                                                let! prevLats = latAcc

                                                if isTextMode && runNum > 1 then
                                                    Console.dim $"  Benchmark run {runNum}/{options.BenchmarkRuns}..."

                                                let! outcome =
                                                    runScenario
                                                        llm
                                                        vectorStore
                                                        scenario
                                                        scenarioNum
                                                        options
                                                        aggregateMetrics

                                                let latency =
                                                    match outcome with
                                                    | Success(_, ms) -> ms
                                                    | _ -> 0L

                                                return latency :: prevLats
                                            })
                                        (Task.FromResult [])

                                let stats = calculateBenchmarkStats latencies

                                if isTextMode then
                                    Console.subHeader "Benchmark Results"

                                    Console.dim
                                        $"Runs: {stats.Runs}, Min: {stats.MinLatencyMs}ms, Max: {stats.MaxLatencyMs}ms"

                                    Console.dim $"Avg: {stats.AvgLatencyMs:F1}ms, StdDev: {stats.StdDevMs:F1}ms"

                                // Add to JSON results
                                let lastOutcome = latencies |> List.head

                                jsonResults.Add(
                                    { ScenarioNumber = scenarioNum
                                      Name = scenario.Name
                                      Query = scenario.Query
                                      Success = true
                                      LatencyMs = lastOutcome
                                      ResultCount = 0
                                      TopScore = 0.0f
                                      Results = []
                                      Error = None
                                      BenchmarkStats = Some stats }
                                )

                                return Success([], lastOutcome) :: prevOutcomes
                            else
                                let! outcome =
                                    runScenario llm vectorStore scenario scenarioNum options aggregateMetrics

                                // Add to JSON results
                                match outcome with
                                | Success(results, ms) ->
                                    jsonResults.Add(
                                        { ScenarioNumber = scenarioNum
                                          Name = scenario.Name
                                          Query = scenario.Query
                                          Success = true
                                          LatencyMs = ms
                                          ResultCount = results.Length
                                          TopScore = if results.Length > 0 then results.[0].Score else 0.0f
                                          Results = results
                                          Error = None
                                          BenchmarkStats = None }
                                    )
                                | Failure err ->
                                    jsonResults.Add(
                                        { ScenarioNumber = scenarioNum
                                          Name = scenario.Name
                                          Query = scenario.Query
                                          Success = false
                                          LatencyMs = 0L
                                          ResultCount = 0
                                          TopScore = 0.0f
                                          Results = []
                                          Error = Some err
                                          BenchmarkStats = None }
                                    )

                                if isTextMode && scenarioNum < scenarios.Length then
                                    Console.waitForEnter options.Quick "Press ENTER for next scenario..."

                                return outcome :: prevOutcomes
                        })
                    (Task.FromResult [])

            let successCount =
                outcomes
                |> List.filter (function
                    | Success _ -> true
                    | _ -> false)
                |> List.length

            // JSON output mode
            // Build JSON result for export/output
            let jsonResult =
                { Timestamp = DateTime.UtcNow
                  TotalScenarios = scenariosToRun.Length
                  SuccessfulScenarios = aggregateMetrics.SuccessfulScenarios
                  FailedScenarios = aggregateMetrics.FailedScenarios
                  TotalRetrievalTimeMs = aggregateMetrics.TotalRetrievalTimeMs
                  TotalDocumentsRetrieved = aggregateMetrics.TotalDocumentsRetrieved
                  Scenarios = jsonResults |> Seq.toList }

            // Handle export if specified (works in both text and JSON mode)
            match options.ExportPath with
            | Some path when options.OutputFormat = Text ->
                let exportOptions = JsonSerializerOptions(WriteIndented = true)
                let json = JsonSerializer.Serialize(jsonResult, exportOptions)
                File.WriteAllText(path, json)
                Console.success $"Results exported to: {path}"
            | _ -> ()

            if options.OutputFormat = Json then
                outputJson jsonResult options.ExportPath
                return (if aggregateMetrics.FailedScenarios > 0 then 1 else 0)
            else
                // Text output mode - final summary
                Console.header "Demo Complete!"

                // Show aggregate metrics
                displayAggregateSummary aggregateMetrics

                Console.subHeader $"RAG Features Demonstrated ({scenariosToRun.Length} Scenarios)"
                Console.info ""
                Console.dim "Part 1 - Core Features:"
                Console.bullet "Basic semantic search with cosine similarity"
                Console.bullet "Hybrid search (semantic + BM25 keyword matching)"
                Console.bullet "Query expansion for broader recall"
                Console.bullet "Query routing for automatic strategy selection"
                Console.bullet "Time decay scoring to favor recent documents"
                Console.info ""
                Console.dim "Part 2 - Advanced Features:"
                Console.bullet "Metadata filtering with operators"
                Console.bullet "Reciprocal Rank Fusion (RRF)"
                Console.bullet "Embedding cache with LRU eviction"
                Console.bullet "Cross-encoder reranking"
                Console.bullet "LLM reranking for maximum precision"
                Console.bullet "Contextual compression"
                Console.bullet "Sentence window expansion"
                Console.bullet "Fallback chain for low-result scenarios"
                Console.info ""
                Console.dim "Part 3 - Full Pipeline:"
                Console.bullet "All features combined with attribution & metrics"
                Console.info ""
                Console.subHeader "Features Available (require Knowledge Graph setup)"
                Console.bullet "Multi-hop retrieval via knowledge graph"
                Console.bullet "Parent document retrieval"
                Console.bullet "Semantic chunking by boundaries"
                Console.info ""

                if aggregateMetrics.FailedScenarios > 0 then
                    Console.warning $"{aggregateMetrics.FailedScenarios} scenario(s) failed - check output above"
                else
                    Console.success "All scenarios completed successfully!"

                Console.info ""
                Console.info "Run 'tars demo-rag' anytime to see this demo again."
                Console.info "Run 'tars demo-rag --quick' for non-interactive mode."
                Console.info "Run 'tars demo-rag --verbose' for detailed output."
                Console.info "Run 'tars demo-rag --scenario 1,5,14' to run specific scenarios."
                Console.info "Run 'tars demo-rag --output json' for JSON output."
                Console.info "Run 'tars demo-rag --benchmark 5' to benchmark each scenario."

                return (if aggregateMetrics.FailedScenarios > 0 then 1 else 0)
    }

/// <summary>Legacy entry point for backward compatibility</summary>
/// <param name="logger">Serilog logger</param>
/// <param name="quick">Skip interactive prompts</param>
/// <returns>Exit code (0 = success, 1 = failure)</returns>
let run (logger: ILogger) (quick: bool) =
    runWithOptions logger { defaultOptions with Quick = quick }
