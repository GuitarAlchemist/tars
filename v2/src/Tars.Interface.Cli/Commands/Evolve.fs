module Tars.Interface.Cli.Commands.Evolve

open System
open System.IO
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Core.Knowledge
open Tars.Kernel
open Tars.Evolution
open System.Net.Http
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Cortex
open Tars.Security
open Tars.Connectors.EpisodeIngestion
open Tars.Interface.Cli // For ConfigurationLoader
open Tars.Interface.Cli.SpectreUI
open Tars.Knowledge

type EvolveOptions =
    { MaxIterations: int
      Quiet: bool
      DemoMode: bool
      Verbose: bool
      Model: string option
      Trace: bool
      Budget: decimal option
      DisableGraphiti: bool
      PlanPath: string option
      Focus: string option
      ResearchEnhanced: bool }

let run (logger: ILogger) (options: EvolveOptions) =
    task {
        // Load configuration
        let config = ConfigurationLoader.load ()

        if not options.Quiet then
            RichOutput.printBanner ()
            RichOutput.dim $"   [Config] Loaded settings from {ConfigurationDefaults.getTarsHome ()}"

            if config.Llm.Provider <> "Ollama" then
                RichOutput.info $"   [Config] Provider: {config.Llm.Provider}"

        logger.Information("Starting TARS v2 Evolution Engine...")

        let registry = AgentRegistry()
        let curriculumId = Guid.NewGuid()
        let executorId = Guid.NewGuid()

        let model = options.Model |> Option.defaultValue config.Llm.Model

        // Log model and agent info
        if not options.Quiet then
            LlmDisplay.printModel model

        logger.Information("Curriculum Agent: {Id}", curriculumId)
        logger.Information("Executor Agent: {Id}", executorId)

        let curriculumCapabilities =
            [ { Kind = CapabilityKind.Planning
                Description = "Can generate curriculum and plan tasks"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.78
                Reputation = Some 0.5 }
              { Kind = CapabilityKind.Reasoning
                Description = "Can reason about task difficulty and progression"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.72
                Reputation = Some 0.5 } ]

        let curriculumAgent =
            AgentFactory.create
                curriculumId
                "Curriculum"
                "0.1.0"
                model
                "You are a curriculum agent that generates progressively harder coding tasks."
                []
                curriculumCapabilities

        let executorCapabilities =
            [ { Kind = CapabilityKind.CodeGeneration
                Description = "Can write code to solve tasks"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.88
                Reputation = Some 0.55 }
              { Kind = CapabilityKind.TaskExecution
                Description = "Can execute tasks using tools"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.82
                Reputation = Some 0.55 }
              { Kind = CapabilityKind.Reasoning
                Description = "Can reflect on solutions and improve them"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.74
                Reputation = Some 0.5 } ]

        // Initialize Tools
        let toolRegistry = Tars.Tools.ToolRegistry()
        toolRegistry.RegisterAssembly(typeof<Tars.Tools.ToolRegistry>.Assembly)

        let semanticTools =
            [ "explore_project"
              "read_code"
              "patch_code"
              "write_code"
              "git_commit"
              "git_status"
              "git_diff"
              "think_step_by_step"
              "plan_task"
              "summarize"
              "lookup_docs"
              "run_tests"
              "generate_test"
              "analyze_code"
              "build_project"
              "improve_prompt"
              "reflect_on_task"
              "report_progress"
              "run_metascript"
              "parse_metascript"
              "create_metascript"
              "create_dynamic_tool"
              "create_grammar"
              "create_block"
              "list_extensions"
              "list_files"
              "search_code"
              "count_lines"
              "find_todos"
              "delegate_task"
              "request_review"
              "query_agent"
              "list_agents"
              "agent_status"
              "debug_hint"
              "trace_error"
              "explain_error"
              "generate_docs"
              "update_readme"
              "save_note"
              "recall_note"
              "list_notes"
              "list_models"
              "switch_model"
              "recommend_model"
              "pull_model"
              "model_info"
              "get_active_model"
              "validate_tool"
              "test_tool"
              "list_tool_errors"
              "introspect_tool"
              "list_all_tools"
              "get_env"
              "list_env"
              "run_shell"
              "http_get"
              "search_web"
              "get_system_info"
              "get_time"
              "get_working_dir"
              "set_working_dir"
              "wikidata_search"
              "wikidata_entity"
              "nuget_search"
              "github_repo"
              "schema_org"
              "log_action"
              "get_action_log"
              "measure_time"
              "sleep_ms"
              "generate_id"
              "format_json"
              "hash_text"
              "base64_encode"
              "base64_decode"
              // Phase 21: Advanced Tools
              "retry_with_backoff"
              "reset_retry"
              "circuit_breaker"
              "cache_set"
              "cache_get"
              "cache_clear"
              "cache_stats"
              "record_metric"
              "get_metrics"
              "health_check"
              "report_status" ]
            |> List.choose toolRegistry.Get

        let executorAgent =
            AgentFactory.create
                executorId
                "Executor"
                "0.1.0"
                model
                "You are a coding assistant that solves programming tasks step by step. Use the provided tools to explore, modify, and save code. Use write_code to save your solutions and git_commit to commit changes."
                semanticTools
                executorCapabilities

        // Define Reviewer Agent for code review
        let reviewerId = Guid.NewGuid()

        let reviewerCapabilities =
            [ { Kind = CapabilityKind.Reasoning
                Description = "Can analyze code for quality and correctness"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.76
                Reputation = Some 0.45 }
              { Kind = CapabilityKind.Planning
                Description = "Can suggest improvements and identify issues"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.66
                Reputation = Some 0.45 } ]

        let reviewerTools =
            [ "read_code"; "git_diff"; "git_status" ] |> List.choose toolRegistry.Get

        let reviewerAgent =
            AgentFactory.create
                reviewerId
                "Reviewer"
                "0.1.0"
                model
                "You are a code reviewer. Analyze code for bugs, style issues, and improvements. Be constructive and specific. Format your review as: APPROVED if code is good, or NEEDS_WORK with specific feedback."
                reviewerTools
                reviewerCapabilities

        registry.Register(curriculumAgent)
        registry.Register(executorAgent)
        registry.Register(reviewerAgent)

        // Initialize LLM Service
        // Ensure secret is registered
        CredentialVault.registerSecret "OLLAMA_BASE_URL" "http://localhost:11434"

        match CredentialVault.getSecret "OLLAMA_BASE_URL" with
        | Microsoft.FSharp.Core.Result.Ok _ -> ()
        | Microsoft.FSharp.Core.Result.Error e -> logger.Warning("Secret registration FAILED: {Error}", e)

        let useLlamaCpp =
            config.Llm.Provider.Equals("LlamaCpp", StringComparison.OrdinalIgnoreCase)
            || config.Llm.Provider.Equals("llama.cpp", StringComparison.OrdinalIgnoreCase)

        let routingCfg: RoutingConfig =
            let ollamaUri =
                config.Llm.BaseUrl |> Option.defaultValue "http://localhost:11434" |> Uri

            // Helper to get API key if provider matches, else fall back to secret
            let getKey provider secretName =
                if config.Llm.Provider.Equals(provider, StringComparison.OrdinalIgnoreCase) then
                    config.Llm.ApiKey
                    |> Option.orElse (CredentialVault.getSecret secretName |> Result.toOption)
                else
                    CredentialVault.getSecret secretName |> Result.toOption

            { RoutingConfig.Default with
                OllamaBaseUri = ollamaUri
                VllmBaseUri = ollamaUri
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = model
                DefaultVllmModel = model
                DefaultOpenAIModel = if config.Llm.Provider = "OpenAI" then model else "gpt-4o"
                DefaultGoogleGeminiModel =
                    if config.Llm.Provider = "Google" then
                        model
                    else
                        "gemini-pro"
                DefaultAnthropicModel =
                    if config.Llm.Provider = "Anthropic" then
                        model
                    else
                        "claude-3-opus-20240229"
                DefaultEmbeddingModel = config.Llm.EmbeddingModel
                OpenAIKey = getKey "OpenAI" "OPENAI_API_KEY"
                GoogleGeminiKey = getKey "Google" "GOOGLE_API_KEY"
                AnthropicKey = getKey "Anthropic" "ANTHROPIC_API_KEY"
                LlamaCppBaseUri =
                    if useLlamaCpp then
                        config.Llm.LlamaCppUrl |> Option.map Uri
                    else
                        None
                DefaultLlamaCppModel =
                    if useLlamaCpp && config.Llm.LlamaCppUrl.IsSome then
                        Some model
                    else
                        None
                DefaultContextWindow = if config.Llm.ContextWindow > 0 then Some config.Llm.ContextWindow else None
                DefaultTemperature = None }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)
        let baseLlmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        // Setup Tracing if enabled
        let traceRecorder = TraceRecorder()

        let llmService =
            if options.Trace then
                if not options.Quiet then
                    RichOutput.info "🔍 Tracing enabled"

                TracingLlmService(baseLlmService, traceRecorder) :> ILlmService
            else
                baseLlmService

        if options.Trace then
            let! traceId = (traceRecorder :> ITraceRecorder).StartTraceAsync() |> Async.StartAsTask
            logger.Information("Started trace {TraceId}", traceId)

        // Initialize Vector Store
        let vectorStore =
            match config.Memory.PostgresConnectionString with
            | Some connStr ->
                if not options.Quiet then
                    RichOutput.info "📁 Using Postgres Vector Store"

                // Align vector dimension with embedding model
                let vecDim =
                    match config.Llm.EmbeddingModel.ToLowerInvariant() with
                    | m when m.Contains("nomic") -> 768
                    | m when m.Contains("mxbai") -> 512
                    | m when m.Contains("text-embedding-3-large") -> 3072
                    | m when m.Contains("text-embedding-3-small") -> 1536
                    | _ -> 1536

                PostgresVectorStore(connStr, vecDim) :> IVectorStore
            | None ->
                let dbPath = config.Memory.VectorStorePath
                let dbDir = Path.GetDirectoryName(dbPath)

                if not (Directory.Exists(dbDir)) then
                    Directory.CreateDirectory(dbDir) |> ignore

                if not options.Quiet then
                    RichOutput.info $"📁 Using persistent memory at {dbPath}"

                Tars.Cortex.SqliteVectorStore(dbPath) :> IVectorStore

        // Initialize Capability Store and register agents
        let capabilityStore = CapabilityStore(vectorStore, llmService)

        if not options.Quiet then
            RichOutput.info "🧠 Registering agent capabilities..."

        for cap in curriculumAgent.Capabilities do
            if not options.Quiet then
                RichOutput.dim $"   Process: {cap.Kind}..."

            do! capabilityStore.RegisterAsync(curriculumAgent.Id, cap)

        for cap in executorAgent.Capabilities do
            if not options.Quiet then
                RichOutput.dim $"   Process: {cap.Kind}..."

            do! capabilityStore.RegisterAsync(executorAgent.Id, cap)

        for cap in reviewerAgent.Capabilities do
            if not options.Quiet then
                RichOutput.dim $"   Process: {cap.Kind}..."

            do! capabilityStore.RegisterAsync(reviewerAgent.Id, cap)

        // Initialize Knowledge Base
        let knowledgePath =
            let candidates =
                [ Path.Combine(Environment.CurrentDirectory, "knowledge")
                  Path.Combine(Environment.CurrentDirectory, "v2", "knowledge")
                  Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "knowledge") ]

            candidates
            |> List.tryFind Directory.Exists
            |> Option.defaultValue candidates.[0]

        let knowledgeBase = KnowledgeBase(knowledgePath)

        if not options.Quiet then
            RichOutput.info $"📚 Knowledge base: {knowledgeBase.BasePath}"

        // Initialize temporal knowledge graph with persistence
        let knowledgeGraph =
            let graphDir =
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "knowledge")

            Directory.CreateDirectory(graphDir) |> ignore
            let graphPath = Path.Combine(graphDir, "temporal_graph.json")
            let graph = Tars.Core.TemporalKnowledgeGraph.TemporalGraph()

            if File.Exists graphPath then
                try
                    graph.Load(graphPath) |> ignore
                with ex ->
                    logger.Warning("Failed to load knowledge graph: {Message}", ex.Message)

            graph

        let knowledgeGraphPath =
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".tars",
                "knowledge",
                "temporal_graph.json"
            )

        if not options.Quiet then
            RichOutput.info
                $"🧠 Knowledge graph: {knowledgeGraphPath} (facts: {knowledgeGraph.GetCurrentFacts().Length})"

            match config.Memory.PostgresConnectionString with
            | Some _ -> RichOutput.info "🧠 Capability index: Postgres (agent_capabilities)"
            | None -> RichOutput.info $"🧠 Capability index: {config.Memory.VectorStorePath} (agent_capabilities)"

        try
            let! ledgerOpt =
                task {
                    let init (ledger: KnowledgeLedger) =
                        task {
                            do! ledger.Initialize()
                            return ledger
                        }

                    let tryInit ledger =
                        task {
                            try
                                let! ready = init ledger
                                return Some ready
                            with ex ->
                                logger.Warning("Knowledge ledger init failed: {Message}", ex.Message)
                                return None
                        }

                    match config.Memory.PostgresConnectionString with
                    | Some connStr ->
                        let storage =
                            PostgresLedgerStorage.createWithConnectionString connStr :> ILedgerStorage

                        let! ledger = tryInit (KnowledgeLedger(storage))

                        match ledger with
                        | Some _ -> return ledger
                        | None -> return! tryInit (KnowledgeLedger.createInMemory ())
                    | None -> return! tryInit (KnowledgeLedger.createInMemory ())
                }

            let runId = ledgerOpt |> Option.map (fun _ -> RunId.New())

            if not options.Quiet then
                match ledgerOpt with
                | Some _ -> RichOutput.info "📒 Knowledge ledger initialized"
                | None -> RichOutput.dim "📒 Knowledge ledger unavailable"

            // Session budget for evolution
            let budget =
                BudgetGovernor(
                    { Budget.Infinite with
                        MaxTokens = Some 1000000<token>
                        MaxMoney =
                            options.Budget
                            |> Option.map (fun m -> m * 1m<usd>)
                            |> Option.orElse (Some 10.0m<usd>) }
                )

            let epistemic =
                if options.DemoMode then
                    None
                else
                    // Create a legacy knowledge graph for code context retrieval
                    // TODO: Migrate to TemporalKnowledgeGraph once EpistemicGovernor is updated
                    let legacyGraph = LegacyKnowledgeGraph.TemporalGraph()
                    Some(Tars.Cortex.EpistemicGovernor(llmService, Some legacyGraph, Some budget) :> IEpistemicGovernor)

            // Initialize Output Guard
            let outputGuard = OutputGuard.defaultGuard

            let evaluator =
                SemanticEvaluation(
                    llmService,
                    minConfidence = 0.6,
                    logger = fun msg -> logger.Information("{Evaluation}", msg)
                )
                :> IEvaluationStrategy

            let evoState: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = Tars.Core.AgentId curriculumId
                  ExecutorAgentId = Tars.Core.AgentId executorId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            if not options.Quiet then
                RichOutput.info "🧠 Ingesting codebase into Knowledge Graph..."

            // Initialize Semantic Memory
            let embedder: Embedder =
                fun text ->
                    async {
                        try
                            let! res = llmService.EmbedAsync text |> Async.AwaitTask
                            return res
                        with _ ->
                            return Array.empty
                    }

            let storageRoot =
                Path.Combine(Environment.CurrentDirectory, "knowledge", "semantic_memory")

            let kernel = KernelBootstrap.createKernel storageRoot embedder llmService

            if not options.Quiet then
                RichOutput.info $"🧠 Semantic Memory initialized at {storageRoot}"

            // Initialize Pre-LLM Pipeline
            let policyNames =
                config.PreLlm.DefaultPolicies
                |> List.append [ "no_destructive_commands" ]
                |> List.distinct

            let policyStage = PolicyGateStage(policyNames) :> IPreLlmStage

            let intentClassifier =
                if config.PreLlm.UseIntentClassifier then
                    LlmIntentClassifier(llmService) :> IIntentClassifier
                else
                    NoopIntentClassifier() :> IIntentClassifier

            let intentStage = IntentClassifierStage(intentClassifier) :> IPreLlmStage

            let entropyMonitor = EntropyMonitor()
            let compressor = ContextCompressor(llmService, entropyMonitor)
            let summarizerStage = ContextSummarizerStage(compressor) :> IPreLlmStage

            let preLlmPipeline = PreLlmPipeline([ policyStage; intentStage; summarizerStage ])

            // Initialize Memory Buffer (Capacitor)
            let onFlush (items: Engine.MemoryItem list) =
                task {
                    for item in items do
                        match item with
                        | Engine.Belief(col, id, vec, pay) -> do! vectorStore.SaveAsync(col, id, vec, pay)
                        | Engine.Legacy(col, id, vec, pay) -> do! vectorStore.SaveAsync(col, id, vec, pay)
                }
                :> Task

            let memoryBuffer =
                BufferAgent<Engine.MemoryItem>(10, TimeSpan.FromSeconds(5.0), onFlush)

            let evoCtx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = llmService
                  VectorStore = vectorStore
                  SemanticMemory = Some kernel.SemanticMemory
                  Epistemic = epistemic
                  PreLlm = Some preLlmPipeline
                  Budget = Some budget
                  OutputGuard = Some outputGuard
                  KnowledgeBase = Some knowledgeBase
                  KnowledgeGraph = Some knowledgeGraph
                  MemoryBuffer = Some memoryBuffer
                  EpisodeService =
                    match options.DisableGraphiti, config.Memory.GraphitiUrl with
                    | true, _ -> None
                    | _, None -> None
                    | _, Some url ->
                        try
                            RichOutput.info $"Graphiti enabled at {url}"
                            Some(createServiceWithUrl url)
                        with ex ->
                            logger.Warning("Graphiti ingestion unavailable: {Message}", ex.Message)
                            None
                  Ledger = ledgerOpt
                  Evaluator = Some evaluator
                  RunId = runId
                  Logger =
                    fun s ->
                        logger.Information("{Evolution}", s)

                        if options.Verbose then
                            RichOutput.dim $"   [LOG] {s}"
                  Verbose = options.Verbose
                  ShowSemanticMessage =
                    match options.Quiet with
                    | true -> (fun _ _ -> ())
                    | false -> DemoVisualization.showSemanticMessage
                  Focus = options.Focus
                  ToolRegistry = Some toolRegistry
                  ResearchEnhanced = options.ResearchEnhanced }

            // Load Plan if provided
            let initialTasks =
                let bootstrapTasks = 
                    match options.Focus with
                    | Some f when f.Contains("analysis tools") ->
                        [ { Tars.Evolution.TaskDefinition.Id = Guid.NewGuid()
                            DifficultyLevel = 1
                            Goal = "Create a new dynamic tool named 'list_fsharp_files' that lists all .fs files in a specified directory using System.IO.Directory.GetFiles. Register it using create_dynamic_tool."
                            Constraints = ["Use create_dynamic_tool"; "Include proper error handling"]
                            ValidationCriteria = "Tool is successfully registered and shows up in list_extensions"
                            Timeout = TimeSpan.FromMinutes(5.0)
                            Score = 1.0 } ]
                    | _ -> []

                match options.PlanPath with
                | Some path ->
                    if File.Exists path then
                        try
                            let json = File.ReadAllText(path)
                            let options = System.Text.Json.JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                            let rawTasks = System.Text.Json.JsonSerializer.Deserialize<{| Instructions: string |}[]>(json, options)
                            let tasks = 
                                rawTasks
                                |> Array.map (fun t -> 
                                     { Tars.Evolution.TaskDefinition.Id = Guid.NewGuid()
                                       DifficultyLevel = 1
                                       Goal = if String.IsNullOrWhiteSpace(t.Instructions) then "Unknown" else t.Instructions
                                       Constraints = ["Use provided tools only"]
                                       ValidationCriteria = "Task completed successfully"
                                       Timeout = TimeSpan.FromMinutes(10.0)
                                       Score = 1.0 })
                                |> Array.toList
                            bootstrapTasks @ tasks
                        with ex ->
                             logger.Error(ex, "Failed to load plan")
                             bootstrapTasks
                    else
                         logger.Warning("Plan file not found: {Path}", path)
                         bootstrapTasks
                | None -> bootstrapTasks

            let mutable currentState = 
                { evoState with TaskQueue = initialTasks }

            for i in 1 .. options.MaxIterations do
                if not options.Quiet then
                    Evolution.printGeneration
                        currentState.Generation
                        currentState.CompletedTasks.Length
                        (Some options.MaxIterations)

                let! nextState =
                    try
                        Engine.step evoCtx currentState
                    with
                    | :? System.Threading.Tasks.TaskCanceledException as ex ->
                        logger.Warning("Evolution step timed out: {Message}", ex.Message)

                        if not options.Quiet then
                            TaskDisplay.printFailure
                                $"TIMEOUT: Task took too long, skipping to next task..."
                                TimeSpan.Zero

                        let failedResult: TaskResult =
                            { TaskId =
                                currentState.CurrentTask
                                |> Option.map (fun t -> t.Id)
                                |> Option.defaultValue (Guid.NewGuid())
                              TaskGoal =
                                currentState.CurrentTask
                                |> Option.map (fun t -> t.Goal)
                                |> Option.defaultValue "Unknown"
                              ExecutorId = currentState.ExecutorAgentId
                              Success = false
                              Output = $"Task timed out: %s{ex.Message}"
                              ExecutionTrace = [ "TIMEOUT" ]
                              Duration = TimeSpan.FromSeconds(120.0)
                              Evaluation = None }

                        Task.FromResult
                            { currentState with
                                CompletedTasks = failedResult :: currentState.CompletedTasks
                                CurrentTask = None }
                    | ex ->
                        logger.Error(ex, "Evolution Step Failed: {Message}", ex.Message)

                        if not options.Quiet then
                            TaskDisplay.printFailure $"ERROR: %s{ex.Message}" TimeSpan.Zero

                        let failedResult: TaskResult =
                            { TaskId =
                                currentState.CurrentTask
                                |> Option.map (fun t -> t.Id)
                                |> Option.defaultValue (Guid.NewGuid())
                              TaskGoal =
                                currentState.CurrentTask
                                |> Option.map (fun t -> t.Goal)
                                |> Option.defaultValue "Unknown"
                              ExecutorId = currentState.ExecutorAgentId
                              Success = false
                              Output = $"Error: %s{ex.Message}"
                              ExecutionTrace = [ ex.GetType().Name ]
                              Duration = TimeSpan.Zero
                              Evaluation = None }

                        Task.FromResult
                            { currentState with
                                CompletedTasks = failedResult :: currentState.CompletedTasks
                                CurrentTask = None }

                currentState <- nextState

                try
                    knowledgeGraph.Save(knowledgeGraphPath)
                with ex ->
                    logger.Warning("Failed to persist knowledge graph: {Message}", ex.Message)

                match currentState.CurrentTask with
                | Some task ->
                    if not options.Quiet then
                        TaskDisplay.printTask task.Goal task.Constraints
                        Evolution.printThinking "Executor"
                | None when not currentState.CompletedTasks.IsEmpty ->
                    let lastResult = currentState.CompletedTasks.Head

                    if not options.Quiet then
                        if lastResult.Success then
                            TaskDisplay.printSuccess lastResult.Output lastResult.Duration true
                        else
                            TaskDisplay.printFailure lastResult.Output lastResult.Duration

                | None -> ()

                do! Task.Delay(500)

            if not options.Quiet then
                let consumedTokens =
                    match budget.Consumed with
                    | { Tokens = t } -> int t

                Summary.printEvolution
                    currentState.Generation
                    currentState.CompletedTasks.Length
                    consumedTokens
                    TimeSpan.Zero

                RichOutput.info
                    $"🧠 Knowledge graph facts: {knowledgeGraph.GetCurrentFacts().Length} (persisted to {knowledgeGraphPath})"

            if options.Trace then
                let timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
                let tracePath = $"trace_{timestamp}.json"

                if not options.Quiet then
                    RichOutput.info $"💾 Saving trace to {tracePath}..."

                try
                    do! traceRecorder.SaveToFileAsync(tracePath) |> Async.StartAsTask

                    if not options.Quiet then
                        RichOutput.info "Trace saved successfully."
                with ex ->
                    RichOutput.error $"Failed to save trace: {ex.Message}"
                    logger.Error(ex, "Trace Saving Failed")

            // Save knowledge graph
            try
                knowledgeGraph.Save(knowledgeGraphPath)

                if not options.Quiet then
                    RichOutput.info $"🧠 Knowledge graph saved to {knowledgeGraphPath}"
            with ex ->
                logger.Warning("Failed to save knowledge graph: {Message}", ex.Message)

            return 0
        with ex ->
            RichOutput.error $"Evolution failed: {ex.Message}"
            logger.Error(ex, "Evolution Engine Failed")
            return 1
    }
