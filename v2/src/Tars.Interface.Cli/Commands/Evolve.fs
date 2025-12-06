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
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Cortex
open Tars.Security

type EvolveOptions =
    { MaxIterations: int
      Quiet: bool
      DemoMode: bool
      Verbose: bool
      Model: string option
      Trace: bool }

/// Console output helpers with colors
module ConsoleUI =
    let private writeColored (color: ConsoleColor) (text: string) =
        let prev = Console.ForegroundColor
        Console.ForegroundColor <- color
        Console.Write(text)
        Console.ForegroundColor <- prev

    let success text = writeColored ConsoleColor.Green text
    let info text = writeColored ConsoleColor.Cyan text
    let warning text = writeColored ConsoleColor.Yellow text
    let error text = writeColored ConsoleColor.Red text
    let dim text = writeColored ConsoleColor.DarkGray text

    let printBanner () =
        Console.ForegroundColor <- ConsoleColor.Cyan
        printfn ""
        printfn "╔════════════════════════════════════════════════════════════╗"
        printfn "║           🧠 TARS v2 Evolution Engine                      ║"
        printfn "║           Self-Improving Agent System                      ║"
        printfn "╚════════════════════════════════════════════════════════════╝"
        Console.ResetColor()
        printfn ""

    let printGeneration (gen: int) (taskCount: int) =
        info $"┌─ Generation {gen} "
        dim $"({taskCount} tasks completed)\n"

    let printTask (goal: string) =
        Console.Write("│ 📋 Task: ")

        success (
            if goal.Length > 60 then
                goal.Substring(0, 57) + "..."
            else
                goal
        )

        printfn ""

    let printProgress (step: string) = dim $"│   ⏳ {step}\n"

    let printSuccess (output: string) =
        Console.Write("│ ✅ Result: ")

        let preview =
            if output.Length > 80 then
                output.Substring(0, 77) + "..."
            else
                output

        printfn "%s" (preview.Replace("\n", " "))

    let printError (err: string) =
        Console.Write("│ ")
        error $"❌ Error: {err}\n"

    let printDivider () =
        dim "└────────────────────────────────────────────────────────────\n"

    let printSummary (completed: int) (budget: BudgetGovernor option) =
        printfn ""
        info "═══ Evolution Summary ═══\n"
        printfn "  Tasks Completed: %d" completed

        match budget with
        | Some b ->
            let consumed = b.Consumed
            printfn "  Tokens Used: %d" (int consumed.Tokens)
        | None -> ()

        printfn ""

let run (logger: ILogger) (options: EvolveOptions) =
    task {
        if not options.Quiet then
            ConsoleUI.printBanner ()

        logger.Information("Starting TARS v2 Evolution Engine...")

        logger.Information("Starting TARS v2 Evolution Engine...")

        let registry = AgentRegistry()
        let curriculumId = Guid.NewGuid()
        let executorId = Guid.NewGuid()

        let model = options.Model |> Option.defaultValue "qwen2.5-coder:1.5b"

        let curriculumCapabilities =
            [ { Kind = CapabilityKind.Planning
                Description = "Can generate curriculum and plan tasks"
                InputSchema = None
                OutputSchema = None }
              { Kind = CapabilityKind.Reasoning
                Description = "Can reason about task difficulty and progression"
                InputSchema = None
                OutputSchema = None } ]

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
                OutputSchema = None }
              { Kind = CapabilityKind.TaskExecution
                Description = "Can execute tasks using tools"
                InputSchema = None
                OutputSchema = None }
              { Kind = CapabilityKind.Reasoning
                Description = "Can reflect on solutions and improve them"
                InputSchema = None
                OutputSchema = None } ]

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
              "report_progress" ]
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
                OutputSchema = None }
              { Kind = CapabilityKind.Planning
                Description = "Can suggest improvements and identify issues"
                InputSchema = None
                OutputSchema = None } ]

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

        let routingCfg: RoutingConfig =
            { OllamaBaseUri = Uri("http://localhost:11434/")
              VllmBaseUri = Uri("http://localhost:11434/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = model
              DefaultVllmModel = model
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text" }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)
        let baseLlmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        // Setup Tracing if enabled
        let traceRecorder = TraceRecorder()

        let llmService =
            if options.Trace then
                if not options.Quiet then
                    ConsoleUI.info "🔍 Tracing enabled\n"

                TracingLlmService(baseLlmService, traceRecorder) :> ILlmService
            else
                baseLlmService

        if options.Trace then
            let! traceId = (traceRecorder :> ITraceRecorder).StartTraceAsync() |> Async.StartAsTask
            logger.Information("Started trace {TraceId}", traceId)

        // Initialize Vector Store
        let tarsDir =
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")

        if not (Directory.Exists(tarsDir)) then
            Directory.CreateDirectory(tarsDir) |> ignore

        let dbPath = Path.Combine(tarsDir, "memory.db")
        let vectorStore = Tars.Cortex.SqliteVectorStore(dbPath) :> IVectorStore

        if not options.Quiet then
            ConsoleUI.info $"📁 Using persistent memory at {dbPath}\n"

        // Initialize Capability Store and register agents
        let capabilityStore = CapabilityStore(vectorStore, llmService)

        if not options.Quiet then
            ConsoleUI.info "🧠 Registering agent capabilities...\n"

        for cap in curriculumAgent.Capabilities do
            do! capabilityStore.RegisterAsync(curriculumAgent.Id, cap)

        for cap in executorAgent.Capabilities do
            do! capabilityStore.RegisterAsync(executorAgent.Id, cap)

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
            ConsoleUI.info $"📚 Knowledge base: {knowledgeBase.BasePath}\n"

        try
            // Session budget for evolution
            let budget =
                BudgetGovernor(
                    { Budget.Infinite with
                        MaxTokens = Some 1000000<token>
                        MaxMoney = Some 10.0m<usd> }
                )

            let epistemic =
                if options.DemoMode then
                    None
                else
                    Some(Tars.Cortex.EpistemicGovernor(llmService, None, Some budget) :> IEpistemicGovernor)

            // Initialize Output Guard
            // Use basic guard for now. In future, we can compose with LlmOutputGuardAnalyzer
            let outputGuard = OutputGuard.defaultGuard

            let evoState: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = AgentId curriculumId
                  ExecutorAgentId = AgentId executorId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            // Initialize Knowledge Graph and Ingest Codebase
            // let knowledgeGraph = Tars.Core.LegacyKnowledgeGraph.TemporalGraph()

            if not options.Quiet then
                ConsoleUI.info "🧠 Ingesting codebase into Knowledge Graph...\n"

            // Ingest current directory
            // do!
            //     CodeGraphIngestor.ingestDirectory knowledgeGraph Environment.CurrentDirectory
            //     |> Async.StartAsTask

            // if not options.Quiet then
            //     ConsoleUI.info $"   Loaded {knowledgeGraph.NodeCount} nodes and {knowledgeGraph.EdgeCount} edges\n"

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
                ConsoleUI.info $"🧠 Semantic Memory initialized at {storageRoot}\n"

            // Initialize Pre-LLM Pipeline
            let safetyStage = SafetyFilterStage() :> IPreLlmStage
            let intentStage = IntentClassifierStage() :> IPreLlmStage
            let entropyMonitor = EntropyMonitor()
            let compressor = ContextCompressor(llmService, entropyMonitor)
            let summarizerStage = ContextSummarizerStage(compressor) :> IPreLlmStage

            let preLlmPipeline = PreLlmPipeline([ safetyStage; intentStage; summarizerStage ])

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
                  KnowledgeGraph = None // Some knowledgeGraph
                  MemoryBuffer = Some memoryBuffer
                  Logger = fun s -> logger.Information("{Evolution}", s)
                  Verbose = options.Verbose
                  ShowSemanticMessage = DemoVisualization.showSemanticMessage }

            let mutable currentState = evoState

            for i in 1 .. options.MaxIterations do
                if not options.Quiet then
                    ConsoleUI.printGeneration currentState.Generation currentState.CompletedTasks.Length

                let! nextState =
                    try
                        Engine.step evoCtx currentState
                    with ex ->
                        logger.Error(ex, "Evolution Step Failed")
                        printfn "CRITICAL ERROR: %s" ex.Message
                        printfn "%s" ex.StackTrace
                        reraise ()

                currentState <- nextState

                match currentState.CurrentTask with
                | Some task ->
                    if not options.Quiet then
                        ConsoleUI.printTask task.Goal
                        ConsoleUI.printProgress "Generating solution..."
                | None when not currentState.CompletedTasks.IsEmpty ->
                    let lastResult = currentState.CompletedTasks.Head

                    if not options.Quiet then
                        if lastResult.Success then
                            ConsoleUI.printSuccess lastResult.Output
                        else
                            ConsoleUI.printError lastResult.Output

                        ConsoleUI.printDivider ()

                // Auto-saved by SqliteVectorStore
                | None -> ()

                do! Task.Delay(500)

            if not options.Quiet then
                ConsoleUI.printSummary currentState.CompletedTasks.Length (Some budget)

            if options.Trace then
                let timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
                let tracePath = $"trace_{timestamp}.json"

                if not options.Quiet then
                    ConsoleUI.info $"💾 Saving trace to {tracePath}...\n"

                try
                    do! traceRecorder.SaveToFileAsync(tracePath) |> Async.StartAsTask

                    if not options.Quiet then
                        ConsoleUI.info "Trace saved successfully.\n"
                with ex ->
                    ConsoleUI.error $"Failed to save trace: {ex.Message}\n"
                    logger.Error(ex, "Trace Saving Failed")

            return 0
        with ex ->
            ConsoleUI.error $"Evolution failed: {ex.Message}\n"
            logger.Error(ex, "Evolution Engine Failed")
            return 1
    }
