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

type EvolveOptions =
    { MaxIterations: int
      Quiet: bool
      DemoMode: bool
      Model: string option }

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

        let ctx = Kernel.init ()
        let curriculumId = Guid.NewGuid()
        let executorId = Guid.NewGuid()

        let model = options.Model |> Option.defaultValue "qwen2.5-coder:1.5b"

        let curriculumAgent =
            Kernel.createAgent
                curriculumId
                "Curriculum"
                "0.1.0"
                model
                "You are a curriculum agent that generates progressively harder coding tasks."
                []

        let executorAgent =
            Kernel.createAgent
                executorId
                "Executor"
                "0.1.0"
                model
                "You are a coding assistant that solves programming tasks step by step."
                []

        let ctx = Kernel.registerAgent curriculumAgent ctx
        let ctx = Kernel.registerAgent executorAgent ctx

        // Initialize LLM Service
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
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        // Initialize Vector Store
        let memoryStore = Tars.Cortex.InMemoryVectorStore()
        let memoryFile = "memory.json"

        let! loaded = memoryStore.LoadFromFileAsync(memoryFile)

        if loaded && not options.Quiet then
            ConsoleUI.info "📁 Loaded existing memory\n"

        let vectorStore = memoryStore :> IVectorStore

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

            let evoState: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = AgentId curriculumId
                  ExecutorAgentId = AgentId executorId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            let evoCtx: Engine.EvolutionContext =
                { Kernel = ctx
                  Llm = llmService
                  VectorStore = vectorStore
                  Epistemic = epistemic
                  Budget = Some budget
                  KnowledgeBase = Some knowledgeBase
                  Logger = fun s -> logger.Information("{Evolution}", s) }

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

                    // Save memory after each task
                    do! memoryStore.PersistToFileAsync(memoryFile)
                | None -> ()

                do! Task.Delay(500)

            if not options.Quiet then
                ConsoleUI.printSummary currentState.CompletedTasks.Length (Some budget)

            return 0
        with ex ->
            ConsoleUI.error $"Evolution failed: {ex.Message}\n"
            logger.Error(ex, "Evolution Engine Failed")
            return 1
    }
