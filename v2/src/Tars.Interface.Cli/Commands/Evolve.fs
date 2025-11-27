module Tars.Interface.Cli.Commands.Evolve

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Evolution
open System.Net.Http
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

let run (logger: ILogger) =
    task {
        logger.Information("Starting TARS v2 Evolution Engine...")

        let ctx = Kernel.init ()

        let curriculumId = Guid.NewGuid()
        let executorId = Guid.NewGuid()

        let curriculumAgent =
            Kernel.createAgent curriculumId "Curriculum" "0.1.0" "llama3.2" "You are a teacher." []

        let executorAgent =
            Kernel.createAgent executorId "Executor" "0.1.0" "llama3.2" "You are a student." []

        let ctx = Kernel.registerAgent curriculumAgent ctx
        let ctx = Kernel.registerAgent executorAgent ctx

        // Initialize LLM Service
        let routingCfg: RoutingConfig =
            { OllamaBaseUri = Uri("http://localhost:11434/")
              VllmBaseUri = Uri("http://localhost:11434/") // Use Ollama as vLLM substitute
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = "qwen2.5-coder:1.5b"
              DefaultVllmModel = "qwen2.5-coder:1.5b"
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

        if loaded then
            logger.Information("Loaded memory from {File}", memoryFile)
        else
            logger.Information("No existing memory file found at {File}", memoryFile)

        let vectorStore = memoryStore :> IVectorStore

        try
            // 3. Initialize Evolution State
            let evoState: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = AgentId curriculumId
                  ExecutorAgentId = AgentId executorId
                  CompletedTasks = []
                  CurrentTask = None }

            let evoCtx: Engine.EvolutionContext =
                { Kernel = ctx
                  Llm = llmService
                  VectorStore = vectorStore }

            let mutable currentState = evoState

            for i in 1..5 do
                printfn "--- Generation %d ---" currentState.Generation
                let! nextState = Engine.step evoCtx currentState
                currentState <- nextState

                match currentState.CurrentTask with
                | Some task -> printfn "Task Generated: %s" task.Goal
                | None ->
                    let lastResult = currentState.CompletedTasks.Head
                    printfn "Task Completed. Output:\n%s" lastResult.Output
                    printfn "Trace:\n%s" (String.Join("\n", lastResult.ExecutionTrace))
                    printfn "History: %d" currentState.CompletedTasks.Length

                    // Save memory after each successful task
                    do! memoryStore.PersistToFileAsync(memoryFile)
                    logger.Information("Memory saved to {File}", memoryFile)

                do! Task.Delay(1000)

            return 0
        with ex ->
            logger.Error(ex, "Evolution Engine Failed")
            return 1
    }
