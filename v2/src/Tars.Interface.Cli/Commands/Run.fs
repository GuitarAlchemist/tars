module Tars.Interface.Cli.Commands.Run

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Metascript
open Tars.Metascript.Domain
open Tars.Metascript.Engine
open System.Net.Http
open Tars.Cortex

let execute (logger: ILogger) (scriptPath: string) =
    task {
        logger.Information("Executing Metascript: {ScriptPath}", scriptPath)

        if not (File.Exists scriptPath) then
            logger.Error("Script file not found: {ScriptPath}", scriptPath)
            return 1
        else
            try
                let json = File.ReadAllText(scriptPath)
                let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                let workflow = JsonSerializer.Deserialize<Workflow>(json, options)

                // Initialize Kernel & LLM

                let routingCfg: RoutingConfig =
                    { OllamaBaseUri = Uri("http://localhost:11434/")
                      VllmBaseUri = Uri("http://localhost:11434/")
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
                let llmService = DefaultLlmService(httpClient, svcCfg)

                // Initialize Tools
                let tools = Tars.Tools.ToolRegistry()
                tools.RegisterAssembly(typeof<Tars.Tools.ToolRegistry>.Assembly)

                let metaBudget =
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 50000<token>
                            MaxCalls = Some 200<requests> }
                    )

                // Initialize Kernel
                let storageRoot =
                    Path.Combine(Environment.CurrentDirectory, "knowledge", "semantic_memory")

                let embedder (text: string) =
                    async {
                        match! (llmService :> ILlmServiceFunctional).EmbedAsync text with
                        | Result.Ok embedding -> return embedding
                        | Result.Error _ -> return Array.empty<float32>
                    }

                let kernel = KernelBootstrap.createKernel storageRoot embedder llmService

                let metaCtx: MetascriptContext =
                    { Llm = llmService
                      Tools = tools
                      Budget = Some metaBudget
                      VectorStore = None
                      KnowledgeGraph = None
                      SemanticMemory = Some kernel.SemanticMemory
                      RagConfig = RagConfig.Default }
                // Ingest Code Structure
                let! codeStructureInput =
                    task {
                        let kg = TemporalKnowledgeGraph.TemporalGraph()
                        let srcDir = Path.Combine(Environment.CurrentDirectory, "src")

                        if Directory.Exists srcDir then
                            do! AstIngestor.ingestDirectory kg srcDir |> Async.StartAsTask
                            let cs = AstIngestor.extractCodeStructure kg
                            return Map [ "code_structure", box cs ]
                        else
                            return Map.empty
                    }

                // Execute
                let! finalState = Engine.run metaCtx workflow codeStructureInput

                printfn "\nWorkflow Completed."
                printfn "Final Outputs:"

                for kvp in finalState.StepOutputs do
                    printfn "Step %s:" kvp.Key

                    for outKvp in kvp.Value do
                        printfn "  %s: %A" outKvp.Key outKvp.Value

                printfn "\nExecution Trace:"

                for trace in finalState.ExecutionTrace do
                    printfn "- %s (%O ms) outputs=%d" trace.StepId trace.Duration.TotalMilliseconds trace.Outputs.Count

                return 0
            with ex ->
                logger.Error(ex, "Failed to execute script")
                return 1
    }
