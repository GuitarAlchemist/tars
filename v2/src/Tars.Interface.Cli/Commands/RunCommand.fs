namespace Tars.Interface.Cli.Commands

open System
open System.IO
open System.Net.Http
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Llm.Routing
open Tars.Metascript
open Tars.Metascript.Domain
open Tars.Metascript.Config
open Tars.Metascript.Engine

open Tars.Evolution
open Tars.Evolution.Reflection
open Tars.Evolution.Optimizer

module RunCommand =

    // Dummy tool registry for now
    type SimpleToolRegistry() =
        interface IToolRegistry with
            member _.Get(name) = None
            member _.GetAll() = []
            member _.Register(tool) = ()

    let run (logger: ILogger) (workflowPath: string) (shouldOptimize: bool) =
        task {
            try
                logger.Information("Loading workflow from {Path}...", workflowPath)

                if not (File.Exists workflowPath) then
                    logger.Error("Workflow file not found: {Path}", workflowPath)
                    return 1
                else
                    let json = File.ReadAllText(workflowPath)

                    match Parser.parseJson json with
                    | Parser.ValidationError errs ->
                        logger.Error("Workflow validation failed:")

                        for e in errs do
                            logger.Error("- {Error}", e)

                        return 1
                    | Parser.ParseError(l, c, msg) ->
                        logger.Error("JSON parse error at line {Line}, col {Col}: {Message}", l, c, msg)
                        return 1
                    | Parser.Success workflow ->
                        logger.Information("Workflow '{Name}' loaded successfully.", workflow.Name)

                        // Initialize Services
                        let routingConfig =
                            { OllamaBaseUri = Uri("http://localhost:11434")
                              VllmBaseUri = Uri("http://localhost:11434/v1")
                              OpenAIBaseUri = Uri("https://api.openai.com")
                              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com")
                              AnthropicBaseUri = Uri("https://api.anthropic.com")
                              DefaultOllamaModel = "qwen2.5-coder:1.5b"
                              DefaultVllmModel = "qwen2.5-coder:1.5b" // Use Ollama for reasoning too
                              DefaultOpenAIModel = "gpt-4o"
                              DefaultGoogleGeminiModel = "gemini-1.5-pro"
                              DefaultAnthropicModel = "claude-3-5-sonnet-20240620"
                              DefaultEmbeddingModel = "nomic-embed-text"
                              OllamaKey = None
                              VllmKey = None
                              OpenAIKey = None
                              GoogleGeminiKey = None
                              AnthropicKey = None }

                        let llmConfig = { Routing = routingConfig }
                        use httpClient = new HttpClient()
                        httpClient.Timeout <- TimeSpan.FromMinutes(2.0)

                        let llm = DefaultLlmService(httpClient, llmConfig)
                        let tools = SimpleToolRegistry()

                        let ctx: MetascriptContext =
                            { Llm = llm
                              Tools = tools
                              Budget = None
                              VectorStore = None
                              KnowledgeGraph = None
                              SemanticMemory = None
                              RagConfig = RagConfig.Default
                              MacroRegistry = None }

                        logger.Information("Starting execution...")

                        let inputs = Map.empty
                        let mutable executionSuccess = false
                        let mutable execTrace = []
                        let mutable finalOutputs = ""

                        try
                            let! resultState = Engine.run ctx workflow inputs
                            executionSuccess <- true
                            execTrace <- resultState.ExecutionTrace |> List.rev // Engine adds to head

                            // Capture output for reflection
                            let outs =
                                resultState.StepOutputs
                                |> Map.toList
                                |> List.map (fun (k, v) -> sprintf "%s: %A" k v)
                                |> String.concat ", "

                            finalOutputs <- outs

                            logger.Information("Workflow completed.")
                            logger.Information("Outputs: {Outputs}", outs)

                        with ex ->
                            logger.Error(ex, "Workflow execution failed")
                            executionSuccess <- false
                            finalOutputs <- ex.Message
                        // We can still try to optimize a failing workflow!

                        if shouldOptimize then
                            logger.Information("🧠 Starting Self-Evolution Loop...")

                            // 1. Reflection
                            let reflectionAgent = LlmReflectionAgent(llm) :> IReflectionAgent

                            let traceItems =
                                execTrace
                                |> List.map (fun t ->
                                    { Step = t.StepId
                                      Input = "N/A"
                                      Output = sprintf "%A" t.Outputs
                                      DurationMs = int64 t.Duration.TotalMilliseconds })

                            logger.Information("🤔 Reflecting on performance...")
                            let! feedback = reflectionAgent.ReflectAsync(workflow.Description, finalOutputs, traceItems)

                            logger.Information("FEEDBACK: [{Type}] Score: {Score}", feedback.Type, feedback.Score)
                            logger.Information("Comment: {Comment}", feedback.Comment)

                            if feedback.Suggestion.IsSome then
                                logger.Information("Suggestion: {Suggestion}", feedback.Suggestion.Value)

                                // 2. Optimization
                                logger.Information("✨ Optimizing workflow...")
                                let optimizer = LlmWorkflowOptimizer(llm) :> IWorkflowOptimizer
                                let! newWorkflow = optimizer.OptimizeAsync(workflow, feedback)

                                match newWorkflow with
                                | Some nw ->
                                    let newPath = Path.ChangeExtension(workflowPath, null) + "_optimized.json"
                                    let jsonOpts = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
                                    let newJson = System.Text.Json.JsonSerializer.Serialize(nw, jsonOpts)
                                    File.WriteAllText(newPath, newJson)
                                    logger.Information("✅ Optimized workflow saved to: {Path}", newPath)
                                | None -> logger.Error("❌ Optimization failed to generate valid workflow.")
                            else
                                logger.Information("No suggestion provided. Workflow is considered optimal.")

                        return if executionSuccess then 0 else 1
            with ex ->
                logger.Error(ex, "Unexpected error")
                return 1
        }
