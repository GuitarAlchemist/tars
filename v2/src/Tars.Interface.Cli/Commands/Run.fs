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
                let ctx = Kernel.init ()

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
                let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

                // Initialize Tools
                let tools = Tars.Tools.ToolRegistry()
                tools.Register(Tars.Tools.Standard.RunCommandTool())

                let metaCtx: MetascriptContext =
                    { Llm = llmService
                      Kernel = ctx
                      Tools = tools }

                // Execute
                let! finalState = Engine.run metaCtx workflow Map.empty

                printfn "\nWorkflow Completed."
                printfn "Final Outputs:"

                for kvp in finalState.StepOutputs do
                    printfn "Step %s:" kvp.Key

                    for outKvp in kvp.Value do
                        printfn "  %s: %A" outKvp.Key outKvp.Value

                return 0
            with ex ->
                logger.Error(ex, "Failed to execute script")
                return 1
    }
