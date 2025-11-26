module Tars.Interface.Cli.Commands.Ask

open System
open System.Net.Http
open System.Threading.Tasks
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

let run (prompt: string) =
    task {
        // TODO: Load this from configuration/secrets
        let routingCfg: RoutingConfig =
            { OllamaBaseUri = Uri("http://localhost:11434/")
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = "qwen2.5-coder:latest"
              DefaultVllmModel = "qwen2.5-72b-instruct"
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229" }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }

        // In a real app, HttpClient should be injected/shared
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)

        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        let req: LlmRequest =
            { ModelHint = None // Let routing decide, or pass "code"/"reasoning" based on args
              MaxTokens = Some 1024
              Temperature = Some 0.7
              Messages = [ { Role = Role.User; Content = prompt } ] }

        try
            let! response = llmService.CompleteAsync(req)

            if response.FinishReason = Some "parse_error" then
                printfn "Error parsing response: %A" response.Raw
                return 1
            else
                printfn "%s" response.Text
                return 0
        with ex ->
            printfn "Error: %s" ex.Message
            return 1
    }
