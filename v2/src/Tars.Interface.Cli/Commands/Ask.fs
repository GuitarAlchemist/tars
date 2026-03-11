module Tars.Interface.Cli.Commands.Ask

open System
open System.Net.Http
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

let run (config: Microsoft.Extensions.Configuration.IConfiguration) (prompt: string) =
    task {
        // Load from user secrets/env; no fallback to hardcoded defaults
        let ollama = config["OLLAMA_BASE_URL"] |> Option.ofObj
        let defaultModel = config["DEFAULT_OLLAMA_MODEL"] |> Option.ofObj

        match ollama, defaultModel with
        | None, _ ->
            printfn "Missing OLLAMA_BASE_URL (set via user secrets or env)."
            return 1
        | _, None ->
            printfn "Missing DEFAULT_OLLAMA_MODEL (set via user secrets or env)."
            return 1
        | Some ollamaUrl, Some model ->
            let routingCfg =
                { RoutingConfig.Default with
                    OllamaBaseUri = Uri(ollamaUrl)
                    DefaultOllamaModel = model
                    DefaultVllmModel = model }

            let svcCfg: LlmServiceConfig = { Routing = routingCfg }

            // In a real app, HttpClient should be injected/shared
            use httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(120.0)

            let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

            let req: LlmRequest =
                { ModelHint = None
                  Model = None
                  SystemPrompt = None
                  MaxTokens = Some 1024
                  Temperature = Some 0.7
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                let! response = llmService.CompleteAsync(req)

                if response.FinishReason = Some "parse_error" then
                    printfn $"Error parsing response: %A{response.Raw}"
                    return 1
                else
                    printfn $"%s{response.Text}"
                    return 0
            with ex ->
                printfn $"Error: %s{ex.Message}"
                return 1
    }
