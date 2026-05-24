module Tars.Interface.Cli.Commands.Ask

open System
open Serilog
open Tars.Llm
open Tars.Interface.Cli

let run (logger: ILogger) (prompt: string) =
    task {
        let llmService = LlmFactory.create logger

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
