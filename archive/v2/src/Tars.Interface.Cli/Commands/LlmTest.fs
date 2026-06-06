module Tars.Interface.Cli.Commands.LlmTest

open System
open System.Threading.Tasks
open Tars.Llm
open Spectre.Console

let runTest (llmService: ILlmService) : Task<int> =
    task {
        AnsiConsole.MarkupLine("[bold cyan]🧪 LLM Smoke Test[/]")
        AnsiConsole.WriteLine()

        let request =
            { Messages =
                [ { Role = "user"
                    Content = "Say 'Hi' and nothing else." } ]
              MaxTokens = Some 10
              Temperature = Some 0.0
              ModelHint = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        try
            AnsiConsole.MarkupLine("[grey]Sending request to LLM...[/]")
            let! response = llmService.CompleteAsync request

            AnsiConsole.MarkupLine("[green]✅ Success![/]")
            AnsiConsole.MarkupLine($"[bold]Response:[/] [cyan]{response.Text}[/]")

            match response.Usage with
            | Some u -> AnsiConsole.MarkupLine($"[grey]Tokens: {u.TotalTokens}[/]")
            | None -> ()

            return 0
        with ex ->
            AnsiConsole.MarkupLine($"[red]❌ Error:[/] {ex.Message}")
            AnsiConsole.WriteException(ex)
            return 1
    }
