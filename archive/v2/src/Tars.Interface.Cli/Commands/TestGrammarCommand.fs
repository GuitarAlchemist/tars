module Tars.Interface.Cli.Commands.TestGrammarCommand

open System
open Microsoft.Extensions.Configuration
open Serilog
open Spectre.Console
open Tars.Llm
open Tars.Cortex
open Tars.Interface.Cli
open System.Text.Json

// Define a test type to enforce
type WeatherReport =
    { Location: string
      Temperature: float
      Condition: string
      Humidity: int }

let run (config: IConfiguration) (args: string array) =
    task {
        let llmService = LlmFactory.create Log.Logger

        AnsiConsole.MarkupLine("[bold cyan]Testing Grammar Distillation...[/]")

        // 1. Generate Schema
        let schemaJson = Structure.generateJsonSchema<WeatherReport> ()
        AnsiConsole.MarkupLine("[bold yellow]Generated Schema:[/]")
        AnsiConsole.WriteLine(schemaJson: string)
        AnsiConsole.WriteLine()

        // 2. Prepare Request
        let req =
            { ModelHint = Some "code"
              Model = None
              SystemPrompt = Some "You are a weather reporter. Output JSON only."
              MaxTokens = Some 500
              Temperature = Some 0.1
              Stop = []
              Messages =
                [ { Role = Role.User
                    Content = "What is the weather in London right now?" } ]
              Tools = []
              ToolChoice = None
              ResponseFormat = Some(ResponseFormat.Constrained(Grammar.JsonSchema schemaJson))
              Stream = false
              JsonMode = false
              Seed = Some 42
              ContextWindow = None }

        // 3. Execute
        AnsiConsole.MarkupLine("[bold green]Sending Request...[/]")

        try
            let! res = llmService.CompleteAsync req
            AnsiConsole.MarkupLine("[bold]Response Text:[/]")
            AnsiConsole.WriteLine(res.Text: string)

            // 4. Validate
            try
                let report =
                    JsonSerializer.Deserialize<WeatherReport>(
                        res.Text,
                        JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                    )

                AnsiConsole.MarkupLine("[bold green]SUCCESS: Parsed valid WeatherReport![/]")
                AnsiConsole.WriteLine($"Location: {report.Location}")
                AnsiConsole.WriteLine($"Condition: {report.Condition}")
                AnsiConsole.WriteLine($"Temp: {report.Temperature}C")
                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[bold red]FAILURE: Could not parse response as WeatherReport[/]")
                AnsiConsole.WriteLine(ex.Message)
                return 1

        with ex ->
            AnsiConsole.MarkupLine($"[bold red]Error calling LLM:[/] {ex.Message}")
            return 1
    }
