module Tars.Interface.Cli.Commands.SmokeTest

open System
open System.Threading.Tasks
open Serilog
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Interface.Cli
open Spectre.Console

let runSmokeTest (logger: ILogger) : Task<int> =
    task {
        AnsiConsole.Write(new Rule("[bold cyan]🧪 TARS LLM Smoke Test[/]"))
        AnsiConsole.WriteLine()
        
        try
            // Load configuration
            let config = ConfigurationLoader.load ()
            
            // Initialize routing config
            let routingCfg = {
                RoutingConfig.Default with
                    OllamaBaseUri = config.Llm.BaseUrl |> Option.map Uri |> Option.defaultValue (Uri "http://localhost:11434")
                    DefaultOllamaModel = config.Llm.Model
                    LlamaCppBaseUri = config.Llm.LlamaCppUrl |> Option.map Uri
                    DefaultLlamaCppModel = if config.Llm.LlamaCppUrl.IsSome then Some config.Llm.Model else None
            }
            
            let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
            use client = new System.Net.Http.HttpClient()
            client.Timeout <- TimeSpan.FromMinutes(2.0)
            let llmService = DefaultLlmService(client, serviceConfig) :> ILlmService
            
            AnsiConsole.MarkupLine($"[bold]Configuration:[/]")
            AnsiConsole.MarkupLine($"  Provider: [cyan]{config.Llm.Provider}[/]")
            AnsiConsole.MarkupLine($"  Model: [cyan]{config.Llm.Model}[/]")
            match config.Llm.LlamaCppUrl with
            | Some url -> AnsiConsole.MarkupLine($"  LlamaCpp URL: [cyan]{url}[/]")
            | None -> ()
            AnsiConsole.WriteLine()
            
            // Test request
            AnsiConsole.MarkupLine("[bold]Test:[/] Asking LLM to say 'Hi'")
            
            let request = {
                ModelHint = None
                Model = None
                SystemPrompt = None
                MaxTokens = Some 10
                Temperature = Some 0.0
                Stop = []
                Messages = [ { Role = Role.User; Content = "Say exactly 'Hi' and nothing else." } ]
                Tools = []
                ToolChoice = None
                ResponseFormat = None
                Stream = false
                JsonMode = false
                Seed = None
                ContextWindow = None
            }
            
            let sw = System.Diagnostics.Stopwatch.StartNew()
            
            let! response =
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .StartAsync("⏳ Calling LLM...", fun _ -> llmService.CompleteAsync request)
            
            sw.Stop()
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[green]✅ Success![/] ({sw.ElapsedMilliseconds}ms)")
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[bold]Response:[/]")
            AnsiConsole.MarkupLine($"[cyan]{Markup.Escape(response.Text)}[/]")
            AnsiConsole.WriteLine()
            
            match response.Usage with
            | Some u ->
                AnsiConsole.MarkupLine($"[grey]Tokens: {u.TotalTokens} (prompt: {u.PromptTokens}, completion: {u.CompletionTokens})[/]")
            | None -> ()
            
            AnsiConsole.WriteLine()
            AnsiConsole.Write(new Rule("[green]✅ Smoke Test Passed[/]"))
            
            return 0
            
        with ex ->
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[red]❌ Smoke Test Failed[/]")
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}")
            AnsiConsole.WriteLine()
            
            if ex.InnerException <> null then
                AnsiConsole.MarkupLine($"[red]Inner:[/] {ex.InnerException.Message}")
            
            AnsiConsole.WriteException(ex)
            
            return 1
    }
