module Tars.Interface.Cli.Commands.TestGrammarCommand

open System
open System.Threading.Tasks
open Microsoft.Extensions.Configuration
open Spectre.Console
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Cortex
open System.Text.Json

// Define a test type to enforce
type WeatherReport =
    { Location: string
      Temperature: float
      Condition: string
      Humidity: int }

let run (config: IConfiguration) (args: string array) =
    task {
        let model =
            config["DEFAULT_GENERATION_MODEL"]
            |> Option.ofObj
            |> Option.defaultValue "qwen2.5-coder:1.5b"

        let routingCfg =
            { Tars.Llm.Routing.RoutingConfig.OllamaBaseUri = Uri("http://localhost:11434")
              Tars.Llm.Routing.RoutingConfig.VllmBaseUri = Uri("http://localhost:8000")
              Tars.Llm.Routing.RoutingConfig.OpenAIBaseUri = Uri("https://api.openai.com")
              Tars.Llm.Routing.RoutingConfig.GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com")
              Tars.Llm.Routing.RoutingConfig.AnthropicBaseUri = Uri("https://api.anthropic.com")
              Tars.Llm.Routing.RoutingConfig.DefaultOllamaModel = model
              Tars.Llm.Routing.RoutingConfig.DefaultVllmModel = model
              Tars.Llm.Routing.RoutingConfig.DefaultOpenAIModel = "gpt-4o-mini" // Support structured output
              Tars.Llm.Routing.RoutingConfig.DefaultGoogleGeminiModel = "gemini-1.5-pro-latest"
              Tars.Llm.Routing.RoutingConfig.DefaultAnthropicModel = "claude-3-opus-20240229"
              Tars.Llm.Routing.RoutingConfig.DefaultEmbeddingModel = "nomic-embed-text"
              Tars.Llm.Routing.RoutingConfig.OllamaKey = None
              Tars.Llm.Routing.RoutingConfig.VllmKey = None
              Tars.Llm.Routing.RoutingConfig.OpenAIKey = Option.ofObj config["OPENAI_API_KEY"]
              Tars.Llm.Routing.RoutingConfig.GoogleGeminiKey = Option.ofObj config["GOOGLE_API_KEY"]
              Tars.Llm.Routing.RoutingConfig.AnthropicKey = Option.ofObj config["ANTHROPIC_API_KEY"]
              Tars.Llm.Routing.RoutingConfig.DockerModelRunnerBaseUri = None
              Tars.Llm.Routing.RoutingConfig.LlamaCppBaseUri = None
              Tars.Llm.Routing.RoutingConfig.DefaultDockerModelRunnerModel = None
              Tars.Llm.Routing.RoutingConfig.DefaultLlamaCppModel = None
              Tars.Llm.Routing.RoutingConfig.DockerModelRunnerKey = None
              Tars.Llm.Routing.RoutingConfig.LlamaCppKey = None }

        let svcCfg = { Routing = routingCfg }
        use httpClient = new System.Net.Http.HttpClient()
        httpClient.Timeout <- TimeSpan.FromMinutes(2.0)
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        AnsiConsole.MarkupLine("[bold cyan]Testing Grammar Distillation...[/]")

        // 1. Generate Schema
        let schemaJson = Structure.generateJsonSchema<WeatherReport> ()
        AnsiConsole.MarkupLine("[bold yellow]Generated Schema:[/]")
        AnsiConsole.WriteLine(schemaJson: string)
        AnsiConsole.WriteLine()

        // 2. Prepare Request
        let req =
            { ModelHint = Some "code"
              Model = None // Use default routed model
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
              JsonMode = false // Driven by ResponseFormat
              Seed = Some 42 }

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
