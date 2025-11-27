module Tars.Interface.Cli.Commands.Chat

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Graph
open Spectre.Console
open System.Net.Http
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

let run (logger: ILogger) =
    task {
        logger.Information("Starting TARS v2 Chat...")

        let ctx = Kernel.init ()

        let agent =
            Kernel.createAgent (Guid.NewGuid()) "TARS" "0.1.0" "llama3.2" "You are a helpful assistant." []

        let ctx = Kernel.registerAgent agent ctx

        // Initialize LLM Service
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
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text" }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        let graphCtx: GraphRuntime.GraphContext =
            { Kernel = ctx
              Llm = llmService
              MaxSteps = 10
              BudgetGovernor = Some(BudgetGovernor(100000)) }

        let mutable currentAgent = agent
        let mutable running = true

        AnsiConsole.MarkupLine("[bold green]TARS v2 Chat initialized.[/] Type [bold red]'exit'[/] to quit.")
        AnsiConsole.WriteLine()

        while running do
            let input = AnsiConsole.Ask<string>("[bold yellow]User>[/]")

            if input = "exit" then
                running <- false
            else
                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Sender = MessageEndpoint.User
                      Receiver = Some(MessageEndpoint.Agent agent.Id)
                      Performative = Performative.Request
                      Constraints = SemanticConstraints.Default
                      Content = input
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.empty }

                currentAgent <- Kernel.receiveMessage msg currentAgent

                let mutable stepAgent = currentAgent
                let mutable stepCount = 0
                let mutable finished = false

                let! (finalAgent, isFinished) =
                    AnsiConsole
                        .Status()
                        .Spinner(Spinner.Known.Dots)
                        .StartAsync(
                            "Thinking...",
                            fun ctx ->
                                task {
                                    let mutable sAgent = stepAgent
                                    let mutable sCount = stepCount
                                    let mutable sFinished = finished

                                    while not sFinished && sCount < graphCtx.MaxSteps do
                                        let! next = GraphRuntime.step sAgent graphCtx
                                        sAgent <- next
                                        sCount <- sCount + 1

                                        match sAgent.State with
                                        | WaitingForUser _
                                        | AgentState.Error _ -> sFinished <- true
                                        | _ -> ()

                                    return (sAgent, sFinished)
                                }
                        )

                currentAgent <- finalAgent

                match currentAgent.State with
                | WaitingForUser prompt ->
                    AnsiConsole.MarkupLine($"[bold cyan]TARS>[/] {Markup.Escape(prompt)}")
                    AnsiConsole.WriteLine()
                | AgentState.Error err ->
                    AnsiConsole.MarkupLine($"[bold red]Agent Error: {Markup.Escape(err)}[/]")
                    logger.Error("Agent Error: {Error}", err)
                | _ -> ()

        return 0
    }
