module Tars.Interface.Cli.Commands.Chat

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Graph
open Spectre.Console
open System.Net.Http
open System.Net.Http.Json
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Security

type AuthRequest = { email: string; password: string }
type AuthResponse = { token: string }

type ChatOptions =
    { Streaming: bool
      Model: string option }

let run (logger: ILogger) : Task<int> =
    task {
        logger.Information("Starting TARS v2 Chat...")

        let registry = Tars.Kernel.AgentRegistry()

        // Load secrets from secrets.json
        let secretsPath = "secrets.json"

        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Result.Ok() -> logger.Information("Secrets loaded successfully")
        | Result.Error err -> logger.Warning("Could not load secrets: {Error}", err)

        let agent =
            Tars.Kernel.AgentFactory.create
                (Guid.NewGuid())
                "TARS"
                "0.1.0"
                "llama3.2"
                "You are a helpful assistant."
                []
                []

        registry.Register(agent)

        // Initialize LLM Service
        let ollamaBaseUri =
            match CredentialVault.getSecret "OLLAMA_BASE_URL" with
            | Result.Ok url ->
                logger.Information("Using Ollama from secrets: {Url}", url)
                Uri(url)
            | Result.Error _ ->
                let errorMsg = "DEBUG: OLLAMA_BASE_URL secret is MISSING! (Modified)"

                logger.Fatal(errorMsg)
                failwith errorMsg

        let routingCfg: RoutingConfig =
            { OllamaBaseUri = ollamaBaseUri
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = "llama3.1:latest"
              DefaultVllmModel = "qwen2.5-72b-instruct"
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text" }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)

        // Try to authenticate if credentials are present
        let emailResult = CredentialVault.getSecret "OPENWEBUI_EMAIL"
        let passwordResult = CredentialVault.getSecret "OPENWEBUI_PASSWORD"

        match emailResult, passwordResult with
        | Result.Ok email, Result.Ok password ->
            try
                logger.Information("Authenticating with Open WebUI...")
                let authReq = { email = email; password = password }
                let authUri = Uri(ollamaBaseUri, "/api/v1/auths/signin")
                use! authResp = httpClient.PostAsJsonAsync(authUri, authReq)

                if authResp.IsSuccessStatusCode then
                    let! authResult = authResp.Content.ReadFromJsonAsync<AuthResponse>()

                    if not (isNull (box authResult)) && not (String.IsNullOrEmpty(authResult.token)) then
                        httpClient.DefaultRequestHeaders.Authorization <-
                            System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", authResult.token)

                        logger.Information("Authentication successful. Bearer token set.")

                        // List models
                        try
                            logger.Information("Connected to: {Host}:{Port}", ollamaBaseUri.Host, ollamaBaseUri.Port)
                            let! models = OllamaClient.getTagsAsync httpClient ollamaBaseUri
                            logger.Information("Available Models: {Models}", String.Join(", ", models))
                        with ex ->
                            logger.Warning("Could not list models: {Error}", ex.Message)
                    else
                        logger.Warning("Authentication response invalid.")
                else
                    logger.Warning("Authentication failed. Status: {Status}", authResp.StatusCode)
            with ex ->
                logger.Warning("Authentication error: {Error}", ex.Message)
        | _ -> logger.Information("No credentials found, skipping authentication.")

        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        let graphCtx: GraphRuntime.GraphContext =
            { Registry = registry
              Llm = llmService
              MaxSteps = 10
              BudgetGovernor =
                Some(
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 100000<token> }
                    )
                )
              OutputGuard = Some OutputGuard.defaultGuard
              Logger = (fun s -> logger.Information("{GraphLog}", s)) }

        // Test LLM connection before starting chat
        let! connectionOk =
            task {
                try
                    logger.Information("Testing LLM connection...")

                    let testRequest: LlmRequest =
                        { ModelHint = Some routingCfg.DefaultOllamaModel
                          Model = None
                          SystemPrompt = None
                          Messages = [ { Role = Role.User; Content = "test" } ]
                          Temperature = Some 0.1
                          MaxTokens = Some 10
                          Stop = []
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = None
                          Stream = false
                          JsonMode = false
                          Seed = None }

                    let! testResponse = llmService.CompleteAsync(testRequest)
                    logger.Information("LLM connection successful")
                    return true
                with ex ->
                    AnsiConsole.MarkupLine("[bold red]❌ Error: Cannot connect to LLM service![/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[yellow]TARS requires Ollama to be running.[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[cyan]To fix this:[/]")
                    AnsiConsole.MarkupLine("  1. Open a new terminal")
                    AnsiConsole.MarkupLine("  2. Run: [green]ollama serve[/]")
                    AnsiConsole.MarkupLine("  3. Run: [green]ollama pull qwen2.5-coder:latest[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine($"[gray]Error details: {Markup.Escape(ex.Message)}[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine($"[dim]Configured endpoint: {ollamaBaseUri}[/]")
                    return false
            }

        if not connectionOk then
            return 1
        else
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
                          Intent = Some AgentIntent.Chat
                          Constraints = SemanticConstraints.Default
                          Ontology = None
                          Language = "text"
                          Content = input
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty }

                    currentAgent <- currentAgent.ReceiveMessage(msg)

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
                                            let! outcome = GraphRuntime.step sAgent graphCtx

                                            match outcome with
                                            | Success next -> sAgent <- next
                                            | PartialSuccess(next, _) -> sAgent <- next
                                            | Failure errs ->
                                                let errStr =
                                                    String.concat "; " (errs |> List.map (fun e -> sprintf "%A" e))

                                                sAgent <-
                                                    { sAgent with
                                                        State = AgentState.Error errStr }

                                                sFinished <- true

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

/// Streaming chat that outputs tokens in real-time
let runStreaming (logger: ILogger) (options: ChatOptions) : Task<int> =
    task {
        logger.Information("Starting TARS v2 Streaming Chat...")

        // Load secrets
        let secretsPath = "secrets.json"

        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Result.Ok() -> ()
        | Result.Error _ -> ()

        let ollamaBaseUri =
            match CredentialVault.getSecret "OLLAMA_BASE_URL" with
            | Result.Ok url -> Uri(url)
            | Result.Error _ -> Uri("http://localhost:11434/")

        let model = options.Model |> Option.defaultValue "qwen2.5-coder:1.5b"

        let routingCfg: RoutingConfig =
            { OllamaBaseUri = ollamaBaseUri
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = model
              DefaultVllmModel = model
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text" }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        AnsiConsole.MarkupLine("[bold green]TARS v2 Streaming Chat[/] (Type [bold red]'exit'[/] to quit)")
        AnsiConsole.MarkupLine($"[dim]Model: {model} | Streaming: enabled[/]")
        AnsiConsole.WriteLine()

        let mutable history: LlmMessage list = []
        let mutable running = true

        while running do
            let input = AnsiConsole.Ask<string>("[bold yellow]You>[/]")

            if input.ToLower() = "exit" then
                running <- false
            else
                history <- history @ [ { Role = Role.User; Content = input } ]

                let req: LlmRequest =
                    { ModelHint = Some model
                      Model = None
                      SystemPrompt = None
                      Messages = history
                      Temperature = Some 0.7
                      MaxTokens = Some 2048
                      Stop = []
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = true
                      JsonMode = false
                      Seed = None }

                AnsiConsole.Markup("[bold cyan]TARS>[/] ")

                try
                    let! response = llmService.CompleteStreamAsync(req, fun token -> Console.Write(token))

                    Console.WriteLine()
                    Console.WriteLine()

                    history <-
                        history
                        @ [ { Role = Role.Assistant
                              Content = response.Text } ]
                with ex ->
                    AnsiConsole.MarkupLine($"[red]Error: {Markup.Escape(ex.Message)}[/]")
                    AnsiConsole.WriteLine()

        return 0
    }
