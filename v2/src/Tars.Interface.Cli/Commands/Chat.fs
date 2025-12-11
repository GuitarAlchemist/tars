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
open Tars.Security
open Tars.Connectors.EpisodeIngestion
open Tars.Cortex
open System.IO
open System.Text.RegularExpressions

type AuthRequest = { email: string; password: string }
type AuthResponse = { token: string }

type ChatOptions =
    { Streaming: bool
      Model: string option }

module ChatHelpers =
    /// Recursively find all relevant files in a directory
    let rec private getFiles (dir: string) (patterns: string list) =
        try
            if Directory.Exists dir then
                let opts =
                    EnumerationOptions(RecurseSubdirectories = true, IgnoreInaccessible = true)

                let allFiles = Directory.GetFiles(dir, "*.*", opts)

                allFiles
                |> Array.filter (fun f ->
                    let ext = Path.GetExtension(f).ToLowerInvariant()
                    let isMatch = patterns |> List.contains ext
                    // Exclude bin/obj folders
                    let isBinObj =
                        f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}")
                        || f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")

                    isMatch && not isBinObj)
                |> Array.toList
            else
                []
        with _ ->
            []

    /// Simple frontmatter parser (extracts title/description if present)
    let private parseDoc (path: string) =
        try
            let content = File.ReadAllText(path)
            let relativePath = Path.GetRelativePath(Environment.CurrentDirectory, path)
            let ext = Path.GetExtension(path).ToLowerInvariant()

            // Extract title
            let title =
                if ext = ".md" then
                    match Regex.Match(content, "^# (.*)$", RegexOptions.Multiline) with
                    | m when m.Success -> m.Groups.[1].Value.Trim()
                    | _ -> Path.GetFileNameWithoutExtension(path)
                else
                    // For code, use filename
                    Path.GetFileName(path)

            (relativePath, title, content)
        with _ ->
            (path, "Unknown", "")

    /// Load docs and code from standard locations
    let loadDocs () =
        let root = Environment.CurrentDirectory
        let docsDir = Path.Combine(root, "docs")
        let srcDir = Path.Combine(root, "src")
        let testsDir = Path.Combine(root, "tests")
        let tarsDir = Path.Combine(root, ".tars") // Include .tars if present

        let extensions = [ ".md"; ".fs"; ".fsi"; ".json" ] // Index these types

        let dirs = [ docsDir; srcDir; testsDir; tarsDir ]

        dirs
        |> List.collect (fun d -> getFiles d extensions)
        |> List.map parseDoc
        |> List.filter (fun (_, _, c) -> String.IsNullOrWhiteSpace(c) |> not)

    /// Index documents into vector store
    let indexDocs (llm: ILlmService) (store: IVectorStore) (docs: (string * string * string) list) =
        task {
            if docs.IsEmpty then
                return 0
            else
                let! _ =
                    docs
                    |> List.indexed
                    |> List.fold
                        (fun (acc: Task<int>) (idx, (id, title, content)) ->
                            task {
                                let! count = acc

                                try
                                    // Simple chunking: take first 2000 chars for embedding to save time/tokens
                                    let snippet =
                                        if content.Length > 2000 then
                                            content.Substring(0, 2000)
                                        else
                                            content

                                    let! embedding = llm.EmbedAsync(snippet)
                                    let metadata = Map [ ("title", title); ("content", content); ("source", id) ]
                                    do! store.SaveAsync("docs", id, embedding, metadata)
                                    return count + 1
                                with _ ->
                                    return count // Skip failures
                            })
                        (Task.FromResult 0)

                return docs.Length
        }

    /// Create the search tool (capturing LLM for query embedding)
    let createSearchTool (llm: ILlmService) (store: IVectorStore) =
        Tars.Core.Tool.Create(
            "search_docs",
            "Searches internal project documentation. Input: search query string.",
            fun args ->
                task {
                    try
                        let query = Tars.Tools.ToolHelpers.parseStringArg args "query"
                        // 1. Embed query (using batch/single appropriately, here assumes single)
                        let! embedding = llm.EmbedAsync(query)
                        // 2. Search
                        let! results = store.SearchAsync("docs", embedding, 5)

                        if List.isEmpty results then
                            return Result.Ok "No relevant documentation found."
                        else
                            let hits =
                                results
                                |> List.map (fun (id, dist, meta) ->
                                    let source = meta |> Map.tryFind "source" |> Option.defaultValue "unknown"
                                    let content = meta |> Map.tryFind "content" |> Option.defaultValue ""
                                    // Truncate content for context window
                                    let preview =
                                        if content.Length > 500 then
                                            content.Substring(0, 500) + "..."
                                        else
                                            content

                                    let score = 1.0f - dist
                                    sprintf "Source: %s (Score: %.2f)\n%s\n" source score preview)
                                |> String.concat "\n---\n"

                            return Result.Ok hits
                    with ex ->
                        return Result.Error $"Error searching docs: {ex.Message}"
                }
        )


    /// Load MCP tools from configured servers
    let loadMcpTools (logger: ILogger) (registry: Tars.Core.IToolRegistry) =
        try
            let mcpConfigPath =
                System.IO.Path.Combine(System.Environment.CurrentDirectory, "mcp_config.json")

            let mcpManager = Tars.Tools.McpManager(mcpConfigPath)

            // 1. Register MCP management tools
            for t in Tars.Tools.McpTools.getTools () do
                registry.Register(t)

            logger.Information("Registered MCP management tools")

            // 2. Connect to configured servers and register their tools
            let mcpServers = mcpManager.GetServers()

            if not mcpServers.IsEmpty then
                logger.Information("Connecting to {Count} MCP servers...", mcpServers.Length)

                for server in mcpServers do
                    try
                        // Connect synchronously for simplicity in initialization
                        let clientTask = mcpManager.ConnectAsync(server.Name)
                        clientTask.Wait()

                        match clientTask.Result with
                        | Some client ->
                            let listTask = client.ListToolsAsync()
                            listTask.Wait()
                            let listResult = listTask.Result

                            if not (isNull (box listResult)) then
                                for mcpTool in listResult.Tools do
                                    let tarsTool = Tars.Connectors.Mcp.McpToolAdapter.toTarsTool client mcpTool
                                    // Prefix with server name to ensure uniqueness/context
                                    let prefixedTool =
                                        { tarsTool with
                                            Name = $"{server.Name}_{tarsTool.Name}" }

                                    registry.Register(prefixedTool)

                                logger.Information(
                                    "Connected to MCP server '{Name}' and registered {Count} tools",
                                    server.Name,
                                    listResult.Tools.Length
                                )
                            else
                                logger.Warning("MCP server '{Name}' returned no tools list", server.Name)
                        | None -> logger.Warning("Failed to connect to MCP server '{Name}'", server.Name)
                    with ex ->
                        logger.Warning("Error connecting to MCP server '{Name}': {Error}", server.Name, ex.Message)
            else
                logger.Debug("No MCP servers configured.")
        with ex ->
            logger.Warning("MCP initialization failed: {Error}", ex.Message)

let run (logger: ILogger) (options: ChatOptions) : Task<int> =
    task {
        logger.Information("Starting TARS v2 Chat...")

        let registry = Tars.Kernel.AgentRegistry()

        // Load secrets from secrets.json
        let secretsPath = "secrets.json"

        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Result.Ok() -> logger.Information("Secrets loaded successfully")
        | Result.Error err -> logger.Warning("Could not load secrets: {Error}", err)

        // Initialize Graphiti episode ingestion (captures conversations to knowledge graph)
        let graphitiUrl =
            match System.Environment.GetEnvironmentVariable("GRAPHITI_URL") with
            | null
            | "" -> "http://localhost:8001"
            | url -> url

        use episodeService = createServiceWithUrl graphitiUrl
        let mutable ingestionEnabled = false

        // Check if Graphiti is available
        let! healthResult = episodeService.HealthCheckAsync()

        match healthResult with
        | Result.Ok status ->
            logger.Information("Graphiti knowledge graph connected: {Status}", status)
            ingestionEnabled <- true
        | Result.Error _ -> logger.Debug("Graphiti not available - episode capture disabled")

        // Initialize Tool Registry and register standard tools
        let toolRegistry = Tars.Tools.ToolRegistry()

        try
            // Register all tools from Tars.Tools assembly
            let toolsAssembly =
                System.Reflection.Assembly.GetAssembly(typeof<Tars.Tools.ToolRegistry>)

            toolRegistry.RegisterAssembly(toolsAssembly)
            logger.Information("Registered tools from Tars.Tools")

            // Register Macro Tools (require dependency injection)
            let macroPath =
                System.IO.Path.Combine(System.Environment.CurrentDirectory, "macros.json")

            let macroRegistry = Tars.Metascript.FileMacroRegistry(macroPath)
            let macroTools = Tars.Tools.MacroTools.getTools (macroRegistry)

            for t in macroTools do
                toolRegistry.Register(t)

            logger.Information("Registered macro tools")

        with ex ->
            logger.Warning("Failed to register tools: {Error}", ex.Message)

        // Initialize MCP Manager and load servers
        ChatHelpers.loadMcpTools logger toolRegistry
        // Register Search Memory Tool if Graphiti is available
        if ingestionEnabled then
            try
                let searchMemTool =
                    Tars.Tools.Standard.KnowledgeTools.createSearchMemoryTool episodeService

                toolRegistry.Register(searchMemTool)

                let saveMemTool =
                    Tars.Tools.Standard.KnowledgeTools.createSaveMemoryTool episodeService

                toolRegistry.Register(saveMemTool)
                logger.Information("Registered search_memory/save_memory tools")
            with ex ->
                logger.Warning("Failed to register search_memory tool: {Error}", ex.Message)

        // =========================================================================================
        // LLM Service Initialization (Moved Up for RAG Support)
        // =========================================================================================
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
              DefaultEmbeddingModel = "nomic-embed-text"

              OllamaKey = None
              VllmKey = None
              OpenAIKey = CredentialVault.getSecret "OPENAI_API_KEY" |> Result.toOption
              GoogleGeminiKey = CredentialVault.getSecret "GOOGLE_API_KEY" |> Result.toOption
              AnthropicKey = CredentialVault.getSecret "ANTHROPIC_API_KEY" |> Result.toOption }

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

        let checkModelConfig (cfg: RoutingConfig) (model: string) =
            let dummyReq: LlmRequest =
                { ModelHint = Some model
                  Model = Some model
                  SystemPrompt = None
                  Messages = []
                  Temperature = None
                  MaxTokens = None
                  Stop = []
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None }

            let routed = chooseBackend cfg dummyReq

            match routed.Backend with
            | OpenAI _ when routed.ApiKey.IsNone ->
                AnsiConsole.MarkupLine(
                    $"[bold yellow]Warning:[/] OpenAI API Key is missing for model [cyan]{model}[/]."
                )

                AnsiConsole.MarkupLine("To fix, run: [green]tars config set openai-key <YOUR_KEY>[/]\n")
            | GoogleGemini _ when routed.ApiKey.IsNone ->
                AnsiConsole.MarkupLine(
                    $"[bold yellow]Warning:[/] Google Gemini API Key is missing for model [cyan]{model}[/]."
                )

                AnsiConsole.MarkupLine("To fix, run: [green]tars config set google-key <YOUR_KEY>[/]\n")
            | Anthropic _ when routed.ApiKey.IsNone ->
                AnsiConsole.MarkupLine(
                    $"[bold yellow]Warning:[/] Anthropic API Key is missing for model [cyan]{model}[/]."
                )

                AnsiConsole.MarkupLine("To fix, run: [green]tars config set anthropic-key <YOUR_KEY>[/]\n")
            | _ -> ()

        let modelName = options.Model |> Option.defaultValue "llama3.2"
        checkModelConfig routingCfg modelName

        // =========================================================================================
        // RAG / Knowledge Initialization
        // =========================================================================================
        let vectorStore = InMemoryVectorStore() :> IVectorStore
        let searchTool = ChatHelpers.createSearchTool llmService vectorStore
        toolRegistry.Register(searchTool)

        // Background Indexing (with spinner)
        Console.WriteLine()

        do!
            AnsiConsole
                .Status()
                .StartAsync(
                    "Indexing documentation...",
                    fun ctx ->
                        task {
                            let docs = ChatHelpers.loadDocs ()
                            let! count = ChatHelpers.indexDocs llmService vectorStore docs

                            if count > 0 then
                                AnsiConsole.MarkupLine($"[green]Indexed {count} documentation files.[/]")
                            else
                                AnsiConsole.MarkupLine("[dim]No documentation found to index.[/]")
                        }
                )

        Console.WriteLine()

        let tools = toolRegistry.GetAll()
        logger.Information("Loaded {Count} tools for agent", tools.Length)

        let systemPrompt =
            """You are TARS v2, an advanced autonomous reasoning system running locally.
Your capabilities include:
- Temporal Knowledge Graph (Graphiti) for long-term memory.
- Internal Documentation Search (Self-RAG) via 'search_docs'.
- Execution of safe local tools (File System, Git, Sandbox, HTTP).
- Specialized reasoning modules.

You are running in an offline-first environment, but you HAVE access to tools that can interact with the outside world (e.g., 'http_get', 'git_status', 'run_command').
If a user asks for something outside your knowledge, USE YOUR TOOLS to find it.
Do NOT claim you cannot access the internet if you have a 'http_get' tool available.
When asked to perform a task, analyze if you have a tool for it. If yes, use it.
If asked about TARS architecture, code, or features, use 'search_docs' to find the answer.
"""

        if not tools.IsEmpty then
            // Inject loaded tools into validation module so list_all_tools works correctly
            Tars.Tools.Standard.ToolValidation.setKnownTools tools
            logger.Information("Registered {Count} tools with ToolValidation", tools.Length)

        let agent =
            Tars.Kernel.AgentFactory.create (Guid.NewGuid()) "TARS" "0.1.0" modelName systemPrompt tools []

        registry.Register(agent)



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

            AnsiConsole.MarkupLine(
                "[dim]Authenticated as: "
                + (emailResult |> Result.defaultValue "Anonymous")
                + "[/]"
            )

            if not tools.IsEmpty then
                AnsiConsole.MarkupLine($"[dim]Verified Tools ({tools.Length}):[/]")
                let toolNames = tools |> List.map (fun t -> t.Name) |> String.concat ", "
                AnsiConsole.WriteLine(Markup.Escape(toolNames))

            AnsiConsole.WriteLine()

            while running do
                let input = AnsiConsole.Ask<string>("[bold yellow]User>[/]")

                if input.StartsWith("/") then
                    match input.ToLowerInvariant().Trim() with
                    | "/quit"
                    | "/exit" -> running <- false
                    | "/clear" -> AnsiConsole.Clear()
                    | "/help" ->
                        AnsiConsole.MarkupLine("[bold cyan]Slash Commands:[/]")
                        AnsiConsole.MarkupLine("  [green]/diagnose[/] - Show system diagnostics and tool status")
                        AnsiConsole.MarkupLine("  [green]/tools[/]    - List all registered tools")
                        AnsiConsole.MarkupLine("  [green]/clear[/]    - Clear the console screen")
                        AnsiConsole.MarkupLine("  [green]/quit[/]     - Exit the application")
                        AnsiConsole.MarkupLine("  [green]/help[/]     - Show this help message")
                        AnsiConsole.WriteLine()
                    | "/tools" ->
                        AnsiConsole.MarkupLine($"[bold]Registered Tools ({tools.Length}):[/]")

                        tools
                        |> List.iter (fun t -> AnsiConsole.MarkupLine($"  - [cyan]{t.Name}[/] ([dim]{t.Version}[/])"))

                        AnsiConsole.WriteLine()
                    | "/diagnose" ->
                        AnsiConsole.MarkupLine("[bold yellow]System Diagnostics:[/]")
                        AnsiConsole.MarkupLine($"  [dim]Tools Loaded:[/] {tools.Length}")
                        AnsiConsole.MarkupLine($"  [dim]Graphiti Connected:[/] {ingestionEnabled}")
                        AnsiConsole.MarkupLine($"  [dim]LLM Endpoint:[/] {ollamaBaseUri}")
                        let activeModel = options.Model |> Option.defaultValue "Default"
                        AnsiConsole.MarkupLine($"  [dim]Active Model:[/] {activeModel}")
                        let mem = System.GC.GetTotalMemory(false) / 1024L / 1024L
                        AnsiConsole.MarkupLine($"  [dim]Memory Usage:[/] {mem} MB")
                        AnsiConsole.MarkupLine($"  [dim]Agent State:[/] {currentAgent.State}")
                        AnsiConsole.WriteLine()
                    | "/test" ->
                        AnsiConsole.MarkupLine("[bold cyan]Running Full Test Suite (dotnet test)...[/]")
                        AnsiConsole.MarkupLine("[dim]This may take a moment.[/]")

                        let! (exitCode, output) =
                            AnsiConsole
                                .Status()
                                .Spinner(Spinner.Known.Dots)
                                .StartAsync(
                                    "Executing tests...",
                                    fun ctx ->
                                        task {
                                            let psi = System.Diagnostics.ProcessStartInfo("dotnet", "test")
                                            psi.RedirectStandardOutput <- true
                                            psi.RedirectStandardError <- true
                                            psi.UseShellExecute <- false
                                            psi.CreateNoWindow <- true
                                            psi.WorkingDirectory <- System.Environment.CurrentDirectory

                                            use proc = new System.Diagnostics.Process()
                                            proc.StartInfo <- psi
                                            proc.Start() |> ignore
                                            let! output = proc.StandardOutput.ReadToEndAsync()
                                            let! error = proc.StandardError.ReadToEndAsync()
                                            let! _ = proc.WaitForExitAsync()
                                            return (proc.ExitCode, output + "\n" + error)
                                        }
                                )

                        if exitCode = 0 then
                            // Extract passed count from output if possible (naive regex or just substring)
                            let passedLines =
                                output.Split('\n')
                                |> Array.filter (fun l -> l.Contains("Passed!") || l.Contains("Passed:"))

                            let summary =
                                if passedLines.Length > 0 then
                                    passedLines |> String.concat "\n"
                                else
                                    "All tests passed."

                            AnsiConsole.MarkupLine($"[bold green]Tests Passed! ✅[/]")
                            AnsiConsole.MarkupLine($"[dim]{summary.Trim()}[/]")
                        else
                            AnsiConsole.MarkupLine($"[bold red]Tests Failed! ❌ (Exit Code {exitCode})[/]")
                            // Show last 10 lines of output for context
                            let lines = output.Split('\n')

                            let lastLines =
                                lines |> Array.skip (max 0 (lines.Length - 10)) |> String.concat "\n"

                            AnsiConsole.MarkupLine($"[red]{Markup.Escape(lastLines)}[/]")

                        AnsiConsole.WriteLine()
                    | _ -> AnsiConsole.MarkupLine($"[red]Unknown command: {input}. Type /help for options.[/]")
                else
                    // Standard Chat Message Logic
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

                        // Capture episode to Graphiti knowledge graph
                        if ingestionEnabled then
                            let episode =
                                Tars.Core.Episode.AgentInteraction("TARS", input, prompt, DateTime.UtcNow)

                            episodeService.Queue(episode)
                            let! _ = episodeService.FlushAsync()
                            ()

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

        let getSecret key =
            match CredentialVault.getSecret key with
            | Result.Ok s -> Some s
            | Result.Error _ -> None

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
              DefaultEmbeddingModel = "nomic-embed-text"

              OllamaKey = None // Usually not needed for local
              VllmKey = None
              OpenAIKey = getSecret "OPENAI_API_KEY"
              GoogleGeminiKey = getSecret "GOOGLE_API_KEY"
              AnthropicKey = getSecret "ANTHROPIC_API_KEY" }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        // Validate API Keys
        let dummyReq: LlmRequest =
            { ModelHint = Some model
              Model = Some model
              SystemPrompt = None
              Messages = []
              Temperature = None
              MaxTokens = None
              Stop = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None }

        let routed = chooseBackend routingCfg dummyReq

        match routed.Backend with
        | OpenAI _ when routed.ApiKey.IsNone ->
            AnsiConsole.MarkupLine(
                "[bold yellow]Warning:[/] OpenAI API Key is missing for model [cyan]"
                + model
                + "[/]."
            )

            AnsiConsole.MarkupLine("To fix, run: [green]tars config set openai-key <YOUR_KEY>[/]")
            AnsiConsole.WriteLine()
        | GoogleGemini _ when routed.ApiKey.IsNone ->
            AnsiConsole.MarkupLine(
                "[bold yellow]Warning:[/] Google Gemini API Key is missing for model [cyan]"
                + model
                + "[/]."
            )

            AnsiConsole.MarkupLine("To fix, run: [green]tars config set google-key <YOUR_KEY>[/]")
            AnsiConsole.WriteLine()
        | Anthropic _ when routed.ApiKey.IsNone ->
            AnsiConsole.MarkupLine(
                "[bold yellow]Warning:[/] Anthropic API Key is missing for model [cyan]"
                + model
                + "[/]."
            )

            AnsiConsole.MarkupLine("To fix, run: [green]tars config set anthropic-key <YOUR_KEY>[/]")
            AnsiConsole.WriteLine()
        | _ -> ()

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
