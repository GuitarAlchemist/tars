module Tars.Interface.Cli.Commands.Chat

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Graph
open Spectre.Console
open System.Net.Http
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
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
                                    $"Source: %s{source} (Score: %.2f{score})\n%s{preview}\n")
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

    /// Helper to register tools and initialize validation
    let initRegistry (logger: ILogger) (toolRegistry: Tars.Tools.ToolRegistry) =
        // Register standard tools from Tars.Tools assembly
        let toolsAssembly =
            System.Reflection.Assembly.GetAssembly(typeof<Tars.Tools.ToolRegistry>)

        toolRegistry.RegisterAssembly(toolsAssembly)

        // Inject loaded tools into validation module so list_all_tools works correctly
        Tars.Tools.Standard.ToolValidation.setRegistry toolRegistry
        logger.Information("Registered tools and initialized ToolValidation layer")

let run (logger: ILogger) (options: ChatOptions) : Task<int> =
    task {
        logger.Information("Starting TARS v2 Chat...")

        let registry = Tars.Kernel.AgentRegistry()

        // Load secrets from secrets.json
        let secretsPath = "secrets.json"

        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Result.Ok() -> logger.Information("Secrets loaded successfully")
        | Result.Error err -> logger.Warning("Could not load secrets: {Error}", err)

        // Initialize Knowledge Graph (Internal)
        let graphPath = Path.Combine(Environment.CurrentDirectory, ".tars", "knowledge")
        let graphService = InternalGraphService(graphPath) :> IGraphService

        // Use Local Ingestion Service by default (Robustness)
        let episodeService = createLocalService graphService
        let ingestionEnabled = true

        logger.Information("Connected to Internal Knowledge Graph at {Path}", graphPath)

        // Ensure graph persists on exit
        AppDomain.CurrentDomain.ProcessExit.Add(fun _ ->
            let task = graphService.PersistAsync()
            task.Wait())

        // Initialize Tool Registry and register standard tools
        let toolRegistry = Tars.Tools.ToolRegistry()
        ChatHelpers.initRegistry logger toolRegistry

        try
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

        // LLM Service Initialization
        let ollamaBaseUri =
            match CredentialVault.getSecret "OLLAMA_BASE_URL" with
            | Result.Ok url ->
                logger.Information("Using Ollama from secrets: {Url}", url)
                Uri(url)
            | Result.Error _ ->
                let errorMsg = "OLLAMA_BASE_URL secret is MISSING!"
                logger.Fatal(errorMsg)
                failwith errorMsg

        let routingCfg: RoutingConfig =
            { RoutingConfig.Default with
                OllamaBaseUri = ollamaBaseUri
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "qwen2.5-coder:latest"
                DefaultVllmModel = "qwen2.5-72b-instruct"
                DefaultOpenAIModel = "gpt-4o"
                DefaultGoogleGeminiModel = "gemini-pro"
                DefaultAnthropicModel = "claude-3-opus-20240229"
                DefaultEmbeddingModel = "nomic-embed-text"
                OpenAIKey = CredentialVault.getSecret "OPENAI_API_KEY" |> Result.toOption
                GoogleGeminiKey = CredentialVault.getSecret "GOOGLE_API_KEY" |> Result.toOption
                AnthropicKey = CredentialVault.getSecret "ANTHROPIC_API_KEY" |> Result.toOption }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        use httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(120.0)
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService

        let modelName = options.Model |> Option.defaultValue "llama3.2"

        // Initialize Vector Store
        let vectorStore =
            match CredentialVault.getSecret "CHROMA_URL" with
            | Result.Ok url ->
                logger.Information("Using Persistent Memory (ChromaDB) at {Url}", url)
                ChromaVectorStore(url) :> IVectorStore
            | Result.Error _ ->
                logger.Information("Using Ephemeral Memory (InMemoryVectorStore)")
                InMemoryVectorStore() :> IVectorStore

        let searchTool = ChatHelpers.createSearchTool llmService vectorStore
        toolRegistry.Register(searchTool)

        // Background Indexing
        do!
            task {
                let docs = ChatHelpers.loadDocs ()
                let! _ = ChatHelpers.indexDocs llmService vectorStore docs
                return ()
            }

        let tools = toolRegistry.GetAll()

        let systemPrompt =
            """You are TARS v2, an advanced autonomous reasoning system running locally.
Your capabilities include:
- Temporal Knowledge Graph (Graphiti) for long-term memory.
- Internal Documentation Search (Self-RAG) via 'search_docs'.
- Execution of safe local tools (File System, Git, Sandbox, HTTP).
- Specialized reasoning modules.

You are running in an offline-first environment, but you HAVE access to tools that can interact with the outside world (e.g., 'http_get', 'git_status', 'run_command').
If a user asks for something outside your knowledge, USE YOUR TOOLS to find it.
"""

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
              CancellationToken = System.Threading.CancellationToken.None
              Logger = (fun s -> logger.Information("{GraphLog}", s)) }

        let mutable currentAgent = agent
        let mutable running = true

        AnsiConsole.MarkupLine("[bold green]TARS v2 Chat initialized.[/] Type [bold red]'exit'[/] to quit.")

        while running do
            let input = AnsiConsole.Ask<string>("[bold yellow]User>[/]")

            if input.StartsWith("/") then
                match input.ToLowerInvariant().Trim() with
                | "/quit"
                | "/exit" -> running <- false
                | "/clear" -> AnsiConsole.Clear()
                | "/tools" ->
                    AnsiConsole.MarkupLine($"[bold]Registered Tools ({tools.Length}):[/]")
                    tools |> List.iter (fun t -> AnsiConsole.MarkupLine($"  - [cyan]{t.Name}[/]"))
                | _ -> AnsiConsole.MarkupLine($"[red]Unknown command: {input}.[/]")
            else
                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Sender = MessageEndpoint.User
                      Receiver = Some(MessageEndpoint.Agent agent.Id)
                      Performative = Performative.Request
                      Intent = Some AgentDomain.Chat
                      Constraints = SemanticConstraints.Default
                      Ontology = None
                      Language = "text"
                      Content = input
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.empty }

                currentAgent <- currentAgent.ReceiveMessage(msg)

                let! (finalAgent, _) =
                    AnsiConsole
                        .Status()
                        .StartAsync(
                            "Thinking...",
                            fun _ ->
                                task {
                                    let mutable sAgent = currentAgent
                                    let mutable sCount = 0
                                    let mutable sFinished = false

                                    while not sFinished && sCount < graphCtx.MaxSteps do
                                        let! outcome = GraphRuntime.step sAgent graphCtx

                                        match outcome with
                                        | Success next -> sAgent <- next
                                        | PartialSuccess(next, _) -> sAgent <- next
                                        | Failure _ -> sFinished <- true

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

                    if ingestionEnabled then
                        let episode =
                            Tars.Core.Episode.AgentInteraction("TARS", input, prompt, DateTime.UtcNow)

                        episodeService.Queue(episode)
                        let! _ = episodeService.FlushAsync()
                        ()
                | AgentState.Error err -> AnsiConsole.MarkupLine($"[bold red]Agent Error: {Markup.Escape(err)}[/]")
                | _ -> ()

        return 0
    }

/// Streaming chat (simplified)
let runStreaming (logger: ILogger) (options: ChatOptions) : Task<int> = task { return 0 }
