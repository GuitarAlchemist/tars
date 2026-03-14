namespace Tars.Interface.Cli.Commands

open System
open Serilog
open Tars.Core
open Tars.Tools
open Tars.Connectors.Mcp
open Tars.Metascript
open Tars.Connectors.EpisodeIngestion
open Tars.Tools.Standard
open Tars.Cortex.WoTTypes
open Tars.Cortex
open Tars.Knowledge

module McpServerCommand =

    let run (logger: ILogger) (args: string array) =
        task {
            let mutable useSse = false
            let mutable port = 8000

            let mutable i = 2 // Skip 'mcp' and 'server'

            while i < args.Length do
                match args.[i] with
                | "--sse" -> useSse <- true
                | "--port" when i + 1 < args.Length ->
                    i <- i + 1
                    port <- int args.[i]
                | _ -> ()

                i <- i + 1

            if useSse then
                logger.Information("Starting TARS in MCP SSE Mode on port {Port}", port)
            else
                logger.Information("Starting TARS in MCP Server Mode (Stdio)")

            let registry = ToolRegistry()
            // ... (rest of tool registration logic)

            // Initialize Graphiti service at top level to ensure it stays alive for the server run
            let graphitiUrl =
                match System.Environment.GetEnvironmentVariable("GRAPHITI_URL") with
                | null
                | "" -> "http://localhost:8001"
                | url -> url

            use episodeService = createServiceWithUrl graphitiUrl

            try
                // Register standard tools from Tars.Tools assembly
                let toolsAssembly = System.Reflection.Assembly.GetAssembly(typeof<ToolRegistry>)
                registry.RegisterAssembly(toolsAssembly)
                logger.Information("Registered standard tools")

                // Register Macros
                let macroPath =
                    System.IO.Path.Combine(System.Environment.CurrentDirectory, "macros.json")

                if System.IO.File.Exists(macroPath) then
                    let macroRegistry = FileMacroRegistry(macroPath)
                    let macroTools = MacroTools.getTools (macroRegistry)

                    for t in macroTools do
                        registry.Register(t)

                    logger.Information("Registered macro tools")

                // Register Memory Tools (RAFmG)
                // Initialize Graphiti episode ingestion
                // Check if Graphiti is available


                // Check if Graphiti is available
                let healthResult =
                    task { return! episodeService.HealthCheckAsync() }
                    |> Async.AwaitTask
                    |> Async.RunSynchronously

                match healthResult with
                | Result.Ok status ->
                    logger.Information("Graphiti connected ({Status}); registering memory tools...", status)
                    let searchMemTool = KnowledgeTools.createSearchMemoryTool episodeService
                    registry.Register(searchMemTool)
                    let saveMemTool = KnowledgeTools.createSaveMemoryTool episodeService
                    registry.Register(saveMemTool)
                | Result.Error _ -> logger.Debug("Graphiti not available; memory tools skipped")

                // Register MCP Management Tools (client tools)
                // This allows the "TARS MCP Server" to manage *other* MCP connections if needed
                for t in McpTools.getTools () do
                    registry.Register(t)

                logger.Information("Registered MCP management tools")

                // Ensure ToolValidation knows about the registry for introspection
                Tars.Tools.Standard.ToolValidation.setRegistry (registry)
                logger.Information("Tool registry finalized and verification layer initialized")

                // --- NEW: Connect to configured MCP servers ---
                let mcpServers = McpTools.Manager.GetServers()

                for serverConfig in mcpServers do
                    try
                        logger.Information(
                            "Connecting to MCP Server: {Name} ({Command})...",
                            serverConfig.Name,
                            serverConfig.Command
                        )

                        let clientOpt =
                            task { return! McpTools.Manager.ConnectAsync(serverConfig.Name) }
                            |> Async.AwaitTask
                            |> Async.RunSynchronously

                        match clientOpt with
                        | Some client ->
                            try
                                let toolsResult =
                                    task { return! client.ListToolsAsync() }
                                    |> Async.AwaitTask
                                    |> Async.RunSynchronously

                                if not (isNull (box toolsResult)) then
                                    for mcpTool in toolsResult.Tools do
                                        let tarsTool = McpToolAdapter.toTarsTool client mcpTool
                                        // Prefix tool name with server name to avoid collisions?
                                        // For now, let's keep original name but maybe log it.
                                        // Ideally: {Prefix}_{ToolName}
                                        let prefixedTool =
                                            { tarsTool with
                                                Name = $"{serverConfig.Name}_{tarsTool.Name}" }

                                        registry.Register(prefixedTool)

                                        logger.Information(
                                            "Registered remote tool: {ToolName} from {Server}",
                                            prefixedTool.Name,
                                            serverConfig.Name
                                        )
                                else
                                    logger.Warning("MCP Server {Name} returned no result.", serverConfig.Name)
                            with ex ->
                                logger.Error("Failed to list tools from {Name}: {Error}", serverConfig.Name, ex.Message)
                        | None -> logger.Error("Failed to connect to MCP Server: {Name}", serverConfig.Name)
                    with ex ->
                        logger.Error("Error integrating MCP Server {Name}: {Error}", serverConfig.Name, ex.Message)

                // --- Register Claude Code bridge tools ---
                let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler
                let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector
                let toolRegistry = registry :> IToolRegistry
                let ledger = KnowledgeLedger.createInMemory ()
                ledger.Initialize() |> Async.AwaitTask |> Async.RunSynchronously

                let bridgeTools : Tool list = [
                    { Name = "tars_compile_plan"
                      Description = "Compile a goal into a WoT execution plan. Input: {\"goal\": \"...\", \"max_steps\": 5}. Returns execution manifest with DAG of typed nodes."
                      Version = "1.0.0"
                      ParentVersion = None
                      CreatedAt = DateTime.UtcNow
                      Execute = fun input -> async { return ClaudeCodeBridge.compilePlan compiler selector toolRegistry input } }
                    { Name = "tars_execute_step"
                      Description = "Execute a Tool/Validate node in an active plan. Input: {\"plan_id\": \"...\", \"node_id\": \"...\", \"input\": \"...\"}. Returns step result with next nodes."
                      Version = "1.0.0"
                      ParentVersion = None
                      CreatedAt = DateTime.UtcNow
                      Execute = fun input -> ClaudeCodeBridge.executeStep toolRegistry input }
                    { Name = "tars_validate_step"
                      Description = "Validate content against a Validate node's invariants. Input: {\"plan_id\": \"...\", \"node_id\": \"...\", \"content\": \"...\"}. Returns pass/fail."
                      Version = "1.0.0"
                      ParentVersion = None
                      CreatedAt = DateTime.UtcNow
                      Execute = fun input -> async { return ClaudeCodeBridge.validateStep input } }
                    { Name = "tars_memory_op"
                      Description = "Knowledge graph operations. Search: {\"operation\": \"search\", \"query\": \"...\"}. Assert: {\"operation\": \"assert\", \"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}. Stats: {\"operation\": \"stats\"}."
                      Version = "1.0.0"
                      ParentVersion = None
                      CreatedAt = DateTime.UtcNow
                      Execute = fun input -> ClaudeCodeBridge.memoryOp ledger input }
                    { Name = "tars_complete_plan"
                      Description = "Complete an active plan. Records outcome for learning, checks golden trace regression. Input: {\"plan_id\": \"...\", \"final_output\": \"...\"}."
                      Version = "1.0.0"
                      ParentVersion = None
                      CreatedAt = DateTime.UtcNow
                      Execute = fun input -> async { return ClaudeCodeBridge.completePlan input } }
                ]

                for tool in bridgeTools do
                    registry.Register(tool)

                logger.Information("Registered {Count} Claude Code bridge tools", bridgeTools.Length)

                // --- Register probabilistic grammar tools ---
                let grammarTools = Tars.Evolution.McpGrammarTools.createTools ()
                for tool in grammarTools do
                    registry.Register(tool)
                logger.Information("Registered {Count} probabilistic grammar tools", grammarTools.Length)

                // --- Register GA trace bridge tools ---
                let gaTraceTools = Tars.Evolution.McpGaTraceBridge.createTools ()
                for tool in gaTraceTools do
                    registry.Register(tool)
                logger.Information("Registered {Count} GA trace bridge tools", gaTraceTools.Length)

            with ex ->
                logger.Error("Failed to register tools: {Error}", ex.Message)
            // We don't throw, just continue with partial tools

            let server = McpServer(registry)

            if useSse then
                let sseServer = Tars.Connectors.Mcp.SseMcpServer(server, port)
                do! sseServer.StartAsync()
            else
                // The server runs until input closes
                do! server.RunAsync()

            return 0
        }
