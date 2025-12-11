namespace Tars.Interface.Cli.Commands

open System
open System.Threading.Tasks
open Serilog
open Tars.Tools
open Tars.Connectors.Mcp
open Tars.Metascript
open Tars.Connectors.EpisodeIngestion
open Tars.Tools.Standard

module McpServerCommand =

    let run (logger: ILogger) =
        task {
            // Note: In MCP Server mode, all logs MUST go to Stderr to avoid corrupting the JSON-RPC stream on Stdout.
            // The logger passed in should ideally be configured for Stderr or File.
            // If it writes to Stdout, we might have issues.
            logger.Information("Starting TARS in MCP Server Mode")

            let registry = ToolRegistry()

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

            with ex ->
                logger.Error("Failed to register tools: {Error}", ex.Message)
            // We don't throw, just continue with partial tools

            let server = McpServer(registry)

            // The server runs until input closes
            do! server.RunAsync()

            return 0
        }
