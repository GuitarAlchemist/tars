// TARS MCP CLI Commands in F#
// Add these to the main TARS CLI command structure

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.CommandLine
open System.Threading.Tasks

module McpCommands =
    
    // MCP Server Commands
    let createServerStartCommand() =
        let transportOption = Option<string>("--transport", "Transport method (stdio, sse)")
        transportOption.SetDefaultValue("stdio")
        
        let portOption = Option<int>("--port", "Port for SSE transport")
        portOption.SetDefaultValue(3000)
        
        let hostOption = Option<string>("--host", "Host address")
        hostOption.SetDefaultValue("localhost")
        
        let cmd = Command("start", "Start TARS as MCP server")
        cmd.AddOption(transportOption)
        cmd.AddOption(portOption)
        cmd.AddOption(hostOption)
        
        cmd.SetHandler(fun (transport: string) (port: int) (host: string) ->
            async {
                printfn $"🚀 Starting TARS MCP Server"
                printfn $"   Transport: {transport}"
                printfn $"   Address: {host}:{port}"
                
                // Start MCP server
                let! result = McpServer.startAsync transport host port
                
                if result.Success then
                    printfn "✅ MCP Server started successfully"
                else
                    printfn $"❌ Failed to start MCP server: {result.Error}"
            } |> Async.RunSynchronously
        , transportOption, portOption, hostOption)
        
        cmd
    
    let createServerCommand() =
        let cmd = Command("server", "MCP server operations")
        cmd.AddCommand(createServerStartCommand())
        // Add other server commands...
        cmd
    
    // MCP Client Commands
    let createClientRegisterCommand() =
        let urlArg = Argument<string>("url", "MCP server URL")
        let nameOption = Option<string>("--name", "Server name")
        let autoDiscoverOption = Option<bool>("--auto-discover", "Auto-discover capabilities")
        
        let cmd = Command("register", "Register external MCP server")
        cmd.AddArgument(urlArg)
        cmd.AddOption(nameOption)
        cmd.AddOption(autoDiscoverOption)
        
        cmd.SetHandler(fun (url: string) (name: string option) (autoDiscover: bool) ->
            async {
                printfn $"📡 Registering MCP server: {url}"
                
                let serverName = name |> Option.defaultValue (extractServerName url)
                let! result = McpClient.registerServerAsync url serverName autoDiscover
                
                if result.Success then
                    printfn $"✅ Successfully registered: {serverName}"
                    printfn $"   Tools: {result.ToolCount}"
                    printfn $"   Resources: {result.ResourceCount}"
                else
                    printfn $"❌ Failed to register server: {result.Error}"
            } |> Async.RunSynchronously
        , urlArg, nameOption, autoDiscoverOption)
        
        cmd
    
    let createClientCommand() =
        let cmd = Command("client", "MCP client operations")
        cmd.AddCommand(createClientRegisterCommand())
        // Add other client commands...
        cmd
    
    // Main MCP Command
    let createMcpCommand() =
        let cmd = Command("mcp", "Model Context Protocol integration")
        cmd.AddCommand(createServerCommand())
        cmd.AddCommand(createClientCommand())
        // Add workflow and integrate commands...
        cmd

// Integration with main CLI
module CliIntegration =
    
    let addMcpCommandsToMainCli (rootCommand: RootCommand) =
        let mcpCommand = McpCommands.createMcpCommand()
        rootCommand.AddCommand(mcpCommand)
        
        printfn "🔗 MCP commands added to TARS CLI"
