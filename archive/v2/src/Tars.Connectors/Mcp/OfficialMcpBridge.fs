namespace Tars.Connectors.Mcp

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open ModelContextProtocol.Server

/// Bridge module that connects TARS's IToolRegistry to the official
/// ModelContextProtocol NuGet SDK, enabling TARS tools to be served
/// via a standards-compliant MCP server.
module OfficialMcpBridge =

    /// Convert a single TARS Tool record into an McpServerTool using the
    /// official SDK's programmatic Create API (delegate overload).
    let toOfficialMcpTool (tool: Tars.Core.Tool) : McpServerTool =
        // Create a delegate that matches the expected signature:
        //   (string, CancellationToken) -> Task<string>
        // The SDK will expose the 'arguments' parameter in the JSON schema
        // and inject CancellationToken automatically.
        let handler =
            Func<string, CancellationToken, Task<string>>(fun arguments _ct ->
                task {
                    let input = if isNull arguments then "" else arguments
                    let! result = tool.Execute(input) |> Async.StartAsTask
                    match result with
                    | Result.Ok output -> return output
                    | Result.Error err -> return $"Error: {err}"
                })

        let options = McpServerToolCreateOptions()
        options.Name <- tool.Name
        options.Description <- tool.Description

        McpServerTool.Create(handler, options)

    /// Convert all tools from an IToolRegistry into official McpServerTool
    /// instances.
    let convertAllTools (registry: Tars.Core.IToolRegistry) : McpServerTool list =
        registry.GetAll()
        |> List.map toOfficialMcpTool

    /// Configure a HostApplicationBuilder to run an official MCP server over
    /// stdio, serving all tools from the given IToolRegistry.
    ///
    /// Usage from F#:
    ///   let builder = HostApplicationBuilder(args)
    ///   OfficialMcpBridge.configureHost registry builder
    ///   let host = builder.Build()
    ///   host.RunAsync() |> Async.AwaitTask |> Async.RunSynchronously
    let configureHost (registry: Tars.Core.IToolRegistry) (builder: HostApplicationBuilder) : unit =
        let mcpTools = convertAllTools registry

        builder.Services
            .AddMcpServer(fun (options: McpServerOptions) ->
                options.ServerInfo <-
                    ModelContextProtocol.Protocol.Implementation(
                        Name = "TARS",
                        Version = "2.0.0"
                    ))
            .WithStdioServerTransport()
            .WithTools(mcpTools)
        |> ignore

        builder.Logging.SetMinimumLevel(LogLevel.Warning) |> ignore

    /// Create and run a fully configured official MCP server over stdio,
    /// blocking until the host shuts down. This is the simplest entry point.
    let runStdioServer (registry: Tars.Core.IToolRegistry) (args: string array) : Task =
        task {
            let builder = HostApplicationBuilder(args)
            configureHost registry builder
            let host = builder.Build()
            do! host.RunAsync()
        }

    /// Create and run a fully configured official MCP server over stdio
    /// as an async workflow.
    let runStdioServerAsync (registry: Tars.Core.IToolRegistry) (args: string array) : Async<unit> =
        runStdioServer registry args |> Async.AwaitTask
