namespace Tars.Connectors

open System
open Tars.Connectors.Mcp

/// Client for Augment Context Engine MCP server.
/// Provides semantic codebase search via the `codebase-retrieval` tool.
type AugmentClient(workspacePath: string) =
    let mutable clientOpt: McpClient option = None
    let mutable transportOpt: IMcpTransport option = None

    /// Connect to the Augment Context Engine MCP server.
    member this.ConnectAsync() =
        task {
            // Start auggie in MCP mode with workspace path
            // Start auggie in MCP mode with workspace path
            let isWindows =
                System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                    System.Runtime.InteropServices.OSPlatform.Windows
                )

            let command, args =
                if isWindows then
                    "cmd", $"/c auggie -w \"{workspacePath}\" --mcp"
                else
                    "auggie", $"-w \"{workspacePath}\" --mcp"

            let transport = StdioTransport(command, args, None) :> IMcpTransport

            let client = McpClient(transport)
            let! initResult = client.ConnectAsync()

            transportOpt <- Some transport
            clientOpt <- Some client

            return initResult
        }

    /// List available tools from the Augment server.
    member this.ListToolsAsync() =
        task {
            match clientOpt with
            | Some client -> return! client.ListToolsAsync()
            | None -> return failwith "Not connected. Call ConnectAsync first."
        }

    /// Perform semantic codebase search using the codebase-retrieval tool.
    member this.CodebaseRetrievalAsync(query: string) =
        task {
            match clientOpt with
            | Some client ->
                let args = Map.ofList [ ("information_request", box query) ]
                let! result = client.CallToolAsync("codebase-retrieval", args)

                // Extract text content from response
                let text = result.Content |> List.choose (fun c -> c.Text) |> String.concat "\n"

                return text
            | None -> return failwith "Not connected. Call ConnectAsync first."
        }

    /// Close the connection to the Augment server.
    member this.CloseAsync() =
        task {
            match clientOpt with
            | Some client ->
                do! client.CloseAsync()
                clientOpt <- None
                transportOpt <- None
            | None -> ()
        }

    interface IDisposable with
        member this.Dispose() =
            this.CloseAsync() |> Async.AwaitTask |> Async.RunSynchronously
