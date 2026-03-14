namespace Tars.Tests

open System.Threading.Tasks
open Xunit
open Tars.Connectors.Mcp

type MockMcpTransport() =
    let mutable sentMessages: JsonRpcRequest list = []
    let mutable responsesToReturn: string list = []
    let mutable requestHandler: (JsonRpcRequest -> string option) option = None

    member this.SentMessages = sentMessages

    member this.SetRequestHandler(handler) = requestHandler <- Some handler

    member this.QueueResponse(response: string) =
        responsesToReturn <- responsesToReturn @ [ response ]

    interface IMcpTransport with
        member this.StartAsync() = Task.CompletedTask

        member this.SendAsync(request) =
            task {
                sentMessages <- sentMessages @ [ request ]

                match requestHandler with
                | Some handler ->
                    match handler request with
                    | Some response -> responsesToReturn <- responsesToReturn @ [ response ]
                    | None -> ()
                | None -> ()
            }

        member this.ReceiveAsync() =
            task {
                while List.isEmpty responsesToReturn do
                    do! Task.Delay(10)

                let head = List.head responsesToReturn
                responsesToReturn <- List.tail responsesToReturn
                return Some head
            }

        member this.CloseAsync() = Task.CompletedTask

module McpTests =

    [<Fact>]
    let ``Client handshake sends initialize`` () =
        task {
            let transport = MockMcpTransport()

            transport.SetRequestHandler(fun req ->
                if req.Method = "initialize" then
                    Some
                        """{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"TestServer","version":"1.0"}}}"""
                else
                    None)

            let client = McpClient(transport)
            let! result = client.ConnectAsync()

            Assert.Equal("TestServer", result.ServerInfo.Name)
            Assert.Equal(2, transport.SentMessages.Length)
        }

    [<Fact>]
    let ``Client can list tools`` () =
        task {
            let transport = MockMcpTransport()

            transport.SetRequestHandler(fun req ->
                match req.Method with
                | "initialize" ->
                    Some
                        """{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"TestServer","version":"1.0"}}}"""
                | "tools/list" ->
                    Some
                        """{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"test-tool","description":"A test tool","inputSchema":{"type":"object"}}]}}"""
                | _ -> None)

            let client = McpClient(transport)
            let! _ = client.ConnectAsync()

            let! toolsResult = client.ListToolsAsync()

            let tool = Assert.Single(toolsResult.Tools)
            Assert.Equal("test-tool", tool.Name)
        }
