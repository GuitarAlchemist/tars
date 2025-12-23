namespace Tars.Connectors.Mcp

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open System.Text.Json
open System.Threading

type McpClient(transport: IMcpTransport) =
    let pendingRequests =
        ConcurrentDictionary<int, TaskCompletionSource<JsonRpcResponse>>()

    let mutable nextId = 0
    let cancellationTokenSource = new CancellationTokenSource()

    let getNextId () = Interlocked.Increment(&nextId)

    let processMessage (line: string) =
        task {
            // Determine if it's a response or request/notification
            try
                let doc = JsonDocument.Parse(line)
                let mutable idProp = Unchecked.defaultof<JsonElement>
                let mutable methodProp = Unchecked.defaultof<JsonElement>

                let hasId = doc.RootElement.TryGetProperty("id", &idProp)
                let hasMethod = doc.RootElement.TryGetProperty("method", &methodProp)

                if hasId && not hasMethod then
                    let response = JsonSerializer.Deserialize<JsonRpcResponse>(line)

                    match pendingRequests.TryRemove(response.Id) with
                    | true, tcs -> tcs.SetResult(response)
                    | false, _ -> Console.Error.WriteLine($"Received response for unknown ID: {response.Id}")
                else
                    // It's a request or notification from server
                    // TODO: Handle server-to-client requests (e.g. sampling, ping)
                    ()
            with ex ->
                Console.Error.WriteLine($"Failed to parse message: {ex.Message}")
        }

    let startLoop () =
        Task.Run(fun () ->
            (task {
                while not cancellationTokenSource.Token.IsCancellationRequested do
                    try
                        let! line = transport.ReceiveAsync()

                        match line with
                        | Some l -> do! processMessage l
                        | None ->
                            // End of stream
                            cancellationTokenSource.Cancel()
                    with ex ->
                        Console.Error.WriteLine($"Transport error: {ex.Message}")
                        cancellationTokenSource.Cancel()
            })
            :> Task)

    member this.SendRequestAsync<'T>(method: string, parameters: obj) =
        task {
            let id = getNextId ()
            let tcs = TaskCompletionSource<JsonRpcResponse>()
            pendingRequests.TryAdd(id, tcs) |> ignore

            let request =
                { JsonRpc = "2.0"
                  Method = method
                  Params = Some(JsonSerializer.SerializeToElement(parameters))
                  Id = Some id }

            do! transport.SendAsync(request)

            // Wait for response with timeout
            // TODO: Add timeout logic
            let! response = tcs.Task

            match response.Error with
            | Some err -> return failwithf $"MCP Error %d{err.Code}: %s{err.Message}"
            | None ->
                match response.Result with
                | Some res -> return JsonSerializer.Deserialize<'T>(res)
                | None -> return Unchecked.defaultof<'T>
        }

    member this.ConnectAsync() =
        task {
            do! transport.StartAsync()
            startLoop () |> ignore

            // Send Initialize
            let initParams =
                { ProtocolVersion = "2024-11-05"
                  Capabilities =
                    { Roots = Some { ListChanged = Some true }
                      Sampling = Some(new obj ()) }
                  ClientInfo = { Name = "TARS"; Version = "2.0.0" } }

            let! result = this.SendRequestAsync<McpInitializeResult>("initialize", initParams)

            // Send Initialized notification
            let initializedMsg =
                { JsonRpc = "2.0"
                  Method = "notifications/initialized"
                  Params = None
                  Id = None }

            do! transport.SendAsync(initializedMsg)

            return result
        }

    member this.ListToolsAsync() =
        this.SendRequestAsync<McpListToolsResult>("tools/list", new obj ())

    member this.CallToolAsync(name: string, args: Map<string, obj>) =
        let params' = { Name = name; Arguments = args }
        this.SendRequestAsync<McpCallToolResult>("tools/call", params')

    member this.CloseAsync() =
        task {
            cancellationTokenSource.Cancel()
            do! transport.CloseAsync()
        }
