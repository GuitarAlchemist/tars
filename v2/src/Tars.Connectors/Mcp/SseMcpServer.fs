namespace Tars.Connectors.Mcp

open System
open System.IO
open System.Net
open System.Text
open System.Threading.Tasks
open System.Text.Json
open Tars.Connectors.Mcp

/// Ultra-Compatible SSE transport for MCP server (Accepts ANY path for POST)
type SseMcpServer(mcpServer: McpServer, port: int) =
    let listener = new HttpListener()

    do listener.Prefixes.Add($"http://localhost:{port}/")

    let log (msg: string) =
        Console.Error.WriteLine($"[TARS SSE] {msg}")

    let addCorsHeaders (response: HttpListenerResponse) =
        response.Headers.Add("Access-Control-Allow-Origin", "*")
        response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.Headers.Add("Access-Control-Allow-Headers", "Content-Type, X-Session-Id, Authorization")

    member _.StartAsync() : Task =
        let runServer () =
            task {
                try
                    listener.Start()
                    log $"SSE MCP Server listening on ALL HOSTS at port {port}"
                    log "--- Handlers Ready ---"
                    log $"GET  /mcp/sse     -> Start SSE Stream"
                    log $"POST (Any Path)  -> Process MCP Message"
                    log $"GET  /           -> Health Check"

                    while listener.IsListening do
                        let! context = listener.GetContextAsync()

                        let handle () =
                            task {
                                try
                                    let request = context.Request
                                    let response = context.Response
                                    let path = request.Url.AbsolutePath.ToLower().TrimEnd('/')

                                    addCorsHeaders response

                                    if request.HttpMethod = "OPTIONS" then
                                        response.StatusCode <- 204
                                        response.Close()
                                    elif request.HttpMethod = "POST" then
                                        // Compatibility: Handle POST to ANY path (root, /sse, /message) as an MCP request
                                        use reader = new StreamReader(request.InputStream)
                                        let! body = reader.ReadToEndAsync()

                                        if not (String.IsNullOrWhiteSpace body) then
                                            log $"[POST] {path}: Handling message..."
                                            let! mcpResponse = mcpServer.HandleRequest(body)

                                            match mcpResponse with
                                            | Some resp ->
                                                response.ContentType <- "application/json"
                                                let rBuffer = Encoding.UTF8.GetBytes(resp)
                                                do! response.OutputStream.WriteAsync(rBuffer, 0, rBuffer.Length)
                                            | None -> response.StatusCode <- 204
                                        else
                                            response.StatusCode <- 400

                                        response.Close()
                                    else
                                        // Handle GET
                                        match path with
                                        | "/mcp/sse"
                                        | "/sse" ->
                                            log $"[GET] {path}: Opening SSE stream..."
                                            response.ContentType <- "text/event-stream"
                                            response.Headers.Add("Cache-Control", "no-cache")
                                            response.Headers.Add("Connection", "keep-alive")
                                            let stream = response.OutputStream

                                            // Tell the client where to POST messages (relative to current host)
                                            let endpoint = "/mcp/message"
                                            let endpointMsg = $"event: endpoint\ndata: {endpoint}\n\n"
                                            let buffer = Encoding.UTF8.GetBytes(endpointMsg)
                                            do! stream.WriteAsync(buffer, 0, buffer.Length)
                                            do! stream.FlushAsync()

                                            try
                                                let mutable active = true

                                                while active && listener.IsListening do
                                                    do! Task.Delay(15000)
                                                    let hb = ": heartbeat\n\n"
                                                    let hbBuffer = Encoding.UTF8.GetBytes(hb)

                                                    try
                                                        do! stream.WriteAsync(hbBuffer, 0, hbBuffer.Length)
                                                        do! stream.FlushAsync()
                                                    with _ ->
                                                        active <- false
                                            with _ ->
                                                ()

                                            response.Close()
                                        | ""
                                        | "/"
                                        | "/about"
                                        | _ ->
                                            // Root GET = Health Check / About
                                            response.ContentType <- "application/json"

                                            let (info: {| instance_id: string; git_commit: string; startup_time: DateTime; uptime: string; tool_count: int; graphiti_status: string; degraded_mode_enabled: bool |}) = 
                                                mcpServer.GetInfo()
                                            let msg = JsonSerializer.Serialize(info)

                                            let buffer = System.Text.Encoding.UTF8.GetBytes(msg)
                                            do! response.OutputStream.WriteAsync(buffer, 0, buffer.Length)
                                            response.Close()
                                with ex ->
                                    log $"Error: {ex.Message}"

                                    try
                                        context.Response.Close()
                                    with _ ->
                                        ()

                                return ()
                            }

                        let _ = Task.Run(fun () -> handle () :> Task)
                        ()

                    return ()
                with ex ->
                    log $"Critical: {ex.Message}"
                    return ()
            }

        runServer () :> Task

    member _.Stop() =
        listener.Stop()
        listener.Close()
