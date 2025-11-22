namespace TarsEngine.FSharp.Cli.MCP

open System
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Diagnostics.TarsRealDiagnostics

/// TARS MCP Server - Real WebSocket integration with Augment Code
module TarsMcpServer =

    /// MCP message types for TARS integration
    type McpMessageType =
        | Initialize
        | GetDiagnostics
        | ExecuteFluxScript
        | GetAgentStatus
        | CoordinateAgents
        | GetSystemHealth
        | ExecuteMetascript
        | GetProjectStatus
        | StreamDiagnostics
        | StopStreaming

    /// MCP request structure
    type McpRequest = {
        Id: string
        Method: string
        Params: JsonElement option
    }

    /// MCP response structure
    type McpResponse = {
        Id: string
        Result: JsonElement option
        Error: string option
    }

    /// MCP notification for streaming data
    type McpNotification = {
        Method: string
        Params: JsonElement
    }

    /// WebSocket connection manager for MCP
    type McpConnectionManager() =
        let connections = System.Collections.Concurrent.ConcurrentDictionary<string, WebSocket * CancellationTokenSource>()
        let logger = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore).CreateLogger<McpConnectionManager>()
        
        member this.AddConnection(id: string, socket: WebSocket) =
            let cts = new CancellationTokenSource()
            connections.TryAdd(id, (socket, cts)) |> ignore
            logger.LogInformation("Added MCP connection: {ConnectionId}", id)
            
        member this.RemoveConnection(id: string) =
            match connections.TryRemove(id) with
            | true, (_, cts) -> 
                cts.Cancel()
                cts.Dispose()
                logger.LogInformation("Removed MCP connection: {ConnectionId}", id)
            | false, _ -> ()
            
        member this.GetConnection(id: string) =
            match connections.TryGetValue(id) with
            | true, (socket, _) -> Some socket
            | false, _ -> None
            
        member this.BroadcastToAll(message: string) =
            task {
                let activeConnections = connections.ToArray()
                for KeyValue(id, (socket, cts)) in activeConnections do
                    if not cts.Token.IsCancellationRequested && socket.State = WebSocketState.Open then
                        try
                            let buffer = Encoding.UTF8.GetBytes(message)
                            do! socket.SendAsync(ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, cts.Token)
                        with
                        | ex -> 
                            logger.LogWarning("Failed to send to connection {ConnectionId}: {Error}", id, ex.Message)
                            this.RemoveConnection(id)
            }
            
        member this.GetActiveConnectionCount() = connections.Count

    /// TARS MCP Server implementation
    type TarsMcpServerService(logger: ILogger<TarsMcpServerService>) =
        let connectionManager = McpConnectionManager()
        let mutable streamingTimer: Timer option = None
        
        /// Handle MCP requests with real TARS integration
        member this.HandleMcpRequest(request: McpRequest): Task<McpResponse> =
            task {
                try
                    logger.LogInformation("Handling MCP request: {Method} ({RequestId})", request.Method, request.Id)
                    
                    match request.Method with
                    | "initialize" ->
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement({|
                                name = "TARS MCP Server"
                                version = "1.0.0"
                                capabilities = [|
                                    "diagnostics"
                                    "real_time_streaming"
                                    "gpu_detection"
                                    "git_health"
                                    "network_diagnostics"
                                    "system_monitoring"
                                    "service_health"
                                |]
                                status = "operational"
                                server_info = {|
                                    active_connections = connectionManager.GetActiveConnectionCount()
                                    uptime = DateTime.UtcNow
                                |}
                            |}))
                            Error = None
                        }
                        
                    | "get_diagnostics" ->
                        // Get REAL diagnostic data from TARS systems
                        let repositoryPath = 
                            match request.Params with
                            | Some params when params.TryGetProperty("repository_path").ValueKind <> JsonValueKind.Undefined ->
                                params.GetProperty("repository_path").GetString()
                            | _ -> Environment.CurrentDirectory
                        
                        let! diagnostics = getComprehensiveDiagnostics repositoryPath
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement(diagnostics))
                            Error = None
                        }
                        
                    | "stream_diagnostics" ->
                        // Start streaming real-time diagnostics
                        this.StartDiagnosticsStreaming()
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement({|
                                status = "streaming_started"
                                interval_ms = 5000
                            |}))
                            Error = None
                        }
                        
                    | "stop_streaming" ->
                        // Stop streaming diagnostics
                        this.StopDiagnosticsStreaming()
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement({|
                                status = "streaming_stopped"
                            |}))
                            Error = None
                        }
                        
                    | "get_gpu_info" ->
                        // Get real GPU information
                        let! gpuInfo = detectGpuInfo()
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement({|
                                gpus = gpuInfo
                                cuda_available = gpuInfo |> List.exists (fun gpu -> gpu.CudaSupported)
                                total_gpu_memory = gpuInfo |> List.sumBy (fun gpu -> gpu.MemoryTotal)
                            |}))
                            Error = None
                        }
                        
                    | "get_git_health" ->
                        // Get real git repository health
                        let repositoryPath = 
                            match request.Params with
                            | Some params when params.TryGetProperty("repository_path").ValueKind <> JsonValueKind.Undefined ->
                                params.GetProperty("repository_path").GetString()
                            | _ -> Environment.CurrentDirectory
                        
                        let! gitHealth = getGitRepositoryHealth repositoryPath
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement(gitHealth))
                            Error = None
                        }
                        
                    | "get_network_diagnostics" ->
                        // Get real network diagnostics
                        let! networkDiagnostics = performNetworkDiagnostics()
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement(networkDiagnostics))
                            Error = None
                        }
                        
                    | "get_system_resources" ->
                        // Get real system resource metrics
                        let systemResources = getSystemResourceMetrics()
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement(systemResources))
                            Error = None
                        }
                        
                    | "get_service_health" ->
                        // Get real service health
                        let! serviceHealth = checkServiceHealth()
                        return {
                            Id = request.Id
                            Result = Some (JsonSerializer.SerializeToElement(serviceHealth))
                            Error = None
                        }
                        
                    | _ ->
                        logger.LogWarning("Unknown MCP method: {Method}", request.Method)
                        return {
                            Id = request.Id
                            Result = None
                            Error = Some $"Unknown method: {request.Method}"
                        }
                with
                | ex ->
                    logger.LogError(ex, "Error handling MCP request: {Method} ({RequestId})", request.Method, request.Id)
                    return {
                        Id = request.Id
                        Result = None
                        Error = Some ex.Message
                    }
            }
            
        /// Start streaming real-time diagnostics
        member private this.StartDiagnosticsStreaming() =
            match streamingTimer with
            | Some _ -> () // Already streaming
            | None ->
                logger.LogInformation("Starting real-time diagnostics streaming")
                let timer = new Timer(this.StreamDiagnosticsCallback, null, TimeSpan.Zero, TimeSpan.FromSeconds(5.0))
                streamingTimer <- Some timer
                
        /// Stop streaming diagnostics
        member private this.StopDiagnosticsStreaming() =
            match streamingTimer with
            | Some timer ->
                logger.LogInformation("Stopping real-time diagnostics streaming")
                timer.Dispose()
                streamingTimer <- None
            | None -> ()
            
        /// Callback for streaming diagnostics
        member private this.StreamDiagnosticsCallback(state: obj) =
            task {
                try
                    let! diagnostics = getComprehensiveDiagnostics Environment.CurrentDirectory
                    let notification = {
                        Method = "diagnostics_update"
                        Params = JsonSerializer.SerializeToElement(diagnostics)
                    }
                    let message = JsonSerializer.Serialize(notification)
                    do! connectionManager.BroadcastToAll(message)
                with
                | ex ->
                    logger.LogError(ex, "Error streaming diagnostics")
            } |> ignore
            
        member this.ConnectionManager = connectionManager

    /// Create and configure the MCP WebSocket server
    let createMcpServer (port: int) (logger: ILogger) =
        let builder = WebApplication.CreateBuilder()
        
        // Configure services
        builder.Services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        
        builder.Services.AddCors() |> ignore
        builder.Services.AddSingleton<TarsMcpServerService>() |> ignore
        
        let app = builder.Build()
        
        // Configure CORS
        app.UseCors(fun policy ->
            policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader() |> ignore
        ) |> ignore
        
        // Configure WebSocket middleware
        app.UseWebSockets() |> ignore
        
        // MCP WebSocket endpoint
        app.Use(fun context next ->
            task {
                if context.Request.Path = "/tars-mcp" then
                    if context.WebSockets.IsWebSocketRequest then
                        let! webSocket = context.WebSockets.AcceptWebSocketAsync()
                        let connectionId = Guid.NewGuid().ToString()
                        let mcpService = context.RequestServices.GetRequiredService<TarsMcpServerService>()
                        
                        mcpService.ConnectionManager.AddConnection(connectionId, webSocket)
                        
                        try
                            // Handle WebSocket communication
                            let buffer = Array.zeroCreate 4096
                            
                            while webSocket.State = WebSocketState.Open do
                                let! result = webSocket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                                
                                if result.MessageType = WebSocketMessageType.Text then
                                    let message = Encoding.UTF8.GetString(buffer, 0, result.Count)
                                    let request = JsonSerializer.Deserialize<McpRequest>(message)
                                    
                                    let! response = mcpService.HandleMcpRequest(request)
                                    
                                    let responseJson = JsonSerializer.Serialize(response)
                                    let responseBuffer = Encoding.UTF8.GetBytes(responseJson)
                                    do! webSocket.SendAsync(ArraySegment<byte>(responseBuffer), WebSocketMessageType.Text, true, CancellationToken.None)
                                elif result.MessageType = WebSocketMessageType.Close then
                                    break
                                    
                        finally
                            mcpService.ConnectionManager.RemoveConnection(connectionId)
                            if webSocket.State = WebSocketState.Open then
                                do! webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Connection closed", CancellationToken.None)
                    else
                        context.Response.StatusCode <- 400
                else
                    do! next.Invoke()
            } :> Task
        ) |> ignore
        
        // Health check endpoint
        app.MapGet("/health", Func<obj>(fun () -> 
            {|
                status = "healthy"
                server = "TARS MCP Server"
                version = "1.0.0"
                timestamp = DateTime.UtcNow
                active_connections = 0 // Would get from service
            |}
        )) |> ignore
        
        app

    /// Start TARS MCP Server
    let startTarsMcpServer (port: int) (logger: ILogger) =
        task {
            let app = createMcpServer port logger
            
            logger.LogInformation("🚀 Starting TARS MCP Server on port {Port}", port)
            logger.LogInformation("🌐 WebSocket endpoint: ws://localhost:{Port}/tars-mcp", port)
            logger.LogInformation("💚 Health check: http://localhost:{Port}/health", port)
            logger.LogInformation("🔗 Ready for Augment Code integration!")
            
            do! app.RunAsync($"http://localhost:{port}")
        }
