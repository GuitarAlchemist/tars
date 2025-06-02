namespace TarsEngine.FSharp.WindowsService.WebSocket

open System
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Tasks

/// <summary>
/// WebSocket message types for TARS communication
/// </summary>
type TarsMessageType =
    | Command = 1
    | Response = 2
    | Event = 3
    | Progress = 4
    | Status = 5
    | Error = 6

/// <summary>
/// WebSocket message structure for TARS communication
/// </summary>
type TarsWebSocketMessage = {
    Id: string
    Type: TarsMessageType
    Command: string option
    Data: JsonElement option
    Timestamp: DateTime
    Source: string
}

/// <summary>
/// WebSocket connection information
/// </summary>
type TarsWebSocketConnection = {
    Id: string
    Socket: WebSocket
    ConnectedAt: DateTime
    LastActivity: DateTime
    ClientType: string // "CLI", "UI", "Monitor", etc.
}

/// <summary>
/// TARS WebSocket Handler for full-duplex communication
/// Manages real-time communication between Windows service and clients
/// </summary>
type TarsWebSocketHandler(logger: ILogger<TarsWebSocketHandler>, taskManager: DocumentationTaskManager) =
    
    let connections = ConcurrentDictionary<string, TarsWebSocketConnection>()
    let mutable isRunning = false
    
    /// Generate unique connection ID
    member private this.GenerateConnectionId() = Guid.NewGuid().ToString("N")[..7]
    
    /// Send message to specific connection
    member private this.SendMessageToConnection(connectionId: string, message: TarsWebSocketMessage) = task {
        match connections.TryGetValue(connectionId) with
        | true, connection when connection.Socket.State = WebSocketState.Open ->
            try
                let json = JsonSerializer.Serialize(message, JsonSerializerOptions(WriteIndented = false))
                let bytes = Encoding.UTF8.GetBytes(json)
                let buffer = ArraySegment<byte>(bytes)
                
                do! connection.Socket.SendAsync(buffer, WebSocketMessageType.Text, true, CancellationToken.None)
                logger.LogDebug($"📤 Sent message to {connectionId}: {message.Type}")
                
                // Update last activity
                let updatedConnection = { connection with LastActivity = DateTime.UtcNow }
                connections.TryUpdate(connectionId, updatedConnection, connection) |> ignore
                
            with
            | ex -> 
                logger.LogError(ex, $"❌ Failed to send message to {connectionId}")
                this.RemoveConnection(connectionId)
        | _ -> 
            logger.LogWarning($"⚠️ Connection {connectionId} not found or not open")
    }
    
    /// Broadcast message to all connections
    member private this.BroadcastMessage(message: TarsWebSocketMessage) = task {
        let tasks = 
            connections.Values
            |> Seq.filter (fun conn -> conn.Socket.State = WebSocketState.Open)
            |> Seq.map (fun conn -> this.SendMessageToConnection(conn.Id, message))
            |> Seq.toArray
        
        do! Task.WhenAll(tasks)
        logger.LogDebug($"📡 Broadcasted {message.Type} to {tasks.Length} connections")
    }
    
    /// Remove connection
    member private this.RemoveConnection(connectionId: string) =
        match connections.TryRemove(connectionId) with
        | true, connection ->
            logger.LogInformation($"🔌 Connection {connectionId} removed")
            if connection.Socket.State = WebSocketState.Open then
                connection.Socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Connection closed", CancellationToken.None) |> ignore
        | false, _ -> ()
    
    /// Handle incoming WebSocket message
    member private this.HandleMessage(connectionId: string, message: TarsWebSocketMessage) = task {
        try
            logger.LogDebug($"📥 Received {message.Type} from {connectionId}: {message.Command}")
            
            match message.Type with
            | TarsMessageType.Command ->
                match message.Command with
                | Some "documentation.start" ->
                    taskManager.StartTask()
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "documentation.start"
                        Data = Some (JsonSerializer.SerializeToElement({| success = true; message = "Documentation task started" |}))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some "documentation.pause" ->
                    taskManager.PauseTask()
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "documentation.pause"
                        Data = Some (JsonSerializer.SerializeToElement({| success = true; message = "Documentation task paused" |}))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some "documentation.resume" ->
                    taskManager.ResumeTask()
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "documentation.resume"
                        Data = Some (JsonSerializer.SerializeToElement({| success = true; message = "Documentation task resumed" |}))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some "documentation.stop" ->
                    taskManager.StopTask()
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "documentation.stop"
                        Data = Some (JsonSerializer.SerializeToElement({| success = true; message = "Documentation task stopped" |}))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some "documentation.status" ->
                    let status = taskManager.GetStatus()
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "documentation.status"
                        Data = Some (JsonSerializer.SerializeToElement(status))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some "service.status" ->
                    let serviceStatus = {|
                        service = "TarsService"
                        status = "Running"
                        uptime = DateTime.UtcNow - DateTime.UtcNow.AddHours(-1.0) // Placeholder
                        connections = connections.Count
                        version = "3.0.0"
                    |}
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "service.status"
                        Data = Some (JsonSerializer.SerializeToElement(serviceStatus))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some "ping" ->
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Response
                        Command = Some "pong"
                        Data = Some (JsonSerializer.SerializeToElement({| timestamp = DateTime.UtcNow |}))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | Some cmd ->
                    logger.LogWarning($"⚠️ Unknown command: {cmd}")
                    let response = {
                        Id = Guid.NewGuid().ToString()
                        Type = TarsMessageType.Error
                        Command = Some cmd
                        Data = Some (JsonSerializer.SerializeToElement({| error = $"Unknown command: {cmd}" |}))
                        Timestamp = DateTime.UtcNow
                        Source = "TarsService"
                    }
                    do! this.SendMessageToConnection(connectionId, response)
                
                | None ->
                    logger.LogWarning("⚠️ Command message without command field")
            
            | TarsMessageType.Event ->
                // Handle client events (e.g., client status updates)
                logger.LogInformation($"📨 Event from {connectionId}: {message.Command}")
            
            | _ ->
                logger.LogDebug($"📋 Received {message.Type} message from {connectionId}")
        
        with
        | ex -> 
            logger.LogError(ex, $"❌ Error handling message from {connectionId}")
            let errorResponse = {
                Id = Guid.NewGuid().ToString()
                Type = TarsMessageType.Error
                Command = message.Command
                Data = Some (JsonSerializer.SerializeToElement({| error = ex.Message |}))
                Timestamp = DateTime.UtcNow
                Source = "TarsService"
            }
            do! this.SendMessageToConnection(connectionId, errorResponse)
    }
    
    /// Handle WebSocket connection
    member this.HandleWebSocketConnection(webSocket: WebSocket, clientType: string) = task {
        let connectionId = this.GenerateConnectionId()
        let connection = {
            Id = connectionId
            Socket = webSocket
            ConnectedAt = DateTime.UtcNow
            LastActivity = DateTime.UtcNow
            ClientType = clientType
        }
        
        connections.TryAdd(connectionId, connection) |> ignore
        logger.LogInformation($"🔌 New WebSocket connection: {connectionId} ({clientType})")
        
        // Send welcome message
        let welcomeMessage = {
            Id = Guid.NewGuid().ToString()
            Type = TarsMessageType.Event
            Command = Some "connected"
            Data = Some (JsonSerializer.SerializeToElement({| 
                connectionId = connectionId
                serverTime = DateTime.UtcNow
                version = "3.0.0"
                capabilities = [| "documentation"; "service"; "monitoring" |]
            |}))
            Timestamp = DateTime.UtcNow
            Source = "TarsService"
        }
        do! this.SendMessageToConnection(connectionId, welcomeMessage)
        
        try
            let buffer = Array.zeroCreate<byte> 4096
            
            while webSocket.State = WebSocketState.Open do
                let result = webSocket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                let! received = result
                
                if received.MessageType = WebSocketMessageType.Text then
                    let json = Encoding.UTF8.GetString(buffer, 0, received.Count)
                    
                    try
                        let message = JsonSerializer.Deserialize<TarsWebSocketMessage>(json)
                        do! this.HandleMessage(connectionId, message)
                    with
                    | ex ->
                        logger.LogError(ex, $"❌ Failed to parse message from {connectionId}: {json}")
                
                elif received.MessageType = WebSocketMessageType.Close then
                    logger.LogInformation($"🔌 WebSocket close requested by {connectionId}")
                    break
        
        with
        | ex -> 
            logger.LogError(ex, $"❌ WebSocket error for {connectionId}")
        
        finally
            this.RemoveConnection(connectionId)
    }
    
    /// Start progress broadcasting
    member this.StartProgressBroadcasting() =
        if not isRunning then
            isRunning <- true
            logger.LogInformation("📡 Starting progress broadcasting")
            
            Task.Run(fun () -> task {
                while isRunning do
                    try
                        if connections.Count > 0 then
                            let status = taskManager.GetStatus()
                            let progressMessage = {
                                Id = Guid.NewGuid().ToString()
                                Type = TarsMessageType.Progress
                                Command = Some "documentation.progress"
                                Data = Some (JsonSerializer.SerializeToElement(status))
                                Timestamp = DateTime.UtcNow
                                Source = "TarsService"
                            }
                            do! this.BroadcastMessage(progressMessage)
                        
                        do! Task.Delay(5000) // Broadcast every 5 seconds
                    with
                    | ex -> logger.LogError(ex, "❌ Error in progress broadcasting")
            }) |> ignore
    
    /// Stop progress broadcasting
    member this.StopProgressBroadcasting() =
        isRunning <- false
        logger.LogInformation("📡 Stopping progress broadcasting")
    
    /// Get connection statistics
    member this.GetConnectionStats() = {|
        totalConnections = connections.Count
        activeConnections = connections.Values |> Seq.filter (fun c -> c.Socket.State = WebSocketState.Open) |> Seq.length
        connectionsByType = 
            connections.Values 
            |> Seq.groupBy (fun c -> c.ClientType)
            |> Seq.map (fun (clientType, conns) -> clientType, Seq.length conns)
            |> Map.ofSeq
        oldestConnection = 
            if connections.Count > 0 then
                Some (connections.Values |> Seq.minBy (fun c -> c.ConnectedAt) |> fun c -> c.ConnectedAt)
            else None
    |}
    
    /// Cleanup inactive connections
    member this.CleanupInactiveConnections() =
        let cutoff = DateTime.UtcNow.AddMinutes(-30.0) // 30 minutes timeout
        let inactiveConnections = 
            connections.Values
            |> Seq.filter (fun c -> c.LastActivity < cutoff || c.Socket.State <> WebSocketState.Open)
            |> Seq.map (fun c -> c.Id)
            |> Seq.toList
        
        for connectionId in inactiveConnections do
            this.RemoveConnection(connectionId)
        
        if inactiveConnections.Length > 0 then
            logger.LogInformation($"🧹 Cleaned up {inactiveConnections.Length} inactive connections")
