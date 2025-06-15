namespace TarsEngine.FSharp.Cli.WebSocket

open System
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

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
/// TARS WebSocket Client for CLI communication with Windows service
/// Provides full-duplex real-time communication
/// </summary>
type TarsWebSocketClient(logger: ILogger<TarsWebSocketClient>) =
    
    let mutable webSocket: ClientWebSocket option = None
    let mutable isConnected = false
    let mutable cancellationTokenSource = new CancellationTokenSource()
    
    /// Event handlers for different message types
    let mutable onProgressUpdate: (JsonElement -> unit) option = None
    let mutable onStatusUpdate: (JsonElement -> unit) option = None
    let mutable onResponse: (string -> JsonElement -> unit) option = None
    let mutable onError: (string -> unit) option = None
    let mutable onConnected: (string -> unit) option = None
    
    /// Generate unique message ID
    member private this.GenerateMessageId() = Guid.NewGuid().ToString("N")[..7]
    
    /// Send message to service
    member private this.SendMessage(message: TarsWebSocketMessage) = task {
        match webSocket with
        | Some ws when ws.State = WebSocketState.Open ->
            try
                let json = JsonSerializer.Serialize(message, JsonSerializerOptions(WriteIndented = false))
                let bytes = Encoding.UTF8.GetBytes(json)
                let buffer = ArraySegment<byte>(bytes)
                
                do! ws.SendAsync(buffer, WebSocketMessageType.Text, true, cancellationTokenSource.Token)
                logger.LogDebug($"📤 Sent {message.Type}: {message.Command}")
                
            with
            | ex -> 
                logger.LogError(ex, "❌ Failed to send message")
                raise ex
        | _ ->
            let error = "WebSocket not connected"
            logger.LogError(error)
            raise (InvalidOperationException(error))
    }
    
    /// Handle incoming message
    member private this.HandleMessage(message: TarsWebSocketMessage) =
        try
            logger.LogDebug($"📥 Received {message.Type}: {message.Command}")
            
            match message.Type with
            | TarsMessageType.Response ->
                match onResponse, message.Command, message.Data with
                | Some handler, Some cmd, Some data -> handler cmd data
                | _ -> ()
            
            | TarsMessageType.Progress ->
                match onProgressUpdate, message.Data with
                | Some handler, Some data -> handler data
                | _ -> ()
            
            | TarsMessageType.Status ->
                match onStatusUpdate, message.Data with
                | Some handler, Some data -> handler data
                | _ -> ()
            
            | TarsMessageType.Error ->
                match onError, message.Data with
                | Some handler, Some data -> 
                    let errorMsg = 
                        try data.GetProperty("error").GetString()
                        with _ -> "Unknown error"
                    handler errorMsg
                | _ -> ()
            
            | TarsMessageType.Event ->
                match message.Command with
                | Some "connected" ->
                    match onConnected, message.Data with
                    | Some handler, Some data ->
                        let connectionId = 
                            try data.GetProperty("connectionId").GetString()
                            with _ -> "unknown"
                        handler connectionId
                    | _ -> ()
                | _ -> ()
            
            | _ ->
                logger.LogDebug($"📋 Unhandled message type: {message.Type}")
        
        with
        | ex -> logger.LogError(ex, "❌ Error handling message")
    
    /// Start receiving messages
    member private this.StartReceiving() = task {
        match webSocket with
        | Some ws ->
            try
                let buffer = Array.zeroCreate<byte> 4096
                
                while ws.State = WebSocketState.Open && not cancellationTokenSource.Token.IsCancellationRequested do
                    let result = ws.ReceiveAsync(ArraySegment<byte>(buffer), cancellationTokenSource.Token)
                    let! received = result
                    
                    if received.MessageType = WebSocketMessageType.Text then
                        let json = Encoding.UTF8.GetString(buffer, 0, received.Count)
                        
                        try
                            let message = JsonSerializer.Deserialize<TarsWebSocketMessage>(json)
                            this.HandleMessage(message)
                        with
                        | ex -> logger.LogError(ex, $"❌ Failed to parse message: {json}")
                    
                    elif received.MessageType = WebSocketMessageType.Close then
                        logger.LogInformation("🔌 WebSocket closed by server")
                        break
            
            with
            | :? OperationCanceledException -> 
                logger.LogInformation("📡 WebSocket receiving cancelled")
            | ex -> 
                logger.LogError(ex, "❌ WebSocket receiving error")
        | None -> ()
    }
    
    /// Connect to TARS service
    member this.ConnectAsync(serviceUrl: string) = task {
        try
            if isConnected then
                logger.LogWarning("⚠️ Already connected to TARS service")
                return true
            
            let ws = new ClientWebSocket()
            let uri = Uri($"ws://{serviceUrl.Replace("http://", "").Replace("https://", "")}/ws")
            
            logger.LogInformation($"🔌 Connecting to TARS service: {uri}")
            
            do! ws.ConnectAsync(uri, cancellationTokenSource.Token)
            
            webSocket <- Some ws
            isConnected <- true
            
            logger.LogInformation("✅ Connected to TARS service")
            
            // Start receiving messages
            Task.Run(fun () -> this.StartReceiving()) |> ignore
            
            return true
            
        with
        | ex ->
            logger.LogError(ex, "❌ Failed to connect to TARS service")
            return false
    }
    
    /// Disconnect from service
    member this.DisconnectAsync() = task {
        try
            isConnected <- false
            cancellationTokenSource.Cancel()
            
            match webSocket with
            | Some ws when ws.State = WebSocketState.Open ->
                do! ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client disconnect", CancellationToken.None)
                logger.LogInformation("🔌 Disconnected from TARS service")
            | _ -> ()
            
            webSocket <- None
            
        with
        | ex -> logger.LogError(ex, "❌ Error during disconnect")
    }
    
    /// Send command to service
    member this.SendCommandAsync(command: string, data: obj option) = task {
        let message = {
            Id = this.GenerateMessageId()
            Type = TarsMessageType.Command
            Command = Some command
            Data = data |> Option.map JsonSerializer.SerializeToElement
            Timestamp = DateTime.UtcNow
            Source = "TarsCLI"
        }
        
        do! this.SendMessage(message)
    }
    
    /// Send event to service
    member this.SendEventAsync(eventName: string, data: obj option) = task {
        let message = {
            Id = this.GenerateMessageId()
            Type = TarsMessageType.Event
            Command = Some eventName
            Data = data |> Option.map JsonSerializer.SerializeToElement
            Timestamp = DateTime.UtcNow
            Source = "TarsCLI"
        }
        
        do! this.SendMessage(message)
    }
    
    /// Ping service
    member this.PingAsync() = task {
        do! this.SendCommandAsync("ping", None)
    }
    
    /// Documentation task commands
    member this.StartDocumentationAsync() = task {
        do! this.SendCommandAsync("documentation.start", None)
    }
    
    member this.PauseDocumentationAsync() = task {
        do! this.SendCommandAsync("documentation.pause", None)
    }
    
    member this.ResumeDocumentationAsync() = task {
        do! this.SendCommandAsync("documentation.resume", None)
    }
    
    member this.StopDocumentationAsync() = task {
        do! this.SendCommandAsync("documentation.stop", None)
    }
    
    member this.GetDocumentationStatusAsync() = task {
        do! this.SendCommandAsync("documentation.status", None)
    }
    
    member this.GetServiceStatusAsync() = task {
        do! this.SendCommandAsync("service.status", None)
    }
    
    /// Event handler setters
    member this.OnProgressUpdate(handler: JsonElement -> unit) =
        onProgressUpdate <- Some handler
    
    member this.OnStatusUpdate(handler: JsonElement -> unit) =
        onStatusUpdate <- Some handler
    
    member this.OnResponse(handler: string -> JsonElement -> unit) =
        onResponse <- Some handler
    
    member this.OnError(handler: string -> unit) =
        onError <- Some handler
    
    member this.OnConnected(handler: string -> unit) =
        onConnected <- Some handler
    
    /// Properties
    member this.IsConnected = isConnected
    
    member this.ConnectionState = 
        match webSocket with
        | Some ws -> ws.State.ToString()
        | None -> "Disconnected"
    
    /// Dispose resources
    interface IDisposable with
        member this.Dispose() =
            try
                this.DisconnectAsync() |> Async.AwaitTask |> Async.RunSynchronously
            with _ -> ()
            
            cancellationTokenSource?.Dispose()
            
            match webSocket with
            | Some ws -> ws.Dispose()
            | None -> ()
