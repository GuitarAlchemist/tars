namespace TarsEngine.FSharp.Cli.WebSocket

open System
open System.Net
open System.Net.WebSockets
open System.Text
open System.Threading
open System.Threading.Tasks
open System.Text.Json
open TarsEngine.FSharp.Cli.BeliefPropagation
open TarsEngine.FSharp.Cli.CognitivePsychology

// ============================================================================
// TARS REAL-TIME WEBSOCKET SERVER - NO SIMULATION
// ============================================================================

type WebSocketMessage = {
    Type: string
    Data: obj
    Timestamp: DateTime
}

type TarsWebSocketServer(beliefBus: TarsBeliefBus, cognitiveEngine: TarsCognitivePsychologyEngine) =
    let mutable httpListener: HttpListener option = None
    let mutable isRunning = false
    let connectedClients = System.Collections.Concurrent.ConcurrentBag<WebSocket>()
    
    // Subscribe to real belief propagation events
    let beliefSubscription = beliefBus.Subscribe(SubsystemId.ApiSandbox) // Use API sandbox as observer
    
    member this.StartServer(port: int) =
        task {
            if not isRunning then
                isRunning <- true
                let listener = new HttpListener()
                listener.Prefixes.Add(sprintf "http://localhost:%d/" port)
                httpListener <- Some listener
                listener.Start()
                
                Console.WriteLine(sprintf "🌐 TARS WebSocket Server started on port %d" port)
                
                // Start belief monitoring task
                Task.Run(fun () -> this.MonitorBeliefs()) |> ignore
                
                // Start client connection handler
                Task.Run(fun () -> this.HandleConnections()) |> ignore
        }
    
    member private this.HandleConnections() =
        task {
            match httpListener with
            | Some listener ->
                while isRunning do
                    try
                        let! context = listener.GetContextAsync()
                        if context.Request.IsWebSocketRequest then
                            Task.Run(fun () -> this.HandleWebSocketConnection(context)) |> ignore
                        else
                            context.Response.StatusCode <- 400
                            context.Response.Close()
                    with
                    | _ -> () // Continue on errors
            | None -> ()
        }
    
    member private this.HandleWebSocketConnection(context: HttpListenerContext) =
        task {
            try
                let! webSocketContext = context.AcceptWebSocketAsync(null)
                let webSocket = webSocketContext.WebSocket
                connectedClients.Add(webSocket)
                
                Console.WriteLine("🔌 WebSocket client connected")
                
                // Send initial state
                do! this.SendInitialState(webSocket)
                
                // Keep connection alive and handle incoming messages
                let buffer = Array.zeroCreate<byte> 4096
                while webSocket.State = WebSocketState.Open do
                    try
                        let! result = webSocket.ReceiveAsync(ArraySegment(buffer), CancellationToken.None)
                        if result.MessageType = WebSocketMessageType.Close then
                            do! webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None)
                        elif result.MessageType = WebSocketMessageType.Text then
                            let message = Encoding.UTF8.GetString(buffer, 0, result.Count)
                            do! this.HandleClientMessage(webSocket, message)
                    with
                    | _ -> 
                        if webSocket.State = WebSocketState.Open then
                            do! webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, "", CancellationToken.None)
            with
            | ex -> Console.WriteLine(sprintf "WebSocket error: %s" ex.Message)
        }
    
    member private this.SendInitialState(webSocket: WebSocket) =
        task {
            // Send current belief state
            let activeBeliefs = beliefBus.GetActiveBeliefs()
            let beliefHistory = beliefBus.GetBeliefHistory()
            // Get REAL cognitive metrics from the engine
            let realMetrics = cognitiveEngine.GetCognitiveMetrics()
            let cognitiveMetrics = {|
                ReasoningQuality = realMetrics.ReasoningQuality
                BiasLevel = realMetrics.BiasLevel
                MentalLoad = realMetrics.MentalLoad
                SelfAwareness = realMetrics.SelfAwareness
                EmotionalIntelligence = realMetrics.EmotionalIntelligence
                DecisionQuality = realMetrics.DecisionQuality
                StressResilience = realMetrics.StressResilience
                LearningRate = realMetrics.LearningRate
                AdaptationSpeed = realMetrics.AdaptationSpeed
                CreativityIndex = realMetrics.CreativityIndex
            |}
            
            let initialState = {
                Type = "initial_state"
                Data = {|
                    ActiveBeliefs = activeBeliefs |> List.map (fun b -> {|
                        Id = b.Id
                        Source = sprintf "%A" b.Source
                        Type = sprintf "%A" b.BeliefType
                        Strength = sprintf "%A" b.Strength
                        Message = b.Message
                        Confidence = b.Confidence
                        Timestamp = b.Timestamp.ToString("HH:mm:ss")
                    |})
                    BeliefHistory = beliefHistory |> List.take (min 10 beliefHistory.Length) |> List.map (fun b -> {|
                        Source = sprintf "%A" b.Source
                        Message = b.Message
                        Timestamp = b.Timestamp.ToString("HH:mm:ss")
                        Strength = sprintf "%A" b.Strength
                    |})
                    CognitiveMetrics = cognitiveMetrics
                |}
                Timestamp = DateTime.UtcNow
            }
            
            do! this.SendMessage(webSocket, initialState)
        }
    
    member private this.HandleClientMessage(webSocket: WebSocket, message: string) =
        task {
            try
                let request = JsonSerializer.Deserialize<{| Type: string; Data: JsonElement |}>(message)
                match request.Type with
                | "request_refresh" ->
                    do! this.SendInitialState(webSocket)
                | "publish_belief" ->
                    // Allow clients to publish beliefs for testing
                    let testBelief = BeliefFactory.CreateInsightBelief(
                        SubsystemId.ApiSandbox,
                        "WebSocket test belief",
                        ["Client-generated belief for testing"],
                        0.8)
                    do! beliefBus.PublishBelief(testBelief)
                | _ -> ()
            with
            | ex -> Console.WriteLine(sprintf "Error handling client message: %s" ex.Message)
        }
    
    member private this.MonitorBeliefs() =
        task {
            while isRunning do
                try
                    let! belief = beliefSubscription.ReadAsync()
                    
                    // Broadcast belief update to all connected clients
                    let beliefUpdate = {
                        Type = "belief_update"
                        Data = {|
                            Id = belief.Id
                            Source = sprintf "%A" belief.Source
                            Type = sprintf "%A" belief.BeliefType
                            Strength = sprintf "%A" belief.Strength
                            Message = belief.Message
                            Confidence = belief.Confidence
                            Timestamp = belief.Timestamp.ToString("HH:mm:ss")
                        |}
                        Timestamp = DateTime.UtcNow
                    }
                    
                    do! this.BroadcastToAllClients(beliefUpdate)

                    // Get REAL cognitive metrics update
                    let realMetrics = cognitiveEngine.GetCognitiveMetrics()
                    let metricsUpdate = {
                        Type = "cognitive_metrics_update"
                        Data = {|
                            ReasoningQuality = realMetrics.ReasoningQuality
                            BiasLevel = realMetrics.BiasLevel
                            MentalLoad = realMetrics.MentalLoad
                            SelfAwareness = realMetrics.SelfAwareness
                            EmotionalIntelligence = realMetrics.EmotionalIntelligence
                            DecisionQuality = realMetrics.DecisionQuality
                            StressResilience = realMetrics.StressResilience
                            LearningRate = realMetrics.LearningRate
                            AdaptationSpeed = realMetrics.AdaptationSpeed
                            CreativityIndex = realMetrics.CreativityIndex
                        |}
                        Timestamp = DateTime.UtcNow
                    }

                    do! this.BroadcastToAllClients(metricsUpdate)
                with
                | _ -> () // Continue on errors
        }
    
    member private this.BroadcastToAllClients(message: WebSocketMessage) =
        task {
            let json = JsonSerializer.Serialize(message)
            let buffer = Encoding.UTF8.GetBytes(json)
            
            for client in connectedClients do
                if client.State = WebSocketState.Open then
                    try
                        do! client.SendAsync(ArraySegment(buffer), WebSocketMessageType.Text, true, CancellationToken.None)
                    with
                    | _ -> () // Ignore failed sends
        }
    
    member private this.SendMessage(webSocket: WebSocket, message: WebSocketMessage) =
        task {
            if webSocket.State = WebSocketState.Open then
                let json = JsonSerializer.Serialize(message)
                let buffer = Encoding.UTF8.GetBytes(json)
                do! webSocket.SendAsync(ArraySegment(buffer), WebSocketMessageType.Text, true, CancellationToken.None)
        }
    
    member this.Stop() =
        isRunning <- false
        match httpListener with
        | Some listener ->
            listener.Stop()
            listener.Close()
            httpListener <- None
        | None -> ()
        
        Console.WriteLine("🔌 TARS WebSocket Server stopped")
    
    interface IDisposable with
        member this.Dispose() = this.Stop()
