namespace TarsEngine.FSharp.WindowsService.Agents

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open System.Threading.Channels
open Microsoft.Extensions.Logging
open System.Text.Json

/// <summary>
/// Message types for inter-agent communication
/// </summary>
type MessageType =
    | Request
    | Response
    | Notification
    | Broadcast
    | Heartbeat

/// <summary>
/// Message priority levels
/// </summary>
type MessagePriority =
    | Critical = 1
    | High = 2
    | Normal = 3
    | Low = 4

/// <summary>
/// Agent message for inter-agent communication
/// </summary>
type AgentMessage = {
    Id: string
    Type: MessageType
    Priority: MessagePriority
    FromAgentId: string
    ToAgentId: string option  // None for broadcast messages
    Subject: string
    Payload: obj
    Timestamp: DateTime
    CorrelationId: string option
    ReplyTo: string option
    ExpiresAt: DateTime option
    Metadata: Map<string, obj>
}

/// <summary>
/// Message delivery result
/// </summary>
type MessageDeliveryResult =
    | Delivered
    | Failed of string
    | Expired
    | AgentNotFound
    | QueueFull

/// <summary>
/// Message handler function type
/// </summary>
type MessageHandler = AgentMessage -> CancellationToken -> Task<Result<obj option, string>>

/// <summary>
/// Message subscription information
/// </summary>
type MessageSubscription = {
    AgentId: string
    Subject: string
    Handler: MessageHandler
    SubscribedAt: DateTime
}

/// <summary>
/// Communication statistics
/// </summary>
type CommunicationStats = {
    TotalMessagesSent: int64
    TotalMessagesReceived: int64
    TotalMessagesDelivered: int64
    TotalMessagesFailed: int64
    TotalMessagesExpired: int64
    AverageDeliveryTimeMs: float
    ActiveSubscriptions: int
    QueueUtilization: float
}

/// <summary>
/// High-performance inter-agent communication system using .NET Channels
/// </summary>
type AgentCommunication(logger: ILogger<AgentCommunication>) =
    
    let agentChannels = ConcurrentDictionary<string, Channel<AgentMessage>>()
    let messageSubscriptions = ConcurrentDictionary<string, ConcurrentDictionary<string, MessageSubscription>>()
    let broadcastSubscriptions = ConcurrentDictionary<string, MessageSubscription>()
    let messageStats = ConcurrentDictionary<string, int64>()
    let deliveryTimes = ConcurrentQueue<float>()
    
    let maxQueueSize = 1000
    let maxDeliveryTimeHistory = 1000
    
    /// Register an agent for communication
    member this.RegisterAgent(agentId: string) =
        try
            logger.LogInformation($"Registering agent for communication: {agentId}")
            
            if agentChannels.ContainsKey(agentId) then
                logger.LogWarning($"Agent {agentId} is already registered for communication")
                Ok ()
            else
                // Create bounded channel for the agent
                let channelOptions = BoundedChannelOptions(maxQueueSize)
                channelOptions.FullMode <- BoundedChannelFullMode.Wait
                channelOptions.SingleReader <- true
                channelOptions.SingleWriter <- false
                
                let channel = Channel.CreateBounded<AgentMessage>(channelOptions)
                agentChannels.[agentId] <- channel
                
                // Initialize subscription dictionary for this agent
                messageSubscriptions.[agentId] <- ConcurrentDictionary<string, MessageSubscription>()
                
                logger.LogInformation($"Agent {agentId} registered successfully for communication")
                Ok ()
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to register agent for communication: {agentId}")
            Error ex.Message
    
    /// Unregister an agent from communication
    member this.UnregisterAgent(agentId: string) =
        try
            logger.LogInformation($"Unregistering agent from communication: {agentId}")
            
            // Remove agent channel
            match agentChannels.TryRemove(agentId) with
            | true, channel ->
                channel.Writer.Complete()
                logger.LogDebug($"Channel closed for agent: {agentId}")
            | false, _ ->
                logger.LogWarning($"No channel found for agent: {agentId}")
            
            // Remove subscriptions
            messageSubscriptions.TryRemove(agentId) |> ignore
            
            // Remove broadcast subscriptions
            let broadcastsToRemove = 
                broadcastSubscriptions.Values
                |> Seq.filter (fun s -> s.AgentId = agentId)
                |> Seq.map (fun s -> s.Subject)
                |> List.ofSeq
            
            for subject in broadcastsToRemove do
                broadcastSubscriptions.TryRemove(subject) |> ignore
            
            logger.LogInformation($"Agent {agentId} unregistered successfully from communication")
            Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to unregister agent from communication: {agentId}")
            Error ex.Message
    
    /// Send a message to a specific agent
    member this.SendMessageAsync(message: AgentMessage) = task {
        try
            let startTime = DateTime.UtcNow
            logger.LogDebug($"Sending message {message.Id} from {message.FromAgentId} to {message.ToAgentId}")
            
            match message.ToAgentId with
            | Some targetAgentId ->
                // Send to specific agent
                match agentChannels.TryGetValue(targetAgentId) with
                | true, channel ->
                    // Check if message has expired
                    match message.ExpiresAt with
                    | Some expiry when DateTime.UtcNow > expiry ->
                        this.IncrementStat("MessagesExpired")
                        logger.LogWarning($"Message {message.Id} expired before delivery")
                        return Expired
                    | _ ->
                        // Try to send the message
                        if channel.Writer.TryWrite(message) then
                            this.IncrementStat("MessagesSent")
                            this.IncrementStat("MessagesDelivered")
                            
                            let deliveryTime = (DateTime.UtcNow - startTime).TotalMilliseconds
                            this.RecordDeliveryTime(deliveryTime)
                            
                            logger.LogDebug($"Message {message.Id} delivered successfully to {targetAgentId}")
                            return Delivered
                        else
                            this.IncrementStat("MessagesFailed")
                            logger.LogWarning($"Failed to deliver message {message.Id} to {targetAgentId} - queue full")
                            return QueueFull
                
                | false, _ ->
                    this.IncrementStat("MessagesFailed")
                    logger.LogWarning($"Target agent {targetAgentId} not found for message {message.Id}")
                    return AgentNotFound
            
            | None ->
                // Broadcast message
                return! this.BroadcastMessageAsync(message)
                
        with
        | ex ->
            this.IncrementStat("MessagesFailed")
            logger.LogError(ex, $"Error sending message {message.Id}")
            return Failed ex.Message
    }
    
    /// Broadcast a message to all subscribed agents
    member private this.BroadcastMessageAsync(message: AgentMessage) = task {
        try
            logger.LogDebug($"Broadcasting message {message.Id} with subject {message.Subject}")
            
            let subscribers = 
                broadcastSubscriptions.Values
                |> Seq.filter (fun s -> s.Subject = message.Subject)
                |> List.ofSeq
            
            if subscribers.IsEmpty then
                logger.LogDebug($"No subscribers found for broadcast subject: {message.Subject}")
                return Delivered
            else
                let mutable deliveredCount = 0
                let mutable failedCount = 0
                
                for subscriber in subscribers do
                    let targetMessage = { message with ToAgentId = Some subscriber.AgentId }
                    let! result = this.SendMessageAsync(targetMessage)
                    
                    match result with
                    | Delivered -> deliveredCount <- deliveredCount + 1
                    | _ -> failedCount <- failedCount + 1
                
                logger.LogInformation($"Broadcast message {message.Id} delivered to {deliveredCount}/{subscribers.Length} subscribers")
                
                if failedCount = 0 then
                    return Delivered
                else
                    return Failed $"Failed to deliver to {failedCount} subscribers"
                    
        with
        | ex ->
            logger.LogError(ex, $"Error broadcasting message {message.Id}")
            return Failed ex.Message
    }
    
    /// Receive messages for an agent
    member this.ReceiveMessagesAsync(agentId: string, cancellationToken: CancellationToken) = 
        task {
            match agentChannels.TryGetValue(agentId) with
            | true, channel ->
                try
                    let! hasMessage = channel.Reader.WaitToReadAsync(cancellationToken).AsTask()
                    if hasMessage then
                        match channel.Reader.TryRead() with
                        | true, message ->
                            this.IncrementStat("MessagesReceived")
                            logger.LogDebug($"Agent {agentId} received message {message.Id}")
                            return Some message
                        | false, _ ->
                            return None
                    else
                        return None
                with
                | :? OperationCanceledException ->
                    logger.LogDebug($"Message receive cancelled for agent: {agentId}")
                    return None
            | false, _ ->
                logger.LogWarning($"No channel found for agent: {agentId}")
                return None
        }
    
    /// Subscribe to messages with a specific subject
    member this.SubscribeToSubject(agentId: string, subject: string, handler: MessageHandler) =
        try
            logger.LogInformation($"Agent {agentId} subscribing to subject: {subject}")
            
            match messageSubscriptions.TryGetValue(agentId) with
            | true, agentSubs ->
                let subscription = {
                    AgentId = agentId
                    Subject = subject
                    Handler = handler
                    SubscribedAt = DateTime.UtcNow
                }
                
                agentSubs.[subject] <- subscription
                logger.LogInformation($"Agent {agentId} subscribed to subject {subject}")
                Ok ()
            
            | false, _ ->
                let error = $"Agent {agentId} is not registered for communication"
                logger.LogError(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to subscribe agent {agentId} to subject {subject}")
            Error ex.Message
    
    /// Subscribe to broadcast messages
    member this.SubscribeToBroadcast(agentId: string, subject: string, handler: MessageHandler) =
        try
            logger.LogInformation($"Agent {agentId} subscribing to broadcast subject: {subject}")
            
            let subscription = {
                AgentId = agentId
                Subject = subject
                Handler = handler
                SubscribedAt = DateTime.UtcNow
            }
            
            let subscriptionKey = $"{agentId}:{subject}"
            broadcastSubscriptions.[subscriptionKey] <- subscription
            
            logger.LogInformation($"Agent {agentId} subscribed to broadcast subject {subject}")
            Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to subscribe agent {agentId} to broadcast subject {subject}")
            Error ex.Message
    
    /// Unsubscribe from a subject
    member this.UnsubscribeFromSubject(agentId: string, subject: string) =
        try
            match messageSubscriptions.TryGetValue(agentId) with
            | true, agentSubs ->
                agentSubs.TryRemove(subject) |> ignore
                logger.LogInformation($"Agent {agentId} unsubscribed from subject {subject}")
                Ok ()
            | false, _ ->
                let error = $"Agent {agentId} is not registered for communication"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to unsubscribe agent {agentId} from subject {subject}")
            Error ex.Message
    
    /// Create a message
    member this.CreateMessage(fromAgentId: string, toAgentId: string option, subject: string, payload: obj, ?messageType: MessageType, ?priority: MessagePriority, ?correlationId: string, ?replyTo: string, ?expiresIn: TimeSpan) =
        {
            Id = Guid.NewGuid().ToString()
            Type = messageType |> Option.defaultValue Request
            Priority = priority |> Option.defaultValue MessagePriority.Normal
            FromAgentId = fromAgentId
            ToAgentId = toAgentId
            Subject = subject
            Payload = payload
            Timestamp = DateTime.UtcNow
            CorrelationId = correlationId
            ReplyTo = replyTo
            ExpiresAt = expiresIn |> Option.map (fun timeSpan -> DateTime.UtcNow.Add(timeSpan))
            Metadata = Map.empty
        }
    
    /// Get communication statistics
    member this.GetStatistics() =
        let totalSent = this.GetStat("MessagesSent")
        let totalReceived = this.GetStat("MessagesReceived")
        let totalDelivered = this.GetStat("MessagesDelivered")
        let totalFailed = this.GetStat("MessagesFailed")
        let totalExpired = this.GetStat("MessagesExpired")
        
        let averageDeliveryTime = 
            if deliveryTimes.Count > 0 then
                deliveryTimes |> Seq.average
            else 0.0
        
        let activeSubscriptions = 
            messageSubscriptions.Values 
            |> Seq.sumBy (fun agentSubs -> agentSubs.Count) 
            + broadcastSubscriptions.Count
        
        let queueUtilization = 
            if agentChannels.Count > 0 then
                let totalCapacity = agentChannels.Count * maxQueueSize
                let totalUsed = 
                    agentChannels.Values 
                    |> Seq.sumBy (fun channel -> 
                        if channel.Reader.CanCount then channel.Reader.Count else 0)
                (float totalUsed / float totalCapacity) * 100.0
            else 0.0
        
        {
            TotalMessagesSent = totalSent
            TotalMessagesReceived = totalReceived
            TotalMessagesDelivered = totalDelivered
            TotalMessagesFailed = totalFailed
            TotalMessagesExpired = totalExpired
            AverageDeliveryTimeMs = averageDeliveryTime
            ActiveSubscriptions = activeSubscriptions
            QueueUtilization = queueUtilization
        }
    
    /// Helper methods for statistics
    member private this.IncrementStat(statName: string) =
        messageStats.AddOrUpdate(statName, 1L, fun _ current -> current + 1L) |> ignore
    
    member private this.GetStat(statName: string) =
        match messageStats.TryGetValue(statName) with
        | true, value -> value
        | false, _ -> 0L
    
    member private this.RecordDeliveryTime(deliveryTimeMs: float) =
        deliveryTimes.Enqueue(deliveryTimeMs)
        
        // Keep only recent delivery times
        while deliveryTimes.Count > maxDeliveryTimeHistory do
            deliveryTimes.TryDequeue() |> ignore
    
    /// Get registered agents
    member this.GetRegisteredAgents() =
        agentChannels.Keys |> List.ofSeq
    
    /// Get agent subscriptions
    member this.GetAgentSubscriptions(agentId: string) =
        match messageSubscriptions.TryGetValue(agentId) with
        | true, agentSubs -> agentSubs.Keys |> List.ofSeq
        | false, _ -> []
