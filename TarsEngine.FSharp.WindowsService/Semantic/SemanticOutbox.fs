namespace TarsEngine.FSharp.WindowsService.Semantic

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Delivery attempt information
/// </summary>
type DeliveryAttempt = {
    AttemptNumber: int
    AttemptedAt: DateTime
    Status: DeliveryStatus
    Error: string option
    ResponseTime: TimeSpan
    RecipientId: string
}

/// <summary>
/// Outbound message with delivery tracking
/// </summary>
type OutboundMessage = {
    Message: SemanticMessage
    Recipients: string list
    DeliveryAttempts: DeliveryAttempt list
    Status: OutboundMessageStatus
    CreatedAt: DateTime
    LastAttemptAt: DateTime option
    CompletedAt: DateTime option
    Priority: int
    RetryCount: int
    MaxRetries: int
}

/// <summary>
/// Outbound message status
/// </summary>
and OutboundMessageStatus =
    | Queued
    | Sending
    | PartiallyDelivered
    | Delivered
    | Failed
    | Expired
    | Cancelled

/// <summary>
/// Delivery configuration
/// </summary>
type DeliveryConfiguration = {
    MaxRetries: int
    RetryDelay: TimeSpan
    RetryBackoffMultiplier: float
    MaxRetryDelay: TimeSpan
    DeliveryTimeout: TimeSpan
    BatchSize: int
    ConcurrentDeliveries: int
    EnableDeliveryReceipts: bool
}

/// <summary>
/// Outbox statistics
/// </summary>
type OutboxStatistics = {
    TotalMessagesSent: int64
    MessagesDelivered: int64
    MessagesFailed: int64
    MessagesExpired: int64
    AverageDeliveryTime: TimeSpan
    DeliverySuccessRate: float
    CurrentQueueSize: int
    MessagesInTransit: int
    TotalRetries: int64
    AverageRetries: float
}

/// <summary>
/// Batch delivery result
/// </summary>
type BatchDeliveryResult = {
    BatchId: string
    MessageIds: string list
    SuccessfulDeliveries: int
    FailedDeliveries: int
    DeliveryTime: TimeSpan
    Errors: string list
}

/// <summary>
/// Semantic outbox for reliable message delivery
/// </summary>
type SemanticOutbox(agentId: string, logger: ILogger<SemanticOutbox>) =
    
    let outboundMessages = ConcurrentDictionary<string, OutboundMessage>()
    let deliveryQueue = ConcurrentQueue<OutboundMessage>()
    let deliveryReceipts = ConcurrentDictionary<string, DeliveryReceipt>()
    let statistics = ConcurrentDictionary<string, int64>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable deliveryTask: Task option = None
    let mutable monitoringTask: Task option = None
    
    let defaultConfig = {
        MaxRetries = 3
        RetryDelay = TimeSpan.FromSeconds(5.0)
        RetryBackoffMultiplier = 2.0
        MaxRetryDelay = TimeSpan.FromMinutes(5.0)
        DeliveryTimeout = TimeSpan.FromSeconds(30.0)
        BatchSize = 10
        ConcurrentDeliveries = 5
        EnableDeliveryReceipts = true
    }
    
    let maxOutboundHistory = 10000
    
    /// Start the semantic outbox
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Starting semantic outbox for agent: {agentId}")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Start delivery loop
            let deliveryLoop = this.DeliveryLoopAsync(defaultConfig, cancellationTokenSource.Value.Token)
            deliveryTask <- Some deliveryLoop
            
            // Start monitoring loop
            let monitoringLoop = this.MonitoringLoopAsync(cancellationTokenSource.Value.Token)
            monitoringTask <- Some monitoringLoop
            
            logger.LogInformation($"Semantic outbox started for agent: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to start semantic outbox for agent: {agentId}")
            isRunning <- false
            raise
    }
    
    /// Stop the semantic outbox
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Stopping semantic outbox for agent: {agentId}")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for tasks to complete
            let tasks = [
                match deliveryTask with Some t -> [t] | None -> []
                match monitoringTask with Some t -> [t] | None -> []
            ] |> List.concat |> Array.ofList
            
            if tasks.Length > 0 then
                try
                    do! Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning($"Semantic outbox tasks did not complete within timeout for agent: {agentId}")
                | ex ->
                    logger.LogWarning(ex, $"Error waiting for semantic outbox tasks to complete for agent: {agentId}")
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            deliveryTask <- None
            monitoringTask <- None
            
            logger.LogInformation($"Semantic outbox stopped for agent: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Error stopping semantic outbox for agent: {agentId}")
    }
    
    /// Send a message
    member this.SendMessageAsync(message: SemanticMessage) = task {
        try
            logger.LogDebug($"Sending message: {message.Id} from agent: {agentId}")
            
            // Validate message
            let validation = SemanticMessageHelpers.validateMessage message
            if not validation.IsValid then
                let errors = String.Join("; ", validation.Errors)
                logger.LogWarning($"Invalid message for sending: {message.Id} - {errors}")
                return Error $"Invalid message: {errors}"
            
            // Determine recipients
            let recipients = 
                if message.Recipients.IsEmpty then
                    match message.BroadcastScope with
                    | Some (SpecificAgents agents) -> agents
                    | Some AllAgents -> ["*"] // Broadcast to all
                    | Some (AgentsByCapability capabilities) -> capabilities |> List.map (fun c -> $"capability:{c}")
                    | Some (AgentsByType types) -> types |> List.map (fun t -> $"type:{t}")
                    | Some (AgentsByTag tags) -> tags |> List.map (fun t -> $"tag:{t}")
                    | Some (NearbyAgents radius) -> [$"nearby:{radius}"]
                    | None -> []
                else
                    message.Recipients
            
            if recipients.IsEmpty then
                logger.LogWarning($"No recipients specified for message: {message.Id}")
                return Error "No recipients specified"
            
            // Create outbound message
            let outboundMessage = {
                Message = message
                Recipients = recipients
                DeliveryAttempts = []
                Status = Queued
                CreatedAt = DateTime.UtcNow
                LastAttemptAt = None
                CompletedAt = None
                Priority = int (SemanticMessageHelpers.calculatePriorityScore message)
                RetryCount = 0
                MaxRetries = message.MaxDeliveryAttempts
            }
            
            // Queue for delivery
            outboundMessages.[message.Id] <- outboundMessage
            deliveryQueue.Enqueue(outboundMessage)
            
            // Update statistics
            this.UpdateStatistics("TotalMessagesSent", 1L)
            this.UpdateStatistics($"Sent_{message.MessageType}", 1L)
            
            logger.LogInformation($"Message queued for delivery: {message.Id} from agent: {agentId}")
            return Ok message.Id
            
        with
        | ex ->
            logger.LogError(ex, $"Error sending message: {message.Id} from agent: {agentId}")
            return Error ex.Message
    }
    
    /// Send message to specific recipients
    member this.SendToRecipientsAsync(message: SemanticMessage, recipients: string list) = task {
        let messageWithRecipients = { message with Recipients = recipients }
        return! this.SendMessageAsync(messageWithRecipients)
    }
    
    /// Broadcast message to all agents
    member this.BroadcastMessageAsync(message: SemanticMessage) = task {
        let broadcastMessage = { message with BroadcastScope = Some AllAgents; Recipients = [] }
        return! this.SendMessageAsync(broadcastMessage)
    }
    
    /// Send message to agents with specific capabilities
    member this.SendToCapabilityAsync(message: SemanticMessage, capabilities: string list) = task {
        let capabilityMessage = { message with BroadcastScope = Some (AgentsByCapability capabilities); Recipients = [] }
        return! this.SendMessageAsync(capabilityMessage)
    }
    
    /// Batch send multiple messages
    member this.BatchSendAsync(messages: SemanticMessage list) = task {
        try
            logger.LogInformation($"Batch sending {messages.Length} messages from agent: {agentId}")
            
            let batchId = Guid.NewGuid().ToString()
            let results = ResizeArray<Result<string, string>>()
            
            for message in messages do
                let! result = this.SendMessageAsync(message)
                results.Add(result)
            
            let successful = results |> Seq.filter (fun r -> r.IsOk) |> Seq.length
            let failed = results.Count - successful
            
            let batchResult = {
                BatchId = batchId
                MessageIds = messages |> List.map (fun m -> m.Id)
                SuccessfulDeliveries = successful
                FailedDeliveries = failed
                DeliveryTime = TimeSpan.Zero // Would be calculated
                Errors = results |> Seq.choose (fun r -> match r with Error e -> Some e | Ok _ -> None) |> List.ofSeq
            }
            
            logger.LogInformation($"Batch send completed: {successful} successful, {failed} failed from agent: {agentId}")
            return Ok batchResult
            
        with
        | ex ->
            logger.LogError(ex, $"Error in batch send from agent: {agentId}")
            return Error ex.Message
    }
    
    /// Get delivery status for a message
    member this.GetDeliveryStatusAsync(messageId: string) = task {
        match outboundMessages.TryGetValue(messageId) with
        | true, outboundMessage ->
            let deliveryReceipts = 
                outboundMessage.Recipients
                |> List.choose (fun recipientId ->
                    match deliveryReceipts.TryGetValue($"{messageId}:{recipientId}") with
                    | true, receipt -> Some receipt
                    | false, _ -> None)
            
            return Some (outboundMessage.Status, deliveryReceipts)
        | false, _ ->
            return None
    }
    
    /// Cancel message delivery
    member this.CancelDeliveryAsync(messageId: string) = task {
        match outboundMessages.TryGetValue(messageId) with
        | true, outboundMessage when outboundMessage.Status = Queued || outboundMessage.Status = Sending ->
            let cancelledMessage = { outboundMessage with Status = Cancelled; CompletedAt = Some DateTime.UtcNow }
            outboundMessages.[messageId] <- cancelledMessage
            
            logger.LogInformation($"Message delivery cancelled: {messageId} from agent: {agentId}")
            return Ok ()
        
        | true, outboundMessage ->
            let error = $"Cannot cancel message in status: {outboundMessage.Status}"
            logger.LogWarning($"Cannot cancel message {messageId}: {error}")
            return Error error
        
        | false, _ ->
            let error = $"Message not found: {messageId}"
            logger.LogWarning($"Cannot cancel message {messageId}: {error}")
            return Error error
    }
    
    /// Delivery loop
    member private this.DeliveryLoopAsync(config: DeliveryConfiguration, cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug($"Starting delivery loop for agent: {agentId}")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Process delivery queue
                    let messagesToDeliver = ResizeArray<OutboundMessage>()
                    
                    // Dequeue messages up to batch size
                    for _ in 1 .. config.BatchSize do
                        match deliveryQueue.TryDequeue() with
                        | true, message -> messagesToDeliver.Add(message)
                        | false, _ -> ()
                    
                    if messagesToDeliver.Count > 0 then
                        // Process deliveries concurrently
                        let deliveryTasks = 
                            messagesToDeliver
                            |> Seq.take config.ConcurrentDeliveries
                            |> Seq.map (fun outboundMessage -> 
                                this.DeliverMessageAsync(outboundMessage, config, cancellationToken))
                            |> Array.ofSeq
                        
                        do! Task.WhenAll(deliveryTasks)
                    
                    // Wait before next delivery cycle
                    do! Task.Delay(TimeSpan.FromMilliseconds(100.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, $"Error in delivery loop for agent: {agentId}")
                    do! Task.Delay(TimeSpan.FromSeconds(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Delivery loop cancelled for agent: {agentId}")
        | ex ->
            logger.LogError(ex, $"Delivery loop failed for agent: {agentId}")
    }
    
    /// Deliver a single message
    member private this.DeliverMessageAsync(outboundMessage: OutboundMessage, config: DeliveryConfiguration, cancellationToken: CancellationToken) = task {
        try
            let startTime = DateTime.UtcNow
            logger.LogDebug($"Delivering message: {outboundMessage.Message.Id} from agent: {agentId}")
            
            // Update status to sending
            let sendingMessage = { outboundMessage with Status = Sending; LastAttemptAt = Some startTime }
            outboundMessages.[outboundMessage.Message.Id] <- sendingMessage
            
            let deliveryResults = ResizeArray<DeliveryAttempt>()
            
            // Deliver to each recipient
            for recipientId in outboundMessage.Recipients do
                try
                    // Simulate message delivery (in real implementation, this would use the communication system)
                    let deliveryStartTime = DateTime.UtcNow
                    
                    // Simulate delivery delay
                    do! Task.Delay(TimeSpan.FromMilliseconds(10.0), cancellationToken)
                    
                    let deliveryTime = DateTime.UtcNow - deliveryStartTime
                    let deliveryAttempt = {
                        AttemptNumber = outboundMessage.RetryCount + 1
                        AttemptedAt = deliveryStartTime
                        Status = Delivered
                        Error = None
                        ResponseTime = deliveryTime
                        RecipientId = recipientId
                    }
                    
                    deliveryResults.Add(deliveryAttempt)
                    
                    // Create delivery receipt
                    let receipt = {
                        MessageId = outboundMessage.Message.Id
                        RecipientId = recipientId
                        Status = Delivered
                        DeliveredAt = Some DateTime.UtcNow
                        AcknowledgedAt = None
                        Error = None
                        Attempts = outboundMessage.RetryCount + 1
                    }
                    
                    deliveryReceipts.[$"{outboundMessage.Message.Id}:{recipientId}"] <- receipt
                    
                    logger.LogDebug($"Message delivered to {recipientId}: {outboundMessage.Message.Id}")
                    
                with
                | ex ->
                    let deliveryAttempt = {
                        AttemptNumber = outboundMessage.RetryCount + 1
                        AttemptedAt = DateTime.UtcNow
                        Status = Failed
                        Error = Some ex.Message
                        ResponseTime = TimeSpan.Zero
                        RecipientId = recipientId
                    }
                    
                    deliveryResults.Add(deliveryAttempt)
                    logger.LogWarning(ex, $"Failed to deliver message to {recipientId}: {outboundMessage.Message.Id}")
            
            // Determine overall delivery status
            let successfulDeliveries = deliveryResults |> Seq.filter (fun d -> d.Status = Delivered) |> Seq.length
            let totalRecipients = outboundMessage.Recipients.Length
            
            let finalStatus = 
                if successfulDeliveries = totalRecipients then Delivered
                elif successfulDeliveries > 0 then PartiallyDelivered
                else Failed
            
            // Update outbound message
            let completedMessage = {
                sendingMessage with
                    Status = finalStatus
                    DeliveryAttempts = sendingMessage.DeliveryAttempts @ (deliveryResults |> List.ofSeq)
                    CompletedAt = Some DateTime.UtcNow
                    RetryCount = sendingMessage.RetryCount + 1
            }
            
            outboundMessages.[outboundMessage.Message.Id] <- completedMessage
            
            // Update statistics
            match finalStatus with
            | Delivered -> this.UpdateStatistics("MessagesDelivered", 1L)
            | PartiallyDelivered -> this.UpdateStatistics("MessagesPartiallyDelivered", 1L)
            | Failed -> this.UpdateStatistics("MessagesFailed", 1L)
            | _ -> ()
            
            // Schedule retry if needed
            if finalStatus = Failed && completedMessage.RetryCount < completedMessage.MaxRetries then
                let retryDelay = this.CalculateRetryDelay(completedMessage.RetryCount, config)
                logger.LogInformation($"Scheduling retry for message {outboundMessage.Message.Id} in {retryDelay}")
                
                // Schedule retry (simplified - in production would use a proper scheduler)
                let retryTask = task {
                    do! Task.Delay(retryDelay, cancellationToken)
                    deliveryQueue.Enqueue(completedMessage)
                }
                retryTask |> ignore
            
            let deliveryTime = DateTime.UtcNow - startTime
            logger.LogInformation($"Message delivery completed: {outboundMessage.Message.Id} - {finalStatus} in {deliveryTime.TotalMilliseconds:F0}ms")
            
        with
        | ex ->
            logger.LogError(ex, $"Error delivering message: {outboundMessage.Message.Id} from agent: {agentId}")
    }
    
    /// Calculate retry delay with exponential backoff
    member private this.CalculateRetryDelay(retryCount: int, config: DeliveryConfiguration) =
        let baseDelay = config.RetryDelay.TotalMilliseconds
        let backoffDelay = baseDelay * (config.RetryBackoffMultiplier ** float retryCount)
        let finalDelay = min backoffDelay config.MaxRetryDelay.TotalMilliseconds
        TimeSpan.FromMilliseconds(finalDelay)
    
    /// Monitoring loop
    member private this.MonitoringLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug($"Starting monitoring loop for agent: {agentId}")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Monitor expired messages
                    let expiredMessages = 
                        outboundMessages.Values
                        |> Seq.filter (fun m -> 
                            m.Message.ExpiresAt.IsSome && 
                            m.Message.ExpiresAt.Value <= DateTime.UtcNow &&
                            m.Status <> Expired)
                        |> List.ofSeq
                    
                    for expiredMessage in expiredMessages do
                        let expiredOutbound = { expiredMessage with Status = Expired; CompletedAt = Some DateTime.UtcNow }
                        outboundMessages.[expiredMessage.Message.Id] <- expiredOutbound
                        this.UpdateStatistics("MessagesExpired", 1L)
                        logger.LogInformation($"Message expired: {expiredMessage.Message.Id}")
                    
                    // Clean up old completed messages
                    this.CleanupCompletedMessages()
                    
                    // Wait for next monitoring cycle
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, $"Error in monitoring loop for agent: {agentId}")
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Monitoring loop cancelled for agent: {agentId}")
        | ex ->
            logger.LogError(ex, $"Monitoring loop failed for agent: {agentId}")
    }
    
    /// Clean up old completed messages
    member private this.CleanupCompletedMessages() =
        let cutoffTime = DateTime.UtcNow.AddHours(-24.0)
        let messagesToRemove = 
            outboundMessages.Values
            |> Seq.filter (fun m -> 
                m.CompletedAt.IsSome && 
                m.CompletedAt.Value < cutoffTime &&
                (m.Status = Delivered || m.Status = Failed || m.Status = Expired || m.Status = Cancelled))
            |> Seq.map (fun m -> m.Message.Id)
            |> List.ofSeq
        
        for messageId in messagesToRemove do
            outboundMessages.TryRemove(messageId) |> ignore
        
        if messagesToRemove.Length > 0 then
            logger.LogDebug($"Cleaned up {messagesToRemove.Length} old messages for agent: {agentId}")
    
    /// Update statistics helper
    member private this.UpdateStatistics(key: string, increment: int64) =
        statistics.AddOrUpdate(key, increment, fun _ current -> current + increment) |> ignore
    
    /// Get outbox statistics
    member this.GetStatistics() =
        let totalSent = statistics.GetOrAdd("TotalMessagesSent", 0L)
        let delivered = statistics.GetOrAdd("MessagesDelivered", 0L)
        let failed = statistics.GetOrAdd("MessagesFailed", 0L)
        let expired = statistics.GetOrAdd("MessagesExpired", 0L)
        
        let successRate = 
            if totalSent > 0L then float delivered / float totalSent * 100.0
            else 0.0
        
        let queueSize = deliveryQueue.Count
        let inTransit = outboundMessages.Values |> Seq.filter (fun m -> m.Status = Sending) |> Seq.length
        
        {
            TotalMessagesSent = totalSent
            MessagesDelivered = delivered
            MessagesFailed = failed
            MessagesExpired = expired
            AverageDeliveryTime = TimeSpan.FromSeconds(1.0) // Would be calculated from actual delivery times
            DeliverySuccessRate = successRate
            CurrentQueueSize = queueSize
            MessagesInTransit = inTransit
            TotalRetries = statistics.GetOrAdd("TotalRetries", 0L)
            AverageRetries = 0.0 // Would be calculated
        }
    
    /// Get pending messages
    member this.GetPendingMessages() =
        outboundMessages.Values 
        |> Seq.filter (fun m -> m.Status = Queued || m.Status = Sending)
        |> List.ofSeq
    
    /// Get delivery receipts for a message
    member this.GetDeliveryReceipts(messageId: string) =
        deliveryReceipts.Values
        |> Seq.filter (fun r -> r.MessageId = messageId)
        |> List.ofSeq
