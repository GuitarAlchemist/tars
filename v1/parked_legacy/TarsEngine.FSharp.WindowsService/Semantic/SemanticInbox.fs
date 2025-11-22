namespace TarsEngine.FSharp.WindowsService.Semantic

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Message filter criteria
/// </summary>
type MessageFilter = {
    MessageTypes: SemanticMessageType list
    Priorities: SemanticPriority list
    Senders: string list
    Keywords: string list
    DateRange: (DateTime * DateTime) option
    HasExpired: bool option
}

/// <summary>
/// Inbox statistics
/// </summary>
type InboxStatistics = {
    TotalMessages: int64
    UnreadMessages: int
    MessagesByType: Map<SemanticMessageType, int>
    MessagesByPriority: Map<SemanticPriority, int>
    AverageProcessingTime: TimeSpan
    OldestMessage: DateTime option
    NewestMessage: DateTime option
    ExpiredMessages: int
    ProcessedToday: int64
}

/// <summary>
/// Message processing result
/// </summary>
type MessageProcessingResult = {
    MessageId: string
    Processed: bool
    ProcessingTime: TimeSpan
    Result: obj option
    Error: string option
    ProcessedAt: DateTime
}

/// <summary>
/// Semantic inbox for intelligent message queuing and processing
/// </summary>
type SemanticInbox(agentId: string, logger: ILogger<SemanticInbox>) =
    
    let messages = ConcurrentDictionary<string, SemanticMessage>()
    let messageQueue = ConcurrentQueue<SemanticMessage>()
    let processedMessages = ConcurrentDictionary<string, MessageProcessingResult>()
    let messageIndex = ConcurrentDictionary<string, string list>() // keyword -> message IDs
    let statistics = ConcurrentDictionary<string, int64>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable processingTask: Task option = None
    let mutable cleanupTask: Task option = None
    
    let maxMessages = 10000
    let maxProcessedHistory = 1000
    let cleanupInterval = TimeSpan.FromMinutes(5.0)
    
    /// Start the semantic inbox
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Starting semantic inbox for agent: {agentId}")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Start message processing loop
            let processingLoop = this.MessageProcessingLoopAsync(cancellationTokenSource.Value.Token)
            processingTask <- Some processingLoop
            
            // Start cleanup loop
            let cleanupLoop = this.CleanupLoopAsync(cancellationTokenSource.Value.Token)
            cleanupTask <- Some cleanupLoop
            
            logger.LogInformation($"Semantic inbox started for agent: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to start semantic inbox for agent: {agentId}")
            isRunning <- false
            raise
    }
    
    /// Stop the semantic inbox
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Stopping semantic inbox for agent: {agentId}")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for tasks to complete
            let tasks = [
                match processingTask with Some t -> [t] | None -> []
                match cleanupTask with Some t -> [t] | None -> []
            ] |> List.concat |> Array.ofList
            
            if tasks.Length > 0 then
                try
                    do! Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning($"Semantic inbox tasks did not complete within timeout for agent: {agentId}")
                | ex ->
                    logger.LogWarning(ex, $"Error waiting for semantic inbox tasks to complete for agent: {agentId}")
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            processingTask <- None
            cleanupTask <- None
            
            logger.LogInformation($"Semantic inbox stopped for agent: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Error stopping semantic inbox for agent: {agentId}")
    }
    
    /// Receive a message
    member this.ReceiveMessageAsync(message: SemanticMessage) = task {
        try
            logger.LogDebug($"Receiving message: {message.Id} for agent: {agentId}")
            
            // Validate message
            let validation = SemanticMessageHelpers.validateMessage message
            if not validation.IsValid then
                let errors = String.Join("; ", validation.Errors)
                logger.LogWarning($"Invalid message received: {message.Id} - {errors}")
                return Error $"Invalid message: {errors}"
            
            // Check if message is expired
            match message.ExpiresAt with
            | Some expiry when expiry <= DateTime.UtcNow ->
                logger.LogWarning($"Expired message received: {message.Id}")
                return Error "Message has expired"
            | _ -> ()
            
            // Check for duplicates
            if messages.ContainsKey(message.Id) then
                logger.LogWarning($"Duplicate message received: {message.Id}")
                return Error "Duplicate message"
            
            // Check inbox capacity
            if messages.Count >= maxMessages then
                logger.LogWarning($"Inbox capacity exceeded for agent: {agentId}")
                return Error "Inbox capacity exceeded"
            
            // Store message
            messages.[message.Id] <- message
            messageQueue.Enqueue(message)
            
            // Update index
            this.UpdateMessageIndex(message)
            
            // Update statistics
            this.UpdateStatistics("TotalMessages", 1L)
            this.UpdateStatistics($"Messages_{message.MessageType}", 1L)
            this.UpdateStatistics($"Priority_{message.Priority}", 1L)
            
            logger.LogInformation($"Message received successfully: {message.Id} for agent: {agentId}")
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Error receiving message: {message.Id} for agent: {agentId}")
            return Error ex.Message
    }
    
    /// Get next message for processing
    member this.GetNextMessageAsync() = task {
        try
            // Get messages sorted by priority
            let sortedMessages = 
                messages.Values
                |> Seq.filter (fun m -> 
                    match m.ExpiresAt with
                    | Some expiry -> expiry > DateTime.UtcNow
                    | None -> true)
                |> Seq.sortByDescending SemanticMessageHelpers.calculatePriorityScore
                |> List.ofSeq
            
            match sortedMessages with
            | message :: _ ->
                // Remove from active messages
                messages.TryRemove(message.Id) |> ignore
                
                logger.LogDebug($"Retrieved next message: {message.Id} for agent: {agentId}")
                return Some message
            
            | [] ->
                return None
                
        with
        | ex ->
            logger.LogError(ex, $"Error getting next message for agent: {agentId}")
            return None
    }
    
    /// Search messages
    member this.SearchMessagesAsync(filter: MessageFilter) = task {
        try
            logger.LogDebug($"Searching messages for agent: {agentId}")
            
            let filteredMessages = 
                messages.Values
                |> Seq.filter (fun message ->
                    // Filter by message type
                    (filter.MessageTypes.IsEmpty || filter.MessageTypes |> List.contains message.MessageType) &&
                    // Filter by priority
                    (filter.Priorities.IsEmpty || filter.Priorities |> List.contains message.Priority) &&
                    // Filter by sender
                    (filter.Senders.IsEmpty || filter.Senders |> List.contains message.SenderId) &&
                    // Filter by date range
                    (filter.DateRange.IsNone || 
                     let (startDate, endDate) = filter.DateRange.Value
                     message.CreatedAt >= startDate && message.CreatedAt <= endDate) &&
                    // Filter by expiration
                    (filter.HasExpired.IsNone ||
                     let hasExpired = message.ExpiresAt.IsSome && message.ExpiresAt.Value <= DateTime.UtcNow
                     filter.HasExpired.Value = hasExpired))
                |> Seq.filter (fun message ->
                    // Filter by keywords
                    if filter.Keywords.IsEmpty then true
                    else
                        let messageText = $"{message.Id} {message.SenderId}"
                        let messageKeywords = SemanticMessageHelpers.extractKeywords messageText
                        filter.Keywords |> List.exists (fun keyword -> 
                            messageKeywords |> List.contains (keyword.ToLower())))
                |> List.ofSeq
            
            logger.LogDebug($"Search returned {filteredMessages.Length} messages for agent: {agentId}")
            return filteredMessages
            
        with
        | ex ->
            logger.LogError(ex, $"Error searching messages for agent: {agentId}")
            return []
    }
    
    /// Get message by ID
    member this.GetMessageAsync(messageId: string) = task {
        match messages.TryGetValue(messageId) with
        | true, message -> return Some message
        | false, _ -> 
            // Check processed messages
            match processedMessages.TryGetValue(messageId) with
            | true, _ -> 
                logger.LogDebug($"Message {messageId} was already processed")
                return None
            | false, _ ->
                logger.LogDebug($"Message {messageId} not found")
                return None
    }
    
    /// Mark message as processed
    member this.MarkMessageProcessedAsync(messageId: string, result: obj option, error: string option) = task {
        try
            let processingResult = {
                MessageId = messageId
                Processed = error.IsNone
                ProcessingTime = TimeSpan.Zero // Would be calculated from processing start
                Result = result
                Error = error
                ProcessedAt = DateTime.UtcNow
            }
            
            processedMessages.[messageId] <- processingResult
            
            // Keep processed history manageable
            while processedMessages.Count > maxProcessedHistory do
                let oldestKey = 
                    processedMessages
                    |> Seq.minBy (fun kvp -> kvp.Value.ProcessedAt)
                    |> fun kvp -> kvp.Key
                processedMessages.TryRemove(oldestKey) |> ignore
            
            // Update statistics
            if error.IsNone then
                this.UpdateStatistics("ProcessedSuccessfully", 1L)
            else
                this.UpdateStatistics("ProcessedWithError", 1L)
            
            logger.LogDebug($"Message marked as processed: {messageId} for agent: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Error marking message as processed: {messageId} for agent: {agentId}")
    }
    
    /// Get inbox statistics
    member this.GetStatistics() =
        let totalMessages = statistics.GetOrAdd("TotalMessages", 0L)
        let unreadMessages = messages.Count
        let processedSuccessfully = statistics.GetOrAdd("ProcessedSuccessfully", 0L)
        let processedWithError = statistics.GetOrAdd("ProcessedWithError", 0L)
        
        let messagesByType = 
            Enum.GetValues<SemanticMessageType>()
            |> Seq.map (fun msgType -> 
                let count = statistics.GetOrAdd($"Messages_{msgType}", 0L) |> int
                (msgType, count))
            |> Map.ofSeq
        
        let messagesByPriority = 
            Enum.GetValues<SemanticPriority>()
            |> Seq.map (fun priority -> 
                let count = statistics.GetOrAdd($"Priority_{priority}", 0L) |> int
                (priority, count))
            |> Map.ofSeq
        
        let (oldestMessage, newestMessage) = 
            if messages.IsEmpty then (None, None)
            else
                let timestamps = messages.Values |> Seq.map (fun m -> m.CreatedAt) |> List.ofSeq
                (Some (List.min timestamps), Some (List.max timestamps))
        
        let expiredMessages = 
            messages.Values
            |> Seq.filter (fun m -> m.ExpiresAt.IsSome && m.ExpiresAt.Value <= DateTime.UtcNow)
            |> Seq.length
        
        let processedToday = 
            let today = DateTime.UtcNow.Date
            processedMessages.Values
            |> Seq.filter (fun p -> p.ProcessedAt.Date = today)
            |> Seq.length
            |> int64
        
        {
            TotalMessages = totalMessages
            UnreadMessages = unreadMessages
            MessagesByType = messagesByType
            MessagesByPriority = messagesByPriority
            AverageProcessingTime = TimeSpan.FromSeconds(1.0) // Would be calculated from actual processing times
            OldestMessage = oldestMessage
            NewestMessage = newestMessage
            ExpiredMessages = expiredMessages
            ProcessedToday = processedToday
        }
    
    /// Update message index for search
    member private this.UpdateMessageIndex(message: SemanticMessage) =
        try
            // Extract searchable text
            let searchableText = $"{message.Id} {message.SenderId}"
            let keywords = SemanticMessageHelpers.extractKeywords searchableText
            
            // Update index
            for keyword in keywords do
                messageIndex.AddOrUpdate(keyword, [message.Id], fun _ existing -> 
                    if existing |> List.contains message.Id then existing 
                    else message.Id :: existing) |> ignore
                    
        with
        | ex ->
            logger.LogWarning(ex, $"Error updating message index for message: {message.Id}")
    
    /// Message processing loop
    member private this.MessageProcessingLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug($"Starting message processing loop for agent: {agentId}")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Process messages if any are available
                    if not messageQueue.IsEmpty then
                        let! nextMessage = this.GetNextMessageAsync()
                        match nextMessage with
                        | Some message ->
                            // Process the message (this would be handled by the agent)
                            logger.LogDebug($"Processing message: {message.Id} for agent: {agentId}")
                            do! this.MarkMessageProcessedAsync(message.Id, Some ("Processed" :> obj), None)
                        | None -> ()
                    
                    // Wait before next processing cycle
                    do! Task.Delay(TimeSpan.FromMilliseconds(100.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, $"Error in message processing loop for agent: {agentId}")
                    do! Task.Delay(TimeSpan.FromSeconds(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Message processing loop cancelled for agent: {agentId}")
        | ex ->
            logger.LogError(ex, $"Message processing loop failed for agent: {agentId}")
    }
    
    /// Cleanup loop for expired messages
    member private this.CleanupLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug($"Starting cleanup loop for agent: {agentId}")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Clean up expired messages
                    let expiredMessages = 
                        messages.Values
                        |> Seq.filter (fun m -> m.ExpiresAt.IsSome && m.ExpiresAt.Value <= DateTime.UtcNow)
                        |> List.ofSeq
                    
                    for expiredMessage in expiredMessages do
                        messages.TryRemove(expiredMessage.Id) |> ignore
                        logger.LogDebug($"Removed expired message: {expiredMessage.Id} for agent: {agentId}")
                    
                    if expiredMessages.Length > 0 then
                        this.UpdateStatistics("ExpiredMessages", int64 expiredMessages.Length)
                        logger.LogInformation($"Cleaned up {expiredMessages.Length} expired messages for agent: {agentId}")
                    
                    // Wait for next cleanup cycle
                    do! Task.Delay(cleanupInterval, cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, $"Error in cleanup loop for agent: {agentId}")
                    do! Task.Delay(cleanupInterval, cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Cleanup loop cancelled for agent: {agentId}")
        | ex ->
            logger.LogError(ex, $"Cleanup loop failed for agent: {agentId}")
    }
    
    /// Update statistics helper
    member private this.UpdateStatistics(key: string, increment: int64) =
        statistics.AddOrUpdate(key, increment, fun _ current -> current + increment) |> ignore
    
    /// Get all messages
    member this.GetAllMessages() =
        messages.Values |> List.ofSeq
    
    /// Get unread message count
    member this.GetUnreadCount() =
        messages.Count
    
    /// Clear all messages
    member this.ClearAllMessages() =
        messages.Clear()
        messageQueue.Clear()
        messageIndex.Clear()
        logger.LogInformation($"Cleared all messages for agent: {agentId}")
    
    /// Get processed messages
    member this.GetProcessedMessages() =
        processedMessages.Values |> List.ofSeq
