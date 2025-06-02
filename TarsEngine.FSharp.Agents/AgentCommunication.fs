namespace TarsEngine.FSharp.Agents

open System
open System.Threading
open System.Threading.Channels
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.Control
open AgentTypes

/// Agent communication system using .NET Channels
module AgentCommunication =
    
    /// Channel-based message bus for inter-agent communication
    type MessageBus(logger: ILogger<MessageBus>) =
        
        let agents = System.Collections.Concurrent.ConcurrentDictionary<AgentId, Channel<AgentMessage>>()
        let broadcastChannel = Channel.CreateUnbounded<AgentMessage>()
        let messageHistory = System.Collections.Concurrent.ConcurrentQueue<AgentMessage>()
        
        /// Register an agent with the message bus
        member this.RegisterAgent(agentId: AgentId) =
            let channel = Channel.CreateUnbounded<AgentMessage>()
            agents.TryAdd(agentId, channel) |> ignore
            logger.LogInformation("Agent {AgentId} registered with message bus", agentId)
            channel
        
        /// Unregister an agent from the message bus
        member this.UnregisterAgent(agentId: AgentId) =
            match agents.TryRemove(agentId) with
            | true, channel ->
                channel.Writer.Complete()
                logger.LogInformation("Agent {AgentId} unregistered from message bus", agentId)
            | false, _ ->
                logger.LogWarning("Agent {AgentId} was not registered", agentId)
        
        /// Send a message to a specific agent
        member this.SendMessageAsync(message: AgentMessage) =
            task {
                try
                    messageHistory.Enqueue(message)
                    
                    match message.ToAgent with
                    | Some targetAgent ->
                        // Direct message to specific agent
                        match agents.TryGetValue(targetAgent) with
                        | true, channel ->
                            do! channel.Writer.WriteAsync(message)
                            logger.LogDebug("Message sent from {FromAgent} to {ToAgent}", 
                                           message.FromAgent, targetAgent)
                        | false, _ ->
                            logger.LogWarning("Target agent {AgentId} not found", targetAgent)
                    | None ->
                        // Broadcast message to all agents
                        do! broadcastChannel.Writer.WriteAsync(message)
                        logger.LogDebug("Broadcast message sent from {FromAgent}", message.FromAgent)
                        
                with
                | ex ->
                    logger.LogError(ex, "Error sending message from {FromAgent}", message.FromAgent)
            }
        
        /// Get message stream for an agent
        member this.GetMessageStream(agentId: AgentId) =
            taskSeq {
                match agents.TryGetValue(agentId) with
                | true, channel ->
                    // Combine direct messages and broadcast messages
                    let directMessages = channel.Reader.ReadAllAsync()
                    let broadcastMessages = broadcastChannel.Reader.ReadAllAsync()
                    
                    // Merge both streams
                    let mutable directEnumerator = directMessages.GetAsyncEnumerator()
                    let mutable broadcastEnumerator = broadcastMessages.GetAsyncEnumerator()
                    
                    let mutable directHasNext = true
                    let mutable broadcastHasNext = true
                    
                    while directHasNext || broadcastHasNext do
                        // Try to get next direct message
                        if directHasNext then
                            try
                                let! hasNext = directEnumerator.MoveNextAsync()
                                if hasNext then
                                    yield directEnumerator.Current
                                else
                                    directHasNext <- false
                            with
                            | _ -> directHasNext <- false
                        
                        // Try to get next broadcast message (filter out own messages)
                        if broadcastHasNext then
                            try
                                let! hasNext = broadcastEnumerator.MoveNextAsync()
                                if hasNext then
                                    let msg = broadcastEnumerator.Current
                                    if msg.FromAgent <> agentId then
                                        yield msg
                                else
                                    broadcastHasNext <- false
                            with
                            | _ -> broadcastHasNext <- false
                        
                        // Small delay to prevent tight loop
                        do! Task.Delay(10)
                        
                | false, _ ->
                    logger.LogError("Agent {AgentId} not registered for message stream", agentId)
            }
        
        /// Get message history
        member this.GetMessageHistory(count: int option) =
            let messages = messageHistory.ToArray()
            match count with
            | Some n -> messages |> Array.rev |> Array.take (min n messages.Length)
            | None -> messages |> Array.rev
        
        /// Get active agents
        member this.GetActiveAgents() =
            agents.Keys |> Seq.toList
    
    /// Agent communication interface
    type IAgentCommunication =
        abstract member SendMessageAsync: AgentMessage -> Task
        abstract member SendRequestAsync: AgentId * string * obj -> Task<obj option>
        abstract member BroadcastAsync: string * obj -> Task
        abstract member GetMessageStream: unit -> seq<AgentMessage>
        abstract member ReplyToAsync: AgentMessage * obj -> Task
    
    /// Agent communication implementation
    type AgentCommunication(agentId: AgentId, messageBus: MessageBus, logger: ILogger) =
        
        let messageStream = messageBus.GetMessageStream(agentId)
        let pendingRequests = System.Collections.Concurrent.ConcurrentDictionary<Guid, TaskCompletionSource<obj>>()
        
        /// Send a message to another agent
        member this.SendMessageAsync(message: AgentMessage) =
            messageBus.SendMessageAsync(message)
        
        /// Send a request and wait for response
        member this.SendRequestAsync(targetAgent: AgentId, messageType: string, content: obj) =
            task {
                let correlationId = Guid.NewGuid()
                let tcs = TaskCompletionSource<obj>()
                pendingRequests.TryAdd(correlationId, tcs) |> ignore
                
                let message = {
                    Id = Guid.NewGuid()
                    FromAgent = agentId
                    ToAgent = Some targetAgent
                    MessageType = messageType
                    Content = content
                    Priority = MessagePriority.Normal
                    Timestamp = DateTime.UtcNow
                    CorrelationId = Some correlationId
                    ReplyTo = Some agentId
                }
                
                do! this.SendMessageAsync(message)
                
                // Wait for response with timeout
                use cts = new CancellationTokenSource(TimeSpan.FromSeconds(30.0))
                try
                    let! result = tcs.Task.WaitAsync(cts.Token)
                    return Some result
                with
                | :? OperationCanceledException ->
                    pendingRequests.TryRemove(correlationId) |> ignore
                    logger.LogWarning("Request to {TargetAgent} timed out", targetAgent)
                    return None
            }
        
        /// Broadcast a message to all agents
        member this.BroadcastAsync(messageType: string, content: obj) =
            let message = {
                Id = Guid.NewGuid()
                FromAgent = agentId
                ToAgent = None
                MessageType = messageType
                Content = content
                Priority = MessagePriority.Normal
                Timestamp = DateTime.UtcNow
                CorrelationId = None
                ReplyTo = None
            }
            this.SendMessageAsync(message)
        
        /// Get message stream for this agent
        member this.GetMessageStream() = Seq.empty<AgentMessage> // Placeholder implementation
        
        /// Reply to a message
        member this.ReplyToAsync(originalMessage: AgentMessage, response: obj) =
            match originalMessage.ReplyTo, originalMessage.CorrelationId with
            | Some replyTo, Some correlationId ->
                let replyMessage = {
                    Id = Guid.NewGuid()
                    FromAgent = agentId
                    ToAgent = Some replyTo
                    MessageType = "Response"
                    Content = response
                    Priority = originalMessage.Priority
                    Timestamp = DateTime.UtcNow
                    CorrelationId = Some correlationId
                    ReplyTo = None
                }
                this.SendMessageAsync(replyMessage)
            | _ ->
                logger.LogWarning("Cannot reply to message without ReplyTo or CorrelationId")
                task { return () }
        
        /// Process incoming messages (internal)
        member internal this.ProcessIncomingMessages() =
            taskSeq {
                for message in messageStream do
                    // Handle responses to pending requests
                    match message.CorrelationId with
                    | Some correlationId when message.MessageType = "Response" ->
                        match pendingRequests.TryRemove(correlationId) with
                        | true, tcs -> tcs.SetResult(message.Content)
                        | false, _ -> ()
                    | _ -> ()
                    
                    yield message
            }
        
        interface IAgentCommunication with
            member this.SendMessageAsync(message) = this.SendMessageAsync(message)
            member this.SendRequestAsync(targetAgent, messageType, content) =
                this.SendRequestAsync(targetAgent, messageType, content)
            member this.BroadcastAsync(messageType, content) = this.BroadcastAsync(messageType, content)
            member this.GetMessageStream() = this.GetMessageStream()
            member this.ReplyToAsync(message, response) = this.ReplyToAsync(message, response)
    
    /// Create a message
    let createMessage fromAgent toAgent messageType content priority =
        {
            Id = Guid.NewGuid()
            FromAgent = fromAgent
            ToAgent = toAgent
            MessageType = messageType
            Content = content
            Priority = priority
            Timestamp = DateTime.UtcNow
            CorrelationId = None
            ReplyTo = None
        }
    
    /// Create a request message
    let createRequest fromAgent toAgent messageType content =
        {
            Id = Guid.NewGuid()
            FromAgent = fromAgent
            ToAgent = Some toAgent
            MessageType = messageType
            Content = content
            Priority = MessagePriority.Normal
            Timestamp = DateTime.UtcNow
            CorrelationId = Some (Guid.NewGuid())
            ReplyTo = Some fromAgent
        }
