namespace TarsEngine.DSL

open System
open System.Collections.Generic
open System.Threading
open Ast

/// Module containing the agent communication framework for the TARS DSL
module AgentCommunication =
    /// Message priority
    type MessagePriority =
        | Low
        | Normal
        | High
        | Critical
    
    /// Message status
    type MessageStatus =
        | Pending
        | Delivered
        | Read
        | Processed
        | Failed
    
    /// Message type
    type MessageType =
        | Command
        | Query
        | Response
        | Notification
        | Error
    
    /// Message
    type Message = {
        Id: Guid
        Sender: string
        Receiver: string
        Type: MessageType
        Priority: MessagePriority
        Content: PropertyValue
        Status: MessageStatus
        Timestamp: DateTime
        CorrelationId: Guid option
        Metadata: Map<string, PropertyValue>
    }
    
    /// Create a new message
    let createMessage sender receiver messageType priority content correlationId =
        {
            Id = Guid.NewGuid()
            Sender = sender
            Receiver = receiver
            Type = messageType
            Priority = priority
            Content = content
            Status = MessageStatus.Pending
            Timestamp = DateTime.UtcNow
            CorrelationId = correlationId
            Metadata = Map.empty
        }
    
    /// Message queue
    type MessageQueue() =
        let messages = Dictionary<string, Queue<Message>>()
        let mutable subscribers = Map.empty<string, (Message -> unit) list>
        let lockObj = Object()
        
        /// Send a message to an agent
        member this.SendMessage(message: Message) =
            lock lockObj (fun () ->
                // Get or create the queue for the receiver
                if not (messages.ContainsKey(message.Receiver)) then
                    messages.[message.Receiver] <- Queue<Message>()
                
                // Add the message to the queue
                messages.[message.Receiver].Enqueue(message)
                
                // Notify subscribers
                match subscribers.TryFind(message.Receiver) with
                | Some(handlers) ->
                    handlers |> List.iter (fun handler -> handler message)
                | None -> ()
            )
        
        /// Receive messages for an agent
        member this.ReceiveMessages(agentName: string, count: int) =
            lock lockObj (fun () ->
                // Get the queue for the agent
                if not (messages.ContainsKey(agentName)) then
                    messages.[agentName] <- Queue<Message>()
                
                // Get the messages
                let queue = messages.[agentName]
                let result = List<Message>()
                
                let mutable i = 0
                while i < count && queue.Count > 0 do
                    let message = queue.Dequeue()
                    result.Add({ message with Status = MessageStatus.Delivered })
                    i <- i + 1
                
                result |> Seq.toList
            )
        
        /// Subscribe to messages for an agent
        member this.Subscribe(agentName: string, handler: Message -> unit) =
            lock lockObj (fun () ->
                // Get or create the subscriber list for the agent
                let handlers = 
                    match subscribers.TryFind(agentName) with
                    | Some(handlers) -> handlers
                    | None -> []
                
                // Add the handler to the list
                subscribers <- subscribers.Add(agentName, handler :: handlers)
            )
        
        /// Unsubscribe from messages for an agent
        member this.Unsubscribe(agentName: string, handler: Message -> unit) =
            lock lockObj (fun () ->
                // Get the subscriber list for the agent
                match subscribers.TryFind(agentName) with
                | Some(handlers) ->
                    // Remove the handler from the list
                    let newHandlers = handlers |> List.filter (fun h -> h <> handler)
                    subscribers <- subscribers.Add(agentName, newHandlers)
                | None -> ()
            )
    
    /// Global message queue
    let messageQueue = MessageQueue()
    
    /// Agent communication channel
    type AgentChannel(agentName: string) =
        let mutable isRunning = false
        let mutable messageHandler: (Message -> unit) option = None
        let cancellationTokenSource = new CancellationTokenSource()
        
        /// Start the channel
        member this.Start(handler: Message -> unit) =
            if not isRunning then
                isRunning <- true
                messageHandler <- Some handler
                
                // Subscribe to messages
                messageQueue.Subscribe(agentName, handler)
                
                // Start the message processing loop
                let token = cancellationTokenSource.Token
                Task.Run(fun () ->
                    while not token.IsCancellationRequested do
                        // Process messages
                        let messages = messageQueue.ReceiveMessages(agentName, 10)
                        
                        // Handle messages
                        messages |> List.iter (fun message ->
                            match messageHandler with
                            | Some handler -> handler message
                            | None -> ()
                        )
                        
                        // Sleep for a short time
                        Thread.Sleep(100)
                )
        
        /// Stop the channel
        member this.Stop() =
            if isRunning then
                isRunning <- false
                
                // Unsubscribe from messages
                match messageHandler with
                | Some handler -> messageQueue.Unsubscribe(agentName, handler)
                | None -> ()
                
                // Cancel the message processing loop
                cancellationTokenSource.Cancel()
        
        /// Send a message
        member this.SendMessage(receiver: string, messageType: MessageType, priority: MessagePriority, content: PropertyValue, ?correlationId: Guid) =
            let message = createMessage agentName receiver messageType priority content (Some correlationId)
            messageQueue.SendMessage(message)
        
        /// Send a command
        member this.SendCommand(receiver: string, command: string, parameters: Map<string, PropertyValue>, ?priority: MessagePriority) =
            let content = ObjectValue(Map.empty
                .Add("command", StringValue(command))
                .Add("parameters", ObjectValue(parameters)))
            
            let priority = defaultArg priority MessagePriority.Normal
            
            this.SendMessage(receiver, MessageType.Command, priority, content)
        
        /// Send a query
        member this.SendQuery(receiver: string, query: string, parameters: Map<string, PropertyValue>, ?priority: MessagePriority) =
            let content = ObjectValue(Map.empty
                .Add("query", StringValue(query))
                .Add("parameters", ObjectValue(parameters)))
            
            let priority = defaultArg priority MessagePriority.Normal
            
            let correlationId = Guid.NewGuid()
            this.SendMessage(receiver, MessageType.Query, priority, content, correlationId)
            correlationId
        
        /// Send a response
        member this.SendResponse(receiver: string, correlationId: Guid, result: PropertyValue, ?priority: MessagePriority) =
            let content = ObjectValue(Map.empty
                .Add("result", result))
            
            let priority = defaultArg priority MessagePriority.Normal
            
            this.SendMessage(receiver, MessageType.Response, priority, content, correlationId)
        
        /// Send a notification
        member this.SendNotification(receiver: string, notificationType: string, data: PropertyValue, ?priority: MessagePriority) =
            let content = ObjectValue(Map.empty
                .Add("type", StringValue(notificationType))
                .Add("data", data))
            
            let priority = defaultArg priority MessagePriority.Normal
            
            this.SendMessage(receiver, MessageType.Notification, priority, content)
        
        /// Send an error
        member this.SendError(receiver: string, errorCode: string, errorMessage: string, ?correlationId: Guid, ?priority: MessagePriority) =
            let content = ObjectValue(Map.empty
                .Add("code", StringValue(errorCode))
                .Add("message", StringValue(errorMessage)))
            
            let priority = defaultArg priority MessagePriority.High
            
            this.SendMessage(receiver, MessageType.Error, priority, content, correlationId)
        
        interface IDisposable with
            member this.Dispose() =
                this.Stop()
                cancellationTokenSource.Dispose()
    
    /// Agent channels
    let mutable private channels = Map.empty<string, AgentChannel>
    
    /// Get or create a channel for an agent
    let getChannel agentName =
        match channels.TryFind(agentName) with
        | Some channel -> channel
        | None ->
            let channel = new AgentChannel(agentName)
            channels <- channels.Add(agentName, channel)
            channel
