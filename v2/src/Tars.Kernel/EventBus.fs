namespace Tars.Kernel

open System
open System.Collections.Concurrent
open System.Threading.Channels
open System.Threading.Tasks
open Serilog

type EventBus(logger: ILogger) =
    let channel = Channel.CreateUnbounded<IMessage>()
    let subscribers = ConcurrentDictionary<string, ConcurrentDictionary<Guid, IMessage -> Task>>()

    // Background loop to process messages
    let processLoop () : Task =
        task {
            logger.Debug("EventBus Loop Started")
            try
                while true do
                    let! msg = channel.Reader.ReadAsync().AsTask()
                    try
                        logger.Debug("Processing message {Id}", msg.Id)
                        // 1. Log the message
                        // logger.Information("EventBus: Processing message {Id} from {Source}", msg.Id, msg.Source)

                        // 2. Route to specific target if present
                        match msg.Target with
                        | Some target ->
                            logger.Debug("Routing to target {Target}", target)
                            if subscribers.ContainsKey(target) then
                                let handlers = subscribers.[target]
                                logger.Debug("Found {Count} subscribers for target {Target}", handlers.Count, target)
                                for handler in handlers.Values do
                                    try
                                        logger.Debug("Invoking handler for message {Id}", msg.Id)
                                        do! handler msg
                                        logger.Debug("Handler invoked for message {Id}", msg.Id)
                                    with ex ->
                                        logger.Error(ex, "Error in handler for message {Id}", msg.Id)
                            else
                                logger.Debug("No subscribers found for target {Target}", target)
                        | None ->
                            // 3. Broadcast to all
                            logger.Debug("Broadcasting message {Id} to all subscribers", msg.Id)
                            for topicHandlers in subscribers.Values do
                                for handler in topicHandlers.Values do
                                    try
                                        do! handler msg
                                    with ex ->
                                        logger.Error(ex, "Error in broadcast handler for message {Id}", msg.Id)

                        // 4. Also route to "System.Diagnostics" if anyone is listening
                        if subscribers.ContainsKey("System.Diagnostics") then
                            for handler in subscribers.["System.Diagnostics"].Values do
                                do! handler msg

                    with ex ->
                        logger.Error(ex, "Error processing message {Id}", msg.Id)
            with 
            | :? ChannelClosedException -> 
                logger.Debug("EventBus Loop Channel Closed")
            | ex ->
                logger.Fatal(ex, "EventBus Loop Fatal Error")
        }
        :> Task

    // Start the processing loop
    let _ = Task.Run(fun () -> processLoop ())

    interface IEventBus with
        member _.PublishAsync(msg) =
            task {
                logger.Debug("Publishing message {Id} to channel", msg.Id)
                do! channel.Writer.WriteAsync(msg)
                logger.Debug("Published message {Id} to channel", msg.Id)
            }

        member _.Subscribe(topic, handler) =
            let topicHandlers = subscribers.GetOrAdd(topic, fun _ -> ConcurrentDictionary<Guid, IMessage -> Task>())
            let subscriptionId = Guid.NewGuid()
            if topicHandlers.TryAdd(subscriptionId, handler) then
                { new IDisposable with
                    member _.Dispose() = 
                        topicHandlers.TryRemove(subscriptionId) |> ignore }
            else
                // Should not happen with Guid
                { new IDisposable with member _.Dispose() = () }
