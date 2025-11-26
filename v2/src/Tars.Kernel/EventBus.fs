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
            let mutable running = true
            logger.Debug("EventBus: Loop started")

            while running do
                let! canRead = channel.Reader.WaitToReadAsync().AsTask()

                if not canRead then
                    running <- false
                else
                    let mutable hasMsg, msg = channel.Reader.TryRead()

                    while hasMsg do
                        try
                            // 1. Log the message
                            logger.Information("EventBus: Processing message {Id} from {Source}", msg.Id, msg.Source)

                            // 2. Route to specific target if present
                            match msg.Target with
                            | Some target ->
                                match subscribers.TryGetValue(target) with
                                | true, handlers ->
                                    for kvp in handlers do
                                        try
                                            do! kvp.Value msg
                                        with ex ->
                                            logger.Error(
                                                ex,
                                                "Error handling message {Id} in target {Target}",
                                                msg.Id,
                                                target
                                            )
                                | false, _ -> ()
                            | None ->
                                // 3. Broadcast to all subscribers
                                for topicHandlers in subscribers.Values do
                                    for kvp in topicHandlers do
                                        try
                                            do! kvp.Value msg
                                        with ex ->
                                            logger.Error(ex, "Error broadcasting message {Id}", msg.Id)

                            // 4. Also route to "System.Diagnostics" if anyone is listening
                            match subscribers.TryGetValue("System.Diagnostics") with
                            | true, handlers ->
                                for kvp in handlers do
                                    do! kvp.Value msg
                            | false, _ -> ()

                        with ex ->
                            logger.Error(ex, "Error processing message {Id}", msg.Id)

                        // Try read next message
                        let next = channel.Reader.TryRead()
                        hasMsg <- fst next

                        if hasMsg then
                            msg <- snd next
        }
        :> Task

    // Start the processing loop
    let _ = Task.Run(fun () -> processLoop ())

    interface IEventBus with
        member _.PublishAsync(msg) =
            task { do! channel.Writer.WriteAsync(msg) }

        member _.Subscribe(topic, handler) =
            let handlers = subscribers.GetOrAdd(topic, fun _ -> ConcurrentDictionary())
            let handlerId = Guid.NewGuid()
            handlers.TryAdd(handlerId, handler) |> ignore

            { new IDisposable with
                member _.Dispose() = 
                    handlers.TryRemove(handlerId) |> ignore }
