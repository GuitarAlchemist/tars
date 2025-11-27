namespace Tars.Kernel

open System
open System.Collections.Concurrent
open System.Threading.Channels
open System.Threading.Tasks
open Serilog
open Tars.Core

type EventBus(logger: ILogger, circuitBreaker: CircuitBreaker, budgetGovernor: BudgetGovernor, router: AgentRouter) =
    let channel = Channel.CreateUnbounded<SemanticMessage<obj>>()

    let subscribers =
        ConcurrentDictionary<string, ConcurrentDictionary<Guid, SemanticMessage<obj> -> Task>>()

    let endpointToString =
        function
        | MessageEndpoint.System -> "System"
        | MessageEndpoint.User -> "User"
        | MessageEndpoint.Agent(AgentId id) -> id.ToString()
        | MessageEndpoint.Alias name -> name

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
                            // 1. Log the message with semantic details
                            let receiverStr =
                                msg.Receiver |> Option.map endpointToString |> Option.defaultValue "ALL"

                            let senderStr = endpointToString msg.Sender

                            logger.Information(
                                "EventBus: [{Performative}] {Id} from {Sender} to {Receiver}",
                                msg.Performative,
                                msg.Id,
                                senderStr,
                                receiverStr
                            )

                            // 2. Check Circuit Breaker for Receiver
                            let canProceed =
                                match msg.Receiver with
                                | Some target ->
                                    let targetStr = endpointToString target

                                    if circuitBreaker.CanRequest(targetStr) then
                                        true
                                    else
                                        logger.Warning(
                                            "EventBus: Circuit Breaker OPEN for {Target}. Message dropped.",
                                            targetStr
                                        )

                                        false
                                | None -> true

                            if canProceed then
                                // 3. Route to specific target if present
                                match msg.Receiver with
                                | Some target ->
                                    let targetStr = endpointToString target

                                    match subscribers.TryGetValue(targetStr) with
                                    | true, handlers ->
                                        for kvp in handlers do
                                            try
                                                do! kvp.Value msg
                                                // If successful, record success for CB
                                                circuitBreaker.RecordSuccess(targetStr)
                                            with ex ->
                                                logger.Error(
                                                    ex,
                                                    "Error handling message {Id} in target {Target}",
                                                    msg.Id,
                                                    targetStr
                                                )
                                                // Record failure for CB
                                                circuitBreaker.RecordFailure(targetStr)
                                    | false, _ ->
                                        logger.Warning(
                                            "EventBus: Message {Id} sent to unknown receiver {Receiver}",
                                            msg.Id,
                                            targetStr
                                        )
                                | None ->
                                    // 4. Broadcast to all subscribers (if Receiver is None)
                                    for topicHandlers in subscribers.Values do
                                        for kvp in topicHandlers do
                                            try
                                                do! kvp.Value msg
                                            with ex ->
                                                logger.Error(ex, "Error broadcasting message {Id}", msg.Id)

                                // 5. Always route to "System.Diagnostics" for observability
                                match subscribers.TryGetValue("System.Diagnostics") with
                                | true, handlers ->
                                    for kvp in handlers do
                                        try
                                            do! kvp.Value msg
                                        with _ ->
                                            ()
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

    new(logger: ILogger) =
        EventBus(logger, CircuitBreaker(5, TimeSpan.FromMinutes(1.0)), BudgetGovernor(100000), AgentRouter())

    interface IEventBus with
        member _.PublishAsync(msg) =
            task {
                // Check Budget before publishing
                let (CorrelationId cid) = msg.CorrelationId

                // 1. Check Budget
                let budgetOk = budgetGovernor.CanSpend(cid, 1<token>)

                // 2. Check Semantic Constraints
                let constraintsOk =
                    match msg.Constraints.MaxTokens with
                    | Some limit ->
                        // Simplified check: if limit is very small, maybe reject?
                        // For now, we just pass this through, as token usage is checked at consumption time
                        true
                    | None -> true

                if budgetOk && constraintsOk then
                    // Resolve Alias if present
                    let resolvedMsg =
                        match msg.Receiver with
                        | Some(MessageEndpoint.Alias name) ->
                            match router.Resolve(name) with
                            | Some agentId ->
                                { msg with
                                    Receiver = Some(MessageEndpoint.Agent agentId) }
                            | None ->
                                logger.Warning("EventBus: Could not resolve alias {Alias}", name)
                                msg // Or drop? For now, keep as is, it will fail in loop
                        | _ -> msg

                    do! channel.Writer.WriteAsync(resolvedMsg)
                else
                    if not budgetOk then
                        logger.Warning(
                            "EventBus: Budget exceeded for CorrelationId {CorrelationId}. Message dropped.",
                            msg.CorrelationId
                        )

                    if not constraintsOk then
                        logger.Warning(
                            "EventBus: Semantic Constraints violated for Message {Id}. Message dropped.",
                            msg.Id
                        )
            }

        member _.Subscribe(topic, handler) =
            let handlers = subscribers.GetOrAdd(topic, fun _ -> ConcurrentDictionary())
            let handlerId = Guid.NewGuid()
            handlers.TryAdd(handlerId, handler) |> ignore

            { new IDisposable with
                member _.Dispose() = handlers.TryRemove(handlerId) |> ignore }
