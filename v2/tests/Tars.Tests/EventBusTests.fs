namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Kernel
open Serilog

module EventBusTests =

    let createNullLogger () =
        LoggerConfiguration().CreateLogger() :> ILogger

    [<Fact>]
    let ``EventBus routes message to subscriber`` () =
        task {
            let logger = createNullLogger ()
            let bus = EventBus(logger) :> IEventBus
            let agentId = "AgentA"
            let tcs = TaskCompletionSource<SemanticMessage<obj>>()

            use sub =
                bus.Subscribe(
                    agentId,
                    fun msg ->
                        tcs.SetResult(msg)
                        Task.CompletedTask
                )

            let msg =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.User
                  Receiver = Some(MessageEndpoint.Alias agentId)
                  Performative = Performative.Request
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Hello" :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            do! bus.PublishAsync(msg)

            let! received = tcs.Task
            Assert.Equal(msg.Id, received.Id)
            Assert.Equal(Performative.Request, received.Performative)
        }

    [<Fact>]
    let ``EventBus broadcasts message when Receiver is None`` () =
        task {
            let logger = createNullLogger ()
            let bus = EventBus(logger) :> IEventBus
            let tcs1 = TaskCompletionSource<SemanticMessage<obj>>()
            let tcs2 = TaskCompletionSource<SemanticMessage<obj>>()

            use sub1 =
                bus.Subscribe(
                    "Topic1",
                    fun msg ->
                        tcs1.SetResult(msg)
                        Task.CompletedTask
                )

            use sub2 =
                bus.Subscribe(
                    "Topic2",
                    fun msg ->
                        tcs2.SetResult(msg)
                        Task.CompletedTask
                )

            let msg =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.User
                  Receiver = None // Broadcast
                  Performative = Performative.Event
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Broadcast" :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            do! bus.PublishAsync(msg)

            let! received1 = tcs1.Task
            let! received2 = tcs2.Task

            Assert.Equal(msg.Id, received1.Id)
            Assert.Equal(msg.Id, received2.Id)
        }
