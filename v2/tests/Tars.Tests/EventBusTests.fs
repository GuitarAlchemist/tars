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
                  Intent = None
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
                  Intent = None
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

    [<Fact>]
    let ``EventBus routes message via Intent when Receiver is None`` () =
        task {
            let logger = createNullLogger ()
            let router = AgentRouter()
            let bus = EventBus(logger, Some router) :> IEventBus
            let agentId = Guid.NewGuid()
            let tcs = TaskCompletionSource<SemanticMessage<obj>>()

            // Register Intent Route
            router.SetRoute("Intent:Coding", Pinned(AgentId agentId))

            use sub =
                bus.Subscribe(
                    agentId.ToString(),
                    fun msg ->
                        tcs.SetResult(msg)
                        Task.CompletedTask
                )

            let msg =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.User
                  Receiver = None // Should fallback to Intent
                  Performative = Performative.Request
                  Intent = Some AgentIntent.Coding
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Code this" :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            do! bus.PublishAsync(msg)

            let! received = tcs.Task
            Assert.Equal(msg.Id, received.Id)
            Assert.Equal(Some(MessageEndpoint.Agent(AgentId agentId)), received.Receiver)
        }
