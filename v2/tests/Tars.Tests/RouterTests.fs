namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Kernel
open Serilog

type RouterTests(output: ITestOutputHelper) =

    [<Fact>]
    member _.``Can route message via Alias``() =
        task {
            output.WriteLine("Starting test: Can route message via Alias")

            // Arrange
            let logger = LoggerConfiguration().CreateLogger()
            let router = AgentRouter()

            let eventBus =
                new EventBus(
                    logger,
                    CircuitBreaker(5, TimeSpan.FromMinutes(1.0)),
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 100000<token> }
                    ),
                    router,
                    1000
                )

            let bus = eventBus :> IEventBus

            let agentId1 = Guid.NewGuid()
            let agentId2 = Guid.NewGuid()

            let mutable received1 = false
            let mutable received2 = false

            // Subscribe agents
            let _ = bus.Subscribe(agentId1.ToString(), fun _ -> task { received1 <- true })
            let _ = bus.Subscribe(agentId2.ToString(), fun _ -> task { received2 <- true })

            // 1. Point "Coder" to Agent 1
            router.SetRoute("Coder", Pinned(AgentId agentId1))
            output.WriteLine($"Routed 'Coder' to {agentId1}")

            let msg1 =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.System
                  Receiver = Some(MessageEndpoint.Alias "Coder")
                  Performative = Performative.Request
                  Intent = None
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Task 1" :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            // Act 1
            do! bus.PublishAsync(msg1)
            do! Task.Delay(500) // Allow processing

            // Assert 1
            Assert.True(received1, "Agent 1 should have received the message")
            Assert.False(received2, "Agent 2 should NOT have received the message")

            // Reset
            received1 <- false
            received2 <- false

            // 2. Switch "Coder" to Agent 2
            router.SetRoute("Coder", Pinned(AgentId agentId2))
            output.WriteLine($"Routed 'Coder' to {agentId2}")

            let msg2 =
                { msg1 with
                    Id = Guid.NewGuid()
                    Content = "Task 2" :> obj }

            // Act 2
            do! bus.PublishAsync(msg2)
            do! Task.Delay(500)

            // Assert 2
            Assert.False(received1, "Agent 1 should NOT have received the message")
            Assert.True(received2, "Agent 2 should have received the message")

            output.WriteLine("Routing test passed.")
        }

    [<Fact>]
    member _.``Can route message via Intent``() =
        task {
            output.WriteLine("Starting test: Can route message via Intent")

            // Arrange
            let logger = LoggerConfiguration().CreateLogger()
            let router = AgentRouter()

            let eventBus =
                new EventBus(
                    logger,
                    CircuitBreaker(5, TimeSpan.FromMinutes(1.0)),
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 100000<token> }
                    ),
                    router,
                    1000
                )

            let bus = eventBus :> IEventBus

            let agentId = Guid.NewGuid()
            let mutable received = false

            // Subscribe agent
            let _ = bus.Subscribe(agentId.ToString(), fun _ -> task { received <- true })

            // Point "Intent:Coding" to Agent
            router.SetRoute("Intent:Coding", Pinned(AgentId agentId))
            output.WriteLine($"Routed 'Intent:Coding' to {agentId}")

            let msg =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.System
                  Receiver = None // Broadcast initially
                  Performative = Performative.Request
                  Intent = Some AgentDomain.Coding
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Write code" :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            // Act
            do! bus.PublishAsync(msg)
            do! Task.Delay(500)

            // Assert
            Assert.True(received, "Agent should have received the message via Intent routing")
            output.WriteLine("Intent routing test passed.")
        }
