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
                new EventBus(logger, CircuitBreaker(5, TimeSpan.FromMinutes(1.0)), BudgetGovernor(100000), router)

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
                  Constraints = SemanticConstraints.Default
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
