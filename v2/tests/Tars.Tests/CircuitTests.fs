namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Kernel
open Tars.Core
open Serilog

module CircuitTests =

    let createLogger () = Serilog.Core.Logger.None

    [<Fact>]
    let ``EventBus applies backpressure when full`` () =
        task {
            let logger = createLogger ()

            let bus =
                EventBus(
                    logger,
                    CircuitBreaker(5, TimeSpan.FromMinutes(1.0)),
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 100000<token> }
                    ),
                    AgentRouter(),
                    1
                ) // Capacity 1

            let msg1 =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.System
                  Receiver = None
                  Performative = Performative.Inform
                  Intent = None
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Message 1" :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            let msg2 =
                { msg1 with
                    Id = Guid.NewGuid()
                    Content = "Message 2" :> obj }

            // Start a reader to consume messages so the bus doesn't block indefinitely
            let tcs = TaskCompletionSource<bool>()

            let sub =
                (bus :> IEventBus)
                    .Subscribe(
                        "System.Diagnostics",
                        fun m ->
                            task {
                                do! Task.Delay(50) // Simulate processing time

                                if m.Content :?> string = "Message 2" then
                                    tcs.SetResult(true)
                            }
                    )

            // Write first message (should succeed immediately or queue)
            do! (bus :> IEventBus).PublishAsync(msg1)

            // Write second message (should block until first is consumed)
            do! (bus :> IEventBus).PublishAsync(msg2)

            let! result = tcs.Task
            Assert.True(result)
        }

    [<Fact>]
    let ``BufferAgent batches items`` () =
        task {
            let tcs = TaskCompletionSource<int>()

            let onFlush (items: int list) =
                task { tcs.SetResult(items.Length) } :> Task

            let buffer = BufferAgent<int>(3, TimeSpan.FromSeconds(10.0), onFlush)

            buffer.Accumulate(1)
            buffer.Accumulate(2)
            buffer.Accumulate(3) // Should trigger flush

            let! count = tcs.Task
            Assert.Equal(3, count)
        }

    [<Fact>]
    let ``Gate waits for condition`` () =
        task {
            let mutable conditionMet = false
            let gate = Gate(fun () -> Task.FromResult(conditionMet))

            let waitTask = gate.WaitForOpen()

            // Should not complete yet (give it a small delay to check)
            do! Task.Delay(50)
            Assert.False(waitTask.IsCompleted)

            conditionMet <- true

            // Should complete now
            do! waitTask
            Assert.True(waitTask.IsCompletedSuccessfully)
        }
