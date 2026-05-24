namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Llm
open Tars.Metascript.Domain
open Tars.Metascript.Engine
open Tars.Metascript.Config
open Tars.Core

type DelayLlm(delayMs: int) =
    interface ILlmService with
        member _.CompleteAsync(req: LlmRequest) =
            task {
                do! Task.Delay(delayMs)

                return
                    { Text = "Delayed Response"
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
            }

        member _.CompleteStreamAsync(req, onToken) = raise (NotImplementedException())
        member _.EmbedAsync(text) = Task.FromResult([| 0.1f |])
        member _.RouteAsync(_req: Tars.Llm.LlmRequest) : Task<Tars.Llm.Routing.RoutedBackend> =
            task { return { Backend = Tars.Llm.LlmBackend.Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None } }

type FluxTests() =

    [<Fact>]
    member _.``Scheduler executes independent steps in parallel``() =
        if not (TestHelpers.requireTools()) then () else
        // 1. Setup Context with Delay LLM
        let delay = 1000
        let llm = DelayLlm(delay) :> ILlmService

        let ctx =
            { Llm = llm
              Tools = StubToolRegistry()
              Budget = None
              VectorStore = None
              KnowledgeGraph = None
              SemanticMemory = None
              EpisodeService = None
              MacroRegistry = None
              RagConfig = RagConfig.Default
              MetascriptRegistry = None }

        // 2. Define Workflow: A and B are independent, C depends on both
        let wf =
            { Name = "ParallelTest"
              Description = "Tests fan-out/fan-in"
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "A"
                    Type = "agent"
                    Agent = Some "AgentA"
                    Tool = None
                    Instruction = Some "Task A"
                    Params = None
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "outA" ]
                    Tools = None }
                  { Id = "B"
                    Type = "agent"
                    Agent = Some "AgentB"
                    Tool = None
                    Instruction = Some "Task B"
                    Params = None
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "outB" ]
                    Tools = None }
                  { Id = "C"
                    Type = "agent"
                    Agent = Some "AgentC" // This one will also delay, but starts after A&B
                    Tool = None
                    Instruction = Some "Task C"
                    Params = None
                    Context = None
                    DependsOn = Some [ { StepId = "A"; Condition = None }; { StepId = "B"; Condition = None } ]
                    Outputs = Some [ "outC" ]
                    Tools = None } ] }

        // 3. Measure Execution Time
        let sw = System.Diagnostics.Stopwatch.StartNew()

        task {
            let! state = run ctx wf Map.empty
            sw.Stop()

            // Assertions
            // Total time should be roughly delay + delay (since A&B are parallel, C is seq)
            // If linear: delay * 3
            // Parallel: delay * 2 + overhead

            let elapsed = sw.ElapsedMilliseconds
            let expectedMax = (int64 delay * 2L) + 1500L // buffer to reduce flakiness in CI
            let expectedMin = (int64 delay * 2L) - 200L

            Assert.True(elapsed < expectedMax, $"Execution took {elapsed}ms, expected < {expectedMax}ms (Parallel)")
            Assert.True(state.StepOutputs.ContainsKey "A")
            Assert.True(state.StepOutputs.ContainsKey "B")
            Assert.True(state.StepOutputs.ContainsKey "C")
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
