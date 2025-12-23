namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Evolution
open Tars.Core

module EvolutionBenchmarkTests =

    type BenchmarkLlm() =
        let mutable callCount = 0
        member _.CallCount = callCount

        interface Tars.Llm.LlmService.ILlmService with
            member _.CompleteAsync(_req) =
                callCount <- callCount + 1

                task {
                    // Return a valid JSON Task for curriculum phase
                    return
                        { Text =
                            "ACT: INFORM: {\"tasks\":[{\"goal\":\"Task "
                            + string callCount
                            + "\",\"constraints\":[],\"validation_criteria\":\"ok\"}]}"
                          FinishReason = Some "stop"
                          Usage =
                            Some
                                { PromptTokens = 100
                                  CompletionTokens = 50
                                  TotalTokens = 150 }
                          Raw = None }
                }

            member _.CompleteStreamAsync(_req, _onToken) =
                task {
                    return
                        { Text = ""
                          FinishReason = None
                          Usage = None
                          Raw = None }
                }

            member _.EmbedAsync(_text) = task { return [| 0.1f; 0.2f; 0.3f |] }

    [<Fact>]
    let ``Evolution Loop - 5 Generation Stability Benchmark`` () =
        task {
            let llm = BenchmarkLlm()
            let registry = Tars.Kernel.AgentRegistry()
            let curriculumId = AgentId(Guid.NewGuid())
            let executorId = AgentId(Guid.NewGuid())

            let agent1 =
                Tars.Kernel.AgentFactory.create
                    (let (AgentId id) = curriculumId in id)
                    "Curriculum"
                    "1.0.0"
                    "test"
                    "System"
                    []
                    []

            let agent2 =
                Tars.Kernel.AgentFactory.create
                    (let (AgentId id) = executorId in id)
                    "Executor"
                    "1.0.0"
                    "test"
                    "System"
                    []
                    []

            registry.Register(agent1)
            registry.Register(agent2)

            // Budget Governor with 5000 tokens limit
            let budget =
                BudgetGovernor(
                    { Budget.Infinite with
                        MaxTokens = Some 5000<token> }
                )

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = llm
                  VectorStore =
                    { new IVectorStore with
                        member _.SaveAsync(_, _, _, _) = Task.CompletedTask
                        member _.SearchAsync(_, _, _) = Task.FromResult([]) }
                  SemanticMemory = None
                  Epistemic = None
                  PreLlm = None
                  Budget = Some budget
                  OutputGuard = None
                  KnowledgeBase = None
                  KnowledgeGraph = None
                  MemoryBuffer = None
                  EpisodeService = None
                  Logger = fun _ -> ()
                  Verbose = false
                  ShowSemanticMessage = fun _ _ -> () }

            let mutable state: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = curriculumId
                  ExecutorAgentId = executorId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            // Run 5 generations
            for i in 1..5 do
                let! nextState = Engine.step ctx state
                state <- nextState

            Assert.Equal(5, state.Generation)
            Assert.True(llm.CallCount >= 5, "Should have called LLM at least once per generation")
            Assert.True(budget.Consumed.Tokens > 0<token>, "Tokens should have been consumed")
        }
