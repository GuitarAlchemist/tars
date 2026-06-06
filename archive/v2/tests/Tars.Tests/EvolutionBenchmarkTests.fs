namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Evolution
open Tars.Core
open Tars.Llm.Routing

module EvolutionBenchmarkTests =

    type BenchmarkLlm(responseText: string) =
        interface Tars.Llm.ILlmService with
            member _.CompleteAsync(_req) =
                task {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.CompleteStreamAsync(_req, _onToken) =
                task {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.EmbedAsync(_text) = task { return [| 0.1f; 0.2f; 0.3f |] }

            member _.RouteAsync _ =
                task {
                    return
                        { Backend = Tars.Llm.LlmBackend.Ollama "mock"
                          Endpoint = Uri "http://localhost:11434"
                          ApiKey = None }
                }

    type BenchmarkRegistry(agents: Agent list) =
        interface IAgentRegistry with
            member _.GetAgent(id) =
                async { return agents |> List.tryFind (fun a -> a.Id = id) }

            member _.FindAgents(_) = async { return [] }
            member _.GetAllAgents() = async { return agents }

    [<Fact>]
    let ``Evolution handles benchmark evaluation`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let curriculumAgentId = AgentId(Guid.NewGuid())
            let executorAgentId = AgentId(Guid.NewGuid())

            let llmJson =
                """{"tasks":[{"goal":"Benchmark Task 1","constraints":[],"validation_criteria":"check"},{"goal":"Benchmark Task 2","constraints":[],"validation_criteria":"check"}]}"""

            let llm = BenchmarkLlm(llmJson)

            let agent1 =
                Tars.Kernel.AgentFactory.create
                    (let (AgentId id) = curriculumAgentId in id)
                    "Curriculum"
                    "1.0.0"
                    "test"
                    "System"
                    []
                    []

            let agent2 =
                Tars.Kernel.AgentFactory.create
                    (let (AgentId id) = executorAgentId in id)
                    "Executor"
                    "1.0.0"
                    "test"
                    "System"
                    []
                    []

            let registry = BenchmarkRegistry([ agent1; agent2 ])

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
                  Budget = None
                  OutputGuard = None
                  KnowledgeBase = None
                  KnowledgeGraph = None
                  MemoryBuffer = None
                  EpisodeService = None
                  Ledger = None
                  Evaluator = None
                  RunId = None
                  Logger = fun msg -> printfn $"LOG: %s{msg}"
                  Verbose = true
                  ShowSemanticMessage = fun _ _ -> ()
                  Focus = None
                  ToolRegistry = None
                  ResearchEnhanced = false
                  SelfImprovement = false }

            let mutable state: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = curriculumAgentId
                  ExecutorAgentId = executorAgentId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            // Step 1: Curriculum Phase
            let! stateAfterCurriculum = Engine.step ctx state
            Assert.NotEmpty(stateAfterCurriculum.TaskQueue)

            // Step 2: Execution Phase
            let! stateAfterExecution = Engine.step ctx stateAfterCurriculum
            Assert.NotEmpty(stateAfterExecution.CompletedTasks)
            Assert.True(stateAfterExecution.CompletedTasks.Head.Success)
        }
