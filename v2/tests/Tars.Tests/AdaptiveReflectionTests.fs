namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Evolution
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Graph
open Tars.Cortex

module AdaptiveReflectionTests =

    let createMockLlm (response: string) =
        { new ILlmService with
            member _.CompleteAsync req =
                task {
                    return
                        { Text = response
                          Usage = None
                          FinishReason = Some "stop"
                          Raw = None }
                }

            member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
            member _.EmbedAsync text = Task.FromResult [| 0.1f |] }

    let createMockRegistry (agent: Agent) =
        { new IAgentRegistry with
            member _.GetAgent id = async { return Some agent }
            member _.FindAgents kind = async { return [ agent ] }
            member _.GetAllAgents() = async { return [ agent ] } }

    let createMockEpistemic (isVerified: bool) (feedback: string) =
        { new IEpistemicGovernor with
            member _.GenerateVariants(desc, count) = Task.FromResult [ "Variant 1" ]

            member _.VerifyGeneralization(desc, sol, vars) =
                Task.FromResult
                    { IsVerified = isVerified
                      Score = if isVerified then 1.0 else 0.0
                      Feedback = feedback
                      FailedVariants = [] }

            member _.ExtractPrinciple(desc, sol) = raise (NotImplementedException())
            member _.SuggestCurriculum(completed, active) = raise (NotImplementedException())
            member _.Verify(stmt) = raise (NotImplementedException())
            member _.GetRelatedCodeContext(query) = Task.FromResult "" }

    let createTestAgent () =
        { Id = AgentId(Guid.NewGuid())
          Name = "TestExecutor"
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "test-model"
          SystemPrompt = "You are a test agent."
          Tools = []
          Capabilities = []
          State = Idle
          Memory = [] }

    [<Fact>]
    let ``Reflection loop stops when Epistemic Governor verifies solution`` () =
        task {
            // Setup
            let agent = createTestAgent ()
            let registry = createMockRegistry agent
            let llm = createMockLlm "Initial Solution" // Agent always returns this
            let epistemic = createMockEpistemic true "Good job" // Verified immediately

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = llm
                  VectorStore = Unchecked.defaultof<_>
                  SemanticMemory = None
                  Epistemic = Some epistemic
                  PreLlm = None
                  Budget = None
                  OutputGuard = None
                  KnowledgeBase = None
                  KnowledgeGraph = None
                  MemoryBuffer = None
                  EpisodeService = None
                  Logger = fun _ -> ()
                  Verbose = false
                  ShowSemanticMessage = fun _ _ -> () }

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Test Task"
                  Constraints = []
                  ValidationCriteria = "None"
                  Timeout = TimeSpan.FromMinutes(1.0)
                  Score = 1.0 }

            let state =
                { Generation = 0
                  CompletedTasks = []
                  TaskQueue = []
                  CurrentTask = Some taskDef
                  ActiveBeliefs = []
                  CurriculumAgentId = AgentId(Guid.NewGuid())
                  ExecutorAgentId = agent.Id }

            // Act
            let! newState = Engine.step ctx state

            // Assert
            // If verified immediately, it should succeed.
            // We can check the trace in the completed task to see if "VERIFIED" is present.
            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success)
                Assert.Contains("--- VERIFIED by Epistemic Governor ---", completed.ExecutionTrace)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Reflection loop continues when Epistemic Governor rejects`` () =
        task {
            // Setup
            let agent = createTestAgent ()
            let registry = createMockRegistry agent

            // LLM returns "Fixed Solution" on second call (reflection)
            // We need a slightly smarter mock LLM to simulate improvement
            let mutable callCount = 0

            let smartLlm =
                { new ILlmService with
                    member _.CompleteAsync req =
                        task {
                            callCount <- callCount + 1
                            let response = if callCount > 1 then "Fixed Solution" else "Bad Solution"

                            return
                                { Text = response
                                  Usage = None
                                  FinishReason = Some "stop"
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |] }

            // Epistemic rejects first time, accepts second time?
            // Since we can't easily change the mock state inside the loop without a mutable ref,
            // let's just test that it DOES reflect at least once if rejected.
            // We'll make it always reject for this test, so it should hit max reflections.
            let epistemic = createMockEpistemic false "Fix this bug"

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = smartLlm
                  VectorStore = Unchecked.defaultof<_>
                  SemanticMemory = None
                  Epistemic = Some epistemic
                  PreLlm = None
                  Budget = None
                  OutputGuard = None
                  KnowledgeBase = None
                  KnowledgeGraph = None
                  MemoryBuffer = None
                  EpisodeService = None
                  Logger = fun _ -> ()
                  Verbose = false
                  ShowSemanticMessage = fun _ _ -> () }

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Test Task"
                  Constraints = []
                  ValidationCriteria = "None"
                  Timeout = TimeSpan.FromMinutes(1.0)
                  Score = 1.0 }

            let state =
                { Generation = 0
                  CompletedTasks = []
                  TaskQueue = []
                  CurrentTask = Some taskDef
                  ActiveBeliefs = []
                  CurriculumAgentId = AgentId(Guid.NewGuid())
                  ExecutorAgentId = agent.Id }

            // Act
            let! newState = Engine.step ctx state

            // Assert
            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success)
                // Should have multiple reflections
                Assert.True(completed.ExecutionTrace |> List.exists (fun t -> t.Contains("--- REFLECTION")))
                // Should NOT have "VERIFIED" since we forced rejection
                Assert.False(completed.ExecutionTrace |> List.exists (fun t -> t.Contains("--- VERIFIED")))
            | [] -> Assert.Fail("Task was not completed")
        }
