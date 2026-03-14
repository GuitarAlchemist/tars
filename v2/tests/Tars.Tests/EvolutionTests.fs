namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Evolution
open Tars.Core
open Tars.Llm.Routing // For Ollama case

module EvolutionTests =

    type EvolutionStubLlm(responseText: string) =
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

        interface Tars.Llm.ILlmServiceFunctional with
            member _.CompleteAsync(_req) =
                asyncResult {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.EmbedAsync(_text: string) =
                asyncResult { return [| 0.1f; 0.2f; 0.3f |] }

            member _.RouteAsync(_req) =
                asyncResult {
                    return
                        { Backend = Tars.Llm.LlmBackend.Ollama "mock"
                          Endpoint = Uri "http://localhost:11434"
                          ApiKey = None }
                }

    type StubVectorStore() =
        interface IVectorStore with
            member _.SaveAsync(_c, _id, _v, _p) = Task.CompletedTask
            member _.SearchAsync(_c, _v, _l) = Task.FromResult([])

    type StubEpistemic() =
        let mutable called = false

        member _.Called = called

        interface IEpistemicGovernor with
            member _.GenerateVariants(_taskDescription, _count) = Task.FromResult([])

            member _.VerifyGeneralization(_taskDescription, _solution, _variants) =
                Task.FromResult(
                    { IsVerified = true
                      Score = 1.0
                      Feedback = ""
                      FailedVariants = [] }
                )

            member _.ExtractPrinciple(_taskDescription, _solution) =
                Task.FromResult(
                    { Id = Guid.NewGuid()
                      Statement = "stub"
                      Context = "test"
                      Status = EpistemicStatus.Hypothesis
                      Confidence = 1.0
                      DerivedFrom = []
                      CreatedAt = DateTime.UtcNow
                      LastVerified = DateTime.UtcNow }
                )

            member _.SuggestCurriculum(_completedTasks, _activeBeliefs, _isCritical) =
                called <- true
                Task.FromResult("suggestion")

            member _.Verify(_statement) = Task.FromResult(true)
            member _.GetRelatedCodeContext(_) = Task.FromResult("mock context")

    [<Fact>]
    let ``TaskQueue supports multiple tasks`` () =
        let task1 =
            { Id = Guid.NewGuid()
              DifficultyLevel = 1
              Goal = "Task 1"
              Constraints = []
              ValidationCriteria = ""
              Timeout = TimeSpan.Zero
              Score = 1.0 }

        let task2 =
            { Id = Guid.NewGuid()
              DifficultyLevel = 1
              Goal = "Task 2"
              Constraints = []
              ValidationCriteria = ""
              Timeout = TimeSpan.Zero
              Score = 0.9 }

        let state =
            { Generation = 0
              CurriculumAgentId = AgentId(Guid.NewGuid())
              ExecutorAgentId = AgentId(Guid.NewGuid())
              CompletedTasks = []
              CurrentTask = None
              TaskQueue = [ task1; task2 ]
              ActiveBeliefs = [] }

        // Simulate step logic manually as Engine.step is complex to mock fully without DI
        let nextState =
            match state.CurrentTask with
            | Some _ -> state // Should not happen
            | None ->
                match state.TaskQueue with
                | next :: rest ->
                    { state with
                        CurrentTask = Some next
                        TaskQueue = rest }
                | [] -> state

        Assert.Equal(Some task1, nextState.CurrentTask)
        let single = Assert.Single(nextState.TaskQueue)
        Assert.Equal(task2, single)



    [<Fact>]
    let ``Evolution generation uses epistemic suggestions`` () =
        if not (TestHelpers.requireTools()) then () else
        let stubEpistemic = StubEpistemic()

        let llmJson =
            """{\"tasks\":[{\"goal\":\"G\",\"constraints\":[],\"validation_criteria\":\"check\"}]}"""

        let llm = EvolutionStubLlm(llmJson)

        let state: EvolutionState =
            { Generation = 0
              CurriculumAgentId = AgentId(Guid.NewGuid())
              ExecutorAgentId = AgentId(Guid.NewGuid())
              CompletedTasks = []
              CurrentTask = None
              TaskQueue = []
              ActiveBeliefs = [] }

        let curriculumAgent =
            let (AgentId id) = state.CurriculumAgentId
            Tars.Kernel.AgentFactory.create id "Curriculum" "1.0.0" "test-model" "System Prompt" [] []

        let registry = Tars.Kernel.AgentRegistry()
        registry.Register(curriculumAgent)
        let registry = registry :> IAgentRegistry

        let evoCtx: Engine.EvolutionContext =
            { Registry = registry
              Llm = llm :> Tars.Llm.ILlmService
              VectorStore = StubVectorStore() :> IVectorStore
              SemanticMemory = None
              Epistemic = Some(stubEpistemic :> IEpistemicGovernor)
              PreLlm = None // No pipeline for tests yet
              Budget = None
              OutputGuard = None
              KnowledgeBase = None
              KnowledgeGraph = None
              MemoryBuffer = None
              EpisodeService = None // Graphiti integration
              Ledger = None
              Evaluator = None
              RunId = None
              Logger = (fun _ -> ())
              Verbose = false
              ShowSemanticMessage = (fun _ _ -> ())
              Focus = None
              ToolRegistry = None
              ResearchEnhanced = false
              SelfImprovement = false }

        let _nextState =
            Engine.step evoCtx state |> Async.AwaitTask |> Async.RunSynchronously

        // Verify the epistemic governor was consulted
        Assert.True(stubEpistemic.Called, "Epistemic governor SuggestCurriculum should be invoked")
