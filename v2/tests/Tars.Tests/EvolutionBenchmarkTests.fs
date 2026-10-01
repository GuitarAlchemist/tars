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
                  Logger = fun msg -> printfn $"LOG: %s{msg}"
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = None
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = true
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      SelfImprovement = false } }

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

module BenchmarkProvenanceTests =

    open System
    open System.Threading.Tasks
    open Xunit
    open Tars.Llm
    open Tars.Cortex.WoTTypes
    open Tars.Evolution

    // A learning curve needs to know which model and which evolve cycle produced each
    // result. Benchmark runs recorded the model as "default" and no cycle, so a
    // per-cycle, per-model pass rate could not be computed from disk.

    /// Routes every request to `backend`; RouteAsync fails when `backend` is None.
    type private RoutedLlm(backend: LlmBackend option) =
        interface ILlmService with
            member _.CompleteAsync(_) =
                Task.FromResult
                    { Text = ""
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }

            member this.CompleteStreamAsync(req, _) = (this :> ILlmService).CompleteAsync req
            member _.EmbedAsync(_) = Task.FromResult([| 0.0f |])

            member _.RouteAsync(_) =
                match backend with
                | Some b ->
                    Task.FromResult(
                        ({ Backend = b
                           Endpoint = Uri "http://localhost:11434"
                           ApiKey = None }
                        : Tars.Llm.Routing.RoutedBackend)
                    )
                | None -> Task.FromException<Tars.Llm.Routing.RoutedBackend>(InvalidOperationException "no backend")

    type private RecordingSelector() =
        let recorded = ResizeArray<PatternOutcome>()
        member _.Recorded = List.ofSeq recorded

        interface IPatternSelector with
            member _.Recommend(_, _) = ChainOfThought
            member _.Score(_) = Map.empty
            member _.RecordOutcome(outcome) = recorded.Add outcome

    let private run (llm: ILlmService) =
        (BenchmarkRunner.runSuiteFromProblems llm [] None None None false ignore).Result

    [<Fact>]
    let ``a run records the model the router chose, not "default"`` () =
        let summary = run (RoutedLlm(Some(Ollama "qwen3-coder:30b")))
        Assert.Equal("ollama/qwen3-coder:30b", summary.ModelUsed)

    [<Fact>]
    let ``a run whose backend cannot be resolved says the model is unknown, and still completes`` () =
        let summary = run (RoutedLlm None)
        Assert.Equal("unknown", summary.ModelUsed)

    [<Fact>]
    let ``a placeholder model name from the router is not credited as a model`` () =
        // ChatClientLlmService routes every request to `Ollama "unknown"`.
        Assert.Equal("unknown", (run (RoutedLlm(Some(Ollama "unknown")))).ModelUsed)
        Assert.Equal("unknown", (run (RoutedLlm(Some(Ollama " ")))).ModelUsed)

    [<Fact>]
    let ``a run outside an evolve cycle has no cycle id`` () =
        Assert.Equal(None, (run (RoutedLlm(Some(Ollama "m")))).CycleId)

    [<Fact>]
    let ``recorded outcomes carry the run's model and cycle`` () =
        let attempt: BenchmarkAttempt =
            { ProblemId = "basic-reverse-string"
              Difficulty = Beginner
              Category = StringManipulation
              GeneratedCode = ""
              Compiled = true
              Validated = true
              CompileErrors = []
              ValidationOutput = ""
              GenerationTimeMs = 10L
              ValidationTimeMs = 5L
              ExecutionNs = None
              PropertiesValidated = None
              Timestamp = DateTime.UtcNow }

        let summary =
            { run (RoutedLlm(Some(Ollama "qwen3-coder:30b"))) with
                Attempts = [ attempt ]
                CycleId = Some "a1b2c3d4/2" }

        let selector = RecordingSelector()
        BenchmarkRunner.recordOutcomes selector summary
        let outcome = Assert.Single(selector.Recorded)
        Assert.Equal(Some "ollama/qwen3-coder:30b", outcome.ModelId)
        Assert.Equal(Some "a1b2c3d4/2", outcome.CycleId)

    [<Fact>]
    let ``an unknown model is not recorded as a model id`` () =
        let summary = run (RoutedLlm None)

        let attempt: BenchmarkAttempt =
            { ProblemId = "p"
              Difficulty = Beginner
              Category = StringManipulation
              GeneratedCode = ""
              Compiled = false
              Validated = false
              CompileErrors = []
              ValidationOutput = ""
              GenerationTimeMs = 0L
              ValidationTimeMs = 0L
              ExecutionNs = None
              PropertiesValidated = None
              Timestamp = DateTime.UtcNow }

        let selector = RecordingSelector()
        BenchmarkRunner.recordOutcomes selector { summary with Attempts = [ attempt ] }
        Assert.Equal(None, (Assert.Single(selector.Recorded)).ModelId)
