namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Evolution
open Tars.Kernel
open Tars.Knowledge
open Tars.Llm

module EvolveIntegrationTests =

    type private ContradictionLlm() =
        interface ILlmService with
            member _.CompleteAsync(_) =
                System.Threading.Tasks.Task.FromResult(
                    { Text = "{\"contradicts\": true, \"reason\": \"Ledger conflict\"}"
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
                )

            member _.EmbedAsync(_) = System.Threading.Tasks.Task.FromResult([| 0.0f |])

            member this.CompleteStreamAsync(req, _) =
                (this :> ILlmService).CompleteAsync(req)
 
            member _.RouteAsync(_) = System.Threading.Tasks.Task.FromResult({ Backend = Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None })

    type private NoOpVectorStore() =
        interface IVectorStore with
            member _.SaveAsync(_, _, _, _) = System.Threading.Tasks.Task.CompletedTask
            member _.SearchAsync(_, _, _) = System.Threading.Tasks.Task.FromResult([])

    [<Fact>]
    let ``Evolution blocks when ledger contradictions violate policy`` () =
        (async {
            if not (TestHelpers.requireTools()) then () else
            let ledger = KnowledgeLedger.createInMemory()
            do! ledger.Initialize() |> Async.AwaitTask

            let belief = Belief.fromTriple "TARS" Contradicts "HeuristicGuide"

            let! assertResult: Result<BeliefId, string> =
                ledger.Assert(belief, AgentId.System) |> Async.AwaitTask
            match assertResult with
            | Microsoft.FSharp.Core.Result.Error err -> Assert.True(false, err)
            | Microsoft.FSharp.Core.Result.Ok _ -> ()

            let registry = AgentRegistry()
            let curriculumGuid = Guid.NewGuid()
            let executorGuid = Guid.NewGuid()

            let executorAgent =
                AgentFactory.create
                    executorGuid
                    "Executor"
                    "0.1.0"
                    "stub-model"
                    "Executor prompt"
                    []
                    []

            let curriculumAgent =
                AgentFactory.create
                    curriculumGuid
                    "Curriculum"
                    "0.1.0"
                    "stub-model"
                    "Curriculum prompt"
                    []
                    []

            registry.Register(executorAgent)
            registry.Register(curriculumAgent)

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Rework TARS heuristics to respect ledger beliefs"
                  Constraints = [ "check_contradictions" ]
                  ValidationCriteria = "Ledger-aligned behavior"
                  Timeout = TimeSpan.FromSeconds(1.0)
                  Score = 1.0 }

            let eCurriculumId = Tars.Core.AgentId curriculumGuid
            let eExecutorId = Tars.Core.AgentId executorGuid

            let state =
                { Generation = 0
                  CurriculumAgentId = eCurriculumId
                  ExecutorAgentId = eExecutorId
                  CompletedTasks = []
                  CurrentTask = Some taskDef
                  TaskQueue = []
                  ActiveBeliefs = [] }

            let context: Engine.EvolutionContext =
                { Registry = registry :> IAgentRegistry
                  Llm = ContradictionLlm() :> ILlmService
                  VectorStore = NoOpVectorStore() :> IVectorStore
                  SemanticMemory = None
                  Epistemic = None
                  PreLlm = None
                  Budget = None
                  OutputGuard = None
                  KnowledgeBase = None
                  KnowledgeGraph = None
                  MemoryBuffer = None
                  EpisodeService = None
                  Ledger = Some ledger
                  Evaluator = None
                  RunId = None
                  Logger = fun _ -> ()
                  Verbose = false
                  ShowSemanticMessage = fun _ _ -> ()
                  Focus = None
                  ToolRegistry = None
                  ResearchEnhanced = false
                  SelfImprovement = false }

            let! nextState = Engine.step context state |> Async.AwaitTask

            Assert.Equal(None, nextState.CurrentTask)
            Assert.NotEmpty(nextState.CompletedTasks)

            let result = nextState.CompletedTasks.Head
            Assert.Contains("Blocked by ledger contradiction policy", result.Output)
            Assert.Contains("LEDGER_CONTRADICTION", result.ExecutionTrace)
        })
        |> Async.StartAsTask
