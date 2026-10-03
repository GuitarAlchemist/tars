namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Evolution
open Tars.Llm

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
            member _.EmbedAsync text = Task.FromResult [| 0.1f |]

            member _.RouteAsync _ =
                Task.FromResult
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None } }

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
            member _.SuggestCurriculum(completed, active, isCritical) = raise (NotImplementedException())
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
          Memory = []
          Fitness = 0.0
          Drives =
            { Accuracy = 0.5
              Speed = 0.5
              Creativity = 0.5
              Safety = 0.5 }
          Constitution = AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) }

    [<Fact>]
    let ``Reflection loop stops when Epistemic Governor verifies solution`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Setup
            let agent = createTestAgent ()
            let registry = createMockRegistry agent
            let llm = createMockLlm "Initial Solution" // Agent always returns this
            let epistemic = createMockEpistemic true "Good job" // Verified immediately

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = llm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some epistemic
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      SelfImprovement = false } }

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
    let ``A task's duration is measured, not derived from its reflection count`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Evolve showed "completed in 20.0s" for every task: the duration was 10 s per
            // reflection plus 10, while the tasks took a second or two.
            let agent = createTestAgent ()

            let slowLlm =
                { new ILlmService with
                    member _.CompleteAsync req =
                        task {
                            do! Task.Delay 200

                            return
                                { Text = "Initial Solution"
                                  Usage = None
                                  FinishReason = Some "stop"
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |]

                    member _.RouteAsync _ =
                        Task.FromResult
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None } }

            let ctx: Engine.EvolutionContext =
                { Registry = createMockRegistry agent
                  Llm = slowLlm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some(createMockEpistemic true "Good job")
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      SelfImprovement = false } }

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

            let! newState = Engine.step ctx state

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Duration >= TimeSpan.FromMilliseconds 200.0, $"duration {completed.Duration}")
                Assert.True(completed.Duration < TimeSpan.FromSeconds 10.0, $"duration {completed.Duration}")
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``An answer with no code is sent back to the executor`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The evolve executor described the function, or asked for the project
            // structure, instead of writing the code; the evaluation then rejected it.
            let agent = createTestAgent ()
            let mutable calls = 0
            let mutable secondRequest = ""

            let llm =
                { new ILlmService with
                    member _.CompleteAsync req =
                        task {
                            calls <- calls + 1

                            if calls = 2 then
                                secondRequest <- req.Messages |> List.map (fun m -> m.Content) |> String.concat "\n"

                            let text =
                                if calls = 1 then
                                    "I will write a recursive factorial function in F#."
                                else
                                    "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"

                            return
                                { Text = text
                                  Usage = None
                                  FinishReason = Some "stop"
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |]

                    member _.RouteAsync _ =
                        Task.FromResult
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None } }

            let ctx: Engine.EvolutionContext =
                { Registry = createMockRegistry agent
                  Llm = llm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some(createMockEpistemic true "Good job")
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      SelfImprovement = false } }

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Write a recursive factorial function in F#"
                  Constraints = []
                  ValidationCriteria = "fact 5 = 120"
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

            let! newState = Engine.step ctx state

            Assert.Contains("contains no code", secondRequest)

            match newState.CompletedTasks with
            | completed :: _ -> Assert.Contains("let rec fact n", completed.Output)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``The first answer is kept when asking for code fails`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let agent = createTestAgent ()
            let mutable calls = 0

            let llm =
                { new ILlmService with
                    member _.CompleteAsync req =
                        task {
                            calls <- calls + 1

                            if calls > 1 then
                                raise (OperationCanceledException())

                            return
                                { Text = "I will write a recursive factorial function in F#."
                                  Usage = None
                                  FinishReason = Some "stop"
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |]

                    member _.RouteAsync _ =
                        Task.FromResult
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None } }

            let ctx: Engine.EvolutionContext =
                { Registry = createMockRegistry agent
                  Llm = llm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some(createMockEpistemic true "Good job")
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      SelfImprovement = false } }

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Write a recursive factorial function in F#"
                  Constraints = []
                  ValidationCriteria = "fact 5 = 120"
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

            let! newState = Engine.step ctx state

            Assert.Equal(2, calls)

            match newState.CompletedTasks with
            | completed :: _ -> Assert.Contains("I will write a recursive factorial function", completed.Output)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Reflection loop continues when Epistemic Governor rejects`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
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
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |]

                    member _.RouteAsync _ =
                        Task.FromResult
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None } }

            // Epistemic rejects first time, accepts second time?
            // Since we can't easily change the mock state inside the loop without a mutable ref,
            // let's just test that it DOES reflect at least once if rejected.
            // We'll make it always reject for this test, so it should hit max reflections.
            let epistemic = createMockEpistemic false "Fix this bug"

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = smartLlm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some epistemic
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      SelfImprovement = false } }

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

    /// Runs one evolve step on a coding task with `goal` and the criteria "fact 5 = 120", whose
    /// executor answers with `answers`, in order (the last one repeats), with `evaluator` and
    /// evolve's --run-code set to `runCode`. Returns the new state and every request the
    /// executor got.
    let private stepWith (goal: string) (evaluator: IEvaluationStrategy option) (runCode: bool) (answers: string list) =
        task {
            let agent = createTestAgent ()
            let requests = Collections.Generic.List<string>()

            let llm =
                { new ILlmService with
                    member _.CompleteAsync req =
                        task {
                            requests.Add(req.Messages |> List.map (fun m -> m.Content) |> String.concat "\n")

                            return
                                { Text = answers.[min (requests.Count - 1) (answers.Length - 1)]
                                  Usage = None
                                  FinishReason = Some "stop"
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |]

                    member _.RouteAsync _ =
                        Task.FromResult
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None } }

            let ctx: Engine.EvolutionContext =
                { Registry = createMockRegistry agent
                  Llm = llm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some(createMockEpistemic true "Good job")
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = evaluator }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      SelfImprovement = false
                      RunCode = runCode } }

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = goal
                  Constraints = []
                  ValidationCriteria = "fact 5 = 120"
                  Timeout = TimeSpan.FromMinutes(2.0)
                  Score = 1.0 }

            let state =
                { Generation = 0
                  CompletedTasks = []
                  TaskQueue = []
                  CurrentTask = Some taskDef
                  ActiveBeliefs = []
                  CurriculumAgentId = AgentId(Guid.NewGuid())
                  ExecutorAgentId = agent.Id }

            let! newState = Engine.step ctx state
            return newState, List.ofSeq requests
        }

    /// A factorial task that names no function, so its criteria hold no example, and no evaluator.
    let private stepWithAnswers (runCode: bool) (answers: string list) =
        stepWith "Write a recursive factorial function in F#" None runCode answers

    /// An evaluator that always gives `passed`, and counts its calls in `calls`.
    let private fixedEvaluator (passed: bool) (calls: int ref) =
        { new IEvaluationStrategy with
            member _.Evaluate(_, _) =
                task {
                    calls.Value <- calls.Value + 1

                    return
                        { Passed = passed
                          Confidence = 1.0
                          Summary = "fixed verdict"
                          Issues = []
                          SuggestedFixes = []
                          EvaluatedAt = DateTime.UtcNow }
                } }

    [<Fact>]
    let ``The examples decide when they run, not the evaluator`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // In live runs the evaluator rejected code that ran and gave the right values,
            // calling it "a description". The criteria's examples are F#, so they can run.
            let goal = "Write `fact : int -> int` in F#"
            let rejecting = ref 0
            let accepting = ref 0

            let! right, _ =
                stepWith
                    goal
                    (Some(fixedEvaluator false rejecting))
                    true
                    [ "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```" ]

            let! wrong, _ = stepWith goal (Some(fixedEvaluator true accepting)) true [ "```fsharp\nlet fact n = n\n```" ]

            match right.CompletedTasks, wrong.CompletedTasks with
            | passed :: _, failed :: _ ->
                Assert.True(passed.Success)
                Assert.False(failed.Success)
                Assert.Contains("fact 5 = 120, got 5", failed.Output)
                Assert.Equal(0, rejecting.Value + accepting.Value)
            | _ -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A failing example goes back to the executor, and its fixed answer is what is checked`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let calls = ref 0

            let! newState, requests =
                stepWith
                    "Write `fact : int -> int` in F#"
                    (Some(fixedEvaluator false calls))
                    true
                    [ "```fsharp\nlet fact n = n\n```"
                      "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```" ]

            Assert.Contains(requests, fun r -> r.Contains "Example failed: fact 5 = 120, got 5")

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success)
                Assert.Equal(0, calls.Value)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``The code runs once, with its examples`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Running it a second time to check the examples repeated its side effects.
            let marks = IO.Path.GetTempFileName()

            try
                let answer =
                    "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\nSystem.IO.File.AppendAllText(@\""
                    + marks
                    + "\", \"x\")\n```"

                let! newState, _ = stepWith "Write `fact : int -> int` in F#" None true [ answer ]

                Assert.Equal("x", IO.File.ReadAllText marks)

                match newState.CompletedTasks with
                | completed :: _ -> Assert.True(completed.Success)
                | [] -> Assert.Fail("Task was not completed")
            finally
                IO.File.Delete marks
        }

    [<Fact>]
    let ``Code that fails to run is sent back to the executor with its errors`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Nothing ran the executor's code: the evaluation only reads the answer.
            let! newState, requests =
                stepWithAnswers
                    true
                    [ "```fsharp\nlet fact (n: int) : int = \"not a number\"\n```"
                      "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```" ]

            Assert.Contains(requests, fun r -> r.Contains "error FS0001")

            match newState.CompletedTasks with
            | completed :: _ -> Assert.Contains("let rec fact n", completed.Output)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Code that runs is not sent back`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let! newState, requests =
                stepWithAnswers true [ "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\nprintfn \"%d\" (fact 5)\n```" ]

            Assert.DoesNotContain(requests, fun r -> r.Contains "Your code was run")

            match newState.CompletedTasks with
            | completed :: _ -> Assert.Contains("let rec fact n", completed.Output)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Code is not run without --run-code`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // It runs with the user's rights, outside any sandbox, so evolve runs it only when asked.
            let! _, requests = stepWithAnswers false [ "```fsharp\nlet fact (n: int) : int = \"not a number\"\n```" ]

            Assert.DoesNotContain(requests, fun r -> r.Contains "Your code was run")
        }

    [<Fact>]
    let ``Vague generated tasks give way to a concrete one`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Tasks the curriculum generated in live evolve runs; every one of them failed.
            let vague =
                """{"tasks":[{"goal":"Refactor an existing F# codebase to use pattern matching instead of if-else statements.","constraints":[],"validation_criteria":"Code is cleaner"},{"goal":"Create a tool to analyze the performance of F# code snippets","constraints":[],"validation_criteria":"It works"}]}"""

            let agent = createTestAgent ()
            let mutable curriculumPrompt = ""

            let llm =
                { new ILlmService with
                    member _.CompleteAsync req =
                        task {
                            let text = req.Messages |> List.map (fun m -> m.Content) |> String.concat "\n"
                            let isCurriculum = text.Contains "generating F# CODING TASKS"

                            if isCurriculum then
                                curriculumPrompt <- text

                            return
                                { Text = (if isCurriculum then vague else "```fsharp\nlet x = 1\n```")
                                  Usage = None
                                  FinishReason = Some "stop"
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
                    member _.EmbedAsync text = Task.FromResult [| 0.1f |]

                    member _.RouteAsync _ =
                        Task.FromResult
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None } }

            let ctx: Engine.EvolutionContext =
                { Registry = createMockRegistry agent
                  Llm = llm
                  VectorStore = Unchecked.defaultof<_>
                  Logger = fun _ -> ()
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = Some(createMockEpistemic true "Good job")
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      SelfImprovement = false } }

            let state =
                { Generation = 0
                  CompletedTasks = []
                  TaskQueue = []
                  CurrentTask = None
                  ActiveBeliefs = []
                  CurriculumAgentId = agent.Id
                  ExecutorAgentId = agent.Id }

            let! newState = Engine.step ctx state

            Assert.Contains("MUST include the complete code to change", curriculumPrompt)
            Assert.Contains("2 or 3 examples", curriculumPrompt)

            match newState.CompletedTasks with
            | completed :: _ -> Assert.Equal(fst Engine.concreteTasks.Head, completed.TaskGoal)
            | [] -> Assert.Fail("No task was run")
        }
