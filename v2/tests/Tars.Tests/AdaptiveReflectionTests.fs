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
                  CurriculumLlm = None
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
                  CurriculumLlm = None
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
                  CurriculumLlm = None
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
                  CurriculumLlm = None
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
                  CurriculumLlm = None
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

    /// Runs one evolve step on a coding task with `goal`, the criteria "fact 5 = 120" and
    /// `timeout`, whose executor answers with `answers`, in order (the last one repeats), with
    /// `evaluator` and evolve's --run-code set to `runCode`. Returns the new state and every
    /// request the executor got.
    let private stepWithin
        (timeout: TimeSpan)
        (goal: string)
        (evaluator: IEvaluationStrategy option)
        (runCode: bool)
        (answers: string list)
        =
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
                  CurriculumLlm = None
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
                  Timeout = timeout
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

    /// stepWithin, with 2 minutes for the task.
    let private stepWith (goal: string) (evaluator: IEvaluationStrategy option) (runCode: bool) (answers: string list) =
        stepWithin (TimeSpan.FromMinutes 2.0) goal evaluator runCode answers

    /// A factorial task that names no function, so its criteria hold no example, and no evaluator.
    let private stepWithAnswers (runCode: bool) (answers: string list) =
        stepWith "Write a recursive factorial function in F#" None runCode answers

    /// An evaluator that always gives `passed`, and records in `seen` the evaluation each task it
    /// judges already had: the examples' verdict, when they ran.
    let private fixedEvaluator (passed: bool) (seen: ResizeArray<EvaluationResult option>) =
        { new IEvaluationStrategy with
            member _.Evaluate(_, result) =
                task {
                    seen.Add result.Evaluation

                    return
                        { Passed = passed
                          Confidence = 1.0
                          Summary = "fixed verdict"
                          Issues = []
                          SuggestedFixes = []
                          EvaluatedAt = DateTime.UtcNow }
                } }

    /// An evaluator that rejects the first answer it judges, with `summary`, the issue "uses List.rev"
    /// and the fix "build the result with an accumulator", and passes the next ones. Records in
    /// `judged` each answer it judges.
    let private rejectsOnce (summary: string) (judged: ResizeArray<string>) =
        { new IEvaluationStrategy with
            member _.Evaluate(_, result) =
                task {
                    judged.Add result.Output

                    return
                        { Passed = judged.Count > 1
                          Confidence = 0.9
                          Summary = summary
                          Issues = [ "uses List.rev" ]
                          SuggestedFixes = [ "build the result with an accumulator" ]
                          EvaluatedAt = DateTime.UtcNow }
                } }

    /// rejectsOnce, whose first verdict takes `wait`.
    let private slowlyRejectsOnce (wait: TimeSpan) (judged: ResizeArray<string>) =
        let reviewer = rejectsOnce "The code uses List.rev, which the constraints forbid." judged

        { new IEvaluationStrategy with
            member _.Evaluate(taskDef, result) =
                task {
                    if judged.Count = 0 then
                        do! Task.Delay wait

                    return! reviewer.Evaluate(taskDef, result)
                } }

    [<Fact>]
    let ``A failing example decides, and passing examples leave the rest to the evaluator`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The criteria's examples are F#, so they can run, and a wrong value fails the task.
            // They do not check what else the goal asks, like a `flatten` written without
            // List.concat: when they pass, the evaluator judges that, told they passed.
            let goal = "Write `fact : int -> int` in F#"
            let right = "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"
            let rejecting = ResizeArray()
            let accepting = ResizeArray()
            let afterWrong = ResizeArray()

            let! rejected, _ = stepWith goal (Some(fixedEvaluator false rejecting)) true [ right ]
            let! accepted, _ = stepWith goal (Some(fixedEvaluator true accepting)) true [ right ]
            let! wrong, _ = stepWith goal (Some(fixedEvaluator true afterWrong)) true [ "```fsharp\nlet fact n = n\n```" ]

            match rejected.CompletedTasks, accepted.CompletedTasks, wrong.CompletedTasks with
            | r :: _, a :: _, w :: _ ->
                Assert.False(r.Success)
                Assert.True(a.Success)
                Assert.False(w.Success)
                Assert.Contains("fact 5 = 120, got 5", w.Output)

                // The rejected answer was answered again, once, and judged again.
                Assert.Equal(2, rejecting.Count)
                Assert.Single(accepting) |> ignore

                for seen in [ rejecting; accepting ] do
                    Assert.True(seen |> Seq.forall (Option.exists (fun e -> e.Passed)))

                Assert.Empty(afterWrong)
            | _ -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A failing example goes back to the executor, and its fixed answer is what is checked`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let seen = ResizeArray()

            let! newState, requests =
                stepWith
                    "Write `fact : int -> int` in F#"
                    (Some(fixedEvaluator true seen))
                    true
                    [ "```fsharp\nlet fact n = n\n```"
                      "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```" ]

            Assert.Contains(requests, fun r -> r.Contains "Example failed: fact 5 = 120, got 5")

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success)
                // The evaluator then saw the fixed answer's examples pass.
                Assert.True(seen |> Seq.exactlyOne |> Option.exists (fun e -> e.Passed))
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A rejected answer goes back to the executor once, with the evaluator's remarks`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // In live runs the judge rejected answers that broke a constraint (List.rev, a mutable),
            // said why, and the task failed: the executor never heard it.
            let judged = ResizeArray()
            let summary = "The code uses List.rev, which the constraints forbid."
            let right = "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"

            let! newState, requests =
                stepWith "Write `fact : int -> int` in F#" (Some(rejectsOnce summary judged)) true [ right ]

            Assert.Equal(2, requests.Length)
            Assert.DoesNotContain(summary, requests.[0])

            for remark in [ summary; "uses List.rev"; "build the result with an accumulator" ] do
                Assert.Contains(remark, requests.[1])

            Assert.Equal(2, judged.Count)

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success, completed.Output)
                // Both attempts' traces are kept, around the rejection that led to the second.
                let trace = completed.ExecutionTrace
                let at = List.findIndex ((=) "--- REJECTED BY THE EVALUATOR, ANSWERED AGAIN ---") trace
                let answered (steps: string list) = steps |> List.exists (fun s -> s.StartsWith "Response:")
                Assert.Equal(summary, trace.[at + 1])
                Assert.True(answered (List.take at trace), $"%A{trace}")
                Assert.True(answered (List.skip (at + 2) trace), $"%A{trace}")
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``An evaluator that could not judge sends nothing back to the executor`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Its verdict says nothing about the code: the judge's answer was not JSON, or its call failed.
            let failing =
                { new ILlmService with
                    member _.CompleteAsync _ =
                        Task.FromException<LlmResponse>(TimeoutException "The judge timed out")

                    member _.CompleteStreamAsync(_, _) = raise (NotImplementedException())
                    member _.EmbedAsync _ = Task.FromResult [| 0.1f |]
                    member _.RouteAsync _ = raise (NotImplementedException()) }

            let right = "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"

            for judge in [ createMockLlm "I cannot judge this."; failing ] do
                let evaluator = SemanticEvaluation(judge) :> IEvaluationStrategy
                let! newState, requests = stepWith "Write `fact : int -> int` in F#" (Some evaluator) true [ right ]

                Assert.Equal(1, requests.Length)

                match newState.CompletedTasks with
                | completed :: _ -> Assert.False(completed.Success)
                | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Only code whose examples passed goes back to the executor after a rejection`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // Nothing else shows that the code ran: evolve without --run-code runs none, and the
            // `fact` examples cannot call a `factorial`. A new answer would double the executor's
            // calls for a verdict on code that may not even run.
            let right = "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"
            let unchecked = "```fsharp\nlet rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)\n```"

            for (runCode, answer) in [ (false, right); (true, unchecked) ] do
                let judged = ResizeArray()

                let! newState, requests =
                    stepWith "Write `fact : int -> int` in F#" (Some(rejectsOnce "Uses List.rev." judged)) runCode [ answer ]

                Assert.Equal(1, requests.Length)
                Assert.Equal(1, judged.Count)

                match newState.CompletedTasks with
                | completed :: _ -> Assert.False(completed.Success)
                | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A rejected answer gets no new answer once the task's time is up`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The verdict comes after the task's 8 s: a new answer would run the task past its limit.
            let judged = ResizeArray()

            let! newState, requests =
                stepWithin
                    (TimeSpan.FromSeconds 8.0)
                    "Write `fact : int -> int` in F#"
                    (Some(slowlyRejectsOnce (TimeSpan.FromSeconds 9.0) judged))
                    true
                    [ "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```" ]

            Assert.Equal(1, requests.Length)

            match newState.CompletedTasks with
            | completed :: _ -> Assert.False(completed.Success)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A rejected answer's new answer runs within the time the task has left`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The first answer and its 6 s verdict use part of the task's 15 s. The new answer's code
            // would run 30 s: it is stopped at the task's deadline, not 15 s after the verdict.
            let judged = ResizeArray()

            let slow =
                "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\nSystem.Threading.Thread.Sleep 30000\n```"

            let watch = Diagnostics.Stopwatch.StartNew()

            let! newState, requests =
                stepWithin
                    (TimeSpan.FromSeconds 15.0)
                    "Write `fact : int -> int` in F#"
                    (Some(slowlyRejectsOnce (TimeSpan.FromSeconds 6.0) judged))
                    true
                    [ "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"; slow ]

            let elapsed = watch.Elapsed
            Assert.Equal(2, requests.Length)
            Assert.True(elapsed < TimeSpan.FromSeconds 20.0, $"{elapsed}")

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.False(completed.Success)
                // The reported duration is the whole task's, the 6 s verdict included.
                Assert.True(completed.Duration > elapsed - TimeSpan.FromSeconds 2.0, $"{completed.Duration} of {elapsed}")
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``When the examples still do not pass, the first new answer that passes them is kept`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // In live runs the executor asked a question or refused instead of writing code, and a
            // fix in the same conversation does not change that; a new answer to the task can.
            let goal = "Write `fact : int -> int` in F#"
            let wrong = "```fsharp\nlet fact n = n\n```"
            let question = "What should fact return for negative numbers?"
            let right = "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"
            let afterWrong = ResizeArray()
            let afterQuestion = ResizeArray()

            // The answer, its fix, then a new answer.
            let! fixedWrong, _ = stepWith goal (Some(fixedEvaluator true afterWrong)) true [ wrong; wrong; right ]
            // The answer, the request for code, then a new answer.
            let! answeredQuestion, _ =
                stepWith goal (Some(fixedEvaluator true afterQuestion)) true [ question; question; right ]

            match fixedWrong.CompletedTasks, answeredQuestion.CompletedTasks with
            | w :: _, q :: _ ->
                Assert.True(w.Success, w.Output)
                Assert.True(q.Success, q.Output)

                for seen in [ afterWrong; afterQuestion ] do
                    Assert.True(seen |> Seq.exactlyOne |> Option.exists (fun e -> e.Passed))
            | _ -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``The executor answers again at most 3 times`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let! newState, requests =
                stepWith "Write `fact : int -> int` in F#" None true [ "```fsharp\nlet fact n = n\n```" ]

            // The answer, its fix, and 3 new answers.
            Assert.Equal(5, requests.Length)

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.False(completed.Success)
                Assert.Contains("fact 5 = 120, got 5", completed.Output)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Code the examples do not compile against is left to the evaluator, with no new answer`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The examples call `fact`, which this answer does not define: they say nothing about it,
            // and no new answer is asked for.
            let seen = ResizeArray()

            let! newState, requests =
                stepWith
                    "Write `fact : int -> int` in F#"
                    (Some(fixedEvaluator true seen))
                    true
                    [ "```fsharp\nlet rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)\n```" ]

            Assert.Equal(1, requests.Length)

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success)
                Assert.True(seen |> Seq.exactlyOne |> Option.isNone)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A new answer the examples cannot check is left to the evaluator rather than a failing one`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The new answers define `factorial`, which the `fact` examples cannot call: they may be
            // right, while the first answer is known to be wrong.
            let seen = ResizeArray()

            let! newState, requests =
                stepWith
                    "Write `fact : int -> int` in F#"
                    (Some(fixedEvaluator true seen))
                    true
                    [ "```fsharp\nlet fact n = n\n```"
                      "```fsharp\nlet fact n = n\n```"
                      "```fsharp\nlet rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)\n```" ]

            // A new answer that passes the examples is still looked for.
            Assert.Equal(5, requests.Length)

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success, completed.Output)
                Assert.Contains("factorial", completed.Output)
                Assert.True(seen |> Seq.exactlyOne |> Option.isNone)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``Code that does not run on its own is left to the evaluator, with no new answer`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // A script cannot load TARS's projects, so this code is not run, but it is code: the
            // evaluator judges it, as in step 5.2.
            let seen = ResizeArray()

            let! newState, requests =
                stepWith
                    "Write `fact : int -> int` in F#"
                    (Some(fixedEvaluator true seen))
                    true
                    [ "```fsharp\nopen Tars.Core\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"
                      "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```" ]

            Assert.Equal(1, requests.Length)

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.True(completed.Success)
                Assert.True(seen |> Seq.exactlyOne |> Option.isNone)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A new answer the examples cannot check is not kept when its code fails to run`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let seen = ResizeArray()

            let! newState, _ =
                stepWith
                    "Write `fact : int -> int` in F#"
                    (Some(fixedEvaluator true seen))
                    true
                    [ "```fsharp\nlet fact n = n\n```"
                      "```fsharp\nlet fact n = n\n```"
                      "```fsharp\nlet rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)\nfailwith \"boom\"\n```" ]

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.False(completed.Success)
                Assert.Contains("fact 5 = 120, got 5", completed.Output)
                Assert.Empty(seen)
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``A new answer's code runs within the task's time`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let wrong = "```fsharp\nlet fact n = n\n```"
            // Right, but its examples can only run after 30 s.
            let slow =
                "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\nSystem.Threading.Thread.Sleep 30000\n```"

            let watch = Diagnostics.Stopwatch.StartNew()

            let! newState, _ =
                stepWithin (TimeSpan.FromSeconds 10.0) "Write `fact : int -> int` in F#" None true [ wrong; wrong; slow ]

            match newState.CompletedTasks with
            | completed :: _ ->
                Assert.False(completed.Success)
                Assert.True(watch.Elapsed < TimeSpan.FromSeconds 25.0, $"{watch.Elapsed}")
            | [] -> Assert.Fail("Task was not completed")
        }

    [<Fact>]
    let ``No new answer is asked for once the task's time is up`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            // The answer's code runs until the task's deadline, 3 s away, and is stopped there.
            let slow = "```fsharp\nlet fact n = n\nSystem.Threading.Thread.Sleep 10000\n```"

            let! newState, requests =
                stepWithin (TimeSpan.FromSeconds 3.0) "Write `fact : int -> int` in F#" None true [ slow ]

            // Only the answer: once the deadline has passed, the task's token is cancelled, and
            // neither the fix request nor a new answer reaches the model.
            Assert.Equal(1, requests.Length)

            match newState.CompletedTasks with
            | completed :: _ -> Assert.False(completed.Success)
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
                  CurriculumLlm = None
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
            // A live Opus teacher wrote "exactly one ```fsharp block, no ACT: prefix", which the
            // executor's protocol can't meet, and the judge failed code whose examples passed.
            Assert.Contains("never about the answer's format", curriculumPrompt)

            match newState.CompletedTasks with
            | completed :: _ -> Assert.Equal(fst Engine.concreteTasks.Head, completed.TaskGoal)
            | [] -> Assert.Fail("No task was run")
        }
