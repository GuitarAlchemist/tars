namespace Tars.Evolution

open System
open System.Threading
open System.Threading.Tasks
open Tars.Core
open Tars.Core.Knowledge
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open System.Text.Json
open Tars.Kernel
open Tars.Cortex
open Tars.Connectors.EpisodeIngestion
open Tars.Knowledge

module Engine =

    /// Item to be buffered for memory storage
    type MemoryItem =
        | Belief of collection: string * id: string * vector: float32[] * payload: Map<string, string>
        | Legacy of collection: string * id: string * vector: float32[] * payload: Map<string, string>

    /// The context for the evolution engine
    type EvolutionContext =
        { Registry: IAgentRegistry
          Llm: ILlmService
          VectorStore: IVectorStore
          SemanticMemory: ISemanticMemory option
          Epistemic: IEpistemicGovernor option
          PreLlm: PreLlmPipeline option
          Budget: BudgetGovernor option
          OutputGuard: IOutputGuard option
          KnowledgeBase: KnowledgeBase option
          KnowledgeGraph: TemporalKnowledgeGraph.TemporalGraph option
          MemoryBuffer: BufferAgent<MemoryItem> option // Added Capacitor
          EpisodeService: IEpisodeIngestionService option // Graphiti integration
          Ledger: KnowledgeLedger option
          Evaluator: IEvaluationStrategy option
          RunId: RunId option
          Logger: string -> unit
          Verbose: bool
          ShowSemanticMessage: Message -> bool -> unit }

    let private scoreTask
        (ctx: EvolutionContext)
        (taskDef: TaskDefinition)
        (recentVectors: float32[] list)
        : Task<float> =
        task {
            // Base Score: Difficulty gives a small boost
            let baseScore = 1.0 + (0.1 * float taskDef.DifficultyLevel)

            if recentVectors.IsEmpty then
                return baseScore
            else
                try
                    // Calculate embedding for the new task
                    let! currentVector = ctx.Llm.EmbedAsync(taskDef.Goal)

                    // Find maximum similarity to any recent task
                    let maxSimilarity =
                        recentVectors
                        |> List.map (fun v -> MetricSpace.cosineSimilarity currentVector v)
                        |> List.max

                    // Semantic Fan-out Limiting:
                    // If similarity is too high (> 0.85), it means we are repeating ourselves.
                    // Penalty scales with similarity.
                    let penalty =
                        if maxSimilarity > 0.9f then 10.0 // Hard block (score becomes negative)
                        elif maxSimilarity > 0.8f then 3.0 // Strong discouragement
                        elif maxSimilarity > 0.7f then 0.5 // Minimal discouragement
                        else 0.0 // Novel task

                    let finalScore = baseScore - penalty
                    // ctx.Logger($"[Scoring] '{taskDef.Goal}' Sim: {maxSimilarity:F2} -> Score: {finalScore:F2}")
                    return finalScore
                with ex ->
                    ctx.Logger($"[Scoring] Failed for '{taskDef.Goal}': {ex.Message}")
                    return baseScore
        }

    let private tryGetPropertyInsensitive (name: string) (elem: JsonElement) =
        if elem.ValueKind <> JsonValueKind.Object then
            None
        else
            elem.EnumerateObject()
            |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun p -> p.Value)

    let private readBoolFromJson names (elem: JsonElement) =
        names
        |> List.tryPick (fun name ->
            match tryGetPropertyInsensitive name elem with
            | Some prop ->
                match prop.ValueKind with
                | JsonValueKind.True -> Some true
                | JsonValueKind.False -> Some false
                | JsonValueKind.String ->
                    match Boolean.TryParse(prop.GetString()) with
                    | true, value -> Some value
                    | _ -> None
                | _ -> None
            | None -> None)

    let private readStringFromJson names (elem: JsonElement) =
        names
        |> List.tryPick (fun name ->
            match tryGetPropertyInsensitive name elem with
            | Some prop when prop.ValueKind = JsonValueKind.String -> Some(prop.GetString())
            | _ -> None)

    let private looksLikeFollowUpRequest (text: string) =
        if String.IsNullOrWhiteSpace text then
            false
        else
            let trimmed = text.Trim()
            let lowered = trimmed.ToLowerInvariant()

            trimmed.EndsWith("?")
            || lowered.StartsWith("please provide")
            || lowered.StartsWith("could you")
            || lowered.StartsWith("can you")
            || lowered.StartsWith("would you")
            || lowered.StartsWith("i need")
            || lowered.StartsWith("i require")
            || lowered.StartsWith("share the")
            || lowered.StartsWith("send the")

    let private tryExtractJsonElement (text: string) =
        let trimmed = text.Trim()

        let tryParse payload =
            match JsonParsing.tryParseElement payload with
            | Result.Ok elem -> Some elem
            | Result.Error _ -> None

        let tryExtract (openChar: char, closeChar: char) =
            let startIdx = trimmed.IndexOf(openChar)
            let endIdx = trimmed.LastIndexOf(closeChar)

            if startIdx >= 0 && endIdx > startIdx then
                trimmed.Substring(startIdx, endIdx - startIdx + 1) |> Some
            else
                None

        match tryParse trimmed with
        | Some elem -> Result.Ok elem
        | None ->
            match tryExtract ('{', '}') |> Option.bind tryParse with
            | Some elem -> Result.Ok elem
            | None ->
                match tryExtract ('[', ']') |> Option.bind tryParse with
                | Some elem -> Result.Ok elem
                | None -> Result.Error "Response was not valid JSON."

    let private formatBelief (belief: Belief) =
        let predicate =
            match belief.Predicate with
            | RelationType.Custom p -> p
            | _ -> belief.Predicate.ToString()

        $"- [{belief.Confidence:F2}] {belief.Subject.Value} {predicate} {belief.Object.Value}"

    let private evaluateContradiction (ctx: EvolutionContext) (goal: string) (beliefs: Belief list) =
        task {
            if beliefs.IsEmpty then
                return None
            else
                let beliefLines = beliefs |> List.map formatBelief |> String.concat "\n"

                let prompt =
                    $"""You maintain a knowledge ledger. Known beliefs:
{beliefLines}

Task: {goal}

Do any of the known beliefs contradict executing this task? Respond in JSON: {{"contradicts": true|false, "reason": "..."}}."""

                let request: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = Some "Check whether a new task violates the known beliefs."
                      MaxTokens = Some 250
                      Temperature = Some 0.0
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = Some ResponseFormat.Json
                      Stream = false
                      JsonMode = true
                      Seed = None

                      ContextWindow = None }

                try
                    let! response = ctx.Llm.CompleteAsync(request)

                    match JsonParsing.tryParseElement response.Text with
                    | Result.Ok elem ->
                        let contradicts =
                            readBoolFromJson [ "contradicts"; "conflicts" ] elem
                            |> Option.defaultValue false

                        if contradicts then
                            let reason =
                                readStringFromJson [ "reason"; "details"; "explanation" ] elem
                                |> Option.defaultValue (response.Text.Trim())

                            return Some reason
                        else
                            return None
                    | Result.Error _ ->
                        let lowered = response.Text.ToLowerInvariant()

                        if lowered.Contains("contradict") && lowered.Contains("yes") then
                            return Some response.Text
                        else
                            return None
                with ex ->
                    ctx.Logger($"[Contradiction] LLM check failed: {ex.Message}")
                    return None
        }

    /// Generates a new task using the Curriculum Agent
    let private generateTask (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            // Check Budget Criticality
            let isCritical =
                match ctx.Budget with
                | Some b -> b.IsCritical(0.1) // Less than 10% remaining
                | None -> false

            // Epistemic Governor: Get Curriculum Suggestions
            let! suggestion =
                match ctx.Epistemic with
                | Some governor ->
                    task {
                        try
                            let recentOutputs =
                                state.CompletedTasks
                                |> List.truncate 5
                                |> List.map (fun t -> t.Output.Substring(0, Math.Min(t.Output.Length, 100)) + "...")

                            return! governor.SuggestCurriculum(recentOutputs, state.ActiveBeliefs, isCritical)
                        with ex ->
                            ctx.Logger($"[Epistemic] SuggestCurriculum failed: {ex.Message}")
                            return "Focus on basic coding tasks."
                    }
                | None -> Task.FromResult "Focus on basic coding tasks."

            let guidance =
                if isCritical then
                    suggestion + " WARNING: Budget is critical. Generate simpler, cheaper tasks."
                else
                    suggestion

            let completedGoals =
                state.CompletedTasks |> List.map (fun t -> t.TaskGoal) |> List.distinct

            let completedList =
                if completedGoals.IsEmpty then
                    "None"
                else
                    completedGoals |> List.truncate 10 |> String.concat " | "

            let prompt =
                $"""IMPORTANT: You are generating F# CODING TASKS. Do NOT ask questions. Output ONLY JSON.

Generation: %d{state.Generation}. Completed tasks: %d{state.CompletedTasks.Length}.

Create 3 concrete, diverse F# programming tasks based on the following guidance.
Guidance: %s{if String.IsNullOrWhiteSpace(guidance) then
                 "basic algorithms"
             else
                 guidance}

Requirements:
- Each task must be a specific coding problem (NOT a question) and solvable with code (NOT a discussion).
- Do NOT repeat or closely rephrase any previous tasks: %s{completedList}
- Vary domains and artifacts (algorithms, refactors, tests, tooling, docs, integrations).
- If guidance sounds like a request for preferences, ignore it and still output tasks.
- Include measurable validation_criteria.

RESPOND WITH THIS EXACT JSON FORMAT (no other text):
{{"tasks":[
  {{"goal":"<concise goal 1>","constraints":["<constraint 1>","<constraint 2>"],"validation_criteria":"<measurable check>"}},
  {{"goal":"<concise goal 2>","constraints":["<constraint 1>","<constraint 2>"],"validation_criteria":"<measurable check>"}},
  {{"goal":"<concise goal 3>","constraints":["<constraint 1>","<constraint 2>"],"validation_criteria":"<measurable check>"}}
]}}"""

            // 1. Retrieve Curriculum Agent
            let! agentOpt = ctx.Registry.GetAgent(state.CurriculumAgentId)

            match agentOpt with
            | None -> return []
            | Some agent ->
                // 2. Initialize Graph Executor
                let graphExecutor =
                    GraphExecutor(ctx.Registry, ctx.Llm, ctx.Budget, ctx.OutputGuard, ctx.Logger)

                // 3. Create Request Message with JSON requirement
                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Sender = MessageEndpoint.System
                      Receiver = Some(MessageEndpoint.Agent agent.Id)
                      Performative = Performative.Request
                      Intent = Some AgentDomain.Planning
                      Constraints = SemanticConstraints.Default
                      Ontology = None
                      Language = "json" // Hint to use JSON mode
                      Content = prompt
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.ofList [ ("response_format", "json"); ("json_mode", "true") ] }

                // Show semantic message in demo mode
                ctx.ShowSemanticMessage msg ctx.Verbose
                let agentWithMsg = agent.ReceiveMessage(msg)

                // 4. Run Execution
                let! outcome = graphExecutor.RunAgentLoop(agentWithMsg, 20)

                // Phase 6.2: Semantic Speech Act Validation
                let responseIntent, responseText =
                    match outcome with
                    | Success(_, o, _)
                    | PartialSuccess((_, o, _), _) ->
                        let requestMsg = SpeechActs.fromSemantic msg

                        let intent, content =
                            match SpeechActs.tryParse o with
                            | Some(i, c) -> i, c
                            | None -> Tell o, o

                        let replyMsg = SpeechActs.createReply requestMsg intent content agent.Id

                        match SpeechActs.validateFlow requestMsg replyMsg with
                        | Result.Ok() ->
                            ctx.Logger
                                $"[Protocol] Verified semantic flow: %A{requestMsg.Intent} -> %A{replyMsg.Intent}"
                        | Result.Error err -> ctx.Logger $"[Protocol] WARNING: Protocol violation: %s{err}"

                        intent, content
                    | Failure err ->
                        let errStr = err |> List.map string |> String.concat "; "
                        ctx.Logger $"[Curriculum] Agent returned failure: %s{errStr}"
                        AgentIntent.Error errStr, ""

                let responseText =
                    match responseIntent with
                    | AgentIntent.Tell _ -> responseText
                    | AgentIntent.Event _ -> responseText
                    | _ ->
                        ctx.Logger("[Curriculum] Invalid response intent for task generation. Using fallback.")
                        ""

                if String.IsNullOrWhiteSpace(responseText) then
                    // Return a fallback task when no response is received
                    ctx.Logger("[Curriculum] No response received, using fallback task")

                    return
                        [ { Id = Guid.NewGuid()
                            DifficultyLevel = state.Generation + 1
                            Goal = "Write a simple F# function that calculates the factorial of a number"
                            Constraints = [ "Use recursion"; "Handle edge cases for 0 and negative numbers" ]
                            ValidationCriteria = "Function returns correct factorial values"
                            Timeout = TimeSpan.FromMinutes(1.0)
                            Score = 1.0 } ]
                else
                    let fallbackPracticalTask () =
                        let practicalTasks =
                            [| "Review the DemoVisualization.fs module and write 3 unit tests for the showSemanticMessage function"
                               "Read the ToolFactory.fs file and add XML documentation comments to all public functions"
                               "Analyze the Engine.fs generateTask function and suggest 2 improvements for better error handling" |]

                        let random = Random()
                        let selectedTask = practicalTasks[random.Next(practicalTasks.Length)]

                        [ { Id = Guid.NewGuid()
                            DifficultyLevel = state.Generation + 1
                            Goal = selectedTask
                            Constraints = [ "Work with the TARS v2 codebase" ]
                            ValidationCriteria = "Provide concrete, actionable output"
                            Timeout = TimeSpan.FromMinutes(2.0)
                            Score = 1.0 } ]

                    try
                        let rootResult = tryExtractJsonElement responseText

                        match rootResult with
                        | Result.Error err ->
                            ctx.Logger($"[Curriculum] Task JSON parse failed: {err}")
                            return fallbackPracticalTask ()
                        | Result.Ok root ->

                            let mutable tasksElem = Unchecked.defaultof<JsonElement>

                            let tasksJson =
                                if root.ValueKind = JsonValueKind.Array then
                                    root.EnumerateArray() |> Seq.map id
                                elif
                                    root.TryGetProperty("tasks", &tasksElem)
                                    && tasksElem.ValueKind = JsonValueKind.Array
                                then
                                    tasksElem.EnumerateArray() |> Seq.map id
                                else
                                    Seq.empty

                            let existingGoals =
                                state.CompletedTasks
                                |> List.map (fun t -> t.TaskGoal.Trim().ToLowerInvariant())
                                |> Set.ofList

                            let parsedTasksRaw =
                                tasksJson
                                |> Seq.map (fun t ->
                                    let goal = t.GetProperty("goal").GetString()

                                    let constraints =
                                        t.GetProperty("constraints").EnumerateArray()
                                        |> Seq.map (fun e -> e.GetString())
                                        |> Seq.toList

                                    let criteria = t.GetProperty("validation_criteria").GetString()

                                    { Id = Guid.NewGuid()
                                      DifficultyLevel = state.Generation + 1
                                      Goal = goal
                                      Constraints = constraints
                                      ValidationCriteria = criteria
                                      Timeout = TimeSpan.FromMinutes(1.0)
                                      Score = 0.0 })
                                |> Seq.toList
                                // Drop exact repeats of completed goals
                                |> List.filter (fun t ->
                                    let key = t.Goal.Trim().ToLowerInvariant()
                                    not (existingGoals.Contains(key)))

                        // 5. Semantic Scoring (Fan-out Limiting)
                        // Pre-calculate embeddings for recent tasks (last 10)
                            let recentTasks = state.CompletedTasks |> List.truncate 10

                            let! recentVectors =
                                task {
                                    if recentTasks.IsEmpty then
                                        return []
                                    else
                                        let! vectors =
                                            recentTasks
                                            |> List.map (fun t -> ctx.Llm.EmbedAsync(t.TaskGoal))
                                            |> Task.WhenAll

                                        return vectors |> Array.toList
                                }

                        // Score all candidates
                            let! scoredTasks =
                                parsedTasksRaw
                                |> List.map (fun t ->
                                    task {
                                        let! score = scoreTask ctx t recentVectors
                                        return { t with Score = score }
                                    })
                                |> Task.WhenAll

                        // Select Top K
                            let k = 3

                            let topK =
                                scoredTasks
                                |> Array.toList
                                |> List.sortByDescending (fun t -> t.Score)
                                |> List.truncate k
                                // Filter out negative scores (hard blocked)
                                |> List.filter (fun t -> t.Score > 0.0)

                        // Budget-aware priority report
                            let remainingTokens =
                                ctx.Budget
                                |> Option.bind (fun b -> b.Remaining.MaxTokens |> Option.map (fun t -> int t))

                            ctx.Logger(TaskPrioritization.priorityReport topK state.CompletedTasks remainingTokens)

                            // Re-prioritize by budget efficiency
                            let budgetPrioritized =
                                TaskPrioritization.prioritizeQueue topK state.CompletedTasks remainingTokens

                            if budgetPrioritized.IsEmpty then
                                ctx.Logger(
                                    "[Curriculum] All generated tasks were rejected by semantic limiter. Using fallback."
                                )

                                return
                                    [ { Id = Guid.NewGuid()
                                        DifficultyLevel = state.Generation + 1
                                        Goal =
                                          "Failed to generate novel tasks. Refactor the existing code for better readability."
                                        Constraints = []
                                        ValidationCriteria = "Code is cleaner"
                                        Timeout = TimeSpan.FromMinutes(1.0)
                                        Score = 0.5 } ]
                            else
                                return budgetPrioritized

                    with ex ->
                        ctx.Logger($"[Curriculum] Task generation failed: {ex.Message}")
                        return fallbackPracticalTask ()

        }

    /// Attempts to solve a task using the Executor Agent
    let private executeTask (ctx: EvolutionContext) (state: EvolutionState) (taskDef: TaskDefinition) =
        task {
            // 1. Retrieve Executor Agent
            let! agentOpt = ctx.Registry.GetAgent(state.ExecutorAgentId)

            match agentOpt with
            | None ->
                return
                    { TaskId = taskDef.Id
                      TaskGoal = taskDef.Goal
                      ExecutorId = state.ExecutorAgentId
                      Success = false
                      Output = "Executor Agent not found in Kernel"
                      ExecutionTrace = []
                      Duration = TimeSpan.Zero
                      Evaluation = None }
            | Some executor ->
                // Log: Curriculum → Request → Executor
                let requestMsg =
                    SpeechActBridge.requestTask state.CurriculumAgentId state.ExecutorAgentId taskDef

                SpeechActBridge.logSpeechAct ctx.Logger requestMsg

                // 2. Initialize Graph Executor
                let graphExecutor =
                    GraphExecutor(ctx.Registry, ctx.Llm, ctx.Budget, ctx.OutputGuard, ctx.Logger)

                // 3. Construct the Task Prompt
                let! codeContext =
                    match ctx.Epistemic with
                    | Some governor -> governor.GetRelatedCodeContext(taskDef.Goal)
                    | None -> Task.FromResult ""

                // 3.1 Retrieve Semantic Memory (Lessons Learned)
                let! memories =
                    match ctx.SemanticMemory with
                    | Some smem ->
                        let query =
                            { TaskId = ""
                              TaskKind = "coding"
                              TextContext = taskDef.Goal
                              Tags = taskDef.Constraints }

                        smem.Retrieve query |> Async.StartAsTask
                    | None -> Task.FromResult []

                let memoryContext =
                    if memories.IsEmpty then
                        ""
                    else
                        let summaries =
                            memories
                            |> List.map (fun m ->
                                let summary =
                                    m.Logical
                                    |> Option.map (fun l -> l.ProblemSummary)
                                    |> Option.defaultValue "Unknown Task"

                                let outcome =
                                    m.Logical
                                    |> Option.map (fun l -> l.OutcomeLabel)
                                    |> Option.defaultValue "unknown"

                                $"- [%s{outcome}] %s{summary}")
                            |> String.concat "\n"

                        $"\nLessons Learned from Past Episodes:\n%s{summaries}\n"

                if not (String.IsNullOrWhiteSpace codeContext) then
                    ctx.Logger($"[Context] Retrieved context for goal '{taskDef.Goal}':\n{codeContext}")

                if not memories.IsEmpty then
                    ctx.Logger($"[Memory] Retrieved {memories.Length} past experiences.")

                let goalLower = taskDef.Goal.ToLowerInvariant()

                let relevantBeliefs =
                    match ctx.Ledger with
                    | Some ledger ->
                        ledger.Query()
                        |> Seq.filter (fun b ->
                            let subj = b.Subject.Value.ToLowerInvariant()
                            let obj = b.Object.Value.ToLowerInvariant()
                            goalLower.Contains(subj) || goalLower.Contains(obj))
                        |> Seq.sortByDescending (fun b -> b.Confidence)
                        |> Seq.truncate 5
                        |> Seq.toList
                    | None -> []

                if not relevantBeliefs.IsEmpty then
                    ctx.Logger($"[Ledger] Retrieved {relevantBeliefs.Length} relevant beliefs.")

                let allowContradictions =
                    taskDef.Constraints
                    |> List.exists (fun c -> c.Equals("allow_contradictions", StringComparison.OrdinalIgnoreCase))

                let requiresContradictionGate =
                    taskDef.Constraints
                    |> List.exists (fun c ->
                        c.Equals("check_contradictions", StringComparison.OrdinalIgnoreCase)
                        || c.Equals("ledger_gate", StringComparison.OrdinalIgnoreCase)
                        || c.Equals("enforce_ledger", StringComparison.OrdinalIgnoreCase))

                let shouldGate =
                    not allowContradictions
                    && requiresContradictionGate
                    && not (List.isEmpty relevantBeliefs)

                let! contradictionReason =
                    if shouldGate then
                        evaluateContradiction ctx taskDef.Goal relevantBeliefs
                    else
                        Task.FromResult None

                match contradictionReason with
                | Some reason ->
                    return
                        { TaskId = taskDef.Id
                          TaskGoal = taskDef.Goal
                          ExecutorId = state.ExecutorAgentId
                          Success = false
                          Output = $"Blocked by ledger contradiction policy: {reason}"
                          ExecutionTrace = [ "LEDGER_CONTRADICTION" ]
                          Duration = TimeSpan.Zero
                          Evaluation = None }
                | None ->

                    let ledgerContext =
                        if relevantBeliefs.IsEmpty then
                            ""
                        else
                            let lines =
                                relevantBeliefs
                                |> List.map (fun b ->
                                    let predicate =
                                        match b.Predicate with
                                        | RelationType.Custom p -> p
                                        | _ -> b.Predicate.ToString()

                                    $"- [{b.Confidence:F2}] {b.Subject.Value} {predicate} {b.Object.Value}")
                                |> String.concat "\n"

                            $"\nKnown Beliefs:\n{lines}\n"

                    let taskPrompt =
                        $"""Task Goal: %s{taskDef.Goal}
    Constraints: %A{taskDef.Constraints}
    Validation Criteria: %s{taskDef.ValidationCriteria}

    %s{codeContext}
    %s{memoryContext}
    %s{ledgerContext}

    Please solve this task. Output your solution code or answer."""

                    // Pre-LLM Pipeline Check
                    let! (finalPrompt, isSafe) =
                        match ctx.PreLlm with
                        | Some pipeline ->
                            task {
                                let! pCtx = pipeline.ExecuteAsync(taskPrompt)

                                if not pCtx.IsSafe then
                                    return (pCtx.BlockReason |> Option.defaultValue "Unsafe", false)
                                else
                                    return (pCtx.CurrentPrompt, true)
                            }
                        | None -> Task.FromResult(taskPrompt, true)

                    if not isSafe then
                        return
                            { TaskId = taskDef.Id
                              TaskGoal = taskDef.Goal
                              ExecutorId = state.ExecutorAgentId
                              Success = false
                              Output = $"Blocked by Safety Filter: {finalPrompt}"
                              ExecutionTrace = []
                              Duration = TimeSpan.Zero
                              Evaluation = None }
                    else
                        let msg =
                            { Id = Guid.NewGuid()
                              CorrelationId = CorrelationId(Guid.NewGuid())
                              Sender = MessageEndpoint.System // System assigns the task
                              Receiver = Some(MessageEndpoint.Agent executor.Id)
                              Performative = Performative.Request
                              Intent = Some AgentDomain.Coding
                              Constraints = SemanticConstraints.Default
                              Ontology = None
                              Language = "text"
                              Content = finalPrompt
                              Timestamp = DateTime.UtcNow
                              Metadata = Map.empty }

                        // 4. Send message to Executor - show semantic message
                        ctx.ShowSemanticMessage msg ctx.Verbose
                        DemoVisualization.showTaskStart taskDef.Goal taskDef.Constraints

                        let agentWithMsg = executor.ReceiveMessage(msg)
                        use cts = new CancellationTokenSource()

                        if taskDef.Timeout > TimeSpan.Zero then
                            cts.CancelAfter(taskDef.Timeout)

                        let deadline =
                            if taskDef.Timeout > TimeSpan.Zero then
                                Some(DateTime.UtcNow + taskDef.Timeout)
                            else
                                None

                        let remaining () =
                            deadline |> Option.map (fun endTime -> endTime - DateTime.UtcNow)

                        let runWithTimeout label (work: Task<'T>) =
                            task {
                                match remaining () with
                                | Some r when r <= TimeSpan.Zero ->
                                    cts.Cancel()
                                    return Choice2Of2($"{label} timeout expired")
                                | Some r ->
                                    let timeoutTask = Task.Delay(r)
                                    let! completed = Task.WhenAny(work, timeoutTask)

                                    if Object.ReferenceEquals(completed, timeoutTask) then
                                        cts.Cancel()
                                        return Choice2Of2($"{label} timed out after {r.TotalSeconds:F1}s")
                                    else
                                        let! result = work
                                        return Choice1Of2 result
                                | None ->
                                    let! result = work
                                    return Choice1Of2 result
                            }

                        // 5. Run Initial Execution
                        let! outcomeResult =
                            runWithTimeout
                                "Execution"
                                (graphExecutor.RunAgentLoop(agentWithMsg, 20, cancellationToken = cts.Token))

                        match outcomeResult with
                        | Choice2Of2 reason ->
                            return
                                { TaskId = taskDef.Id
                                  TaskGoal = taskDef.Goal
                                  ExecutorId = state.ExecutorAgentId
                                  Success = false
                                  Output = $"Task timed out: {reason}"
                                  ExecutionTrace = [ "TIMEOUT" ]
                                  Duration = taskDef.Timeout
                                  Evaluation = None }
                        | Choice1Of2 outcome ->

                            // Phase 6.2: Semantic Speech Act Validation
                            let (success, output, trace) =
                                match outcome with
                                | Success(_, o, t) -> (true, o, t)
                                | PartialSuccess((_, o, t), _) -> (true, o, t)
                                | Failure err -> (false, String.concat "; " (err |> List.map (fun e -> $"%A{e}")), [])

                            let replyIssue =
                                if success then
                                    let requestMsg = SpeechActs.fromSemantic msg

                                    let intent, content =
                                        match SpeechActs.tryParse output with
                                        | Some(i, c) -> i, c
                                        | None -> Tell output, output // Fallback for legacy outputs

                                    let replyMsg = SpeechActs.createReply requestMsg intent content executor.Id

                                    match SpeechActs.validateFlow requestMsg replyMsg with
                                    | Result.Ok() ->
                                        ctx.Logger
                                            $"[Protocol] Verified semantic flow: %A{requestMsg.Intent} -> %A{replyMsg.Intent}"
                                    | Result.Error err ->
                                        ctx.Logger $"[Protocol] WARNING: Protocol violation: %s{err}"

                                    let issue =
                                        match intent with
                                        | AgentIntent.Tell _ when looksLikeFollowUpRequest content ->
                                            Some "Agent requested additional input instead of completing the task."
                                        | AgentIntent.Tell _ -> None
                                        | AgentIntent.Ask _ -> Some "Agent asked a follow-up question instead of answering."
                                        | AgentIntent.Error _ -> Some "Agent returned an error response."
                                        | AgentIntent.Propose _ -> Some "Agent proposed a plan instead of providing a result."
                                        | AgentIntent.Accept _ -> Some "Agent accepted a plan instead of providing a result."
                                        | AgentIntent.Reject _ -> Some "Agent rejected a plan instead of providing a result."
                                        | AgentIntent.Act _ -> Some "Agent returned an action instead of a result."
                                        | AgentIntent.Event _ -> Some "Agent returned an event instead of a result."

                                    issue |> Option.map (fun msg -> (msg, content))
                                else
                                    None

                            match replyIssue with
                            | Some(issue, content) ->
                                let issueOutput =
                                    if String.IsNullOrWhiteSpace content then
                                        issue
                                    else
                                        issue + "\n" + content

                                return
                                    { TaskId = taskDef.Id
                                      TaskGoal = taskDef.Goal
                                      ExecutorId = state.ExecutorAgentId
                                      Success = false
                                      Output = issueOutput
                                      ExecutionTrace = trace @ [ "PROTOCOL_VIOLATION" ]
                                      Duration = TimeSpan.FromSeconds(5.0)
                                      Evaluation = None }
                            | None ->
                                let (agentAfterExec, _, output, trace) =
                                    match outcome with
                                    | Success(a, o, t) -> (a, true, o, t)
                                    | PartialSuccess((a, o, t), _) -> (a, true, o, t)
                                    | Failure err ->
                                        (agentWithMsg, false, String.concat "; " (err |> List.map (fun e -> $"%A{e}")), [])

                                if not success then
                                    return
                                        { TaskId = taskDef.Id
                                          TaskGoal = taskDef.Goal
                                          ExecutorId = state.ExecutorAgentId
                                          Success = false
                                          Output = output
                                          ExecutionTrace = trace
                                          Duration = TimeSpan.FromSeconds(5.0)
                                          Evaluation = None }
                                else
                                // Handle Speech Act prefix in output using new helper
                                let mutable currentOutput =
                                    match SpeechActs.tryParse output with
                                    | Some(_, c) -> c
                                    | None -> output

                                let mutable currentTrace = trace
                                let mutable reflectionCount = 0
                                let mutable isOptimal = false
                                let mutable currentAgent = agentAfterExec
                                let mutable timeoutOccurred = false
                                let maxReflections = 3

                                while reflectionCount < maxReflections && not isOptimal do
                                    reflectionCount <- reflectionCount + 1

                                    match remaining () with
                                    | Some r when r <= TimeSpan.Zero ->
                                        timeoutOccurred <- true
                                        currentTrace <- currentTrace @ [ "TIMEOUT before reflection" ]
                                        reflectionCount <- maxReflections
                                    | _ -> ()

                                    if not timeoutOccurred then
                                        // 6.1 Epistemic Verification (if available)
                                        let! (verificationFeedback, isVerified) =
                                            task {
                                                match ctx.Epistemic with
                                                | Some governor ->
                                                    try
                                                        // Generate minimal variants for quick check
                                                        let! variants = governor.GenerateVariants(taskDef.Goal, 1)

                                                        let! result =
                                                            governor.VerifyGeneralization(
                                                                taskDef.Goal,
                                                                currentOutput,
                                                                variants
                                                            )

                                                        return (result.Feedback, result.IsVerified)
                                                    with ex ->
                                                        return ($"Verification failed: {ex.Message}", false)
                                                | None -> return ("", false)
                                            }

                                        if isVerified then
                                            isOptimal <- true
                                            currentTrace <- currentTrace @ [ $"--- VERIFIED by Epistemic Governor ---" ]
                                        else
                                            // 6.2 Construct Reflection Prompt
                                            // Truncate output to prevent HTTP 400 errors from excessive length
                                            let maxOutputLength = 2000

                                            let truncatedOutput =
                                                if currentOutput.Length > maxOutputLength then
                                                    currentOutput.Substring(0, maxOutputLength)
                                                    + "\n... [output truncated]"
                                                else
                                                    currentOutput

                                            let reflectionPrompt =
                                                if String.IsNullOrEmpty verificationFeedback then
                                                    // Standard Reflection
                                                    $"""You have generated a solution.
    Current Output:
    %s{truncatedOutput}

    Please reflect on this solution.
    1. Identify any potential bugs or inefficiencies.
    2. Verify if it meets all constraints: %A{taskDef.Constraints}
    3. If you can improve it, output the IMPROVED solution.
    4. If it is already optimal, output "OPTIMAL"."""
                                                else
                                                    // Epistemic Feedback Reflection
                                                    $"""Your solution failed verification.
    Current Output:
    %s{truncatedOutput}

    Feedback from Epistemic Governor:
    %s{verificationFeedback}

    Please fix the solution based on this feedback. Output the IMPROVED solution."""

                                            let reflectionMsg =
                                                { Id = Guid.NewGuid()
                                                  CorrelationId = CorrelationId(Guid.NewGuid())
                                                  Sender = MessageEndpoint.System
                                                  Receiver = Some(MessageEndpoint.Agent executor.Id)
                                                  Performative = Performative.Request
                                                  Intent = Some AgentDomain.Reasoning
                                                  Constraints = SemanticConstraints.Default
                                                  Ontology = None
                                                  Language = "text"
                                                  Content = reflectionPrompt
                                                  Timestamp = DateTime.UtcNow
                                                  Metadata = Map.empty }

                                            // Show reflection visualization
                                            DemoVisualization.showReflection
                                                reflectionCount
                                                maxReflections
                                                (Some verificationFeedback)

                                            // Phase 6.8: Epistemic Reflection Recording
                                            match ctx.EpisodeService with
                                            | Some svc ->
                                                let episode =
                                                    Tars.Core.Episode.Reflection(
                                                        state.ExecutorAgentId.ToString(),
                                                        reflectionPrompt,
                                                        DateTime.UtcNow
                                                    )

                                                svc.Queue(episode)
                                            | None -> ()

                                            let agentWithReflection = currentAgent.ReceiveMessage(reflectionMsg)

                                            let! reflectOutcomeResult =
                                                runWithTimeout
                                                    "Reflection"
                                                    (graphExecutor.RunAgentLoop(
                                                        agentWithReflection,
                                                        20,
                                                        cancellationToken = cts.Token
                                                    ))

                                            match reflectOutcomeResult with
                                            | Choice2Of2 reason ->
                                                timeoutOccurred <- true

                                                currentTrace <-
                                                    currentTrace @ [ $"TIMEOUT during reflection: {reason}" ]

                                                reflectionCount <- maxReflections
                                                currentOutput <- currentOutput + "\n[TIMEOUT during reflection]"
                                            | Choice1Of2 reflectOutcome ->
                                                match reflectOutcome with
                                                | Success(nextAgent, reflectOutput, reflectTrace) ->
                                                    currentAgent <- nextAgent

                                                    currentTrace <-
                                                        currentTrace
                                                        @ [ $"--- REFLECTION {reflectionCount} ---" ]
                                                        @ reflectTrace

                                                    if
                                                        reflectOutput.Contains("OPTIMAL")
                                                        && String.IsNullOrEmpty verificationFeedback
                                                    then
                                                        isOptimal <- true
                                                    else
                                                        currentOutput <- reflectOutput
                                                | PartialSuccess((nextAgent, reflectOutput, reflectTrace), _) ->
                                                    currentAgent <- nextAgent

                                                    currentTrace <-
                                                        currentTrace
                                                        @ [ $"--- REFLECTION {reflectionCount} (Partial) ---" ]
                                                        @ reflectTrace

                                                    currentOutput <- reflectOutput
                                                | Failure err ->
                                                    // If reflection fails, stop and keep previous result
                                                    currentTrace <-
                                                        currentTrace
                                                        @ [ $"--- REFLECTION {reflectionCount} FAILED ---" ]
                                                    // Don't update output, just stop
                                                    reflectionCount <- maxReflections

                                return
                                    { TaskId = taskDef.Id
                                      TaskGoal = taskDef.Goal
                                      ExecutorId = state.ExecutorAgentId
                                      Success = not timeoutOccurred
                                      Output =
                                        if timeoutOccurred then
                                            currentOutput + "\n[TIMEOUT]"
                                        else
                                            currentOutput
                                      ExecutionTrace = currentTrace
                                      Duration = TimeSpan.FromSeconds(10.0 * float (reflectionCount + 1))
                                      Evaluation = None }
        }

    /// The main tick of the evolutionary loop
    let rec step (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            match state.CurrentTask with
            | Some taskDef ->
                // 2. Execution Phase: Attempt to solve
                let! result = executeTask ctx state taskDef

                let! evaluation =
                    match ctx.Evaluator with
                    | Some evaluator ->
                        task {
                            try
                                let! evaluated = evaluator.Evaluate(taskDef, result)
                                return Some evaluated
                            with ex ->
                                ctx.Logger($"[Evaluation] Failed: {ex.Message}")
                                return None
                        }
                    | None -> Task.FromResult None

                let resultWithEvaluation = { result with Evaluation = evaluation }

                let evaluationPassed =
                    evaluation
                    |> Option.map (fun e -> e.Passed)
                    |> Option.defaultValue result.Success

                // Update success flag based on BOTH execution and semantic evaluation
                // A task is only truly successful if it executed AND passed semantic validation
                let finalResult =
                    { resultWithEvaluation with
                        Success = result.Success && evaluationPassed }

                match ctx.Ledger with
                | Some ledger -> do! LedgerIngestion.recordTaskResult ledger ctx.RunId taskDef finalResult ctx.Logger
                | None -> ()

                let resultForDisplay =
                    match evaluation with
                    | Some e when not e.Passed ->
                        { finalResult with
                            Output = finalResult.Output + "\n[SEMANTIC EVAL FAILED] " + e.Summary }
                    | _ -> finalResult

                // 3. Evaluation Phase (semantic)
                if result.Success && evaluationPassed then
                    let mutable newBeliefs = state.ActiveBeliefs

                    // Epistemic Governor: Extract Principle
                    match ctx.Epistemic with
                    | Some governor ->
                        try
                            let! belief = governor.ExtractPrinciple(taskDef.Goal, finalResult.Output)

                            // Store belief in VectorStore (Buffered)
                            let! embedding = ctx.Llm.EmbedAsync belief.Statement

                            let payload =
                                Map
                                    [ "type", "belief"
                                      "statement", belief.Statement
                                      "context", belief.Context
                                      "confidence", string belief.Confidence
                                      "derived_from", string taskDef.Id ]

                            match ctx.MemoryBuffer with
                            | Some buffer ->
                                buffer.Accumulate(Belief("tars-beliefs", string belief.Id, embedding, payload))
                            | None ->
                                do! ctx.VectorStore.SaveAsync("tars-beliefs", string belief.Id, embedding, payload)

                            // Phase 6.8: Record Belief Update to Knowledge Graph
                            match ctx.EpisodeService with
                            | Some svc ->
                                let episode =
                                    Tars.Core.Episode.BeliefUpdate(
                                        "EpistemicGovernor",
                                        belief.Statement,
                                        belief.Confidence,
                                        DateTime.UtcNow
                                    )

                                svc.Queue(episode)
                            | None -> ()

                            // Update Active Beliefs (keep last 10)
                            newBeliefs <- (belief.Statement :: newBeliefs) |> List.truncate 10

                            match ctx.Ledger with
                            | Some ledger ->
                                do!
                                    LedgerIngestion.recordEpistemicBelief
                                        ledger
                                        ctx.RunId
                                        belief
                                        (Some taskDef.Id)
                                        ctx.Logger
                            | None -> ()
                        with ex ->
                            // Log error but continue
                            printfn $"Epistemic extraction failed: %s{ex.Message}"
                    | None -> ()

                    // Construct a trace object (simplified for now)
                    let trace: MemoryTrace =
                        { TaskId = string taskDef.Id
                          Variables =
                            Map
                                [ "output", box resultWithEvaluation.Output
                                  "trace", box resultWithEvaluation.ExecutionTrace ]
                          StepOutputs = Map.empty }

                    // Save to Semantic Memory (Grow)
                    match ctx.SemanticMemory with
                    | Some smem ->
                        try
                            let! schemaId = smem.Grow(trace, obj ())
                            ctx.Logger($"[Memory] Grew new memory schema: {schemaId}")
                        with ex ->
                            ctx.Logger($"[Memory] Failed to grow memory: {ex.Message}")
                    | None -> ()

                    // Save to Knowledge            // Ingest trace into KG
                    // Save to Knowledge            // Ingest trace into KG
                    match ctx.KnowledgeGraph with
                    | Some kg ->
                        try
                            let taskEntity =
                                TarsEntity.ConceptE
                                    { Name = $"Task: {taskDef.Goal}"
                                      Description = taskDef.Goal
                                      RelatedConcepts = [] }

                            let _ = kg.AddNode(taskEntity)

                            let resultEntity =
                                if resultWithEvaluation.Success then
                                    TarsEntity.ConceptE
                                        { Name = "Success"
                                          Description = "Task Success"
                                          RelatedConcepts = [] }
                                else
                                    TarsEntity.ConceptE
                                        { Name = "Failure"
                                          Description = "Task Failure"
                                          RelatedConcepts = [] }

                            let _ = kg.AddFact(TarsFact.DerivedFrom(resultEntity, taskEntity))

                            // Map code structure if available
                            match trace.Variables |> Map.tryFind "code_structure" with
                            | Some(:? CodeStructure as cs) ->
                                for m in cs.Modules do
                                    let modEntity =
                                        TarsEntity.CodeModuleE
                                            { Path = m
                                              Namespace = ""
                                              Dependencies = []
                                              Complexity = 0.0
                                              LineCount = 0 }

                                    let _ = kg.AddFact(TarsFact.BelongsTo(taskEntity, m))
                                    ()

                                for t in cs.Types do
                                    let typeEntity =
                                        TarsEntity.ConceptE
                                            { Name = t
                                              Description = "Type"
                                              RelatedConcepts = [] }

                                    let _ = kg.AddFact(TarsFact.DerivedFrom(typeEntity, taskEntity))
                                    ()

                                for f in cs.Functions do
                                    let funcEntity =
                                        TarsEntity.ConceptE
                                            { Name = f
                                              Description = "Function"
                                              RelatedConcepts = [] }

                                    let _ = kg.AddFact(TarsFact.DerivedFrom(funcEntity, taskEntity))
                                    ()
                            | _ -> ()

                            ctx.Logger($"[KnowledgeGraph] Ingested episode for task: {taskDef.Id}")
                        with ex ->
                            ctx.Logger($"[KnowledgeGraph] Failed to ingest episode: {ex.Message}")
                    | None -> ()

                    // Save to Legacy Memory (Backup)
                    try
                        let! embedding = ctx.Llm.EmbedAsync taskDef.Goal

                        let payload =
                            Map
                                [ "goal", taskDef.Goal
                                  "output", resultWithEvaluation.Output
                                  "generation", string state.Generation ]

                        match ctx.MemoryBuffer with
                        | Some buffer ->
                            buffer.Accumulate(Legacy("tars-evolution-memory", string taskDef.Id, embedding, payload))
                        | None ->
                            do!
                                ctx.VectorStore.SaveAsync(
                                    "tars-evolution-memory",
                                    string taskDef.Id,
                                    embedding,
                                    payload
                                )
                    with ex ->
                        printfn $"Failed to save to memory: %s{ex.Message}"

                    // Log: Executor → Inform/Failure → Curriculum
                    let responseMsg =
                        SpeechActBridge.informResult
                            state.ExecutorAgentId
                            state.CurriculumAgentId
                            taskDef.Id
                            resultWithEvaluation

                    SpeechActBridge.logSpeechAct ctx.Logger responseMsg

                    // Feature C: Epistemic Verification Checkpoint
                    let! isVerified =
                        match ctx.Epistemic, resultWithEvaluation.Success with
                        | Some governor, true ->
                            task {
                                try
                                    let statement =
                                        sprintf
                                            "Task '%s' was completed with output: %s"
                                            (taskDef.Goal.Substring(0, min 50 taskDef.Goal.Length))
                                            (resultWithEvaluation.Output.Substring(
                                                0,
                                                min 100 resultWithEvaluation.Output.Length
                                            ))

                                    let! verified = governor.Verify(statement)

                                    if not verified then
                                        ctx.Logger("[Epistemic] ⚠️ Output verification FAILED - possible quality issue")
                                    else
                                        ctx.Logger("[Epistemic] ✓ Output verified")

                                    return verified
                                with ex ->
                                    ctx.Logger($"[Epistemic] Verification skipped: {ex.Message}")
                                    return true // Skip on error, don't block
                            }
                        | _ -> Task.FromResult(true)

                    // Adjust result based on verification (add metadata)
                    let verifiedResult =
                        let baseResult = resultForDisplay

                        if isVerified then
                            baseResult
                        else
                            { baseResult with
                                Output = baseResult.Output + "\n[UNVERIFIED - Review Recommended]" }

                    // Display task completion with generated solution
                    DemoVisualization.showTaskComplete
                        verifiedResult.Success
                        verifiedResult.Output
                        verifiedResult.Duration

                    // Capture episode to Graphiti knowledge graph
                    match ctx.EpisodeService with
                    | Some svc ->
                        let episode =
                            Tars.Core.Episode.AgentInteraction(
                                "Evolution",
                                taskDef.Goal,
                                (if finalResult.Success then
                                     "SUCCESS: "
                                 else
                                     "FAILED: ")
                                + resultForDisplay.Output,
                                DateTime.UtcNow
                            )

                        svc.Queue(episode)
                        let! _ = svc.FlushAsync()
                        ()
                    | None -> ()

                    return
                        { state with
                            Generation = state.Generation + 1
                            CompletedTasks = resultForDisplay :: state.CompletedTasks
                            CurrentTask = None
                            ActiveBeliefs = newBeliefs }
                else
                    // Retry or fail? For now, just log and clear
                    DemoVisualization.showTaskComplete
                        resultForDisplay.Success
                        resultForDisplay.Output
                        resultForDisplay.Duration

                    return
                        { state with
                            CompletedTasks = resultForDisplay :: state.CompletedTasks
                            CurrentTask = None }

            | None ->
                // Check Queue first
                match state.TaskQueue with
                | nextTask :: remainingQueue ->
                    // Execute task immediately instead of just setting it
                    ctx.Logger
                        $"[Evolution] Picking task from queue: {nextTask.Goal.Substring(0, Math.Min(nextTask.Goal.Length, 50))}..."

                    let stateWithTask =
                        { state with
                            CurrentTask = Some nextTask
                            TaskQueue = remainingQueue }
                    // Recursively call step to execute the task
                    return! step ctx stateWithTask
                | [] ->
                    // 1. Curriculum Phase: Generate new tasks
                    let! newTasks = generateTask ctx state

                    match newTasks with
                    | first :: rest ->
                        ctx.Logger
                            $"[Evolution] Generated {newTasks.Length} tasks, executing first: {first.Goal.Substring(0, Math.Min(first.Goal.Length, 50))}..."

                        let stateWithTask =
                            { state with
                                CurrentTask = Some first
                                TaskQueue = rest }
                        // Recursively call step to execute the task
                        return! step ctx stateWithTask
                    | [] ->
                        ctx.Logger "[Evolution] No tasks generated"
                        return state
        }
