namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Core.Knowledge
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open System.Text.Json
open Tars.Kernel
open Tars.Cortex

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
          KnowledgeGraph: LegacyKnowledgeGraph.TemporalGraph option
          MemoryBuffer: BufferAgent<MemoryItem> option // Added Capacitor
          Logger: string -> unit }

    let private scoreTask (task: TaskDefinition) : float =
        // Simple heuristic for now:
        // Score = 1.0 + (0.1 * float task.DifficultyLevel)
        // In future, this would compare embeddings with previous tasks to ensure novelty.
        1.0 + (0.1 * float task.DifficultyLevel)

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
                    let recentOutputs =
                        state.CompletedTasks
                        |> List.truncate 5
                        |> List.map (fun t -> t.Output.Substring(0, Math.Min(t.Output.Length, 100)) + "...")

                    governor.SuggestCurriculum(recentOutputs, state.ActiveBeliefs)
                | None -> Task.FromResult "Focus on basic coding tasks."

            let guidance =
                if isCritical then
                    suggestion + " WARNING: Budget is critical. Generate simpler, cheaper tasks."
                else
                    suggestion

            let prompt =
                sprintf
                    """You are a Curriculum Agent. Your goal is to generate a new coding task for an AI agent to solve.
The current generation is %d.
Previous completed tasks: %d.

Guidance from the Epistemic Governor:
"%s"

Generate a JSON object containing a list of 3 potential tasks under the key "tasks".
Each task should have:
- goal: A clear description of the task
- constraints: A list of strings
- validation_criteria: A string describing how to verify success

JSON:"""
                    state.Generation
                    state.CompletedTasks.Length
                    guidance

            // 1. Retrieve Curriculum Agent
            let! agentOpt = ctx.Registry.GetAgent(state.CurriculumAgentId)

            match agentOpt with
            | None -> return []
            | Some agent ->
                // 2. Initialize Graph Executor
                let graphExecutor =
                    GraphExecutor(ctx.Registry, ctx.Llm, ctx.Budget, ctx.OutputGuard, ctx.Logger)

                // 3. Create Request Message
                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Sender = MessageEndpoint.System
                      Receiver = Some(MessageEndpoint.Agent agent.Id)
                      Performative = Performative.Request
                      Intent = Some AgentIntent.Planning
                      Constraints = SemanticConstraints.Default
                      Ontology = None
                      Language = "text"
                      Content = prompt
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.empty }

                let agentWithMsg = agent.ReceiveMessage(msg)

                // 4. Run Execution
                let! outcome = graphExecutor.RunAgentLoop agentWithMsg 20

                let responseText =
                    match outcome with
                    | Success(_, output, _) -> output
                    | PartialSuccess((_, output, _), _) -> output
                    | Failure err -> ""

                if String.IsNullOrWhiteSpace(responseText) then
                    return []
                else
                    try
                        let json = responseText.Trim()

                        let json =
                            if json.StartsWith("```json") then
                                json.Substring(7, json.Length - 10).Trim()
                            elif json.StartsWith("```") then
                                json.Substring(3, json.Length - 6).Trim()
                            else
                                json

                        let doc = JsonDocument.Parse(json)
                        let root = doc.RootElement

                        let mutable tasksElem = Unchecked.defaultof<JsonElement>

                        let tasks =
                            if root.ValueKind = JsonValueKind.Array then
                                root.EnumerateArray() |> Seq.map id
                            elif
                                root.TryGetProperty("tasks", &tasksElem)
                                && tasksElem.ValueKind = JsonValueKind.Array
                            then
                                tasksElem.EnumerateArray() |> Seq.map id
                            else
                                Seq.empty

                        let parsedTasks =
                            tasks
                            |> Seq.map (fun t ->
                                let goal = t.GetProperty("goal").GetString()

                                let constraints =
                                    t.GetProperty("constraints").EnumerateArray()
                                    |> Seq.map (fun e -> e.GetString())
                                    |> Seq.toList

                                let criteria = t.GetProperty("validation_criteria").GetString()

                                let taskDef =
                                    { Id = Guid.NewGuid()
                                      DifficultyLevel = state.Generation + 1
                                      Goal = goal
                                      Constraints = constraints
                                      ValidationCriteria = criteria
                                      Timeout = TimeSpan.FromMinutes(1.0)
                                      Score = 0.0 }

                                let score = scoreTask taskDef
                                { taskDef with Score = score })
                            |> Seq.toList

                        // Select Top K (Fan-out Limiter)
                        let k = 3

                        let topK =
                            parsedTasks |> List.sortByDescending (fun t -> t.Score) |> List.truncate k

                        if topK.IsEmpty then
                            return
                                [ { Id = Guid.NewGuid()
                                    DifficultyLevel = state.Generation + 1
                                    Goal = "Failed to parse tasks. Write a hello world script."
                                    Constraints = []
                                    ValidationCriteria = "Output 'Hello World'"
                                    Timeout = TimeSpan.FromMinutes(1.0)
                                    Score = 0.0 } ]
                        else
                            return topK

                    with ex ->
                        return
                            [ { Id = Guid.NewGuid()
                                DifficultyLevel = state.Generation + 1
                                Goal = "Failed to parse task. Write a hello world script."
                                Constraints = []
                                ValidationCriteria = "Output 'Hello World'"
                                Timeout = TimeSpan.FromMinutes(1.0)
                                Score = 0.0 } ]


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
                      ExecutorId = state.ExecutorAgentId
                      Success = false
                      Output = "Executor Agent not found in Kernel"
                      ExecutionTrace = []
                      Duration = TimeSpan.Zero }
            | Some executor ->
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

                                sprintf "- [%s] %s" outcome summary)
                            |> String.concat "\n"

                        sprintf "\nLessons Learned from Past Episodes:\n%s\n" summaries

                if not (String.IsNullOrWhiteSpace codeContext) then
                    ctx.Logger($"[Context] Retrieved context for goal '{taskDef.Goal}':\n{codeContext}")

                if not memories.IsEmpty then
                    ctx.Logger($"[Memory] Retrieved {memories.Length} past experiences.")

                let taskPrompt =
                    sprintf
                        """Task Goal: %s
Constraints: %A
Validation Criteria: %s

%s
%s

Please solve this task. Output your solution code or answer."""
                        taskDef.Goal
                        taskDef.Constraints
                        taskDef.ValidationCriteria
                        codeContext
                        memoryContext

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
                          ExecutorId = state.ExecutorAgentId
                          Success = false
                          Output = $"Blocked by Safety Filter: {finalPrompt}"
                          ExecutionTrace = []
                          Duration = TimeSpan.Zero }
                else
                    let msg =
                        { Id = Guid.NewGuid()
                          CorrelationId = CorrelationId(Guid.NewGuid())
                          Sender = MessageEndpoint.System // System assigns the task
                          Receiver = Some(MessageEndpoint.Agent executor.Id)
                          Performative = Performative.Request
                          Intent = Some AgentIntent.Coding
                          Constraints = SemanticConstraints.Default
                          Ontology = None
                          Language = "text"
                          Content = finalPrompt
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty }

                    // 4. Send message to Executor
                    let agentWithMsg = executor.ReceiveMessage(msg)

                    // 5. Run Initial Execution
                    let! outcome = graphExecutor.RunAgentLoop agentWithMsg 20

                    let (agentAfterExec, success, output, trace) =
                        match outcome with
                        | Success(a, o, t) -> (a, true, o, t)
                        | PartialSuccess((a, o, t), _) -> (a, true, o, t)
                        | Failure err ->
                            (agentWithMsg, false, String.concat "; " (err |> List.map (fun e -> sprintf "%A" e)), [])

                    if not success then
                        return
                            { TaskId = taskDef.Id
                              ExecutorId = state.ExecutorAgentId
                              Success = false
                              Output = output
                              ExecutionTrace = trace
                              Duration = TimeSpan.FromSeconds(5.0) }
                    else
                        // 6. Adaptive Reflection Loop (Phase 6.4)
                        let maxReflections = 3
                        let mutable currentOutput = output
                        let mutable currentTrace = trace
                        let mutable reflectionCount = 0
                        let mutable isOptimal = false
                        let mutable currentAgent = agentAfterExec

                        while reflectionCount < maxReflections && not isOptimal do
                            reflectionCount <- reflectionCount + 1

                            // 6.1 Epistemic Verification (if available)
                            let! (verificationFeedback, isVerified) =
                                task {
                                    match ctx.Epistemic with
                                    | Some governor ->
                                        try
                                            // Generate minimal variants for quick check
                                            let! variants = governor.GenerateVariants(taskDef.Goal, 1)

                                            let! result =
                                                governor.VerifyGeneralization(taskDef.Goal, currentOutput, variants)

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
                                let reflectionPrompt =
                                    if String.IsNullOrEmpty verificationFeedback then
                                        // Standard Reflection
                                        sprintf
                                            """You have generated a solution.
Current Output:
%s

Please reflect on this solution.
1. Identify any potential bugs or inefficiencies.
2. Verify if it meets all constraints: %A
3. If you can improve it, output the IMPROVED solution.
4. If it is already optimal, output "OPTIMAL"."""
                                            currentOutput
                                            taskDef.Constraints
                                    else
                                        // Epistemic Feedback Reflection
                                        sprintf
                                            """Your solution failed verification.
Current Output:
%s

Feedback from Epistemic Governor:
%s

Please fix the solution based on this feedback. Output the IMPROVED solution."""
                                            currentOutput
                                            verificationFeedback

                                let reflectionMsg =
                                    { Id = Guid.NewGuid()
                                      CorrelationId = CorrelationId(Guid.NewGuid())
                                      Sender = MessageEndpoint.System
                                      Receiver = Some(MessageEndpoint.Agent executor.Id)
                                      Performative = Performative.Request
                                      Intent = Some AgentIntent.Reasoning
                                      Constraints = SemanticConstraints.Default
                                      Ontology = None
                                      Language = "text"
                                      Content = reflectionPrompt
                                      Timestamp = DateTime.UtcNow
                                      Metadata = Map.empty }

                                let agentWithReflection = currentAgent.ReceiveMessage(reflectionMsg)
                                let! reflectOutcome = graphExecutor.RunAgentLoop agentWithReflection 20

                                match reflectOutcome with
                                | Success(nextAgent, reflectOutput, reflectTrace) ->
                                    currentAgent <- nextAgent

                                    currentTrace <-
                                        currentTrace @ [ $"--- REFLECTION {reflectionCount} ---" ] @ reflectTrace

                                    if
                                        reflectOutput.Contains("OPTIMAL") && String.IsNullOrEmpty verificationFeedback
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
                                    currentTrace <- currentTrace @ [ $"--- REFLECTION {reflectionCount} FAILED ---" ]
                                    // Don't update output, just stop
                                    reflectionCount <- maxReflections

                        return
                            { TaskId = taskDef.Id
                              ExecutorId = state.ExecutorAgentId
                              Success = true
                              Output = currentOutput
                              ExecutionTrace = currentTrace
                              Duration = TimeSpan.FromSeconds(10.0 * float (reflectionCount + 1)) }
        }

    /// The main tick of the evolutionary loop
    let step (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            match state.CurrentTask with
            | Some taskDef ->
                // 2. Execution Phase: Attempt to solve
                let! result = executeTask ctx state taskDef

                // 3. Evaluation Phase (simplified)
                if result.Success then
                    let mutable newBeliefs = state.ActiveBeliefs

                    // Epistemic Governor: Extract Principle
                    match ctx.Epistemic with
                    | Some governor ->
                        try
                            let! belief = governor.ExtractPrinciple(taskDef.Goal, result.Output)

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

                            // Update Active Beliefs (keep last 10)
                            newBeliefs <- (belief.Statement :: newBeliefs) |> List.truncate 10
                        with ex ->
                            // Log error but continue
                            printfn "Epistemic extraction failed: %s" ex.Message
                    | None -> ()

                    // Construct a trace object (simplified for now)
                    let trace: MemoryTrace =
                        { TaskId = string taskDef.Id
                          Variables = Map [ "output", box result.Output; "trace", box result.ExecutionTrace ]
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

                    // Save to Knowledge Graph (Ingest Episode)
                    match ctx.KnowledgeGraph with
                    | Some kg ->
                        try
                            kg.IngestEpisode(trace)
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
                                  "output", result.Output
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
                        printfn "Failed to save to memory: %s" ex.Message

                    return
                        { state with
                            Generation = state.Generation + 1
                            CompletedTasks = result :: state.CompletedTasks
                            CurrentTask = None
                            ActiveBeliefs = newBeliefs }
                else
                    // Retry or fail? For now, just log and clear
                    return
                        { state with
                            CompletedTasks = result :: state.CompletedTasks
                            CurrentTask = None }

            | None ->
                // Check Queue first
                match state.TaskQueue with
                | nextTask :: remainingQueue ->
                    return
                        { state with
                            CurrentTask = Some nextTask
                            TaskQueue = remainingQueue }
                | [] ->
                    // 1. Curriculum Phase: Generate new tasks
                    let! newTasks = generateTask ctx state

                    match newTasks with
                    | first :: rest ->
                        return
                            { state with
                                CurrentTask = Some first
                                TaskQueue = rest }
                    | [] -> return state
        }
