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
          KnowledgeGraph: obj option // Tars.Core.LegacyKnowledgeGraph.TemporalGraph option
          MemoryBuffer: BufferAgent<MemoryItem> option // Added Capacitor
          Logger: string -> unit
          Verbose: bool
          ShowSemanticMessage: Message -> bool -> unit }

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
                    task {
                        try
                            let recentOutputs =
                                state.CompletedTasks
                                |> List.truncate 5
                                |> List.map (fun t -> t.Output.Substring(0, Math.Min(t.Output.Length, 100)) + "...")

                            return! governor.SuggestCurriculum(recentOutputs, state.ActiveBeliefs)
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

            let prompt =
                sprintf
                    """IMPORTANT: You are generating F# CODING TASKS. Do NOT ask questions. Output ONLY JSON.

Generation: %d. Completed tasks: %d.

Create 3 concrete F# programming tasks. Each task must be:
- A specific coding problem (NOT a question)
- Solvable with code (NOT a discussion)
- Related to: %s

RESPOND WITH THIS EXACT JSON FORMAT (no other text):
{"tasks":[
  {"goal":"Write an F# function that reverses a list using recursion","constraints":["Pure functional","No mutable state"],"validation_criteria":"Returns reversed list"},
  {"goal":"Implement a binary search function in F#","constraints":["Handle empty lists","Return Option type"],"validation_criteria":"Finds element or returns None"},
  {"goal":"Create a function to calculate Fibonacci numbers","constraints":["Use tail recursion","Handle negative inputs"],"validation_criteria":"Returns correct Fibonacci number"}
]}"""
                    state.Generation
                    state.CompletedTasks.Length
                    (if String.IsNullOrWhiteSpace(guidance) then
                         "basic algorithms"
                     else
                         guidance.Substring(0, Math.Min(guidance.Length, 50)))

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

                // Show semantic message in demo mode
                ctx.ShowSemanticMessage msg ctx.Verbose
                let agentWithMsg = agent.ReceiveMessage(msg)

                // 4. Run Execution
                let! outcome = graphExecutor.RunAgentLoop agentWithMsg 20

                // Log the outcome for debugging
                match outcome with
                | Success(_, output, _) ->
                    ctx.Logger(
                        $"[Curriculum] Agent returned success: {output.Substring(0, Math.Min(output.Length, 100))}..."
                    )
                | PartialSuccess((_, output, _), errors) ->
                    ctx.Logger($"[Curriculum] Agent returned partial success with {errors.Length} errors")
                | Failure err ->
                    let errStr = err |> List.map string |> String.concat "; "
                    ctx.Logger(sprintf "[Curriculum] Agent returned failure: %s" errStr)

                let responseText =
                    match outcome with
                    | Success(_, output, _) -> output
                    | PartialSuccess((_, output, _), _) -> output
                    | Failure err -> ""

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
                    try
                        let json = responseText.Trim()
                        ctx.Logger($"[Curriculum] Raw response: {json.Substring(0, Math.Min(json.Length, 200))}...")

                        // Handle Speech Act prefix (ACT: <performative>: <content>)
                        // Strip any prefix before the JSON
                        let json =
                            // Look for the start of JSON (either { or [)
                            let jsonStart =
                                let braceIdx = json.IndexOf('{')
                                let bracketIdx = json.IndexOf('[')

                                match braceIdx, bracketIdx with
                                | -1, -1 -> -1
                                | -1, b -> b
                                | b, -1 -> b
                                | a, b -> Math.Min(a, b)

                            if jsonStart > 0 then json.Substring(jsonStart) else json

                        // More robust backtick removal
                        let json =
                            let mutable cleaned = json
                            // Remove opening ```json or ```
                            if cleaned.StartsWith("```json") then
                                cleaned <- cleaned.Substring(7)
                            elif cleaned.StartsWith("```") then
                                cleaned <- cleaned.Substring(3)

                            // Remove closing ``` (may be on its own line)
                            cleaned <- cleaned.TrimEnd()

                            if cleaned.EndsWith("```") then
                                cleaned <- cleaned.Substring(0, cleaned.Length - 3)

                            // Also handle case where ``` is followed by newline content
                            let lastBackticks = cleaned.LastIndexOf("\n```")

                            if lastBackticks > 0 then
                                cleaned <- cleaned.Substring(0, lastBackticks)

                            cleaned.Trim()

                        ctx.Logger($"[Curriculum] Cleaned JSON: {json.Substring(0, Math.Min(json.Length, 200))}...")
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
                        // Use practical, useful tasks as fallback
                        ctx.Logger($"[Curriculum] JSON parse failed: {ex.Message}")

                        let practicalTasks =
                            [| // Test Generation
                               "Review the DemoVisualization.fs module and write 3 unit tests for the showSemanticMessage function"
                               // Documentation
                               "Read the ToolFactory.fs file and add XML documentation comments to all public functions"
                               // Code Improvement
                               "Analyze the Engine.fs generateTask function and suggest 2 improvements for better error handling"
                               // Bug Finding
                               "Review the ResponseParser module in Graph.fs and identify any edge cases that might cause parsing failures"
                               // Refactoring
                               "Examine the showTaskComplete function in DemoVisualization.fs and refactor to reduce code duplication" |]

                        let random = Random()
                        let selectedTask = practicalTasks[random.Next(practicalTasks.Length)]
                        ctx.Logger($"[Curriculum] Assigned practical task: {selectedTask}")

                        return
                            [ { Id = Guid.NewGuid()
                                DifficultyLevel = state.Generation + 1
                                Goal = selectedTask
                                Constraints =
                                  [ "Work with the TARS v2 codebase"; "Use explore_project and read_code tools" ]
                                ValidationCriteria = "Provide concrete, actionable output"
                                Timeout = TimeSpan.FromMinutes(2.0)
                                Score = 1.0 } ]


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

                    // 4. Send message to Executor - show semantic message
                    ctx.ShowSemanticMessage msg ctx.Verbose
                    DemoVisualization.showTaskStart taskDef.Goal taskDef.Constraints

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
                        // Handle Speech Act prefix in output
                        let cleanOutput (text: string) =
                            if text.StartsWith("ACT:") then
                                let parts = text.Split(':', 3)
                                if parts.Length = 3 then parts[2].Trim() else text
                            else
                                text

                        let mutable currentOutput = cleanOutput output
                        let mutable currentTrace = trace
                        let mutable reflectionCount = 0
                        let mutable isOptimal = false
                        let mutable currentAgent = agentAfterExec
                        let maxReflections = 3

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

                                // Show reflection visualization
                                DemoVisualization.showReflection
                                    reflectionCount
                                    maxReflections
                                    (Some verificationFeedback)

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
    let rec step (ctx: EvolutionContext) (state: EvolutionState) =
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

                    // Save to Knowledge            // Ingest trace into KG
                    match ctx.KnowledgeGraph with
                    | Some kgObj ->
                        let kg = kgObj :?> Tars.Core.LegacyKnowledgeGraph.TemporalGraph

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

                    // Display task completion with generated solution
                    DemoVisualization.showTaskComplete result.Success result.Output result.Duration

                    return
                        { state with
                            Generation = state.Generation + 1
                            CompletedTasks = result :: state.CompletedTasks
                            CurrentTask = None
                            ActiveBeliefs = newBeliefs }
                else
                    // Retry or fail? For now, just log and clear
                    DemoVisualization.showTaskComplete result.Success result.Output result.Duration

                    return
                        { state with
                            CompletedTasks = result :: state.CompletedTasks
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
