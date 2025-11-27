namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open System.Text.Json
open Tars.Kernel

module Engine =

    /// The context for the evolution engine
    type EvolutionContext =
        { Kernel: KernelContext
          Llm: ILlmService
          VectorStore: IVectorStore }

    let private scoreTask (task: TaskDefinition) : float =
        // Simple heuristic for now:
        // Score = 1.0 + (0.1 * float task.DifficultyLevel)
        // In future, this would compare embeddings with previous tasks to ensure novelty.
        1.0 + (0.1 * float task.DifficultyLevel)

    /// Generates a new task using the Curriculum Agent
    let private generateTask (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            let prompt =
                sprintf
                    """You are a Curriculum Agent. Your goal is to generate a new coding task for an AI agent to solve.
The current generation is %d.
Previous completed tasks: %d.

Generate a JSON object containing a list of 3 potential tasks under the key "tasks".
Each task should have:
- goal: A clear description of the task
- constraints: A list of strings
- validation_criteria: A string describing how to verify success

JSON:"""
                    state.Generation
                    state.CompletedTasks.Length

            let req =
                { ModelHint = Some "code" // Use smart model for curriculum
                  MaxTokens = None // Some 500
                  Temperature = None // Some 0.7
                  Messages = [ { Role = Role.User; Content = prompt } ] }

            let! response = ctx.Llm.CompleteAsync req

            try
                let json = response.Text.Trim()

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

                // Select Top 1 (Fan-out Limiter)
                match parsedTasks |> List.sortByDescending (fun t -> t.Score) with
                | best :: _ -> return best
                | [] ->
                    return
                        { Id = Guid.NewGuid()
                          DifficultyLevel = state.Generation + 1
                          Goal = "Failed to parse tasks. Write a hello world script."
                          Constraints = []
                          ValidationCriteria = "Output 'Hello World'"
                          Timeout = TimeSpan.FromMinutes(1.0)
                          Score = 0.0 }

            with ex ->
                return
                    { Id = Guid.NewGuid()
                      DifficultyLevel = state.Generation + 1
                      Goal = "Failed to parse task. Write a hello world script."
                      Constraints = []
                      ValidationCriteria = "Output 'Hello World'"
                      Timeout = TimeSpan.FromMinutes(1.0)
                      Score = 0.0 }
        }

    /// Helper to run the agent loop until it produces a response or errors
    let private runAgentLoop (agent: Agent) (graphCtx: GraphRuntime.GraphContext) =
        task {
            let mutable currentAgent = agent
            let mutable stepCount = 0
            let mutable finished = false
            let mutable trace = []
            let mutable resultOutput = ""
            let mutable success = false

            while not finished && stepCount < graphCtx.MaxSteps do
                trace <- trace @ [ sprintf "Step %d: %A" stepCount currentAgent.State ]
                let! next = GraphRuntime.step currentAgent graphCtx
                currentAgent <- next
                stepCount <- stepCount + 1

                match currentAgent.State with
                | WaitingForUser response ->
                    trace <- trace @ [ sprintf "Response: %s" response ]
                    resultOutput <- response
                    success <- true
                    finished <- true
                | AgentState.Error err ->
                    trace <- trace @ [ sprintf "Error: %s" err ]
                    resultOutput <- err
                    success <- false
                    finished <- true
                | _ -> ()

            if not finished then
                resultOutput <- "Timeout or incomplete"
                success <- false

            return (currentAgent, success, resultOutput, trace)
        }

    /// Attempts to solve a task using the Executor Agent
    let private executeTask (ctx: EvolutionContext) (state: EvolutionState) (taskDef: TaskDefinition) =
        task {
            // 1. Retrieve Executor Agent
            match Kernel.getAgent state.ExecutorAgentId ctx.Kernel with
            | None ->
                return
                    { TaskId = taskDef.Id
                      ExecutorId = state.ExecutorAgentId
                      Success = false
                      Output = "Executor Agent not found in Kernel"
                      ExecutionTrace = []
                      Duration = TimeSpan.Zero }
            | Some executor ->
                // 2. Initialize Graph Context for the task
                let graphCtx: GraphRuntime.GraphContext =
                    { Kernel = ctx.Kernel
                      Llm = ctx.Llm
                      MaxSteps = 20
                      BudgetGovernor = Some(BudgetGovernor(100000)) }

                // 3. Construct the Task Prompt
                let taskPrompt =
                    sprintf
                        """Task Goal: %s
Constraints: %A
Validation Criteria: %s

Please solve this task. Output your solution code or answer."""
                        taskDef.Goal
                        taskDef.Constraints
                        taskDef.ValidationCriteria

                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Sender = MessageEndpoint.System // System assigns the task
                      Receiver = Some(MessageEndpoint.Agent executor.Id)
                      Performative = Performative.Request
                      Constraints = SemanticConstraints.Default
                      Content = taskPrompt
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.empty }

                // 4. Send message to Executor
                let agentWithMsg = Kernel.receiveMessage msg executor

                // 5. Run Initial Execution
                let! (agentAfterExec, success, output, trace) = runAgentLoop agentWithMsg graphCtx

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
                    // We will try to reflect once for now.
                    let reflectionPrompt =
                        sprintf
                            """You have generated a solution.
Previous Output:
%s

Please reflect on this solution.
1. Identify any potential bugs or inefficiencies.
2. Verify if it meets all constraints: %A
3. If you can improve it, output the IMPROVED solution.
4. If it is already optimal, output "OPTIMAL"."""
                            output
                            taskDef.Constraints

                    let reflectionMsg =
                        { Id = Guid.NewGuid()
                          CorrelationId = CorrelationId(Guid.NewGuid())
                          Sender = MessageEndpoint.System
                          Receiver = Some(MessageEndpoint.Agent executor.Id)
                          Performative = Performative.Request
                          Constraints = SemanticConstraints.Default
                          Content = reflectionPrompt
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty }

                    let agentWithReflection = Kernel.receiveMessage reflectionMsg agentAfterExec

                    let! (agentAfterReflect, reflectSuccess, reflectOutput, reflectTrace) =
                        runAgentLoop agentWithReflection graphCtx

                    let finalOutput =
                        if reflectSuccess && not (reflectOutput.Contains("OPTIMAL")) then
                            reflectOutput
                        else
                            output

                    let fullTrace = trace @ [ "--- REFLECTION ---" ] @ reflectTrace

                    return
                        { TaskId = taskDef.Id
                          ExecutorId = state.ExecutorAgentId
                          Success = true
                          Output = finalOutput
                          ExecutionTrace = fullTrace
                          Duration = TimeSpan.FromSeconds(10.0) }
        }

    /// The main tick of the evolutionary loop
    let step (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            match state.CurrentTask with
            | None ->
                // 1. Curriculum Phase: Generate new task
                let! newTask = generateTask ctx state

                return
                    { state with
                        CurrentTask = Some newTask }

            | Some taskDef ->
                // 2. Execution Phase: Attempt to solve
                let! result = executeTask ctx state taskDef

                // 3. Evaluation Phase (simplified)
                if result.Success then
                    // Save to Memory
                    try
                        let! embedding = ctx.Llm.EmbedAsync taskDef.Goal

                        let payload =
                            Map
                                [ "goal", taskDef.Goal
                                  "output", result.Output
                                  "generation", string state.Generation ]

                        do! ctx.VectorStore.SaveAsync("tars-evolution-memory", string taskDef.Id, embedding, payload)
                    with ex ->
                        // Log error but continue
                        printfn "Failed to save to memory: %s" ex.Message

                    return
                        { state with
                            Generation = state.Generation + 1
                            CompletedTasks = result :: state.CompletedTasks
                            CurrentTask = None }
                else
                    // Retry or fail? For now, just log and clear
                    return
                        { state with
                            CompletedTasks = result :: state.CompletedTasks
                            CurrentTask = None } // Move to next task anyway for this demo
        }
