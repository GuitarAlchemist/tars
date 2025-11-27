namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open System.Text.Json

module Engine =

    /// The context for the evolution engine
    type EvolutionContext =
        { Kernel: KernelContext
          Llm: ILlmService
          VectorStore: IVectorStore }

    /// Generates a new task using the Curriculum Agent
    let private generateTask (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            let prompt =
                sprintf
                    """You are a Curriculum Agent. Your goal is to generate a new coding task for an AI agent to solve.
The current generation is %d.
Previous completed tasks: %d.

Generate a JSON object with the following fields:
- goal: A clear description of the task (e.g., "Write a python script to calculate fibonacci")
- constraints: A list of strings (e.g., ["Must use recursion", "Max time 1s"])
- validation_criteria: A string describing how to verify success (e.g., "Output should be 55 for input 10")

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

                let goal = root.GetProperty("goal").GetString()

                let constraints =
                    root.GetProperty("constraints").EnumerateArray()
                    |> Seq.map (fun e -> e.GetString())
                    |> Seq.toList

                let criteria = root.GetProperty("validation_criteria").GetString()

                return
                    { Id = Guid.NewGuid()
                      DifficultyLevel = state.Generation + 1
                      Goal = goal
                      Constraints = constraints
                      ValidationCriteria = criteria
                      Timeout = TimeSpan.FromMinutes(1.0) }
            with ex ->
                return
                    { Id = Guid.NewGuid()
                      DifficultyLevel = state.Generation + 1
                      Goal = "Failed to parse task. Write a hello world script."
                      Constraints = []
                      ValidationCriteria = "Output 'Hello World'"
                      Timeout = TimeSpan.FromMinutes(1.0) }
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
                let graphCtx: Graph.GraphContext =
                    { Kernel = ctx.Kernel
                      Llm = ctx.Llm
                      MaxSteps = 20 }

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
                      Source = MessageEndpoint.System // System assigns the task
                      Target = MessageEndpoint.Agent executor.Id
                      Content = taskPrompt
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.empty }

                // 4. Send message to Executor
                let mutable currentAgent = Kernel.receiveMessage msg executor
                let mutable stepCount = 0
                let mutable finished = false
                let mutable trace = []

                // 5. Run the Graph Loop
                while not finished && stepCount < graphCtx.MaxSteps do
                    trace <- trace @ [ sprintf "Step %d: %A" stepCount currentAgent.State ]
                    let! next = Graph.step currentAgent graphCtx
                    currentAgent <- next
                    stepCount <- stepCount + 1

                    match currentAgent.State with
                    | WaitingForUser response ->
                        trace <- trace @ [ sprintf "Response: %s" response ]
                        finished <- true
                    | AgentState.Error err ->
                        trace <- trace @ [ sprintf "Error: %s" err ]
                        finished <- true
                    | _ -> ()

                // 6. Extract Result
                let (success, output) =
                    match currentAgent.State with
                    | WaitingForUser response -> (true, response)
                    | AgentState.Error err -> (false, err)
                    | _ -> (false, "Timeout or incomplete")

                return
                    { TaskId = taskDef.Id
                      ExecutorId = state.ExecutorAgentId
                      Success = success
                      Output = output
                      ExecutionTrace = trace
                      Duration = TimeSpan.FromSeconds(5.0) }
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
