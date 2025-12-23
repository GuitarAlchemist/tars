namespace Tars.Cortex

open System
open System.Text.Json
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService
open System.Text

/// <summary>
/// Agentic Patterns: Composable reasoning patterns for autonomous agents.
/// Implements Chain of Thought, ReAct, and Plan & Execute patterns.
/// </summary>
module Patterns =

    // =========================================================================
    // Types for ReAct Pattern
    // =========================================================================

    /// Represents a single step in the ReAct loop
    type ReActStep =
        { Thought: string
          Action: string
          ActionInput: string
          Observation: string option }

    /// The result of parsing an LLM response in ReAct format
    type ReActParse =
        | Continue of thought: string * action: string * actionInput: string
        | Finish of thought: string * finalAnswer: string
        | ParseError of raw: string

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Builds the ReAct system prompt with available tools
    let private buildReActSystemPrompt (tools: Tool list) =
        let toolDescs =
            tools
            |> List.map (fun t -> $"- {t.Name}: {t.Description}")
            |> String.concat "\n"

        "You are a ReAct agent. You solve problems by thinking step-by-step and using tools.\n\n"
        + "Available Tools:\n"
        + toolDescs
        + "\n"
        + "- Finish: Use this when you have the final answer. Input is your final response.\n\n"
        + "For each step, respond in EXACTLY this format:\n"
        + "Thought: [Your reasoning about what to do next]\n"
        + "Action: [Tool name - exactly as listed above]\n"
        + "Action Input: [The input to pass to the tool]\n\n"
        + "When you have the final answer, use:\n"
        + "Thought: [Your final reasoning]\n"
        + "Action: Finish\n"
        + "Action Input: [Your final answer to the user's question]\n\n"
        + "Important:\n"
        + "- Always start with a Thought\n"
        + "- Use exactly one Action per response\n"
        + "- Wait for the Observation before continuing"

    /// Parses the LLM response to extract Thought, Action, and Action Input
    let private parseReActResponse (response: string) : ReActParse =
        let lines = response.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)

        let findValue prefix =
            lines
            |> Array.tryFind (fun l -> l.Trim().StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun l -> l.Substring(l.IndexOf(':') + 1).Trim())

        match findValue "Thought", findValue "Action", findValue "Action Input" with
        | Some thought, Some action, Some input when action.Equals("Finish", StringComparison.OrdinalIgnoreCase) ->
            Finish(thought, input)
        | Some thought, Some action, Some input -> Continue(thought, action, input)
        | _ -> ParseError response

    /// Formats the conversation history for the LLM
    let private formatHistory (steps: ReActStep list) (goal: string) =
        let stepStrings =
            steps
            |> List.mapi (fun i step ->
                let obs =
                    match step.Observation with
                    | Some o -> "\nObservation: " + o
                    | None -> ""

                sprintf
                    "Step %d:\nThought: %s\nAction: %s\nAction Input: %s%s"
                    (i + 1)
                    step.Thought
                    step.Action
                    step.ActionInput
                    obs)
            |> String.concat "\n\n"

        if steps.IsEmpty then
            goal
        else
            goal + "\n\n" + stepStrings + "\n\nContinue from here:"

    // =========================================================================
    // Pattern Implementations
    // =========================================================================

    /// <summary>
    /// Chain of Thought: Sequential reasoning where each step's output feeds the next.
    /// </summary>
    /// <param name="steps">List of reasoning functions, each taking input and returning output.</param>
    /// <param name="input">Initial input to the chain.</param>
    /// <returns>An AgentWorkflow that executes the chain.</returns>
    let chainOfThought (steps: (string -> AgentWorkflow<string>) list) (input: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                let rec loop remaining current =
                    async {
                        match remaining with
                        | [] -> return Success current
                        | step :: rest ->
                            let! result = step current ctx

                            match result with
                            | Success output -> return! loop rest output
                            | PartialSuccess(output, warnings) ->
                                let! nextResult = loop rest output

                                match nextResult with
                                | Success final -> return PartialSuccess(final, warnings)
                                | PartialSuccess(final, moreWarnings) ->
                                    return PartialSuccess(final, warnings @ moreWarnings)
                                | Failure errors -> return Failure(warnings @ errors)
                            | Failure errors -> return Failure errors
                    }

                return! loop steps input
        }

    /// <summary>
    /// ReAct Pattern: Reason -> Act -> Observe loop for tool-augmented reasoning.
    /// The agent reasons about the problem, takes an action (calls a tool),
    /// observes the result, and repeats until it reaches a final answer.
    /// </summary>
    /// <param name="llm">The LLM service for generating reasoning.</param>
    /// <param name="tools">The tool registry for executing actions.</param>
    /// <param name="maxSteps">Maximum number of reasoning steps before stopping.</param>
    /// <param name="goal">The user's goal or question to solve.</param>
    /// <returns>An AgentWorkflow that executes the ReAct loop.</returns>
    let reAct (llm: ILlmService) (tools: IToolRegistry) (maxSteps: int) (goal: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                let! contextPrelude =
                    async {
                        let! memories =
                            match ctx.SemanticMemory with
                            | Some smem ->
                                let query =
                                    { TaskId = ""
                                      TaskKind = "coding"
                                      TextContext = goal
                                      Tags = [] }

                                smem.Retrieve query
                            | None -> async { return [] }

                        let memoryText =
                            memories
                            |> List.truncate 3
                            |> List.choose (fun m ->
                                m.Logical
                                |> Option.map (fun l -> $"{l.ProblemSummary} | {l.StrategySummary} | {l.OutcomeLabel}"))
                            |> function
                                | [] -> ""
                                | xs -> "Lessons:\n" + (String.concat "\n- " xs |> fun s -> "- " + s)

                        let! facts =
                            match ctx.KnowledgeGraph with
                            | Some kg -> kg.QueryAsync(goal) |> Async.AwaitTask
                            | None -> async { return [] }

                        let factText =
                            facts
                            |> List.truncate 3
                            |> List.map (fun f -> f.ToString())
                            |> function
                                | [] -> ""
                                | xs -> "Facts:\n" + (String.concat "\n- " xs |> fun s -> "- " + s)

                        let combined = [ memoryText; factText ] |> List.filter (fun s -> s <> "") |> String.concat "\n"

                        if String.IsNullOrWhiteSpace combined then
                            return ""
                        else
                            return combined + "\n\n"
                    }

                let allTools = tools.GetAll()
                let systemPrompt = buildReActSystemPrompt allTools
                let mutable steps: ReActStep list = []
                let mutable stepCount = 0
                let mutable finalAnswer: string option = None
                let mutable allWarnings: PartialFailure list = []

                ctx.Logger(sprintf "[ReAct] Starting with goal: %s" goal)
                let toolNames = allTools |> List.map (fun t -> t.Name) |> String.concat ", "
                ctx.Logger(sprintf "[ReAct] Available tools: %s" toolNames)

                while stepCount < maxSteps && finalAnswer.IsNone do
                    stepCount <- stepCount + 1
                    ctx.Logger(sprintf "[ReAct] Step %d/%d" stepCount maxSteps)

                    // Build the conversation
                    let userContent = contextPrelude + formatHistory steps goal

                    let request: LlmRequest =
                        { ModelHint = None
                          Model = None
                          SystemPrompt = Some systemPrompt
                          MaxTokens = Some 500
                          Temperature = Some 0.3
                          Stop = [ "Observation:" ]
                          Messages =
                            [ { Role = Role.User
                                Content = userContent } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = None
                          Stream = false
                          JsonMode = false
                          Seed = None }

                    // Get LLM response
                    let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                    ctx.Logger(sprintf "[ReAct] LLM Response: %s" response.Text)

                    // Parse the response
                    match parseReActResponse response.Text with
                    | Finish(thought, answer) ->
                        ctx.Logger(sprintf "[ReAct] Finished with answer: %s" answer)
                        finalAnswer <- Some answer

                        steps <-
                            steps
                            @ [ { Thought = thought
                                  Action = "Finish"
                                  ActionInput = answer
                                  Observation = None } ]

                    | Continue(thought, action, actionInput) ->
                        ctx.Logger(sprintf "[ReAct] Action: %s(%s)" action actionInput)

                        // Execute the tool
                        let! observation =
                            async {
                                match tools.Get(action) with
                                | Some tool ->
                                    try
                                        let riskyTools =
                                            set [ "write_code"; "patch_code"; "run_shell"; "build_project"; "git_commit" ]

                                        if riskyTools.Contains action then
                                            let preview =
                                                if actionInput.Length > 240 then
                                                    actionInput.Substring(0, 240) + "..."
                                                else
                                                    actionInput

                                            ctx.Logger(sprintf "[Safety] %s preview: %s" action preview)
                                            allWarnings <- allWarnings @ [ PartialFailure.Warning $"SafetyGate preview logged for {action}" ]

                                        let! result = tool.Execute actionInput

                                        match result with
                                        | Result.Ok output ->
                                            let preview = output.Substring(0, min 200 output.Length)
                                            ctx.Logger(sprintf "[ReAct] Tool result: %s..." preview)
                                            return output
                                        | Result.Error err ->
                                            allWarnings <- allWarnings @ [ PartialFailure.ToolError(action, err) ]

                                            return sprintf "Error: %s" err
                                    with ex ->
                                        allWarnings <- allWarnings @ [ PartialFailure.ToolError(action, ex.Message) ]

                                        return sprintf "Exception: %s" ex.Message
                                | None ->
                                    allWarnings <-
                                        allWarnings @ [ PartialFailure.Warning(sprintf "Unknown tool: %s" action) ]

                                    let availableTools = allTools |> List.map (fun t -> t.Name) |> String.concat ", "
                                    return sprintf "Unknown tool: %s. Available tools: %s" action availableTools
                            }

                        steps <-
                            steps
                            @ [ { Thought = thought
                                  Action = action
                                  ActionInput = actionInput
                                  Observation = Some observation } ]

                    | ParseError raw ->
                        ctx.Logger(sprintf "[ReAct] Parse error: %s" raw)

                        let preview = raw.Substring(0, min 100 raw.Length)

                        allWarnings <-
                            allWarnings
                            @ [ PartialFailure.Warning(sprintf "Failed to parse LLM response: %s" preview) ]

                        // Try to recover by treating the whole response as a thought
                        steps <-
                            steps
                            @ [ { Thought = raw
                                  Action = "ParseError"
                                  ActionInput = ""
                                  Observation =
                                    Some "Please respond in the correct format with Thought, Action, and Action Input." } ]

                // Return result
                match finalAnswer with
                | Some answer ->
                    if allWarnings.IsEmpty then
                        return Success answer
                    else
                        return PartialSuccess(answer, allWarnings)
                | None ->
                    let lastThought =
                        steps
                        |> List.tryLast
                        |> Option.map (fun s -> s.Thought)
                        |> Option.defaultValue "No reasoning captured"

                    let budgetExceeded =
                        PartialFailure.Warning(
                            sprintf "ReAct loop reached max steps (%d). Last thought: %s" maxSteps lastThought
                        )

                    return Failure(allWarnings @ [ budgetExceeded ])
            }

    /// <summary>
    /// Plan & Execute: Generate a plan first, then execute each step.
    /// </summary>
    /// <param name="planner">Workflow that generates a list of steps.</param>
    /// <param name="executor">Function that executes a single step.</param>
    /// <returns>An AgentWorkflow that plans and executes.</returns>
    let planAndExecute
        (planner: AgentWorkflow<string list>)
        (executor: string -> AgentWorkflow<string>)
        : AgentWorkflow<string list> =
        fun ctx ->
            async {
                ctx.Logger "[PlanAndExecute] Generating plan..."

                // Generate the plan
                let! planResult = planner ctx

                match planResult with
                | Failure errors -> return Failure errors
                | Success steps
                | PartialSuccess(steps, _) ->
                    ctx.Logger(sprintf "[PlanAndExecute] Plan has %d steps" steps.Length)

                    let planWarnings =
                        match planResult with
                        | PartialSuccess(_, w) -> w
                        | _ -> []

                    // Execute each step
                    let results = ResizeArray<string>()
                    let mutable allWarnings = planWarnings
                    let mutable failed = false
                    let mutable failErrors = []

                    for i, step in steps |> List.indexed do
                        if not failed then
                            ctx.Logger(sprintf "[PlanAndExecute] Executing step %d: %s" (i + 1) step)
                            let! stepResult = executor step ctx

                            match stepResult with
                            | Success output -> results.Add(output)
                            | PartialSuccess(output, warnings) ->
                                results.Add(output)
                                allWarnings <- allWarnings @ warnings
                            | Failure errors ->
                                failed <- true
                                failErrors <- errors

                    if failed then
                        return Failure(allWarnings @ failErrors)
                    elif allWarnings.IsEmpty then
                        return Success(List.ofSeq results)
                    else
                        return PartialSuccess(List.ofSeq results, allWarnings)
            }

    // =========================================================================
    // Convenience Builders
    // =========================================================================

    /// Creates a simple reasoning step that calls the LLM
    let llmStep (llm: ILlmService) (systemPrompt: string) : string -> AgentWorkflow<string> =
        fun input ->
            fun ctx ->
                async {
                    let request: LlmRequest =
                        { ModelHint = None
                          Model = None
                          SystemPrompt = Some systemPrompt
                          MaxTokens = Some 500
                          Temperature = Some 0.7
                          Stop = []
                          Messages = [ { Role = Role.User; Content = input } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = None
                          Stream = false
                          JsonMode = false
                          Seed = None }

                    let! response = llm.CompleteAsync(request) |> Async.AwaitTask
                    return Success response.Text
                }

    /// Creates a planner that uses the LLM to generate steps
    let llmPlanner (llm: ILlmService) (goal: string) : AgentWorkflow<string list> =
        fun ctx ->
            async {
                let request: LlmRequest =
                    { ModelHint = None
                      Model = None
                      SystemPrompt =
                        Some
                            "You are a planning assistant. Generate a numbered list of steps to accomplish the goal. Output ONLY the steps, one per line, numbered like '1. Step one'"
                      MaxTokens = Some 300
                      Temperature = Some 0.5
                      Stop = []
                      Messages =
                        let content = sprintf "Create a plan to: %s" goal
                        [ { Role = Role.User; Content = content } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None }

                let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                // Parse numbered steps
                let steps =
                    response.Text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.filter (fun l -> l.Trim().Length > 0)
                    |> Array.map (fun l ->
                        // Remove numbering like "1." or "1)"
                        let trimmed = l.Trim()

                        if trimmed.Length > 2 && Char.IsDigit(trimmed[0]) then
                            trimmed.Substring(trimmed.IndexOfAny([| '.'; ')' |]) + 1).Trim()
                        else
                            trimmed)
                    |> Array.toList

                ctx.Logger(sprintf "[Planner] Generated %d steps" steps.Length)
                return Success steps
            }
