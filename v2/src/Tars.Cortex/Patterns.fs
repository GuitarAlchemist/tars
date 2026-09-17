namespace Tars.Cortex

open System
open System.Text.Json
open Tars.Core
open Tars.Llm
open System.Text
open Tars.Cortex.ThoughtParsing
open Tars.Cortex.ThoughtGraph

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

                $"Step %d{i + 1}:\nThought: %s{step.Thought}\nAction: %s{step.Action}\nAction Input: %s{step.ActionInput}%s{obs}")
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

                        let combined =
                            [ memoryText; factText ] |> List.filter (fun s -> s <> "") |> String.concat "\n"

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

                ctx.Logger $"[ReAct] Starting with goal: %s{goal}"
                let toolNames = allTools |> List.map (fun t -> t.Name) |> String.concat ", "
                ctx.Logger $"[ReAct] Available tools: %s{toolNames}"

                while stepCount < maxSteps && finalAnswer.IsNone do
                    stepCount <- stepCount + 1
                    ctx.Logger $"[ReAct] Step %d{stepCount}/%d{maxSteps}"

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
                          Seed = None

                          ContextWindow = None }

                    // Get LLM response
                    let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                    ctx.Logger $"[ReAct] LLM Response: %s{response.Text}"

                    // Parse the response
                    match parseReActResponse response.Text with
                    | Finish(thought, answer) ->
                        ctx.Logger $"[ReAct] Finished with answer: %s{answer}"
                        finalAnswer <- Some answer

                        steps <-
                            steps
                            @ [ { Thought = thought
                                  Action = "Finish"
                                  ActionInput = answer
                                  Observation = None } ]

                    | Continue(thought, action, actionInput) ->
                        ctx.Logger $"[ReAct] Action: %s{action}(%s{actionInput})"

                        // Execute the tool
                        let! observation =
                            async {
                                match tools.Get(action) with
                                | Some tool ->
                                    try
                                        let riskyTools =
                                            set
                                                [ "write_code"
                                                  "patch_code"
                                                  "run_shell"
                                                  "build_project"
                                                  "git_commit" ]

                                        if riskyTools.Contains action then
                                            let preview =
                                                if actionInput.Length > 240 then
                                                    actionInput.Substring(0, 240) + "..."
                                                else
                                                    actionInput

                                            ctx.Logger $"[Safety] %s{action} preview: %s{preview}"

                                            allWarnings <-
                                                allWarnings
                                                @ [ PartialFailure.Warning $"SafetyGate preview logged for {action}" ]

                                        let! result = Tars.Core.ToolExecution.runDefault tool actionInput

                                        match result with
                                        | Result.Ok output ->
                                            let preview = output.Substring(0, min 200 output.Length)
                                            ctx.Logger $"[ReAct] Tool result: %s{preview}..."
                                            return output
                                        | Result.Error err ->
                                            allWarnings <- allWarnings @ [ PartialFailure.ToolError(action, err) ]

                                            return $"Error: %s{err}"
                                    with ex ->
                                        allWarnings <- allWarnings @ [ PartialFailure.ToolError(action, ex.Message) ]

                                        return $"Exception: %s{ex.Message}"
                                | None ->
                                    allWarnings <- allWarnings @ [ PartialFailure.Warning $"Unknown tool: %s{action}" ]

                                    let availableTools = allTools |> List.map (fun t -> t.Name) |> String.concat ", "
                                    return $"Unknown tool: %s{action}. Available tools: %s{availableTools}"
                            }

                        steps <-
                            steps
                            @ [ { Thought = thought
                                  Action = action
                                  ActionInput = actionInput
                                  Observation = Some observation } ]

                    | ParseError raw ->
                        ctx.Logger $"[ReAct] Parse error: %s{raw}"

                        let preview = raw.Substring(0, min 100 raw.Length)

                        allWarnings <-
                            allWarnings
                            @ [ PartialFailure.Warning $"Failed to parse LLM response: %s{preview}" ]

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
                        PartialFailure.Warning
                            $"ReAct loop reached max steps (%d{maxSteps}). Last thought: %s{lastThought}"

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
                    ctx.Logger $"[PlanAndExecute] Plan has %d{steps.Length} steps"

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
                            ctx.Logger $"[PlanAndExecute] Executing step %d{i + 1}: %s{step}"
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
                          Seed = None

                          ContextWindow = None }

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
                        let content = $"Create a plan to: %s{goal}"
                        [ { Role = Role.User; Content = content } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

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

                ctx.Logger $"[Planner] Generated %d{steps.Length} steps"
                return Success steps
            }

    /// <summary>
    /// Graph of Thoughts: Graph-structured reasoning with branching and aggregation.
    /// </summary>
    let graphOfThoughts (llm: ILlmService) (config: GoTConfig) (goal: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                ctx.Logger "[GoT] Starting Graph-of-Thoughts reasoning"
                let mutable nodes: Map<Guid, ThoughtNode> = Map.empty
                let mutable frontier: Guid list = []
                let mutable edges: GoTEdge list = []

                let recordEdge sourceId targetId edgeType evidence =
                    if config.TrackEdges then
                        edges <-
                            { Id = Guid.NewGuid()
                              SourceId = sourceId
                              TargetId = targetId
                              EdgeType = edgeType
                              Weight = None
                              Evidence = evidence
                              CreatedAt = DateTime.UtcNow }
                            :: edges

                let! contextPrelude =
                    if config.EnableMemoryRecall then
                        buildContextPrelude ctx goal
                    else
                        async { return "" }

                // Phase 1: Generate initial thoughts
                ctx.Logger "[GoT] Phase 1: Generating initial thoughts"
                let! initialThoughts = generateThoughts llm ctx config goal contextPrelude [] [] 0

                for thought in initialThoughts do
                    nodes <- nodes.Add(thought.Id, thought)
                    frontier <- thought.Id :: frontier

                    ctx.Logger $"[GoT] Generated: %s{thought.Content.Substring(0, min 60 thought.Content.Length)}..."

                // Phase 2: Iterative expansion
                for depth in 1 .. config.MaxDepth - 1 do
                    ctx.Logger $"[GoT] Phase 2.%d{depth}: Scoring and expanding (frontier size: %d{frontier.Length})"

                    // Score all frontier nodes
                    let! scoredNodes =
                        frontier
                        |> List.map (fun id ->
                            async {
                                match nodes.TryFind id with
                                | Some node when node.Score.IsNone ->
                                    let peers =
                                        nodes
                                        |> Map.toList
                                        |> List.map snd
                                        |> List.filter (fun n -> n.Id <> node.Id && n.Score.IsSome)

                                    let! scored = scoreThought llm ctx config goal contextPrelude peers node
                                    return Some scored
                                | _ -> return None
                            })
                        |> Async.Parallel

                    for maybeNode in scoredNodes do
                        match maybeNode with
                        | Some node ->
                            nodes <- nodes.Add(node.Id, node)

                            ctx.Logger
                                $"[GoT] Scored %.2f{node.Score |> Option.defaultValue 0.0}: %s{node.Content.Substring(0, min 40 node.Content.Length)}..."
                        | None -> ()

                    // Select top-K thoughts above threshold
                    let passesThreshold (n: ThoughtNode) =
                        let score = n.Score |> Option.defaultValue 0.0

                        let confidence =
                            n.Evaluation |> Option.map (fun e -> e.Confidence) |> Option.defaultValue 0.0

                        let passed = score >= config.ScoreThreshold && confidence >= config.MinConfidence
                        logThresholdDecision ctx n passed
                        passed

                    let scoredFrontier = frontier |> List.choose (fun id -> nodes.TryFind id)

                    let topThoughts =
                        scoredFrontier
                        |> List.filter passesThreshold
                        |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                        |> List.truncate config.TopK

                    let topThoughts =
                        if topThoughts.IsEmpty then
                            let fallback =
                                scoredFrontier
                                |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                                |> List.truncate (max 1 config.TopK)

                            if not fallback.IsEmpty then
                                ctx.Logger "[GoT] No thoughts above threshold; continuing with top scored candidates"

                            fallback
                        else
                            topThoughts

                    if topThoughts.IsEmpty then
                        ctx.Logger "[GoT] No thoughts available for expansion, stopping early"
                    else
                        let! refinedNodes =
                            topThoughts
                            |> List.map (fun n -> refineThought llm ctx config goal contextPrelude n)
                            |> Async.Parallel

                        frontier <- []

                        for refined in refinedNodes do
                            nodes <- nodes.Add(refined.Id, refined)
                            frontier <- refined.Id :: frontier

                            for parentId in refined.ParentIds do
                                recordEdge parentId refined.Id Refines None

                // Phase 3: Final aggregation
                ctx.Logger "[GoT] Phase 3: Aggregating best thoughts"

                let candidates =
                    nodes
                    |> Map.toList
                    |> List.map snd
                    |> List.filter (fun n ->
                        n.Score.IsSome
                        && (n.Evaluation |> Option.map (fun e -> e.Confidence) |> Option.defaultValue 0.0)
                           >= config.MinConfidence)
                    |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                    |> List.truncate 3

                let bestThoughts =
                    if not candidates.IsEmpty then
                        candidates
                    else
                        ctx.Logger
                            "[GoT] No thoughts passed confidence threshold; falling back to best effort selection"

                        nodes
                        |> Map.toList
                        |> List.map snd
                        |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                        |> List.truncate 1

                let! finalNode =
                    async {
                        if bestThoughts.Length >= 2 then
                            let! aggregated = aggregateThoughts llm ctx config goal contextPrelude bestThoughts
                            nodes <- nodes.Add(aggregated.Id, aggregated)

                            if config.TrackEdges then
                                for parentId in aggregated.ParentIds do
                                    recordEdge parentId aggregated.Id Merges None

                            return aggregated
                        elif bestThoughts.Length = 1 then
                            return bestThoughts.Head
                        else
                            let allNodes = nodes |> Map.toList |> List.map snd

                            if allNodes.IsEmpty then
                                return
                                    { Id = Guid.NewGuid()
                                      Content = "No solution found."
                                      Score = Some 0.0
                                      Evaluation = None
                                      Embedding = None
                                      ParentIds = []
                                      Depth = 0
                                      Operation = Generate
                                      Path = [] }
                            else
                                return allNodes.Head
                    }

                let finalPeers =
                    nodes
                    |> Map.toList
                    |> List.map snd
                    |> List.filter (fun n -> n.Id <> finalNode.Id)

                let! scoredFinal = scoreThought llm ctx config goal contextPrelude finalPeers finalNode

                if config.TrackEdges then
                    ctx.Logger $"[GoT] Recorded {edges.Length} edges"

                return Success scoredFinal.Content
            }

    /// <summary>
    /// Tree of Thoughts (ToT): Systematic search over reasoning steps.
    /// Uses BFS to explore multiple reasoning paths and selects the best leaf.
    /// </summary>
    let treeOfThoughts (llm: ILlmService) (config: GoTConfig) (goal: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                ctx.Logger "[ToT] Starting Tree-of-Thoughts reasoning"
                let mutable nodes: Map<Guid, ThoughtNode> = Map.empty
                let mutable frontier: Guid list = []
                let mutable edges: GoTEdge list = []

                let recordEdge sourceId targetId edgeType evidence =
                    if config.TrackEdges then
                        edges <-
                            { Id = Guid.NewGuid()
                              SourceId = sourceId
                              TargetId = targetId
                              EdgeType = edgeType
                              Weight = None
                              Evidence = evidence
                              CreatedAt = DateTime.UtcNow }
                            :: edges

                let! contextPrelude =
                    if config.EnableMemoryRecall then
                        buildContextPrelude ctx goal
                    else
                        async { return "" }

                // Phase 1: Propose initial candidates
                ctx.Logger "[ToT] Step 1: Proposing initial thought candidates"
                let! initialThoughts = generateThoughts llm ctx config goal contextPrelude [] [] 0

                for thought in initialThoughts do
                    nodes <- nodes.Add(thought.Id, thought)
                    frontier <- thought.Id :: frontier

                    ctx.Logger $"[ToT] Proposed: %s{thought.Content.Substring(0, min 60 thought.Content.Length)}..."

                // Phase 2: Systematic expansion and evaluation (BFS)
                for depth in 1 .. config.MaxDepth - 1 do
                    ctx.Logger $"[ToT] Step %d{depth + 1}: Evaluating and expanding best paths"

                    // Score current leaf candidates
                    let! scoredNodes =
                        frontier
                        |> List.map (fun id ->
                            async {
                                match nodes.TryFind id with
                                | Some node when node.Score.IsNone ->
                                    let peers =
                                        nodes
                                        |> Map.toList
                                        |> List.map snd
                                        |> List.filter (fun n -> n.Id <> node.Id && n.Score.IsSome)

                                    let! scored = scoreThought llm ctx config goal contextPrelude peers node
                                    return Some scored
                                | _ -> return None
                            })
                        |> Async.Parallel

                    for maybeNode in scoredNodes do
                        match maybeNode with
                        | Some node ->
                            nodes <- nodes.Add(node.Id, node)

                            ctx.Logger
                                $"[ToT] Valued %.2f{node.Score |> Option.defaultValue 0.0}: %s{node.Content.Substring(0, min 40 node.Content.Length)}..."
                        | None -> ()

                    // Pruning: Select top-K best thoughts
                    let passesThreshold (n: ThoughtNode) =
                        let score = n.Score |> Option.defaultValue 0.0

                        let confidence =
                            n.Evaluation |> Option.map (fun e -> e.Confidence) |> Option.defaultValue 0.0

                        let passed = score >= config.ScoreThreshold && confidence >= config.MinConfidence
                        logThresholdDecision ctx n passed
                        passed

                    let scoredFrontier = frontier |> List.choose (fun id -> nodes.TryFind id)

                    let topThoughts =
                        scoredFrontier
                        |> List.filter passesThreshold
                        |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                        |> List.truncate config.TopK

                    let topThoughts =
                        if topThoughts.IsEmpty then
                            let fallback =
                                scoredFrontier
                                |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                                |> List.truncate (max 1 config.TopK)

                            if not fallback.IsEmpty then
                                ctx.Logger "[ToT] No thoughts passed threshold; continuing with top scored candidates"

                            fallback
                        else
                            topThoughts

                    if topThoughts.IsEmpty then
                        ctx.Logger "[ToT] No candidates available for expansion, stopping search"
                    else
                        // Expand: Generate next steps from top thoughts
                        let! expandedNodes =
                            topThoughts
                            |> List.map (fun n ->
                                generateThoughts llm ctx config goal contextPrelude n.Path [ n.Id ] n.Depth)
                            |> Async.Parallel

                        frontier <- []

                        for thoughtList in expandedNodes do
                            for thought in thoughtList do
                                nodes <- nodes.Add(thought.Id, thought)
                                frontier <- thought.Id :: frontier

                                for parentId in thought.ParentIds do
                                    recordEdge parentId thought.Id DependsOn None

                // Phase 3: Selection of best final thought
                ctx.Logger "[ToT] Final Step: Selecting best reasoning path"

                let candidates =
                    nodes
                    |> Map.toList
                    |> List.map snd
                    |> List.filter (fun n ->
                        n.Score.IsSome
                        && (n.Evaluation |> Option.map (fun e -> e.Confidence) |> Option.defaultValue 0.0)
                           >= config.MinConfidence)
                    |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)

                let bestNode =
                    match candidates with
                    | n :: _ -> Some n
                    | [] ->
                        ctx.Logger "[ToT] No paths passed confidence threshold; falling back to best effort selection"

                        nodes
                        |> Map.toList
                        |> List.map snd
                        |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                        |> List.tryHead

                match bestNode with
                | Some node ->
                    ctx.Logger
                        $"[ToT] Selected best path (score: %.2f{node.Score |> Option.defaultValue 0.0}): %s{node.Content.Substring(0, min 60 node.Content.Length)}..."

                    if config.TrackEdges then
                        ctx.Logger $"[ToT] Recorded {edges.Length} edges"

                    return Success node.Content
                | None -> return Failure [ PartialFailure.Error "No suitable reasoning path found." ]
            }

    /// <summary>
    /// Workflow of Thoughts (WoT): Minimal controller that layers policy/memory/tool hooks on GoT.
    /// </summary>
    let workflowOfThought (llm: ILlmService) (config: WoTConfig) (goal: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                ctx.Logger "[WoT] Starting Workflow-of-Thought reasoning"
                let baseConfig = config.BaseConfig
                let policyRequired = not config.RequiredPolicies.IsEmpty
                let policyEnabled = baseConfig.EnablePolicyChecks || policyRequired

                let policyConfig =
                    if policyEnabled then
                        { baseConfig with
                            EnablePolicyChecks = true }
                    else
                        baseConfig

                if not config.RequiredPolicies.IsEmpty then
                    let requiredPolicies = String.concat "; " config.RequiredPolicies
                    ctx.Logger $"[WoT] Required policies: {requiredPolicies}"

                let! contextPrelude =
                    if policyConfig.EnableMemoryRecall then
                        buildContextPrelude ctx goal
                    else
                        async { return "" }

                let availableTools =
                    config.AvailableTools
                    |> List.choose (fun name -> ctx.Self.Tools |> List.tryFind (fun t -> t.Name = name))

                if not availableTools.IsEmpty then
                    ctx.Logger $"[WoT] Tools enabled: {availableTools.Length}"

                let! initialThoughts = generateThoughts llm ctx policyConfig goal contextPrelude [] [] 0
                ctx.Logger $"[WoT] Initial thoughts: {initialThoughts.Length}"

                let! scoredInitial =
                    initialThoughts
                    |> List.map (fun node -> scoreThought llm ctx policyConfig goal contextPrelude [] node)
                    |> Async.Parallel

                let passesThreshold (n: ThoughtNode) =
                    let score = n.Score |> Option.defaultValue 0.0

                    let confidence =
                        n.Evaluation |> Option.map (fun e -> e.Confidence) |> Option.defaultValue 0.0

                    let passed =
                        score >= policyConfig.ScoreThreshold && confidence >= policyConfig.MinConfidence

                    logThresholdDecision ctx n passed
                    passed

                let evaluateRequiredPolicies (content: string) =
                    async {
                        if not policyRequired then
                            return []
                        else
                            let input: PolicyEngine.PolicyInput = { Text = content; Metadata = Map.empty }

                            return PolicyEngine.evaluateDefault config.RequiredPolicies input
                    }

                let mutable working =
                    scoredInitial
                    |> Array.toList
                    |> List.filter passesThreshold
                    |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                    |> List.truncate policyConfig.TopK

                ctx.Logger $"[WoT] Nodes after threshold: {working.Length}/{scoredInitial.Length}"

                if working.IsEmpty && scoredInitial.Length > 0 then
                    let best =
                        scoredInitial |> Array.maxBy (fun n -> n.Score |> Option.defaultValue 0.0)

                    ctx.Logger
                        $"[WoT] Best score was {best.Score |> Option.defaultValue 0.0} (Threshold: {policyConfig.ScoreThreshold})"

                return!
                    async {
                        let mutable policyFailure: string option = None

                        if policyRequired && not working.IsEmpty then
                            let! gated =
                                working
                                |> List.map (fun n ->
                                    async {
                                        let! outcomes = evaluateRequiredPolicies n.Content
                                        return n, outcomes
                                    })
                                |> Async.Parallel

                            let passed, failed =
                                gated
                                |> Array.toList
                                |> List.partition (fun (_, outcomes) -> not (PolicyEngine.anyFailed outcomes))

                            working <- passed |> List.map fst

                            if working.IsEmpty then
                                let reasons =
                                    failed
                                    |> List.collect (fun (_, outcomes) ->
                                        outcomes
                                        |> List.filter (fun o -> not o.Passed)
                                        |> List.collect (fun o -> o.Messages))
                                    |> List.distinct
                                    |> function
                                        | [] -> [ "policy_gate_failed" ]
                                        | xs -> xs

                                let reasonText = String.concat "; " reasons
                                policyFailure <- Some reasonText

                        match policyFailure with
                        | Some reasonText ->
                            return Failure [ PartialFailure.Error($"[WoT] Required policies failed: {reasonText}") ]
                        | None ->
                            if working.IsEmpty then
                                working <-
                                    scoredInitial
                                    |> Array.toList
                                    |> List.sortByDescending (fun n -> n.Score |> Option.defaultValue 0.0)
                                    |> List.truncate 1

                            let! refined =
                                if policyConfig.EnableCritique && not working.IsEmpty then
                                    async {
                                        let! refinedArr =
                                            working
                                            |> List.map (fun n ->
                                                refineThought llm ctx policyConfig goal contextPrelude n)
                                            |> Async.Parallel

                                        return refinedArr |> Array.toList
                                    }
                                else
                                    async { return working }

                            let! toolContext =
                                if availableTools.IsEmpty then
                                    async { return None }
                                else
                                    async {
                                        match tryConsumeCall ctx "[WoT] tool decision" with
                                        | false -> return None
                                        | true ->
                                            let toolList =
                                                availableTools
                                                |> List.map (fun t -> $"- {t.Name}: {t.Description}")
                                                |> String.concat "\n"

                                            let state =
                                                buildThoughtState goal policyConfig.Constraints [] contextPrelude

                                            let seedThought =
                                                refined
                                                |> List.tryHead
                                                |> Option.map (fun n -> n.Content)
                                                |> Option.defaultValue goal

                                            let! policyBlocked =
                                                if not policyRequired then
                                                    async { return false }
                                                else
                                                    async {
                                                        let! outcomes = evaluateRequiredPolicies seedThought

                                                        if PolicyEngine.anyFailed outcomes then
                                                            let reasons =
                                                                outcomes
                                                                |> List.filter (fun o -> not o.Passed)
                                                                |> List.collect (fun o -> o.Messages)
                                                                |> List.distinct
                                                                |> String.concat "; "

                                                            ctx.Logger
                                                                $"[WoT] Skipping tool call due to policy failure: {reasons}"

                                                            return true
                                                        else
                                                            return false
                                                    }

                                            if policyBlocked then
                                                return None
                                            else
                                                let request: LlmRequest =
                                                    { ModelHint = Some "fast"
                                                      Model = None
                                                      SystemPrompt =
                                                        Some
                                                            "Decide whether a tool call is necessary. Return JSON: {\"tool\":\"name or null\",\"input\":\"...\",\"reason\":\"...\"}."
                                                      MaxTokens = Some 120
                                                      Temperature = Some 0.1
                                                      Stop = []
                                                      Messages =
                                                        [ { Role = Role.User
                                                            Content =
                                                              $"{state}\n\nCandidate thought:\n{seedThought}\n\nAvailable tools:\n{toolList}" } ]
                                                      Tools = []
                                                      ToolChoice = None
                                                      ResponseFormat = Some ResponseFormat.Json
                                                      Stream = false
                                                      JsonMode = true
                                                      Seed = None

                                                      ContextWindow = None }

                                                let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                                                match JsonParsing.tryParseElement response.Text with
                                                | Result.Ok elem ->
                                                    let toolName =
                                                        tryGetPropertyInsensitive "tool" elem
                                                        |> Option.filter (fun p -> p.ValueKind = JsonValueKind.String)
                                                        |> Option.map (fun p -> p.GetString())
                                                        |> Option.defaultValue ""

                                                    let input =
                                                        tryGetPropertyInsensitive "input" elem
                                                        |> Option.filter (fun p -> p.ValueKind = JsonValueKind.String)
                                                        |> Option.map (fun p -> p.GetString())
                                                        |> Option.defaultValue ""

                                                    let selected =
                                                        if String.IsNullOrWhiteSpace toolName then
                                                            None
                                                        else
                                                            availableTools
                                                            |> List.tryFind (fun t ->
                                                                t.Name.Equals(
                                                                    toolName,
                                                                    StringComparison.OrdinalIgnoreCase
                                                                ))

                                                    match selected with
                                                    | Some tool ->
                                                        match tryConsumeCall ctx "[WoT] tool execution" with
                                                        | false -> return None
                                                        | true ->
                                                            let! result = Tars.Core.ToolExecution.runDefault tool input

                                                            match result with
                                                            | Result.Ok output ->
                                                                return Some($"{tool.Name}({input}) => {output}")
                                                            | Result.Error err ->
                                                                return Some($"{tool.Name}({input}) => ERROR: {err}")
                                                    | None -> return None
                                                | Result.Error _ -> return None
                                    }

                            let contextWithTool =
                                match toolContext with
                                | Some toolInfo -> contextPrelude + "\nTool output:\n" + toolInfo
                                | None -> contextPrelude

                            // Always synthesize, even from a single survivor: a thought is one
                            // step toward the goal, not an answer to it (#263).
                            let! finalNode =
                                async {
                                    if not refined.IsEmpty then
                                        return! aggregateThoughts llm ctx policyConfig goal contextWithTool refined
                                    else
                                        return
                                            { Id = Guid.NewGuid()
                                              Content = "No solution found."
                                              Score = Some 0.0
                                              Evaluation = None
                                              Embedding = None
                                              ParentIds = []
                                              Depth = 0
                                              Operation = Generate
                                              Path = [] }
                                }

                            let peers = refined |> List.filter (fun n -> n.Id <> finalNode.Id)
                            let! scoredFinal = scoreThought llm ctx policyConfig goal contextWithTool peers finalNode

                            // --- Trace Ingestion ---
                            try
                                match ctx.KnowledgeGraph with
                                | Some kg ->
                                    let runId = Guid.NewGuid()

                                    let runEnt =
                                        { Id = runId
                                          Goal = goal
                                          Pattern = "Workflow-of-Thoughts"
                                          Timestamp = DateTime.UtcNow }

                                    let ingestTask =
                                        task {
                                            let! _ = kg.AddNodeAsync(TarsEntity.RunE runEnt)

                                            let ingestNode (n: ThoughtNode) =
                                                task {
                                                    let stepEnt =
                                                        { RunId = runId
                                                          StepId = n.Id.ToString()
                                                          NodeType = n.Operation.ToString()
                                                          Content = n.Content
                                                          Timestamp = DateTime.UtcNow }

                                                    let nodeE = TarsEntity.StepE stepEnt
                                                    let! _ = kg.AddNodeAsync(nodeE)

                                                    let! _ =
                                                        kg.AddFactAsync(
                                                            TarsFact.Contains(TarsEntity.RunE runEnt, nodeE)
                                                        )

                                                    for pid in n.ParentIds do
                                                        let parentProb =
                                                            { RunId = runId
                                                              StepId = pid.ToString()
                                                              NodeType = "Unknown"
                                                              Content = ""
                                                              Timestamp = DateTime.MinValue }

                                                        let parentE = TarsEntity.StepE parentProb
                                                        let! _ = kg.AddFactAsync(TarsFact.NextStep(parentE, nodeE))
                                                        ()
                                                }

                                            for n in scoredInitial do
                                                do! ingestNode n

                                            for n in refined do
                                                do! ingestNode n

                                            do! ingestNode finalNode
                                        }

                                    do! ingestTask |> Async.AwaitTask
                                    ctx.Logger($"[WoT] Persisted trace run:{runId}")
                                | None -> ()
                            with ex ->
                                ctx.Logger($"[WoT] Trace ingestion failed: {ex.Message}")
                            // -----------------------

                            let! policyOutcomes = evaluateRequiredPolicies scoredFinal.Content

                            if policyRequired && PolicyEngine.anyFailed policyOutcomes then
                                let reasonText =
                                    policyOutcomes
                                    |> List.filter (fun o -> not o.Passed)
                                    |> List.collect (fun o -> o.Messages)
                                    |> List.distinct
                                    |> String.concat "; "

                                return Failure [ PartialFailure.Error($"[WoT] Required policies failed: {reasonText}") ]
                            else
                                return Success scoredFinal.Content
                    }
            }

    /// Run Tree-of-Thoughts with default configuration
    let treeOfThoughtsDefault (llm: ILlmService) (goal: string) : AgentWorkflow<string> =
        treeOfThoughts llm defaultGoTConfig goal

    /// Run Graph-of-Thoughts with default configuration
    let graphOfThoughtsDefault (llm: ILlmService) (goal: string) : AgentWorkflow<string> =
        graphOfThoughts llm defaultGoTConfig goal
