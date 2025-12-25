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
                          Seed = None

                          ContextWindow = None }

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

                                            ctx.Logger(sprintf "[Safety] %s preview: %s" action preview)

                                            allWarnings <-
                                                allWarnings
                                                @ [ PartialFailure.Warning $"SafetyGate preview logged for {action}" ]

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
                        let content = sprintf "Create a plan to: %s" goal
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

                ctx.Logger(sprintf "[Planner] Generated %d steps" steps.Length)
                return Success steps
            }

    // =========================================================================
    // Graph of Thoughts (GoT) Pattern
    // Reference: https://github.com/spcl/graph-of-thoughts
    // Paper: "Graph of Thoughts: Solving Elaborate Problems with LLMs"
    // Extended with Workflow-of-Thought (WoT) patterns for enterprise production
    // Reference: https://blog.bytebytego.com/p/top-ai-agentic-workflow-patterns
    // =========================================================================

    /// The 5 essential agentic workflow patterns
    type AgenticWorkflowPattern =
        | Reflection // Iterative self-improvement (Generate -> Critique -> Revise)
        | ToolUse // Dynamic selection and invocation of external capabilities
        | ReAct // Interleaved reasoning and acting (Reasoning -> Action -> Observation)
        | Planning // Strategic decomposition and dependency management
        | MultiAgent // Collaboration between specialized specialist agents

    // ----- Edge Types (Relationships between nodes) -----

    /// Edge types representing relationships between thought/work nodes
    /// These are the "arrows" connecting nodes in the reasoning graph
    type GoTEdgeType =
        | Supports // This node provides evidence supporting another
        | Contradicts // This node conflicts with another (triggers reconciliation)
        | DependsOn // This node requires another to be resolved first
        | Refines // This node improves/clarifies another
        | Merges // This node combines multiple nodes into one
        | Critiques // This node evaluates/judges another
        | Validates // This node confirms correctness of another
        | Escalates // This node triggers human review of another

    /// An edge in the reasoning graph
    type GoTEdge =
        { Id: Guid
          SourceId: Guid
          TargetId: Guid
          EdgeType: GoTEdgeType
          Weight: float option // Confidence/strength of relationship
          Evidence: string option // Why this relationship exists
          CreatedAt: DateTime }

    // ----- Node Types (for WoT Work Nodes) -----

    /// Types of nodes in a Workflow-of-Thought (WoT) graph
    /// Extends basic thought nodes with operational work nodes
    type WoTNodeType =
        | ThoughtNode // Reasoning step, claim, or sub-problem
        | ToolNode // Web search, RAG, code execution, API call
        | PolicyNode // PII check, legal clause, brand style, risk threshold
        | RoleNode // Analyst, Reviewer, Approver, Human-in-the-loop
        | MemoryNode // Past decisions, precedent cases, reusable snippets
        | VerifierNode // Schema check, grammar, unit test, constraint solver
        | CritiqueNode // Evaluation of another node's quality/correctness
        | AggregationNode // Merges multiple nodes into synthesized output

    /// Policy check status for policy nodes
    type PolicyStatus =
        | Pending // Not yet checked
        | Passed // Policy constraint satisfied
        | Failed of reason: string // Policy violated with reason
        | Waived of by: string * reason: string // Explicitly waived by authority

    /// A node in the WoT reasoning/workflow graph
    type WoTNode =
        { Id: Guid
          NodeType: WoTNodeType
          Content: string
          Score: float option
          PolicyStatus: PolicyStatus option
          Metadata: Map<string, string> // Flexible key-value metadata
          ParentIds: Guid list // Incoming edges (for quick traversal)
          ChildIds: Guid list // Outgoing edges
          Depth: int
          CreatedAt: DateTime
          CreatedBy: string option } // Agent/tool/human that created this

    // ----- Controller Components -----

    /// Router decisions in WoT execution
    type RouterDecision =
        | Expand of nodeId: Guid // Generate children from this node
        | Merge of nodeIds: Guid list // Combine these nodes
        | Rollback of toNodeId: Guid // Backtrack to this node
        | Escalate of nodeId: Guid * reason: string // Send to human
        | Finalize of nodeId: Guid // This is the answer
        | CallTool of toolName: string * input: string // Invoke external tool
        | ApplyPolicy of policyName: string * nodeId: Guid // Run policy check

    /// Verification result from a verifier node
    type VerificationResult =
        | Valid
        | Invalid of errors: string list
        | PartiallyValid of warnings: string list

    // ----- Original GoT types (preserved for backward compatibility) -----

    /// A thought node in the reasoning graph (simplified view)
    type ThoughtEvaluation =
        { Score: float
          Confidence: float
          Reasons: string list
          Risks: string list }

    type ThoughtNode =
        { Id: Guid
          Content: string
          Score: float option
          Evaluation: ThoughtEvaluation option
          Embedding: float32[] option
          ParentIds: Guid list
          Depth: int
          Operation: GoTOperation
          Path: string list }

    /// Operations in Graph-of-Thoughts
    and GoTOperation =
        | Generate // Create new thoughts from prompt
        | Aggregate // Combine multiple thoughts into one
        | Refine // Improve an existing thought
        | Score // Evaluate a thought's quality

    /// Configuration for GoT execution
    type GoTConfig =
        { BranchingFactor: int // How many thoughts to generate per step
          MaxDepth: int // Maximum reasoning depth
          TopK: int // Keep top K thoughts for expansion
          ScoreThreshold: float // Minimum score to keep a thought
          MinConfidence: float // Minimum confidence to keep a thought
          DiversityThreshold: float // Max cosine similarity before penalizing
          DiversityPenalty: float // Penalty for near-duplicate thoughts
          Constraints: string list // Optional constraints for the task
          EnableCritique: bool // Whether to add critique nodes
          EnablePolicyChecks: bool // Whether to run policy validators
          EnableMemoryRecall: bool // Whether to consult memory for precedents
          TrackEdges: bool } // Whether to track edge relationships

    /// Default GoT configuration
    let defaultGoTConfig =
        { BranchingFactor = 3
          MaxDepth = 3
          TopK = 2
          ScoreThreshold = 0.1 // Lowered for more robustness with local LLMs
          MinConfidence = 0.4
          DiversityThreshold = 0.85
          DiversityPenalty = 0.25
          Constraints = []
          EnableCritique = false
          EnablePolicyChecks = false
          EnableMemoryRecall = false
          TrackEdges = false }

    /// Extended WoT configuration for production workflows
    type WoTConfig =
        { BaseConfig: GoTConfig
          RequiredPolicies: string list // Policy checks that must pass
          AvailableTools: string list // Tools this workflow can invoke
          RoleAssignments: Map<string, string> // Role -> Agent/Human ID
          MemoryNamespace: string option // Namespace for memory operations
          MaxEscalations: int // Max human escalations before abort
          TimeoutMs: int option } // Optional timeout

    /// Default WoT configuration
    let defaultWoTConfig =
        { BaseConfig =
            { defaultGoTConfig with
                EnableCritique = true
                TrackEdges = true }
          RequiredPolicies = []
          AvailableTools = []
          RoleAssignments = Map.empty
          MemoryNamespace = None
          MaxEscalations = 3
          TimeoutMs = Some 300000 } // 5 minute default

    let private recordBranchDecision (ctx: AgentContext) (node: ThoughtNode) (action: string) (status: string) =
        let evaluation = node.Evaluation

        let decision =
            { BranchDecision.NodeId = node.Id
              Content = node.Content
              NodeType = node.Operation.ToString()
              Action = action
              Status = status
              Score = node.Score
              Confidence = evaluation |> Option.map (fun e -> e.Confidence)
              Reasons = evaluation |> Option.map (fun e -> e.Reasons) |> Option.defaultValue []
              Risks = evaluation |> Option.map (fun e -> e.Risks) |> Option.defaultValue []
              Timestamp = DateTime.UtcNow }

        match ctx.Audit with
        | None -> ()
        | Some audit ->
            ReasoningAudit.record audit decision

            let truncatedContent =
                if String.IsNullOrWhiteSpace decision.Content then ""
                elif decision.Content.Length <= 120 then decision.Content
                else decision.Content.Substring(0, 120) + "..."

            let scoreText =
                decision.Score |> Option.map (sprintf "%.2f") |> Option.defaultValue "n/a"

            let confText =
                decision.Confidence |> Option.map (sprintf "%.2f") |> Option.defaultValue "n/a"

            let reasons =
                if decision.Reasons.IsEmpty then
                    "none"
                else
                    String.Join("; ", decision.Reasons)

            let risks =
                if decision.Risks.IsEmpty then
                    "none"
                else
                    String.Join("; ", decision.Risks)

            ctx.Logger(
                sprintf
                    "[ReasoningAudit] action=%s status=%s score=%s conf=%s node=\"%s\" reasons=%s risks=%s"
                    decision.Action
                    decision.Status
                    scoreText
                    confText
                    truncatedContent
                    reasons
                    risks
            )

    let private logEvaluation (ctx: AgentContext) (node: ThoughtNode) =
        recordBranchDecision ctx node "score" "scored"

    let private logThresholdDecision (ctx: AgentContext) (node: ThoughtNode) (passed: bool) =
        let status = if passed then "kept" else "pruned"
        recordBranchDecision ctx node "threshold" status



    // =========================================================================
    // Graph-of-Thoughts & Tree-of-Thoughts Internals
    // =========================================================================

    let private buildContextPrelude (ctx: AgentContext) (goal: string) =
        async {
            let! memories =
                match ctx.SemanticMemory with
                | Some smem ->
                    let query =
                        { TaskId = ""
                          TaskKind = "reasoning"
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
                return combined
        }

    let private truncateForPrompt (value: string) =
        let trimmed = value.Trim()

        if trimmed.Length <= 160 then
            trimmed
        else
            trimmed.Substring(0, 160) + "..."

    let private renderConstraints (constraints: string list) =
        if constraints.IsEmpty then
            "None"
        else
            constraints |> List.map truncateForPrompt |> String.concat "; "

    let private renderPath (path: string list) =
        if path.IsEmpty then
            "None"
        else
            path
            |> List.rev
            |> List.truncate 4
            |> List.rev
            |> List.mapi (fun i step -> $"{i + 1}. {truncateForPrompt step}")
            |> String.concat " | "

    let private buildThoughtState
        (goal: string)
        (constraints: string list)
        (path: string list)
        (contextPrelude: string)
        =
        let baseLines =
            [ $"Goal: {goal}"
              $"Constraints: {renderConstraints constraints}"
              $"Path so far: {renderPath path}" ]

        let allLines =
            if String.IsNullOrWhiteSpace contextPrelude then
                baseLines
            else
                baseLines @ [ "Context:"; contextPrelude ]

        String.concat "\n" allLines

    let private tryConsumeCall (ctx: AgentContext) (label: string) =
        match ctx.Budget with
        | Some budget ->
            match budget.TryConsumeCall() with
            | Result.Ok() -> true
            | Result.Error err ->
                ctx.Logger $"[Budget] {label} skipped: {err}"
                false
        | None -> true

    let private maybeEmbed (llm: ILlmService) (ctx: AgentContext) (config: GoTConfig) (label: string) (text: string) =
        async {
            if config.DiversityPenalty <= 0.0 && config.DiversityThreshold >= 1.0 then
                return None
            elif String.IsNullOrWhiteSpace text then
                ctx.Logger $"[GoT] {label} skipped: empty text"
                return None
            elif not (tryConsumeCall ctx label) then
                return None
            else
                try
                    let! embed = llm.EmbedAsync text |> Async.AwaitTask

                    if isNull embed || embed.Length = 0 then
                        ctx.Logger $"[GoT] {label} produced empty embedding"
                        return None
                    else
                        return Some embed
                with _ ->
                    return None
        }

    let private tryCosineSimilarity (ctx: AgentContext) (label: string) (v1: float32[]) (v2: float32[]) =
        if v1.Length <> v2.Length then
            ctx.Logger $"[GoT] {label} embedding length mismatch: {v1.Length} vs {v2.Length}"
            None
        else
            Some(MetricSpace.cosineSimilarity v1 v2)

    let private tryGetPropertyInsensitive (name: string) (elem: JsonElement) =
        if elem.ValueKind = JsonValueKind.Object then
            elem.EnumerateObject()
            |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun p -> p.Value)
        else
            None

    let private getDoubleWithDefault (names: string list) (defaultValue: float) (elem: JsonElement) =
        names
        |> List.tryPick (fun name ->
            match tryGetPropertyInsensitive name elem with
            | Some prop ->
                match prop.ValueKind with
                | JsonValueKind.Number -> Some(prop.GetDouble())
                | JsonValueKind.String ->
                    match Double.TryParse(prop.GetString()) with
                    | true, v -> Some v
                    | _ -> None
                | _ -> None
            | None -> None)
        |> Option.defaultValue defaultValue

    let private getStringList (name: string) (elem: JsonElement) =
        match tryGetPropertyInsensitive name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.Array ->
                prop.EnumerateArray()
                |> Seq.choose (fun item ->
                    if item.ValueKind = JsonValueKind.String then
                        Some(item.GetString())
                    else
                        None)
                |> Seq.toList
            | JsonValueKind.String -> [ prop.GetString() ]
            | _ -> []
        | None -> []

    let private clamp01 (value: float) = value |> max 0.0 |> min 1.0

    let private stripCodeFence (value: string) =
        let trimmed = value.Trim()

        if trimmed.StartsWith("```", StringComparison.Ordinal) then
            let withoutTicks = trimmed.Substring(3)
            let newLineIdx = withoutTicks.IndexOfAny([| '\n'; '\r' |])

            let body =
                if newLineIdx >= 0 then
                    withoutTicks.Substring(newLineIdx + 1).Trim()
                else
                    withoutTicks.Trim()

            if body.EndsWith("```", StringComparison.Ordinal) then
                body.Substring(0, body.Length - 3).Trim()
            else
                body
        else
            trimmed

    let private tryParseJsonWithFallback (text: string) =
        let cleaned = stripCodeFence text

        if String.IsNullOrWhiteSpace cleaned then
            Result.Error "empty response"
        else
            match JsonParsing.tryParseElement cleaned with
            | Result.Ok elem -> Result.Ok elem
            | Result.Error firstError ->
                let startIdx = cleaned.IndexOf('{')
                let endIdx = cleaned.LastIndexOf('}')

                if startIdx >= 0 && endIdx > startIdx then
                    let slice = cleaned.Substring(startIdx, endIdx - startIdx + 1)

                    match JsonParsing.tryParseElement slice with
                    | Result.Ok elem -> Result.Ok elem
                    | Result.Error secondError -> Result.Error($"{firstError}; {secondError}")
                else
                    Result.Error firstError

    let private parseThoughts (text: string) =
        let lines = text.Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)
        let mutable thoughts = []
        let mutable currentThought = StringBuilder()

        for line in lines do
            let trimmed = line.Trim()

            if trimmed.StartsWith("THOUGHT", StringComparison.OrdinalIgnoreCase) then
                if currentThought.Length > 0 then
                    thoughts <- currentThought.ToString().Trim() :: thoughts
                    currentThought.Clear() |> ignore

                let colonIdx = trimmed.IndexOf(':')

                if colonIdx > 0 && colonIdx < trimmed.Length - 1 then
                    currentThought.Append(trimmed.Substring(colonIdx + 1).Trim()) |> ignore
            else if currentThought.Length > 0 || trimmed.Length > 20 then
                if currentThought.Length > 0 then
                    currentThought.Append(" ") |> ignore

                currentThought.Append(trimmed) |> ignore

        if currentThought.Length > 0 then
            thoughts <- currentThought.ToString().Trim() :: thoughts

        thoughts |> List.rev |> List.filter (fun s -> s.Length > 10)

    let private generateThoughts
        (llm: ILlmService)
        (ctx: AgentContext)
        (config: GoTConfig)
        (goal: string)
        (contextPrelude: string)
        (path: string list)
        (parentIds: Guid list)
        (depth: int)
        =
        async {
            match tryConsumeCall ctx "[GoT] generateThoughts" with
            | false -> return []
            | true ->
                let state = buildThoughtState goal config.Constraints path contextPrelude

                let request: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt =
                        Some
                            $"""You are a reasoning engine exploring multiple solution paths.
Generate {config.BranchingFactor} DIFFERENT next-step thoughts to solve the goal.
Each thought should be a single step or hypothesis (not a full final answer).
Ensure each thought explores a distinct angle or strategy.
Format your response as:
THOUGHT 1: [first approach]
THOUGHT 2: [second approach]
THOUGHT 3: [third approach]
Be creative and diverse in your approaches."""
                      MaxTokens = Some 1000
                      Temperature = Some 0.9
                      Stop = []
                      Messages = [ { Role = Role.User; Content = state } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response = llm.CompleteAsync(request) |> Async.AwaitTask
                let initialText = response.Text.Trim()

                let! responseText =
                    if String.IsNullOrWhiteSpace initialText then
                        async {
                            ctx.Logger "[GoT] generateThoughts empty response; retrying with default model."

                            if not (tryConsumeCall ctx "[GoT] generateThoughts retry") then
                                return initialText
                            else
                                let retryRequest =
                                    { request with
                                        ModelHint = None
                                        Temperature = Some 0.7
                                        ResponseFormat = None
                                        JsonMode = false }

                                let! retryResponse = llm.CompleteAsync(retryRequest) |> Async.AwaitTask
                                return retryResponse.Text.Trim()
                        }
                    else
                        async { return initialText }

                let thoughts = parseThoughts responseText

                let finalThoughts =
                    if thoughts.Length < 2 then
                        responseText.Split([| "\n\n" |], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.map (fun s -> s.Trim())
                        |> Array.filter (fun s -> s.Length > 5)
                        |> Array.truncate config.BranchingFactor
                        |> Array.toList
                    else
                        thoughts |> List.truncate config.BranchingFactor

                let cleanedThoughts =
                    finalThoughts
                    |> List.map (fun content -> content.Trim())
                    |> List.filter (fun content -> not (String.IsNullOrWhiteSpace content))

                if cleanedThoughts.IsEmpty then
                    ctx.Logger "[GoT] No thoughts generated; LLM response empty."
                    return []
                else
                    let! embeddings =
                        cleanedThoughts
                        |> List.map (fun content -> maybeEmbed llm ctx config "[GoT] embed" content)
                        |> Async.Parallel

                    return
                        cleanedThoughts
                        |> List.mapi (fun idx content ->
                            { Id = Guid.NewGuid()
                              Content = content
                              Score = None
                              Evaluation = None
                              Embedding = embeddings.[idx]
                              ParentIds = parentIds
                              Depth = depth
                              Operation = Generate
                              Path = path @ [ content ] })
        }

    let private heuristicScore
        (llm: ILlmService)
        (ctx: AgentContext)
        (config: GoTConfig)
        (goal: string)
        (contextPrelude: string)
        (node: ThoughtNode)
        =
        async {
            let goalText =
                if String.IsNullOrWhiteSpace contextPrelude then
                    goal
                else
                    $"{goal} {contextPrelude}"

            let! similarity, hasMismatch =
                match node.Embedding with
                | Some embed ->
                    async {
                        let! maybeGoalEmbed = maybeEmbed llm ctx config "[GoT] heuristic-goal" goalText

                        match maybeGoalEmbed with
                        | Some goalEmbed ->
                            match tryCosineSimilarity ctx "heuristic" embed goalEmbed with
                            | Some value -> return float value, false
                            | None -> return 0.0, true
                        | None -> return 0.0, false
                    }
                | None -> async { return 0.0, false }

            let baseScore = 0.4 + (similarity * 0.3) |> clamp01

            let baseConfidence = 0.45 + (similarity * 0.25) |> clamp01

            let reasons =
                if hasMismatch then
                    [ "heuristic_embed_mismatch" ]
                else if similarity >= 0.5 then
                    [ "heuristic_cosine_alignment" ]
                else if node.Content.Length > 100 then
                    [ "heuristic_long_form" ]
                else
                    [ "heuristic_fallback" ]

            let risks =
                if hasMismatch then
                    [ "score_parse_failed"; "embedding_dim_mismatch" ]
                else
                    [ "score_parse_failed" ]

            return baseScore, baseConfidence, reasons, risks
        }

    let private scoreThought
        (llm: ILlmService)
        (ctx: AgentContext)
        (config: GoTConfig)
        (goal: string)
        (contextPrelude: string)
        (peers: ThoughtNode list)
        (node: ThoughtNode)
        =
        async {
            match tryConsumeCall ctx "[GoT] scoreThought" with
            | false ->
                let evaluation =
                    { Score = 0.0
                      Confidence = 0.0
                      Reasons = []
                      Risks = [ "budget_exceeded" ] }

                let updated =
                    { node with
                        Score = Some 0.0
                        Evaluation = Some evaluation
                        Operation = Score }

                logEvaluation ctx updated
                return updated
            | true ->
                let state = buildThoughtState goal config.Constraints node.Path contextPrelude

                let request: LlmRequest =
                    { ModelHint = Some "fast"
                      Model = None
                      SystemPrompt =
                        Some
                            $"""You are a strict evaluator of reasoning steps.
Return ONLY a JSON object with fields:
- score: 0.0 to 1.0
- confidence: 0.0 to 1.0
- reasons: array of short strings
- risks: array of short strings
Do not include markdown, code fences, or extra text."""
                      MaxTokens = Some 200
                      Temperature = Some 0.1
                      Stop = []
                      Messages =
                        [ { Role = Role.User
                            Content = $"{state}\n\nThought:\n{node.Content}" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = Some ResponseFormat.Json
                      Stream = false
                      JsonMode = true
                      Seed = None

                      ContextWindow = None }

                let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                let! parsed =
                    match tryParseJsonWithFallback response.Text with
                    | Result.Ok elem -> async { return Result.Ok elem }
                    | Result.Error err ->
                        async {
                            ctx.Logger $"[GoT] Failed to parse score JSON: {err}"

                            if not (tryConsumeCall ctx "[GoT] scoreThought retry") then
                                return Result.Error err
                            else
                                let retryRequest =
                                    { request with
                                        ModelHint = None
                                        ResponseFormat = None
                                        JsonMode = false
                                        Temperature = Some 0.0 }

                                let! retryResponse = llm.CompleteAsync(retryRequest) |> Async.AwaitTask

                                match tryParseJsonWithFallback retryResponse.Text with
                                | Result.Ok elem -> return Result.Ok elem
                                | Result.Error retryErr ->
                                    ctx.Logger $"[GoT] Failed to parse retry score JSON: {retryErr}"
                                    return Result.Error($"{err}; retry: {retryErr}")
                        }

                let! baseScore, confidence, reasons, risks =
                    match parsed with
                    | Result.Ok elem ->
                        async {
                            let score = getDoubleWithDefault [ "score"; "rating"; "value" ] 0.0 elem |> clamp01
                            let confidence = getDoubleWithDefault [ "confidence"; "conf" ] 0.0 elem |> clamp01
                            let reasons = getStringList "reasons" elem
                            let risks = getStringList "risks" elem
                            return score, confidence, reasons, risks
                        }
                    | Result.Error _ -> async { return! heuristicScore llm ctx config goal contextPrelude node }

                let! guardResult =
                    if config.EnablePolicyChecks then
                        let input =
                            { ResponseText = node.Content
                              Grammar = None
                              ExpectedJsonFields = None
                              RequireCitations = false
                              Citations = None
                              AllowExtraFields = true
                              Metadata = Map.empty }

                        OutputGuard.defaultGuard.Evaluate input
                    else
                        async {
                            return
                                { Risk = 0.0
                                  Action = GuardAction.Accept
                                  Messages = [] }
                        }

                let diversityPenalty =
                    match node.Embedding with
                    | Some embed ->
                        let similarities =
                            peers
                            |> List.choose (fun p ->
                                p.Embedding
                                |> Option.bind (fun e -> tryCosineSimilarity ctx "diversity" embed e))

                        match similarities with
                        | [] -> 0.0
                        | sims ->
                            let maxSim = sims |> List.max |> float

                            if maxSim >= config.DiversityThreshold then
                                config.DiversityPenalty * maxSim
                            else
                                0.0
                    | None -> 0.0

                let scoreAfterDiversity = max 0.0 (baseScore - diversityPenalty)
                let adjustedScore = scoreAfterDiversity * (1.0 - guardResult.Risk)

                let policyRisks =
                    match guardResult.Action with
                    | GuardAction.Reject reason -> [ reason ]
                    | GuardAction.RetryWithHint hint -> [ hint ]
                    | GuardAction.AskForEvidence msg -> [ msg ]
                    | GuardAction.Fallback msg -> [ msg ]
                    | GuardAction.Accept -> []

                let evaluation =
                    { Score = adjustedScore
                      Confidence = confidence
                      Reasons = reasons
                      Risks =
                        risks
                        @ (if diversityPenalty > 0.0 then
                               [ "diversity_penalty_applied" ]
                           else
                               [])
                        @ policyRisks }

                let updated =
                    { node with
                        Score = Some adjustedScore
                        Evaluation = Some evaluation
                        Operation = Score }

                logEvaluation ctx updated
                return updated
        }

    let private refineThought
        (llm: ILlmService)
        (ctx: AgentContext)
        (config: GoTConfig)
        (goal: string)
        (contextPrelude: string)
        (node: ThoughtNode)
        =
        async {
            match tryConsumeCall ctx "[GoT] refineThought" with
            | false ->
                return
                    { Id = Guid.NewGuid()
                      Content = node.Content
                      Score = None
                      Evaluation = None
                      Embedding = node.Embedding
                      ParentIds = [ node.Id ]
                      Depth = node.Depth + 1
                      Operation = Refine
                      Path = node.Path @ [ node.Content ] }
            | true ->
                let critique =
                    if config.EnableCritique then
                        node.Evaluation
                        |> Option.map (fun e ->
                            let reasons =
                                if e.Reasons.IsEmpty then
                                    ""
                                else
                                    "Reasons: " + (String.concat "; " e.Reasons)

                            let risks =
                                if e.Risks.IsEmpty then
                                    ""
                                else
                                    "Risks: " + (String.concat "; " e.Risks)

                            String.concat " | " [ reasons; risks ] |> fun s -> s.Trim())
                        |> Option.filter (fun s -> not (String.IsNullOrWhiteSpace s))
                        |> Option.defaultValue ""
                    else
                        ""

                let state = buildThoughtState goal config.Constraints node.Path contextPrelude

                let request: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt =
                        Some
                            $"""Improve and refine this reasoning step to better achieve the goal.
Fix errors, add missing details, and make it more precise and actionable.
Output ONLY the improved thought."""
                      MaxTokens = Some 500
                      Temperature = Some 0.4
                      Stop = []
                      Messages =
                        [ { Role = Role.User
                            Content =
                              if String.IsNullOrWhiteSpace critique then
                                  $"{state}\n\nCurrent thought:\n{node.Content}"
                              else
                                  $"{state}\n\nCurrent thought:\n{node.Content}\n\nCritique:\n{critique}" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                let refined =
                    let trimmed = response.Text.Trim()

                    if String.IsNullOrWhiteSpace trimmed then
                        node.Content
                    else
                        trimmed

                let! embedding = maybeEmbed llm ctx config "[GoT] embed" refined

                return
                    { Id = Guid.NewGuid()
                      Content = refined
                      Score = None
                      Evaluation = None
                      Embedding = embedding
                      ParentIds = [ node.Id ]
                      Depth = node.Depth + 1
                      Operation = Refine
                      Path = node.Path @ [ refined ] }
        }

    let private aggregateThoughts
        (llm: ILlmService)
        (ctx: AgentContext)
        (config: GoTConfig)
        (goal: string)
        (contextPrelude: string)
        (selectedNodes: ThoughtNode list)
        =
        async {
            let fallback =
                selectedNodes
                |> List.tryHead
                |> Option.map (fun n -> n.Content)
                |> Option.defaultValue "No solution found."

            match tryConsumeCall ctx "[GoT] aggregateThoughts" with
            | false ->
                return
                    { Id = Guid.NewGuid()
                      Content = fallback
                      Score = None
                      Evaluation = None
                      Embedding = None
                      ParentIds = selectedNodes |> List.map (fun n -> n.Id)
                      Depth = (selectedNodes |> List.map (fun n -> n.Depth) |> List.max) + 1
                      Operation = Aggregate
                      Path = selectedNodes |> List.map (fun n -> n.Content) }
            | true ->
                let thoughtsList =
                    selectedNodes
                    |> List.mapi (fun i n ->
                        let score =
                            n.Score |> Option.map (fun s -> $" (score: {s:F2})") |> Option.defaultValue ""

                        $"Approach {i + 1}{score}: {n.Content}")
                    |> String.concat "\n\n"

                let state =
                    buildThoughtState
                        goal
                        config.Constraints
                        (selectedNodes |> List.collect (fun n -> n.Path))
                        contextPrelude

                let request: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt =
                        Some
                            $"""Synthesize these approaches into a single coherent solution.
Take the best ideas from each approach and honor the constraints.
Output ONLY the synthesized solution."""
                      MaxTokens = Some 800
                      Temperature = Some 0.3
                      Stop = []
                      Messages =
                        [ { Role = Role.User
                            Content = $"{state}\n\nCandidate approaches:\n{thoughtsList}" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response = llm.CompleteAsync(request) |> Async.AwaitTask

                let content =
                    let trimmed = response.Text.Trim()

                    if String.IsNullOrWhiteSpace trimmed then
                        fallback
                    else
                        trimmed

                let! embedding = maybeEmbed llm ctx config "[GoT] embed" content

                return
                    { Id = Guid.NewGuid()
                      Content = content
                      Score = None
                      Evaluation = None
                      Embedding = embedding
                      ParentIds = selectedNodes |> List.map (fun n -> n.Id)
                      Depth = (selectedNodes |> List.map (fun n -> n.Depth) |> List.max) + 1
                      Operation = Aggregate
                      Path = selectedNodes |> List.map (fun n -> n.Content) }
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

                    ctx.Logger(
                        sprintf "[GoT] Generated: %s..." (thought.Content.Substring(0, min 60 thought.Content.Length))
                    )

                // Phase 2: Iterative expansion
                for depth in 1 .. config.MaxDepth - 1 do
                    ctx.Logger(
                        sprintf "[GoT] Phase 2.%d: Scoring and expanding (frontier size: %d)" depth frontier.Length
                    )

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

                            ctx.Logger(
                                sprintf
                                    "[GoT] Scored %.2f: %s..."
                                    (node.Score |> Option.defaultValue 0.0)
                                    (node.Content.Substring(0, min 40 node.Content.Length))
                            )
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

                    ctx.Logger(
                        sprintf "[ToT] Proposed: %s..." (thought.Content.Substring(0, min 60 thought.Content.Length))
                    )

                // Phase 2: Systematic expansion and evaluation (BFS)
                for depth in 1 .. config.MaxDepth - 1 do
                    ctx.Logger(sprintf "[ToT] Step %d: Evaluating and expanding best paths" (depth + 1))

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

                            ctx.Logger(
                                sprintf
                                    "[ToT] Valued %.2f: %s..."
                                    (node.Score |> Option.defaultValue 0.0)
                                    (node.Content.Substring(0, min 40 node.Content.Length))
                            )
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
                    ctx.Logger(
                        sprintf
                            "[ToT] Selected best path (score: %.2f): %s..."
                            (node.Score |> Option.defaultValue 0.0)
                            (node.Content.Substring(0, min 60 node.Content.Length))
                    )

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
                                                            let! result = tool.Execute input

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

                            let! finalNode =
                                async {
                                    if refined.Length >= 2 then
                                        return! aggregateThoughts llm ctx policyConfig goal contextWithTool refined
                                    elif refined.Length = 1 then
                                        return refined.Head
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
