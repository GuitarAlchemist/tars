namespace Tars.Cortex

open System
open System.Text.Json
open Tars.Core
open Tars.Llm
open System.Text
open Tars.Cortex.ThoughtParsing

/// The Graph-of-Thoughts engine shared by the Graph-, Tree- and Workflow-of-Thought
/// patterns: node and edge types, configs, and the generate/score/refine/aggregate
/// helpers. Split out of Patterns.fs so the helpers can be tested directly (#280).
module ThoughtGraph =
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

    let recordBranchDecision (ctx: AgentContext) (node: ThoughtNode) (action: string) (status: string) =
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

            ctx.Logger
                $"[ReasoningAudit] action=%s{decision.Action} status=%s{decision.Status} score=%s{scoreText} conf=%s{confText} node=\"%s{truncatedContent}\" reasons=%s{reasons} risks=%s{risks}"

    let logEvaluation (ctx: AgentContext) (node: ThoughtNode) =
        recordBranchDecision ctx node "score" "scored"

    let logThresholdDecision (ctx: AgentContext) (node: ThoughtNode) (passed: bool) =
        let status = if passed then "kept" else "pruned"
        recordBranchDecision ctx node "threshold" status



    // =========================================================================
    // Graph-of-Thoughts & Tree-of-Thoughts Internals
    // =========================================================================

    let buildContextPrelude (ctx: AgentContext) (goal: string) =
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

    let truncateForPrompt (value: string) =
        let trimmed = value.Trim()

        if trimmed.Length <= 160 then
            trimmed
        else
            trimmed.Substring(0, 160) + "..."

    let renderConstraints (constraints: string list) =
        if constraints.IsEmpty then
            "None"
        else
            constraints |> List.map truncateForPrompt |> String.concat "; "

    let renderPath (path: string list) =
        if path.IsEmpty then
            "None"
        else
            path
            |> List.rev
            |> List.truncate 4
            |> List.rev
            |> List.mapi (fun i step -> $"{i + 1}. {truncateForPrompt step}")
            |> String.concat " | "

    let buildThoughtState
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

    let tryConsumeCall (ctx: AgentContext) (label: string) =
        match ctx.Budget with
        | Some budget ->
            match budget.TryConsumeCall() with
            | Result.Ok() -> true
            | Result.Error err ->
                ctx.Logger $"[Budget] {label} skipped: {err}"
                false
        | None -> true

    let maybeEmbed (llm: ILlmService) (ctx: AgentContext) (config: GoTConfig) (label: string) (text: string) =
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

    let tryCosineSimilarity (ctx: AgentContext) (label: string) (v1: float32[]) (v2: float32[]) =
        if v1.Length <> v2.Length then
            ctx.Logger $"[GoT] {label} embedding length mismatch: {v1.Length} vs {v2.Length}"
            None
        else
            Some(MetricSpace.cosineSimilarity v1 v2)

    let generateThoughts
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
                      MaxTokens = Some 4000
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

                ctx.Logger
                    $"[GoT] generateThoughts prompt length: {state.Length}, response length: {initialText.Length}"

                if initialText.Length > 0 && initialText.Length < 100 then
                    ctx.Logger $"[GoT] Short response: {initialText}"

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
                    ctx.Logger $"[GoT] No thoughts generated; LLM response was: \"{responseText}\""
                    return []
                else
                    ctx.Logger $"[GoT] Successfully generated {cleanedThoughts.Length} thoughts."

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

    let heuristicScore
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

    let scoreThought
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

    let refineThought
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
                      MaxTokens = Some 2000
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
                    let rawText = response.Text.Trim()

                    let stripped =
                        System.Text.RegularExpressions.Regex
                            .Replace(rawText, @"(?s)<thinking>.*?</thinking>", "")
                            .Trim()

                    if String.IsNullOrWhiteSpace stripped then
                        if String.IsNullOrWhiteSpace rawText then
                            node.Content
                        else
                            rawText
                    else
                        stripped

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

    let aggregateThoughts
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
                      MaxTokens = Some 4000
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
                    let rawText = response.Text.Trim()

                    let stripped =
                        System.Text.RegularExpressions.Regex
                            .Replace(rawText, @"(?s)<thinking>.*?</thinking>", "")
                            .Trim()

                    if String.IsNullOrWhiteSpace stripped then
                        if String.IsNullOrWhiteSpace rawText then
                            fallback
                        else
                            rawText
                    else
                        stripped

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
