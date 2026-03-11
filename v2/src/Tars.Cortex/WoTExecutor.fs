namespace Tars.Cortex

open System
open System.Diagnostics
open System.Text.Json
open Tars.Llm
open Tars.Core
open Tars.Cortex.WoTTypes

/// <summary>
/// Executes WoT plans with full tracing and cognitive state management.
/// This is the universal execution engine for all reasoning patterns.
/// </summary>
module WoTExecutor =

    // =========================================================================
    // Execution Context
    // =========================================================================

    /// <summary>
    /// Context for WoT execution, containing all necessary services.
    /// </summary>
    type ExecutionContext =
        { Llm: ILlmService
          Tools: Tars.Core.IToolRegistry
          Logger: string -> unit
          OnProgress: WoTTraceStep -> unit
          CancellationToken: System.Threading.CancellationToken
          KnowledgeGraph: Tars.Core.IGraphService option
          Reflector: Tars.Core.ISymbolicReflector option }

    // =========================================================================
    // Context Engineering: Progressive Context Disclosure
    // =========================================================================

    /// Select only tools relevant to the current step based on node type and prompt keywords.
    let private selectRelevantTools (tools: Tars.Core.IToolRegistry) (node: WoTNode) : Tars.Core.Tool list =
        let allTools = tools.GetAll()
        match node.Kind with
        | Tool ->
            // For Tool nodes, only include the specific tool being called + related tools
            match node.Payload with
            | :? ToolPayload as payload ->
                allTools
                |> List.filter (fun t ->
                    t.Name = payload.Tool
                    || payload.Args |> Map.exists (fun _ v ->
                        let vs = string v
                        vs.Contains(t.Name)))
            | _ -> allTools |> List.truncate 10
        | Reason ->
            // For Reason nodes, include tools whose descriptions match prompt keywords
            match node.Payload with
            | :? ReasonPayload as payload ->
                let promptLower = payload.Prompt.ToLowerInvariant()
                let keywords =
                    promptLower.Split([|' '; ','; '.'; '?'; '!'|], System.StringSplitOptions.RemoveEmptyEntries)
                let relevant =
                    allTools
                    |> List.filter (fun t ->
                        let descLower = t.Description.ToLowerInvariant()
                        keywords |> Array.exists (fun kw -> kw.Length > 3 && descLower.Contains(kw)))
                if relevant.IsEmpty then allTools |> List.truncate 5
                else relevant |> List.truncate 15
            | _ -> allTools |> List.truncate 10
        | Validate -> [] // Validation nodes don't need tools
        | Memory ->
            allTools
            |> List.filter (fun t ->
                t.Name.Contains("memory")
                || t.Name.Contains("search")
                || t.Name.Contains("know"))
        | Control -> [] // Control nodes don't need tools

    /// Build a focused system prompt for the current step, including only relevant previous outputs.
    let private buildStepContext
        (node: WoTNode)
        (edges: WoTEdge list)
        (stepOutputs: Map<string, string>)
        : string =
        let contextParts = ResizeArray<string>()

        // Find which node IDs feed into the current node via edges
        let incomingIds =
            edges
            |> List.filter (fun e -> e.To = node.Id)
            |> List.map (fun e -> e.From)
            |> Set.ofList

        // Select only outputs from upstream nodes (or all if no edges exist)
        let relevantOutputs =
            if incomingIds.IsEmpty then
                // No explicit edges — include nothing extra (rely on sequential lastOutput)
                Map.empty
            else
                stepOutputs
                |> Map.filter (fun key _ -> incomingIds.Contains(key))

        if not relevantOutputs.IsEmpty then
            contextParts.Add("Previous results:")
            for KeyValue(key, value) in relevantOutputs do
                let preview = if value.Length > 500 then value.[..499] + "..." else value
                contextParts.Add(sprintf "  [%s]: %s" key preview)

        String.concat "\n" contextParts

    // =========================================================================
    // Node Execution
    // =========================================================================

    /// Execute a Think node
    let private executeThink
        (ctx: ExecutionContext)
        (id: string)
        (prompt: string)
        (hint: ModelHint option)
        (stepContext: string)
        : Async<Result<string, string>> =
        async {
            let promptPreview =
                if prompt.Length > 50 then
                    prompt.Substring(0, 50) + "..."
                else
                    prompt

            ctx.Logger $"[WoT] Think: %s{promptPreview}"

            let modelHint =
                match hint with
                | Some Fast -> Some "fast"
                | Some Smart -> Some "smart"
                | Some Reasoning -> Some "reasoning"
                | Some(Specific m) -> Some m
                | None -> None

            // Progressive context: prepend relevant step context to the prompt
            let enrichedPrompt =
                if String.IsNullOrWhiteSpace(stepContext) then prompt
                else sprintf "%s\n\n%s" stepContext prompt

            let request: LlmRequest =
                { LlmRequest.Default with
                    ModelHint = modelHint
                    SystemPrompt = Some "You are TARS, an autonomous reasoning agent."
                    Messages = [ { Role = Role.User; Content = enrichedPrompt } ]
                    MaxTokens = Some 1024
                    Temperature = Some 0.7 }

            try
                let! response = ctx.Llm.CompleteAsync(request) |> Async.AwaitTask
                ctx.Logger $"[WoT] Think completed: %d{response.Text.Length} chars"
                return Result.Ok response.Text
            with ex ->
                ctx.Logger $"[WoT] Think failed: %s{ex.Message}"
                return Result.Error ex.Message
        }

    /// Execute an Act node
    let private executeAct
        (ctx: ExecutionContext)
        (id: string)
        (toolName: string)
        (args: Map<string, obj>)
        : Async<Result<string, string>> =
        async {
            ctx.Logger $"[WoT] Act: %s{toolName}"

            match ctx.Tools.Get(toolName) with
            | Some tool ->
                let input =
                    if args.IsEmpty then
                        ""
                    else
                        args
                        |> Map.toList
                        |> List.map (fun (k, v) -> $"\"%s{k}\": \"{v}\"")
                        |> String.concat ", "
                        |> sprintf "{ %s }"

                let! toolResult = tool.Execute(input)

                match toolResult with
                | Result.Ok output ->
                    ctx.Logger $"[WoT] Tool %s{toolName} succeeded: %d{output.Length} chars"
                    return Result.Ok output
                | Result.Error err ->
                    ctx.Logger $"[WoT] Tool %s{toolName} failed: %s{err}"
                    return Result.Error err
            | None ->
                let err = $"Tool not found: %s{toolName}"
                ctx.Logger $"[WoT] %s{err}"
                return Result.Error err
        }

    /// Execute an Observe node
    let private executeObserve
        (ctx: ExecutionContext)
        (id: string)
        (input: string)
        (transform: (string -> string) option)
        : Async<Result<string, string>> =
        async {
            ctx.Logger $"[WoT] Observe: %d{input.Length} chars"

            let output =
                match transform with
                | Some f -> f input
                | None -> input

            return Result.Ok output
        }

    /// Execute a Decide node
    let private executeDecide
        (ctx: ExecutionContext)
        (id: string)
        (candidates: string list)
        (criteria: string list)
        : Async<Result<string, string>> =
        async {
            ctx.Logger $"[WoT] Decide: %d{candidates.Length} candidates"

            let candidateList =
                candidates |> List.mapi (fun i c -> $"%d{i + 1}. %s{c}") |> String.concat "\n"

            let criteriaText =
                if criteria.IsEmpty then
                    "Choose the best option."
                else
                    criteria |> String.concat "; "

            let prompt =
                $"""Select the best option based on: %s{criteriaText}

Options:
%s{candidateList}

Respond with ONLY the number of your choice."""

            let! result = executeThink ctx id prompt (Some Fast) ""

            match result with
            | Result.Ok response ->
                // Parse the selection
                let cleaned = response.Trim()

                let selected =
                    if cleaned.Length > 0 then
                        match Int32.TryParse(cleaned.Chars(0).ToString()) with
                        | true, idx when idx > 0 && idx <= candidates.Length -> candidates.[idx - 1]
                        | _ -> candidates.Head
                    else
                        candidates.Head

                ctx.Logger $"[WoT] Decided: %s{selected}"
                return Result.Ok selected
            | Result.Error e -> return Result.Error e
        }

    /// Execute a Sync node
    let private executeSync (ctx: ExecutionContext) (id: string) (op: MemoryOp) : Async<Result<string, string>> =
        async {
            ctx.Logger $"[WoT] Sync: %A{op}"

            // Memory operations are currently stubs - will integrate with KG
            match op with
            | Query sparql ->
                ctx.Logger $"[WoT] Query: %s{sparql}"
                return Result.Ok "[]"
            | Assert(s, p, o) ->
                ctx.Logger $"[WoT] Assert: (%s{s}, %s{p}, %s{o})"
                return Result.Ok $"Asserted: %s{s} %s{p} %s{o}"
            | Retract(s, p, o) ->
                ctx.Logger $"[WoT] Retract: (%s{s}, %s{p}, %s{o})"
                return Result.Ok $"Retracted: %s{s} %s{p} %s{o}"
            | Search(_, topK) ->
                ctx.Logger $"[WoT] Search: top %d{topK}"
                return Result.Ok "[]"
        }

    /// Execute a Validate node
    let private executeValidate
        (ctx: ExecutionContext)
        (id: string)
        (invariants: WoTInvariant list)
        (lastOutput: string)
        : Async<Result<string, string>> =
        async {
            ctx.Logger $"[WoT] Validate: %d{invariants.Length} invariants"

            let toolExecutor name (args: Map<string, obj>) =
                async {
                    match ctx.Tools.Get(name) with
                    | Some tool ->
                        let input =
                            if args.IsEmpty then
                                "{}"
                            else
                                JsonSerializer.Serialize(args)

                        return! tool.Execute(input)
                    | None -> return Result.Error $"Tool '{name}' not found"
                }

            let! checks =
                invariants
                |> List.map (fun inv ->
                    async {
                        let! res = Verification.verify lastOutput inv.Op toolExecutor

                        let passed =
                            match res with
                            | Result.Ok p -> p
                            | Result.Error _ -> false

                        return (inv.Name, passed, inv.Weight)
                    })
                |> Async.Sequential

            let results = checks |> List.ofArray

            let totalWeight = results |> List.sumBy (fun (_, _, w) -> w)

            let passedWeight =
                results |> List.filter (fun (_, p, _) -> p) |> List.sumBy (fun (_, _, w) -> w)

            let score =
                if totalWeight > 0.0 then
                    passedWeight / totalWeight
                else
                    1.0

            let failed =
                results |> List.filter (fun (_, p, _) -> not p) |> List.map (fun (n, _, _) -> n)

            if failed.IsEmpty then
                ctx.Logger $"[WoT] Validation passed: score %.2f{score}"
                return Result.Ok $"Validation passed with score %.2f{score}"
            else
                let msg =
                    sprintf "Validation failed: %s (score %.2f)" (String.concat ", " failed) score

                ctx.Logger $"[WoT] %s{msg}"
                return Result.Error msg
        }

    // =========================================================================
    // Condition Evaluation
    // =========================================================================

    /// Resolve ${variable} references in a string using step outputs as context.
    let private resolveVariables (stepOutputs: Map<string, string>) (text: string) : string =
        let varRx = System.Text.RegularExpressions.Regex(@"\$\{([^}]+)\}")
        varRx.Replace(text, fun (m: System.Text.RegularExpressions.Match) ->
            let key = m.Groups.[1].Value.Trim()
            match stepOutputs.TryFind key with
            | Some v -> v
            | None -> m.Value)

    /// Try to parse a string as a float, returning None on failure.
    let private tryParseFloat (s: string) =
        match Double.TryParse(s.Trim(), Globalization.NumberStyles.Any, Globalization.CultureInfo.InvariantCulture) with
        | true, v -> Some v
        | false, _ -> None

    /// Evaluate a simple condition expression after variable resolution.
    /// Supports: comparisons (>, <, >=, <=, ==, !=) with numeric or string values,
    /// and boolean literals (true/false).
    let private evaluateConditionExpr (expr: string) : bool =
        let trimmed = expr.Trim()
        // Boolean literals
        match trimmed.ToLowerInvariant() with
        | "true" -> true
        | "false" -> false
        | _ ->
            // Try comparison operators (ordered longest-first to match >= before >)
            let ops = [ ">="; "<="; "!="; "=="; ">"; "<" ]
            let found =
                ops |> List.tryPick (fun op ->
                    let idx = trimmed.IndexOf(op)
                    if idx >= 0 then
                        let lhs = trimmed.Substring(0, idx).Trim().Trim('"')
                        let rhs = trimmed.Substring(idx + op.Length).Trim().Trim('"')
                        Some (lhs, op, rhs)
                    else None)
            match found with
            | Some (lhs, op, rhs) ->
                // Try numeric comparison first
                match tryParseFloat lhs, tryParseFloat rhs with
                | Some l, Some r ->
                    match op with
                    | ">"  -> l > r
                    | "<"  -> l < r
                    | ">=" -> l >= r
                    | "<=" -> l <= r
                    | "==" -> abs (l - r) < 1e-9
                    | "!=" -> abs (l - r) >= 1e-9
                    | _    -> false
                | _ ->
                    // Fall back to string comparison
                    match op with
                    | "==" -> lhs = rhs
                    | "!=" -> lhs <> rhs
                    | _    -> false // String ordering comparisons not supported
            | None ->
                // No operator found; treat non-empty as true
                not (String.IsNullOrWhiteSpace trimmed)

    /// Check if a node's condition (from metadata Extra) is satisfied.
    /// Returns true if no condition exists (unconditional execution).
    let private evaluateCondition
        (ctx: ExecutionContext)
        (node: WoTNode)
        (stepOutputs: Map<string, string>)
        : bool =
        match node.Metadata.Extra.TryFind "condition" with
        | None -> true // No condition = always execute
        | Some condExpr ->
            let resolved = resolveVariables stepOutputs condExpr
            let result = evaluateConditionExpr resolved
            if not result then
                ctx.Logger $"[WoT] Condition not met for %s{node.Id}: %s{condExpr} (resolved: %s{resolved})"
            result

    // =========================================================================
    // Parallel Grouping
    // =========================================================================

    /// A segment of the execution: either a single node or a parallel group.
    type private ExecutionSegment =
        | Single of WoTNode
        | ParallelGroup of group: string * nodes: WoTNode list

    /// Group a list of nodes into execution segments, merging consecutive nodes
    /// that share the same parallel_group metadata.
    let private groupIntoSegments (nodes: WoTNode list) : ExecutionSegment list =
        let folder (acc: ExecutionSegment list) (node: WoTNode) =
            match node.Metadata.Extra.TryFind "parallel_group" with
            | None -> acc @ [ Single node ]
            | Some groupId ->
                match acc with
                | [] -> [ ParallelGroup(groupId, [ node ]) ]
                | _ ->
                    let last = List.last acc
                    let init = acc |> List.take (acc.Length - 1)
                    match last with
                    | ParallelGroup(gid, members) when gid = groupId ->
                        init @ [ ParallelGroup(gid, members @ [ node ]) ]
                    | _ ->
                        acc @ [ ParallelGroup(groupId, [ node ]) ]
        nodes |> List.fold folder []

    // =========================================================================
    // Main Executor
    // =========================================================================

    /// Execute a single node and return a trace step
    let private executeNode
        (ctx: ExecutionContext)
        (node: WoTNode)
        (lastOutput: string)
        (edges: WoTEdge list)
        (stepOutputs: Map<string, string>)
        : Async<WoTTraceStep * string> =
        async {
            let sw = Stopwatch.StartNew()
            let startedAt = DateTime.UtcNow
            let id = PatternCompiler.nodeId node

            let nodeType = node.Kind.ToString()

            // Progressive context disclosure: build focused context for this step
            let stepContext = buildStepContext node edges stepOutputs
            let relevantTools = selectRelevantTools ctx.Tools node
            let toolCount = relevantTools |> List.length
            let allToolCount = ctx.Tools.GetAll() |> List.length
            if toolCount < allToolCount then
                ctx.Logger $"[WoT] Context: %d{toolCount}/%d{allToolCount} tools selected for %s{nodeType} node %s{id}"

            let! result =
                match node.Kind with
                | Reason ->
                    match node.Payload with
                    | :? ReasonPayload as p -> executeThink ctx id p.Prompt p.Hint stepContext
                    | _ -> async { return Result.Error "Invalid Reason Payload" }

                | Tool ->
                    match node.Payload with
                    | :? ToolPayload as p -> executeAct ctx id p.Tool p.Args
                    | _ -> async { return Result.Error "Invalid Tool Payload" }

                | Validate ->
                    match node.Payload with
                    | :? ValidatePayload as p -> executeValidate ctx id p.Invariants lastOutput
                    | _ -> async { return Result.Error "Invalid Validate Payload" }

                | Memory ->
                    match node.Payload with
                    | :? MemoryPayload as p -> executeSync ctx id p.Operation
                    | _ -> async { return Result.Error "Invalid Memory Payload" }

                | Control ->
                    match node.Payload with
                    | :? ControlPayload as p ->
                        match p with
                        | Decide(candidates, criteria) -> executeDecide ctx id candidates criteria
                        | Observe(input, transform) ->
                            let actualInput = if String.IsNullOrEmpty input then lastOutput else input
                            executeObserve ctx id actualInput transform
                        | Parallel _ -> async { return Result.Ok "Parallel execution completed" }
                        | Branch _ -> async { return Result.Ok "Branch evaluated" }
                        | Loop _ -> async { return Result.Ok "Loop completed" }
                    | _ -> async { return Result.Error "Invalid Control Payload" }

            sw.Stop()

            let (status, output) =
                match result with
                | Result.Ok out -> (Completed(out, sw.ElapsedMilliseconds), Some out)
                | Result.Error err -> (Failed(err, sw.ElapsedMilliseconds), None)

            let step =
                { NodeId = id
                  NodeType = nodeType
                  StartedAt = startedAt
                  Status = status
                  Input = Some lastOutput
                  Output = output
                  Confidence = None
                  TokensUsed = None }

            ctx.OnProgress(step)

            return (step, output |> Option.defaultValue lastOutput)
        }

    /// Execute a WoT plan
    let execute (ctx: ExecutionContext) (plan: WoTPlan) : Async<WoTResult> =
        async {
            let runId = Guid.NewGuid()
            let startedAt = DateTime.UtcNow
            let mutable steps: WoTTraceStep list = []
            let mutable currentOutput = ""
            let mutable toolsUsed: string list = []
            let mutable warnings: string list = []
            let mutable errors: string list = []

            ctx.Logger $"[WoT] Starting execution: %A{plan.Metadata.Kind} (%d{plan.Nodes.Length} nodes)"

            // Register run in KH
            match ctx.KnowledgeGraph with
            | Some kg ->
                let runE =
                    RunE
                        { Id = runId
                          Goal = plan.Metadata.SourceGoal
                          Pattern = plan.Metadata.Kind.ToString()
                          Timestamp = startedAt }

                let! _ = kg.AddNodeAsync(runE) |> Async.AwaitTask
                ()
            | None -> ()

            // Track per-node outputs for progressive context disclosure
            let mutable stepOutputs: Map<string, string> = Map.empty

            // Helper: process a single executed node result (KG recording, tracking)
            let recordNodeResult (node: WoTNode) (step: WoTTraceStep) (output: string) =
                async {
                    let nodeId = PatternCompiler.nodeId node
                    stepOutputs <- stepOutputs |> Map.add nodeId output

                    // Record step in KG
                    match ctx.KnowledgeGraph with
                    | Some kg ->
                        let stepE =
                            StepE
                                { RunId = runId
                                  StepId = node.Id
                                  NodeType = node.Kind.ToString()
                                  Content = output
                                  Timestamp = DateTime.UtcNow }

                        let! _ = kg.AddNodeAsync(stepE) |> Async.AwaitTask

                        let runESkeleton =
                            RunE
                                { Id = runId
                                  Goal = ""
                                  Pattern = ""
                                  Timestamp = DateTime.MinValue }

                        let! _ = kg.AddFactAsync(TarsFact.Contains(runESkeleton, stepE)) |> Async.AwaitTask

                        if not steps.IsEmpty then
                            let prev = steps |> List.last

                            let prevE =
                                StepE
                                    { RunId = runId
                                      StepId = prev.NodeId
                                      NodeType = ""
                                      Content = ""
                                      Timestamp = DateTime.MinValue }

                            let! _ = kg.AddFactAsync(TarsFact.NextStep(prevE, stepE)) |> Async.AwaitTask
                            ()
                        else
                            ()
                    | None -> ()

                    steps <- steps @ [ step ]
                    currentOutput <- output

                    match node.Kind with
                    | Tool ->
                        match node.Payload with
                        | :? ToolPayload as p -> toolsUsed <- p.Tool :: toolsUsed
                        | _ -> ()
                    | _ -> ()

                    match step.Status with
                    | Failed(err, _) -> errors <- err :: errors
                    | Completed _ -> ()
                    | Pending -> ()
                    | Running -> ()
                    | Skipped _ -> ()
                }

            // Group nodes into sequential and parallel segments
            let segments = groupIntoSegments plan.Nodes

            for segment in segments do
                if not ctx.CancellationToken.IsCancellationRequested then
                    match segment with
                    | Single node ->
                        // Check condition before executing
                        if evaluateCondition ctx node stepOutputs then
                            let! (step, output) = executeNode ctx node currentOutput plan.Edges stepOutputs
                            do! recordNodeResult node step output
                        else
                            // Skip node: record a Skipped trace step
                            let skipStep =
                                { NodeId = node.Id
                                  NodeType = node.Kind.ToString()
                                  StartedAt = DateTime.UtcNow
                                  Status = Skipped "Condition not met"
                                  Input = Some currentOutput
                                  Output = None
                                  Confidence = None
                                  TokensUsed = None }
                            ctx.OnProgress(skipStep)
                            steps <- steps @ [ skipStep ]

                    | ParallelGroup(groupId, nodes) ->
                        ctx.Logger $"[WoT] Executing parallel group '%s{groupId}' with %d{nodes.Length} nodes"

                        // Filter nodes by condition, skip those that don't pass
                        let executableNodes =
                            nodes |> List.filter (fun n -> evaluateCondition ctx n stepOutputs)
                        let skippedNodes =
                            nodes |> List.filter (fun n -> not (evaluateCondition ctx n stepOutputs))

                        // Record skipped nodes
                        for skipped in skippedNodes do
                            let skipStep =
                                { NodeId = skipped.Id
                                  NodeType = skipped.Kind.ToString()
                                  StartedAt = DateTime.UtcNow
                                  Status = Skipped "Condition not met"
                                  Input = Some currentOutput
                                  Output = None
                                  Confidence = None
                                  TokensUsed = None }
                            ctx.OnProgress(skipStep)
                            steps <- steps @ [ skipStep ]

                        // Execute all eligible nodes in parallel
                        if not executableNodes.IsEmpty then
                            let! parallelResults =
                                executableNodes
                                |> List.map (fun node ->
                                    executeNode ctx node currentOutput plan.Edges stepOutputs)
                                |> Async.Parallel

                            // Record results from all parallel nodes
                            for (node, (step, output)) in List.zip executableNodes (Array.toList parallelResults) do
                                do! recordNodeResult node step output

                            // Set currentOutput to the combined output of the last parallel node
                            let lastOutput =
                                parallelResults
                                |> Array.toList
                                |> List.map snd
                                |> List.last
                            currentOutput <- lastOutput

            if ctx.CancellationToken.IsCancellationRequested then
                warnings <- "Execution cancelled" :: warnings

            let completedAt = DateTime.UtcNow
            let totalDuration = int64 ((completedAt - startedAt).TotalMilliseconds)

            // Calculate metrics
            let successCount =
                steps
                |> List.filter (fun s ->
                    match s.Status with
                    | Completed _ -> true
                    | _ -> false)
                |> List.length

            let failedCount =
                steps
                |> List.filter (fun s ->
                    match s.Status with
                    | Failed _ -> true
                    | _ -> false)
                |> List.length

            let totalTokens = steps |> List.choose (fun s -> s.TokensUsed) |> List.sum

            let branchingFactor =
                if plan.Nodes.Length > 0 then
                    float plan.Edges.Length / float plan.Nodes.Length
                else
                    1.0

            let trace =
                { RunId = runId
                  Plan = plan
                  Steps = steps
                  StartedAt = startedAt
                  CompletedAt = Some completedAt
                  FinalStatus = if errors.IsEmpty then "Success" else "Failed" }

            let metrics =
                { TotalSteps = steps.Length
                  SuccessfulSteps = successCount
                  FailedSteps = failedCount
                  TotalTokens = totalTokens
                  TotalDurationMs = totalDuration
                  BranchingFactor = branchingFactor
                  ConstraintScore = None }

            ctx.Logger
                $"[WoT] Execution completed: %d{steps.Length} steps, %d{successCount} successful, %d{failedCount} failed"

            let tempResult =
                { Output = currentOutput
                  Success = errors.IsEmpty
                  Trace = trace
                  TriplesDelta = []
                  ToolsUsed = toolsUsed |> List.distinct
                  Metrics = metrics
                  Warnings = warnings
                  Errors = errors
                  CognitiveStateAfter = None }

            // Calculate final cognitive state
            let finalState =
                CognitiveStateManager.update CognitiveStateManager.initialState tempResult

            ctx.Logger
                $"[WoT] Cognitive State: %A{finalState.Mode} (Entropy: %.2f{finalState.Entropy}, Eigen: %.2f{finalState.Eigenvalue})"

            // Ingest state into Knowledge Graph
            match ctx.KnowledgeGraph with
            | Some kg ->
                // 1. Ingest State
                let episode =
                    Tars.Core.Episode.CognitiveStateUpdate(
                        runId,
                        finalState.Mode.ToString(),
                        finalState.Entropy,
                        finalState.Eigenvalue,
                        DateTime.UtcNow
                    )

                // Ingest async
                try
                    do! kg.AddEpisodeAsync(episode) |> Async.AwaitTask |> Async.Ignore

                    // 2. Ingest Trace
                    let runEnt: Tars.Core.RunEntity =
                        { Id = runId
                          Goal = plan.Metadata.SourceGoal
                          Pattern = plan.Metadata.Kind.ToString()
                          Timestamp = startedAt }

                    let runNode = Tars.Core.TarsEntity.RunE runEnt
                    do! kg.AddNodeAsync(runNode) |> Async.AwaitTask |> Async.Ignore

                    let mutable prevNode = None

                    for step in steps do
                        let stepEnt: Tars.Core.StepEntity =
                            { RunId = runId
                              StepId = step.NodeId
                              NodeType = step.NodeType
                              Content = step.Output |> Option.defaultValue ""
                              Timestamp = step.StartedAt }

                        let stepNode = Tars.Core.TarsEntity.StepE stepEnt
                        do! kg.AddNodeAsync(stepNode) |> Async.AwaitTask |> Async.Ignore

                        // Run -> Contains -> Step
                        let contains = Tars.Core.TarsFact.Contains(runNode, stepNode)
                        do! kg.AddFactAsync(contains) |> Async.AwaitTask |> Async.Ignore

                        // Prev -> NextStep -> Current
                        match prevNode with
                        | Some prev ->
                            let next = Tars.Core.TarsFact.NextStep(prev, stepNode)
                            do! kg.AddFactAsync(next) |> Async.AwaitTask |> Async.Ignore
                        | None -> ()

                        prevNode <- Some stepNode

                with ex ->
                    ctx.Logger $"[WoT] Failed to ingest trace: %s{ex.Message}"
            | None -> ()

            // Perform symbolic reflection if available
            match ctx.Reflector with
            | Some reflector ->
                try
                    ctx.Logger(sprintf "[WoT] Triggering symbolic reflection for run: %A" runId)
                    let! reflectionRes = reflector.ReflectOnRunAsync(runId, []) |> Async.AwaitTask

                    match reflectionRes with
                    | Microsoft.FSharp.Core.Ok reflection ->
                        ctx.Logger(
                            sprintf "[WoT] Reflection completed. Found %d observations." reflection.Observations.Length
                        )
                    | Microsoft.FSharp.Core.Error err -> ctx.Logger(sprintf "[WoT] Reflection failed: %s" err)
                with ex ->
                    ctx.Logger(sprintf "[WoT] Reflection error: %s" ex.Message)
            | None -> ()

            return
                { tempResult with
                    CognitiveStateAfter = Some finalState }
        }

    // =========================================================================
    // Executor Implementation
    // =========================================================================

    /// <summary>
    /// Default WoT executor implementation.
    /// </summary>
    type DefaultWoTExecutor(llm: ILlmService, tools: Tars.Core.IToolRegistry) =

        member private this.CreateContext(logger, onProgress, ct, kg, reflector) =
            { Llm = llm
              Tools = tools
              Logger = logger
              OnProgress = onProgress
              CancellationToken = ct
              KnowledgeGraph = kg
              Reflector = reflector }

        interface IWoTExecutor with
            member this.Execute(plan, context) =
                let execCtx =
                    this.CreateContext(
                        context.Logger,
                        ignore,
                        context.CancellationToken,
                        context.KnowledgeGraph,
                        context.SymbolicReflector
                    )

                execute execCtx plan

            member this.ExecuteWithProgress(plan, context, onProgress) =
                let execCtx =
                    this.CreateContext(
                        context.Logger,
                        onProgress,
                        context.CancellationToken,
                        context.KnowledgeGraph,
                        context.SymbolicReflector
                    )

                execute execCtx plan

    // =========================================================================
    // Convenience Functions
    // =========================================================================

    /// Create a default executor
    let createExecutor llm tools =
        DefaultWoTExecutor(llm, tools) :> IWoTExecutor

    /// Execute a pattern directly (compile + run)
    let executePattern
        (compiler: IPatternCompiler)
        (executor: IWoTExecutor)
        (pattern: PatternKind)
        (goal: string)
        (context: Tars.Core.AgentContext)
        : Async<WoTResult> =
        async {
            let plan =
                match pattern with
                | ChainOfThought -> compiler.CompileChainOfThought(5, goal)
                | ReAct -> compiler.CompileReAct([ "search"; "calculate"; "read" ], 10, goal)
                | GraphOfThoughts -> compiler.CompileGraphOfThoughts(3, 3, goal)
                | TreeOfThoughts -> compiler.CompileTreeOfThoughts(3, 2, goal)
                | WorkflowOfThought -> failwith "WoT requires explicit nodes"
                | PlanAndExecute -> compiler.CompileChainOfThought(3, goal)
                | Custom _ -> compiler.CompileChainOfThought(3, goal)

            return! executor.Execute(plan, context)
        }
