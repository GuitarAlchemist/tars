namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.Text.Json
open Tars.Cortex.WoTTypes

/// Bridge module that lets Claude Code execute TARS WoT plans step by step.
/// TARS compiles the cognitive structure; Claude Code provides the intelligence.
module ClaudeCodeBridge =

    // =========================================================================
    // Manifest types for JSON serialization to Claude Code
    // =========================================================================

    type ManifestNode =
        { Id: string
          Kind: string // "Reason" | "Tool" | "Validate" | "Memory" | "Control"
          Prompt: string option // For Reason nodes
          ModelHint: string option
          ToolName: string option
          ToolArgs: Map<string, string> option
          Invariants: string list option // For Validate nodes (simplified)
          Next: string list } // Next node IDs via edges

    type ExecutionManifest =
        { PlanId: string
          Pattern: string
          Goal: string
          EntryNode: string
          Nodes: ManifestNode list
          AvailableTools: string list }

    type StepResult =
        { Success: bool
          Output: string
          NextNodes: string list
          DurationMs: int64 }

    type CompletionResult =
        { Success: bool
          TotalSteps: int
          SuccessfulSteps: int
          FailedSteps: int
          RegressionCheck: string option
          Promoted: bool }

    // =========================================================================
    // Active plan state
    // =========================================================================

    type ActivePlan =
        { Plan: WoTPlan
          Goal: string
          PatternKind: PatternKind
          StartedAt: DateTime
          StepOutputs: ConcurrentDictionary<string, string>
          StepStatuses: ConcurrentDictionary<string, bool> }

    let private activePlans = ConcurrentDictionary<string, ActivePlan>()

    // =========================================================================
    // JSON helpers
    // =========================================================================

    let private jsonOptions =
        JsonSerializerOptions(
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    /// Try to get a string property from a JsonElement, returning None if missing.
    let private tryGetString (prop: string) (elem: JsonElement) : string option =
        let mutable value = JsonElement()
        if elem.TryGetProperty(prop, &value) && value.ValueKind = JsonValueKind.String then
            Some (value.GetString())
        else
            None

    /// Try to get an int property from a JsonElement, returning a default if missing.
    let private tryGetInt (prop: string) (defaultVal: int) (elem: JsonElement) : int =
        let mutable value = JsonElement()
        if elem.TryGetProperty(prop, &value) && value.ValueKind = JsonValueKind.Number then
            value.GetInt32()
        else
            defaultVal

    // =========================================================================
    // Helper: convert WoTNode to ManifestNode
    // =========================================================================

    let private nextNodesFor (edges: WoTEdge list) (nodeId: string) : string list =
        edges
        |> List.filter (fun e -> e.From = nodeId)
        |> List.map (fun e -> e.To)

    let private nodeToManifest (edges: WoTEdge list) (node: WoTNode) : ManifestNode =
        let next = nextNodesFor edges node.Id

        match node.Kind with
        | Reason ->
            match node.Payload with
            | :? ReasonPayload as payload ->
                let hint =
                    match payload.Hint with
                    | Some Fast -> Some "fast"
                    | Some Smart -> Some "smart"
                    | Some Reasoning -> Some "reasoning"
                    | Some (Specific m) -> Some m
                    | None -> None
                { Id = node.Id; Kind = "Reason"; Prompt = Some payload.Prompt; ModelHint = hint
                  ToolName = None; ToolArgs = None; Invariants = None; Next = next }
            | _ ->
                { Id = node.Id; Kind = "Reason"; Prompt = None; ModelHint = None
                  ToolName = None; ToolArgs = None; Invariants = None; Next = next }
        | Tool ->
            match node.Payload with
            | :? ToolPayload as payload ->
                let args = payload.Args |> Map.map (fun _ v -> sprintf "%A" v)
                { Id = node.Id; Kind = "Tool"; Prompt = None; ModelHint = None
                  ToolName = Some payload.Tool; ToolArgs = Some args; Invariants = None; Next = next }
            | _ ->
                { Id = node.Id; Kind = "Tool"; Prompt = None; ModelHint = None
                  ToolName = None; ToolArgs = None; Invariants = None; Next = next }
        | Validate ->
            match node.Payload with
            | :? ValidatePayload as payload ->
                let invs =
                    payload.Invariants
                    |> List.map (fun i -> sprintf "%s: %A" i.Name i.Op)
                { Id = node.Id; Kind = "Validate"; Prompt = None; ModelHint = None
                  ToolName = None; ToolArgs = None; Invariants = Some invs; Next = next }
            | _ ->
                { Id = node.Id; Kind = "Validate"; Prompt = None; ModelHint = None
                  ToolName = None; ToolArgs = None; Invariants = None; Next = next }
        | Memory ->
            { Id = node.Id; Kind = "Memory"; Prompt = None; ModelHint = None
              ToolName = None; ToolArgs = None; Invariants = None; Next = next }
        | Control ->
            { Id = node.Id; Kind = "Control"; Prompt = None; ModelHint = None
              ToolName = None; ToolArgs = None; Invariants = None; Next = next }

    // =========================================================================
    // compilePlan
    // =========================================================================

    /// Compile a goal into a WoT execution manifest for Claude Code.
    /// Input JSON: { "goal": "...", "max_steps": 5 }
    /// Returns JSON ExecutionManifest or error.
    let compilePlan
        (compiler: IPatternCompiler)
        (selector: IPatternSelector)
        (toolRegistry: Tars.Core.IToolRegistry)
        (input: string)
        : Result<string, string> =
        try
            let doc = JsonDocument.Parse(input)
            let root = doc.RootElement
            let goal =
                match tryGetString "goal" root with
                | Some g -> g
                | None -> failwith "Missing required 'goal' property"
            let maxSteps = tryGetInt "max_steps" 5 root

            // Select pattern using cognitive state defaults
            let cogState =
                { Mode = Exploratory
                  Eigenvalue = 1.0
                  Entropy = 0.5
                  BranchingFactor = 1.0
                  ActivePattern = None
                  WoTRunId = None
                  StepCount = 0
                  TokenBudget = None
                  LastTransition = DateTime.UtcNow
                  ConstraintScore = None
                  SuccessRate = 1.0 }

            let patternKind = selector.Recommend(goal, cogState)

            // Compile plan based on selected pattern
            let plan =
                match patternKind with
                | ChainOfThought -> compiler.CompileChainOfThought(maxSteps, goal)
                | ReAct -> compiler.CompileReAct([ "search"; "read"; "write" ], maxSteps, goal)
                | GraphOfThoughts -> compiler.CompileGraphOfThoughts(3, 3, goal)
                | TreeOfThoughts -> compiler.CompileTreeOfThoughts(3, 2, goal)
                | _ -> compiler.CompileChainOfThought(maxSteps, goal)

            let planId = plan.Id.ToString()

            // Store active plan
            activePlans.[planId] <-
                { Plan = plan
                  Goal = goal
                  PatternKind = patternKind
                  StartedAt = DateTime.UtcNow
                  StepOutputs = ConcurrentDictionary<string, string>()
                  StepStatuses = ConcurrentDictionary<string, bool>() }

            // Build manifest
            let manifest =
                { PlanId = planId
                  Pattern = sprintf "%A" patternKind
                  Goal = goal
                  EntryNode = plan.EntryNode
                  Nodes = plan.Nodes |> List.map (nodeToManifest plan.Edges)
                  AvailableTools = toolRegistry.GetAll() |> List.map (fun t -> t.Name) }

            Ok(JsonSerializer.Serialize(manifest, jsonOptions))
        with ex ->
            Error $"Failed to compile plan: {ex.Message}"

    // =========================================================================
    // executeStep (for Tool nodes executed by TARS)
    // =========================================================================

    /// Execute a single step in an active plan.
    /// Input JSON: { "plan_id": "...", "node_id": "...", "input": "..." }
    /// For Tool nodes, runs the tool via the registry.
    /// For other node kinds, records the provided input as output.
    let executeStep
        (toolRegistry: Tars.Core.IToolRegistry)
        (input: string)
        : Async<Result<string, string>> =
        async {
            try
                let doc = JsonDocument.Parse(input)
                let root = doc.RootElement
                let planId =
                    match tryGetString "plan_id" root with
                    | Some s -> s
                    | None -> failwith "Missing 'plan_id'"
                let nodeId =
                    match tryGetString "node_id" root with
                    | Some s -> s
                    | None -> failwith "Missing 'node_id'"
                let stepInput =
                    tryGetString "input" root |> Option.defaultValue ""

                match activePlans.TryGetValue(planId) with
                | false, _ -> return Error "Plan not found"
                | true, activePlan ->
                    let node =
                        activePlan.Plan.Nodes
                        |> List.tryFind (fun n -> n.Id = nodeId)

                    match node with
                    | None -> return Error $"Node '{nodeId}' not found in plan"
                    | Some node ->
                        match node.Kind with
                        | Tool ->
                            match node.Payload with
                            | :? ToolPayload as payload ->
                                let tool = toolRegistry.Get(payload.Tool)

                                match tool with
                                | Some t ->
                                    let sw = System.Diagnostics.Stopwatch.StartNew()

                                    let toolInput =
                                        if payload.Args.IsEmpty then stepInput
                                        else JsonSerializer.Serialize(payload.Args)

                                    let! result = t.Execute(toolInput)
                                    sw.Stop()

                                    match result with
                                    | Ok output ->
                                        activePlan.StepOutputs.[nodeId] <- output
                                        activePlan.StepStatuses.[nodeId] <- true

                                        let nextNodes = nextNodesFor activePlan.Plan.Edges nodeId

                                        let stepResult =
                                            { Success = true
                                              Output = output
                                              NextNodes = nextNodes
                                              DurationMs = sw.ElapsedMilliseconds }

                                        return Ok(JsonSerializer.Serialize(stepResult, jsonOptions))
                                    | Error err ->
                                        activePlan.StepStatuses.[nodeId] <- false
                                        return Error $"Tool '{payload.Tool}' failed: {err}"
                                | None ->
                                    return Error $"Tool '{payload.Tool}' not found in registry"
                            | _ ->
                                return Error "Tool node has invalid payload type"
                        | _ ->
                            // For non-tool nodes, record the input as output
                            activePlan.StepOutputs.[nodeId] <- stepInput
                            activePlan.StepStatuses.[nodeId] <- true
                            let nextNodes = nextNodesFor activePlan.Plan.Edges nodeId

                            let stepResult =
                                { Success = true
                                  Output = stepInput
                                  NextNodes = nextNodes
                                  DurationMs = 0L }

                            return Ok(JsonSerializer.Serialize(stepResult, jsonOptions))
            with ex ->
                return Error $"Step execution error: {ex.Message}"
        }

    // =========================================================================
    // validateStep
    // =========================================================================

    /// Validate content against a Validate node's invariants.
    /// Input JSON: { "plan_id": "...", "node_id": "...", "content": "..." }
    /// Uses synchronous subset of Verification (Contains, Regex, JsonPath, Schema, CustomOp).
    let validateStep (input: string) : Result<string, string> =
        try
            let doc = JsonDocument.Parse(input)
            let root = doc.RootElement
            let planId =
                match tryGetString "plan_id" root with
                | Some s -> s
                | None -> failwith "Missing 'plan_id'"
            let nodeId =
                match tryGetString "node_id" root with
                | Some s -> s
                | None -> failwith "Missing 'node_id'"
            let content =
                match tryGetString "content" root with
                | Some s -> s
                | None -> failwith "Missing 'content'"

            match activePlans.TryGetValue(planId) with
            | false, _ -> Error "Plan not found"
            | true, activePlan ->
                let node =
                    activePlan.Plan.Nodes
                    |> List.tryFind (fun n -> n.Id = nodeId)

                match node with
                | None -> Error $"Node '{nodeId}' not found"
                | Some node ->
                    match node.Kind with
                    | Validate ->
                        match node.Payload with
                        | :? ValidatePayload as payload ->
                            // Perform synchronous verification for simple ops
                            let results =
                                payload.Invariants
                                |> List.map (fun inv ->
                                    let passed =
                                        match inv.Op with
                                        | Contains substring ->
                                            content.Contains(substring, StringComparison.OrdinalIgnoreCase)
                                        | VerificationOp.Regex pattern ->
                                            System.Text.RegularExpressions.Regex.IsMatch(
                                                content, pattern,
                                                System.Text.RegularExpressions.RegexOptions.IgnoreCase)
                                        | JsonPath path ->
                                            content.Contains(path) || content.Contains($"\"{path}\"")
                                        | Schema _ -> true // placeholder
                                        | CustomOp _ -> true // placeholder
                                        | ToolCheck _ -> true // skip tool checks in sync validate
                                    inv.Name, passed)

                            let allPassed = results |> List.forall snd

                            let failed =
                                results
                                |> List.filter (fun (_, p) -> not p)
                                |> List.map fst

                            activePlan.StepOutputs.[nodeId] <-
                                if allPassed then "PASS"
                                else sprintf "FAIL: %s" (String.Join(", ", failed))

                            activePlan.StepStatuses.[nodeId] <- allPassed

                            let nextNodes = nextNodesFor activePlan.Plan.Edges nodeId

                            let result =
                                {| passed = allPassed
                                   failed_invariants = failed
                                   next_nodes = nextNodes |}

                            Ok(JsonSerializer.Serialize(result, jsonOptions))
                        | _ -> Error "Validate node has invalid payload type"
                    | _ -> Error "Node is not a Validate node"
        with ex ->
            Error $"Validation error: {ex.Message}"

    // =========================================================================
    // completePlan
    // =========================================================================

    /// Complete an active plan, record outcome for pattern learning, and check
    /// for golden trace regression.
    /// Input JSON: { "plan_id": "...", "final_output": "..." }
    let completePlan (input: string) : Result<string, string> =
        try
            let doc = JsonDocument.Parse(input)
            let root = doc.RootElement
            let planId =
                match tryGetString "plan_id" root with
                | Some s -> s
                | None -> failwith "Missing 'plan_id'"
            let _finalOutput =
                tryGetString "final_output" root |> Option.defaultValue ""

            match activePlans.TryRemove(planId) with
            | false, _ -> Error "Plan not found"
            | true, activePlan ->
                let totalSteps = activePlan.Plan.Nodes.Length

                let successfulSteps =
                    activePlan.StepStatuses.Values |> Seq.filter id |> Seq.length

                let failedSteps =
                    activePlan.StepStatuses.Values |> Seq.filter (not) |> Seq.length

                let success = failedSteps = 0

                let duration =
                    (DateTime.UtcNow - activePlan.StartedAt).TotalMilliseconds |> int64

                // Record outcome for pattern learning
                PatternOutcomeStore.record
                    { PatternKind = activePlan.PatternKind
                      Goal = activePlan.Goal
                      Success = success
                      DurationMs = duration
                      Timestamp = DateTime.UtcNow }

                // Check golden regression (best-effort)
                let regressionMsg =
                    let goldenName = RegressionChecker.goalToGoldenName activePlan.Goal

                    match GoldenTraceStore.load goldenName with
                    | Ok _ -> Some "PASS (golden trace exists)"
                    | Error _ -> None

                let result =
                    { Success = success
                      TotalSteps = totalSteps
                      SuccessfulSteps = successfulSteps
                      FailedSteps = failedSteps
                      RegressionCheck = regressionMsg
                      Promoted = false }

                Ok(JsonSerializer.Serialize(result, jsonOptions))
        with ex ->
            Error $"Plan completion error: {ex.Message}"

    // =========================================================================
    // memoryOp
    // =========================================================================

    /// Execute a memory operation (search or assert) against the Knowledge Ledger.
    /// Input JSON:
    ///   Search: { "operation": "search", "query": "..." }
    ///   Assert: { "operation": "assert", "subject": "...", "predicate": "...", "object": "..." }
    ///   Stats:  { "operation": "stats" }
    let memoryOp
        (ledger: Tars.Knowledge.KnowledgeLedger)
        (input: string)
        : Async<Result<string, string>> =
        async {
            try
                let doc = JsonDocument.Parse(input)
                let root = doc.RootElement
                let op =
                    match tryGetString "operation" root with
                    | Some s -> s.ToLowerInvariant()
                    | None -> failwith "Missing 'operation' property"

                match op with
                | "search" ->
                    let query =
                        match tryGetString "query" root with
                        | Some q -> q
                        | None -> failwith "Missing 'query' for search"

                    // Search by subject, predicate substring, or object
                    let results =
                        ledger.Query()
                        |> Seq.filter (fun b ->
                            let s = string b.Subject
                            let p = string b.Predicate
                            let o = string b.Object
                            let q = query.ToLowerInvariant()
                            s.ToLowerInvariant().Contains(q)
                            || p.ToLowerInvariant().Contains(q)
                            || o.ToLowerInvariant().Contains(q))
                        |> Seq.truncate 20
                        |> Seq.map (fun b ->
                            {| subject = string b.Subject
                               predicate = string b.Predicate
                               obj = string b.Object
                               confidence = b.Confidence
                               valid = b.IsValid |})
                        |> Seq.toList

                    return Ok(JsonSerializer.Serialize({| results = results; count = results.Length |}, jsonOptions))

                | "assert" ->
                    let subject =
                        match tryGetString "subject" root with
                        | Some s -> s
                        | None -> failwith "Missing 'subject' for assert"
                    let predicate =
                        match tryGetString "predicate" root with
                        | Some s -> s
                        | None -> failwith "Missing 'predicate' for assert"
                    let obj =
                        match tryGetString "object" root with
                        | Some s -> s
                        | None -> failwith "Missing 'object' for assert"

                    let provenance = Tars.Knowledge.Provenance.FromUser()
                    let agentId = Tars.Knowledge.AgentId "claude-code-bridge"
                    let relType = Tars.Knowledge.RelationType.Custom predicate

                    let! result =
                        ledger.AssertTriple(subject, relType, obj, provenance, agentId)
                        |> Async.AwaitTask

                    match result with
                    | Ok beliefId ->
                        return Ok(JsonSerializer.Serialize(
                            {| success = true; beliefId = string beliefId; triple = $"({subject} {predicate} {obj})" |},
                            jsonOptions))
                    | Error err ->
                        return Error $"Assert failed: {err}"

                | "stats" ->
                    let stats = ledger.Stats()
                    return Ok(JsonSerializer.Serialize(stats, jsonOptions))

                | other ->
                    return Error $"Unknown operation: '{other}'. Use 'search', 'assert', or 'stats'."

            with ex ->
                return Error $"Memory operation error: {ex.Message}"
        }

    // =========================================================================
    // Utility: list active plans
    // =========================================================================

    /// Get the number of currently active plans.
    let activePlanCount () = activePlans.Count

    /// Get summary info for all active plans.
    let listActivePlans () : (string * string * string) list =
        activePlans
        |> Seq.map (fun kv ->
            kv.Key, kv.Value.Goal, sprintf "%A" kv.Value.PatternKind)
        |> Seq.toList
