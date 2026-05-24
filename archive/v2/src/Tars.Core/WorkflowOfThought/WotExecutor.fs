namespace Tars.Core.WorkflowOfThought

open System
open VariableResolution
open Tars.Core.WorkflowOfThought // Explicit open

type VerifyResult = { Passed: bool; Errors: string list }

module WotExecutor =
    // ... (omitted)

    let private storeSingleOutput (ctx: ExecContext) (outputs: string list) (value: obj) : Result<ExecContext, string> =
        match outputs with
        | [] -> Result.Ok ctx
        | [ name ] ->
            Result.Ok
                { ctx with
                    Vars = ctx.Vars.Add(name, value) }
        | many ->
            let joined = String.Join(", ", many)
            Result.Error $"v0 supports at most one output var per step. Got: {joined}"

    let private runVerify (toolInvoker: IToolInvoker) (ctx: ExecContext) (checks: WotCheck list) : Async<VerifyResult> =
        async {
            let mutable errs = []

            for check in checks do
                let! res =
                    match check with
                    | WotCheck.NonEmpty v ->
                        async {
                            match ctx.Vars.TryFind v with
                            | Some(:? string as s) when not (String.IsNullOrWhiteSpace s) -> return None
                            | Some(:? string) -> return Some $"NonEmpty failed: '{v}' is empty."
                            | Some _ -> return Some $"NonEmpty failed: '{v}' is not a string."
                            | None -> return Some $"NonEmpty failed: missing var '{v}'."
                        }
                    | WotCheck.Contains(v, needle) ->
                        async {
                            match resolveString ctx needle with
                            | Error e -> return Some $"Contains check failed: could not resolve needle - {e}"
                            | Ok resolvedNeedle ->
                                match ctx.Vars.TryFind v with
                                | Some(:? string as s) when s.Contains resolvedNeedle -> return None
                                | Some(:? string) ->
                                    return Some $"Contains failed: '{v}' missing needle '{resolvedNeedle}'."
                                | Some _ -> return Some $"Contains failed: '{v}' is not a string."
                                | None -> return Some $"Contains failed: missing var '{v}'."
                        }
                    | WotCheck.RegexMatch(v, pattern) ->
                        async {
                            match ctx.Vars.TryFind v with
                            | Some(:? string as s) ->
                                try
                                    let re = System.Text.RegularExpressions.Regex(pattern)

                                    if re.IsMatch s then
                                        return None
                                    else
                                        return Some $"RegexMatch failed: '{s}' does not match pattern '{pattern}'."
                                with ex ->
                                    return Some $"RegexMatch error: invalid pattern '{pattern}' - {ex.Message}"
                            | Some _ -> return Some $"RegexMatch failed: '{v}' is not a string."
                            | None -> return Some $"RegexMatch failed: missing var '{v}'."
                        }
                    | WotCheck.SchemaMatch(v, schema) ->
                        async {
                            match ctx.Vars.TryFind v with
                            | Some(:? string as s) ->
                                try
                                    use doc = System.Text.Json.JsonDocument.Parse(s)
                                    let root = doc.RootElement
                                    // Parse schema as JSON to extract required fields and types
                                    try
                                        use schemaParsed = System.Text.Json.JsonDocument.Parse(schema)
                                        let schemaRoot = schemaParsed.RootElement
                                        let mutable errors = []

                                        // Check "type" constraint
                                        match schemaRoot.TryGetProperty("type") with
                                        | true, typeProp ->
                                            let expectedType = typeProp.GetString()
                                            let actualKind = root.ValueKind
                                            let typeMatch =
                                                match expectedType with
                                                | "object" -> actualKind = System.Text.Json.JsonValueKind.Object
                                                | "array" -> actualKind = System.Text.Json.JsonValueKind.Array
                                                | "string" -> actualKind = System.Text.Json.JsonValueKind.String
                                                | "number" | "integer" -> actualKind = System.Text.Json.JsonValueKind.Number
                                                | "boolean" -> actualKind = System.Text.Json.JsonValueKind.True || actualKind = System.Text.Json.JsonValueKind.False
                                                | _ -> true
                                            if not typeMatch then
                                                errors <- $"Expected type '%s{expectedType}' but got '%A{actualKind}'" :: errors
                                        | false, _ -> ()

                                        // Check "required" fields
                                        match schemaRoot.TryGetProperty("required") with
                                        | true, reqProp when reqProp.ValueKind = System.Text.Json.JsonValueKind.Array ->
                                            for reqField in reqProp.EnumerateArray() do
                                                let fieldName = reqField.GetString()
                                                match root.TryGetProperty(fieldName) with
                                                | true, _ -> ()
                                                | false, _ -> errors <- $"Missing required field '%s{fieldName}'" :: errors
                                        | _ -> ()

                                        // Check "properties" field types
                                        match schemaRoot.TryGetProperty("properties") with
                                        | true, propsDef when root.ValueKind = System.Text.Json.JsonValueKind.Object ->
                                            for prop in propsDef.EnumerateObject() do
                                                match root.TryGetProperty(prop.Name) with
                                                | true, actualVal ->
                                                    match prop.Value.TryGetProperty("type") with
                                                    | true, fieldType ->
                                                        let ft = fieldType.GetString()
                                                        let ok =
                                                            match ft with
                                                            | "string" -> actualVal.ValueKind = System.Text.Json.JsonValueKind.String
                                                            | "number" | "integer" -> actualVal.ValueKind = System.Text.Json.JsonValueKind.Number
                                                            | "boolean" -> actualVal.ValueKind = System.Text.Json.JsonValueKind.True || actualVal.ValueKind = System.Text.Json.JsonValueKind.False
                                                            | "array" -> actualVal.ValueKind = System.Text.Json.JsonValueKind.Array
                                                            | "object" -> actualVal.ValueKind = System.Text.Json.JsonValueKind.Object
                                                            | _ -> true
                                                        if not ok then
                                                            errors <- $"Field '%s{prop.Name}' expected type '%s{ft}' but got '%A{actualVal.ValueKind}'" :: errors
                                                    | false, _ -> ()
                                                | false, _ -> () // Not required, skip
                                        | _ -> ()

                                        if errors.IsEmpty then
                                            return None
                                        else
                                            let errMsg = String.Join("; ", errors)
                                            return Some $"SchemaMatch failed: {errMsg}"
                                    with _ ->
                                        // Schema isn't valid JSON — fall back to basic JSON parse check
                                        return None
                                with ex ->
                                    return Some $"SchemaMatch failed: invalid JSON - {ex.Message}"
                            | Some _ -> return Some $"SchemaMatch failed: '{v}' is not a string."
                            | None -> return Some $"SchemaMatch failed: missing var '{v}'."
                        }
                    | WotCheck.ToolResult(toolName, args, check) ->
                        async {
                            // Convert string args to obj args for resolver
                            let boxedArgs = args |> Map.map (fun _ v -> box v)

                            match resolveToolArgs ctx boxedArgs with
                            | Result.Error e -> return Some $"ToolResult check failed to resolve args: {e}"
                            | Result.Ok resolvedArgs ->
                                let! r = toolInvoker.Invoke(toolName, resolvedArgs)

                                match r with
                                | Result.Error e -> return Some $"ToolResult failed: Tool '{toolName}' error: {e}"
                                | Result.Ok res ->
                                    let resStr = res.ToString()

                                    if resStr.Contains check then
                                        return None
                                    else
                                        return
                                            Some $"ToolResult failed: Tool result '{resStr}' does not contain '{check}'"
                        }
                    | other -> async { return Some $"Check not implemented in v0: {other}" }

                match res with
                | Some e -> errs <- e :: errs
                | None -> ()

            return
                { Passed = errs.IsEmpty
                  Errors = List.rev errs }
        }

    type ExecutionPolicy =
        { AllowedTools: Set<string>
          MaxToolCalls: int }

    let executePlanV0
        (toolInvoker: IToolInvoker)
        (reasoner: IReasoner)
        (graphService: Tars.Core.IGraphService option)
        (reflector: Tars.Core.ISymbolicReflector option)
        (constitution: Tars.Core.AgentConstitution option)
        (mode: ReasonStepMode)
        (policy: ExecutionPolicy)
        (inputs: Map<string, string>)
        (planSteps: Step list)
        : Async<Result<ExecContext * VerifyResult option * TraceEvent list, string * TraceEvent list>> =
        async {
            let runId = Guid.NewGuid()
            let startedAt = DateTime.UtcNow

            // Register run in KG
            match graphService with
            | Some kg ->
                let runE =
                    Tars.Core.RunE
                        { Id = runId
                          Goal = "WotV0 Execution"
                          Pattern = "WorkflowOfThought"
                          Timestamp = startedAt }

                let! _ = kg.AddNodeAsync(runE) |> Async.AwaitTask
                ()
            | None -> ()

            let mutable ctx = { Inputs = inputs; Vars = Map.empty }
            let mutable lastVerify: VerifyResult option = None
            let mutable toolCallCount = 0
            let traces = ResizeArray<TraceEvent>()

            let addTrace
                (stepId: string)
                kind
                (startUtc: DateTime)
                (endUtc: DateTime)
                (toolName: string option)
                (args: Map<string, string> option)
                outputs
                status
                err
                usage
                metadata
                =
                let duration = int64 (endUtc - startUtc).TotalMilliseconds

                traces.Add(
                    { StepId = stepId
                      Kind = kind
                      StartedAtUtc = startUtc
                      EndedAtUtc = endUtc
                      DurationMs = duration
                      ToolName = toolName
                      ResolvedArgs = args
                      Outputs = outputs
                      Status = status
                      Error = err
                      Usage = usage
                      Metadata = metadata }
                )

            // Using a recursive loop for clean early exit
            let rec loop steps currentCtx (fbState: FeedbackState) =
                async {
                    match steps with
                    | [] -> return Result.Ok(currentCtx, lastVerify, Seq.toList traces)
                    | (step: Step) :: rest ->
                        let startUtc = DateTime.UtcNow

                        // Helper to record error and return
                        let fail
                            (msg: string)
                            (kind: string)
                            (tool: string option)
                            (args: Map<string, string> option)
                            (meta: Meta)
                            =
                            let endUtc = DateTime.UtcNow

                            addTrace
                                step.Id
                                kind
                                startUtc
                                endUtc
                                tool
                                args
                                step.Outputs
                                StepStatus.Error
                                (Some msg)
                                None
                                meta

                            Result.Error(msg, Seq.toList traces)

                        // Helper to record success
                        let succeed
                            (kind: string)
                            (tool: string option)
                            (args: Map<string, string> option)
                            resultValue
                            usage
                            meta
                            =
                            async {
                                let endUtc = DateTime.UtcNow

                                addTrace
                                    step.Id
                                    kind
                                    startUtc
                                    endUtc
                                    tool
                                    args
                                    step.Outputs
                                    StepStatus.Ok
                                    None
                                    usage
                                    meta

                                // Record step in KG
                                match graphService with
                                | Some kg ->
                                    let stepE =
                                        Tars.Core.StepE
                                            { RunId = runId
                                              StepId = step.Id
                                              NodeType = kind
                                              Content = sprintf "%A" resultValue
                                              Timestamp = DateTime.UtcNow }

                                    let! _ = kg.AddNodeAsync(stepE) |> Async.AwaitTask

                                    // Link to run
                                    let runESkeleton =
                                        Tars.Core.RunE
                                            { Id = runId
                                              Goal = ""
                                              Pattern = ""
                                              Timestamp = DateTime.MinValue }

                                    let! _ =
                                        kg.AddFactAsync(Tars.Core.TarsFact.Contains(runESkeleton, stepE))
                                        |> Async.AwaitTask

                                    // Link to previous step
                                    if traces.Count > 1 then
                                        let prev = traces.[traces.Count - 2]

                                        let prevE =
                                            Tars.Core.StepE
                                                { RunId = runId
                                                  StepId = prev.StepId
                                                  NodeType = ""
                                                  Content = ""
                                                  Timestamp = DateTime.MinValue }

                                        let! _ =
                                            kg.AddFactAsync(Tars.Core.TarsFact.NextStep(prevE, stepE))
                                            |> Async.AwaitTask

                                        ()
                                    else
                                        ()
                                | None -> ()
                            }

                        match step.Action with
                        | StepAction.Work(WorkOperation.ToolCall(toolName, args)) ->
                            // Phase 14: Constitution Enforcement
                            let constiResult =
                                match constitution with
                                | Some consti ->
                                    let action =
                                        match toolName.ToLowerInvariant() with
                                        | "read_code"
                                        | "read_file" ->
                                            let path = args.TryFind "path" |> Option.defaultValue "" |> sprintf "%A"
                                            Tars.Core.AgentAction.ReadFile path
                                        | "write_to_file"
                                        | "patch_code" ->
                                            let path =
                                                args.TryFind "TargetFile"
                                                |> Option.defaultValue (args.TryFind "path" |> Option.defaultValue "")
                                                |> sprintf "%A"

                                            Tars.Core.AgentAction.WriteFile path
                                        | "fetch_webpage"
                                        | "search_web" ->
                                            let url =
                                                args.TryFind "url"
                                                |> Option.defaultValue (args.TryFind "query" |> Option.defaultValue "")
                                                |> sprintf "%A"

                                            Tars.Core.AgentAction.NetworkRequest url
                                        | _ -> Tars.Core.AgentAction.ExecuteTool(toolName, sprintf "%A" args)

                                    Tars.Core.ContractEnforcement.validateAction consti action
                                | None -> Result.Ok()

                            match constiResult with
                            | Result.Error violation ->
                                return
                                    fail
                                        $"ConstitutionViolation: %A{violation}"
                                        "tool"
                                        (Some toolName)
                                        None
                                        step.Metadata
                            | Result.Ok() ->
                                // Legacy Policy Enforcement
                                if not (policy.AllowedTools.Contains toolName) then
                                    return
                                        fail
                                            $"PolicyViolation: Tool '{toolName}' is not in the allowed list."
                                            "tool"
                                            (Some toolName)
                                            None
                                            step.Metadata
                                elif toolCallCount >= policy.MaxToolCalls then
                                    return
                                        fail
                                            $"PolicyViolation: Maximum tool calls ({policy.MaxToolCalls}) exceeded."
                                            "tool"
                                            (Some toolName)
                                            None
                                            step.Metadata
                                else
                                    toolCallCount <- toolCallCount + 1

                                    match resolveToolArgs currentCtx args with
                                    | Result.Error e ->
                                        return
                                            fail
                                                $"Step '{step.Id}' resolve args failed: {e}"
                                                "tool"
                                                (Some toolName)
                                                None
                                                step.Metadata
                                    | Result.Ok resolvedArgs ->
                                        let! r = toolInvoker.Invoke(toolName, resolvedArgs)

                                        match r with
                                        | Result.Error e ->
                                            return
                                                fail
                                                    $"Tool '{toolName}' failed: {e}"
                                                    "tool"
                                                    (Some toolName)
                                                    (Some resolvedArgs)
                                                    step.Metadata
                                        | Result.Ok value ->
                                            match storeSingleOutput currentCtx step.Outputs value with
                                            | Result.Error e ->
                                                return
                                                    fail
                                                        $"Step '{step.Id}' output failed: {e}"
                                                        "tool"
                                                        (Some toolName)
                                                        (Some resolvedArgs)
                                                        step.Metadata
                                            | Result.Ok ctx2 ->
                                                // Phase 17.4: Update feedback loop with rich evidence
                                                let ev =
                                                    { Id = step.Id
                                                      Source = ToolContribution(toolName, step.Id)
                                                      Content = $"Result from tool '{toolName}'"
                                                      Confidence = 1.0
                                                      Weight = 1.0
                                                      IsContradiction = false
                                                      ParentIds =
                                                        step.Inputs
                                                        |> List.choose (fun inp ->
                                                            FeedbackLoop.findEvidenceBySourceId inp fbState
                                                            |> Option.map (fun e -> e.Id))
                                                      Timestamp = DateTime.UtcNow }

                                                let fbState2 = FeedbackLoop.addEvidence ev fbState

                                                do!
                                                    succeed
                                                        "tool"
                                                        (Some toolName)
                                                        (Some resolvedArgs)
                                                        value
                                                        None
                                                        step.Metadata

                                                return! loop rest ctx2 fbState2

                        | StepAction.Work(WorkOperation.Verify checks) ->
                            let! vr = runVerify toolInvoker currentCtx checks
                            lastVerify <- Some vr
                            // store verification output too (optional but handy)
                            match storeSingleOutput currentCtx step.Outputs (box vr) with
                            | Result.Error e ->
                                return fail $"Step '{step.Id}' output failed: {e}" "verify" None None step.Metadata
                            | Result.Ok ctx2 ->
                                // Phase 15.2: Log failure if verification failed
                                if not vr.Passed then
                                    let combinedErrors = vr.Errors |> String.concat "; "
                                    do! SymbolicMemory.logFailure runId (Some step.Id) combinedErrors Map.empty

                                // Verification failure is NOT execution failure in v0
                                do! succeed "verify" None None vr None step.Metadata
                                return! loop rest ctx2 fbState

                        | StepAction.Reason op ->
                            let goal, instruction =
                                match op with
                                | ReasonOperation.Generate t -> Some $"Generate: {t}", None
                                | ReasonOperation.Plan g -> Some g, None
                                | ReasonOperation.Critique t -> Some $"Critique %A{t}", None
                                | ReasonOperation.Synthesize src -> Some $"Synthesize nodes: %A{src}", None
                                | ReasonOperation.Explain t -> Some $"Explain {t}", None
                                | ReasonOperation.Rewrite(t, i) -> Some $"Rewrite %A{t}", Some i
                                | ReasonOperation.Aggregate src ->
                                    let ids =
                                        src
                                        |> List.map (function
                                            | NodeId s -> s)

                                    let content = FeedbackLoop.aggregateEvidence ids fbState
                                    Some $"Aggregate evidence from: %A{src}", Some $"Context from sources:\n{content}"
                                | ReasonOperation.Refine t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    let context = FeedbackLoop.summarizeEvidence tid fbState

                                    Some $"Refine thought: %A{t}",
                                    Some $"Previous evidence for this thought:\n{context}"
                                | ReasonOperation.Contradict t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    let context = FeedbackLoop.summarizeEvidence tid fbState

                                    Some $"Find contradictions for: %A{t}",
                                    Some $"Current evidence to challenge:\n{context}"
                                | ReasonOperation.Distill t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    let context = FeedbackLoop.summarizeEvidence tid fbState
                                    Some $"Distill core claims from: %A{t}", Some $"Raw evidence to distill:\n{context}"
                                | ReasonOperation.Backtrack t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    Some $"Backtrack from failed path: %A{t}",
                                    Some
                                        $"The reasoning path leading to '{tid}' has been flagged as problematic. Propose an alternative strategy."
                                | ReasonOperation.Score t -> Some $"Score hypothesis at: %A{t}", None
                                | ReasonOperation.VerifyConsensus p ->
                                    let reached, explain = FeedbackLoop.checkConsensus p fbState
                                    let statusStr = if reached then "Passed" else "Failed"
                                    Some $"Verify multi-agent consensus ({statusStr}): {explain}", None

                            match mode with
                            | ReasonStepMode.Stub ->
                                let stubValue =
                                    { Content = "<reason-step-stub>"
                                      Usage = None }

                                match storeSingleOutput currentCtx step.Outputs stubValue.Content with
                                | Result.Error e ->
                                    return fail $"Step '{step.Id}' output failed: {e}" "reason" None None step.Metadata
                                | Result.Ok ctx2 ->
                                    // Phase 17.4: Evidence Provenance logic (Stub version)
                                    let fbState2 =
                                        match op with
                                        | ReasonOperation.Score t ->
                                            let h: HypothesisScore =
                                                FeedbackLoop.score
                                                    (string t)
                                                    "Stub Hypothesis"
                                                    0.5
                                                    0.5
                                                    0.5
                                                    0
                                                    0.5
                                                    stubValue.Content
                                                    []
                                                    []
                                                    0 // Volume

                                            FeedbackLoop.registerHypothesis h fbState
                                        | ReasonOperation.Refine t ->
                                            let ev =
                                                { Id = $"stub_ev_{step.Id}"
                                                  Source = ReasonerThought step.Id
                                                  Content = stubValue.Content
                                                  Confidence = 1.0
                                                  Weight = 0.5
                                                  IsContradiction = false
                                                  ParentIds =
                                                    match FeedbackLoop.findEvidenceBySourceId (string t) fbState with
                                                    | Some p -> [ p.Id ]
                                                    | None -> []
                                                  Timestamp = DateTime.UtcNow }

                                            let s2, _ = FeedbackLoop.updateHypothesis (string t) ev fbState
                                            s2
                                        | ReasonOperation.Contradict t ->
                                            let ev =
                                                { Id = $"stub_ev_con_{step.Id}"
                                                  Source = ReasonerThought step.Id
                                                  Content = stubValue.Content
                                                  Confidence = 1.0
                                                  Weight = 0.5
                                                  IsContradiction = true
                                                  ParentIds =
                                                    match FeedbackLoop.findEvidenceBySourceId (string t) fbState with
                                                    | Some p -> [ p.Id ]
                                                    | None -> []
                                                  Timestamp = DateTime.UtcNow }

                                            let s2, _ = FeedbackLoop.updateHypothesis (string t) ev fbState
                                            s2
                                        | _ -> fbState

                                    do! succeed "reason" None None stubValue.Content None step.Metadata


                                    return! loop rest ctx2 fbState2
                            | ReasonStepMode.Llm
                            | ReasonStepMode.Replay ->
                                // Call reasoner with stepId - Llm calls LLM, Replay reads journal
                                let! r = reasoner.Reason(step.Id, currentCtx, goal, instruction, step.Agent)

                                match r with
                                | Result.Error e ->
                                    // Phase 15.2: Log to symbolic memory
                                    do! SymbolicMemory.logFailure runId (Some step.Id) e Map.empty
                                    return fail $"Reason step '{step.Id}' failed: {e}" "reason" None None step.Metadata
                                | Result.Ok(res: ReasoningResult) ->
                                    match storeSingleOutput currentCtx step.Outputs res.Content with
                                    | Result.Error e ->
                                        return
                                            fail $"Step '{step.Id}' output failed: {e}" "reason" None None step.Metadata
                                    | Result.Ok ctx2 ->
                                        // Phase 17.4: Evidence Provenance logic
                                        let fbState2, injectedSteps =
                                            match op with
                                            | ReasonOperation.Score t ->
                                                // Register new hypothesis from scoring node
                                                // Calculate initial volume from parent 't'
                                                let volume =
                                                    match FeedbackLoop.findEvidenceBySourceId (string t) fbState with
                                                    | Some ev -> 1 // Or compute volume of ev
                                                    | None -> 0

                                                let parseScore (c: string) =
                                                    let lines = c.Split('\n')

                                                    let scoreLine =
                                                        lines |> Array.tryFind (fun l -> l.StartsWith("Score:"))

                                                    match scoreLine with
                                                    | Some s ->
                                                        match Double.TryParse(s.Replace("Score:", "").Trim()) with
                                                        | true, v -> v
                                                        | _ -> 0.5
                                                    | None -> 0.5

                                                let scoreVal = parseScore res.Content

                                                let h: HypothesisScore =
                                                    FeedbackLoop.score
                                                        (string t)
                                                        "Hypothesis"
                                                        scoreVal
                                                        scoreVal
                                                        scoreVal
                                                        0
                                                        scoreVal
                                                        res.Content
                                                        [] // No initial evidence
                                                        [] // No conflicts yet
                                                        volume

                                                let st = FeedbackLoop.registerHypothesis h fbState

                                                // ---------------------------------------------------------
                                                // CONTROLLER INTEGRATION
                                                // ---------------------------------------------------------
                                                // Ask Router what to do with this scored hypothesis
                                                let decision = WotController.Router.decide h 0.7 10

                                                let newSteps =
                                                    match decision with
                                                    | WotController.Expand(nid, k) ->
                                                        [ 1..k ]
                                                        |> List.map (fun i ->
                                                            { step with
                                                                Id = Guid.NewGuid().ToString()
                                                                Action =
                                                                    StepAction.Reason(
                                                                        ReasonOperation.Generate(
                                                                            $"Expansion {i} of {nid}"
                                                                        )
                                                                    ) })
                                                    | WotController.Finalize nid ->
                                                        [ { step with
                                                              Id = Guid.NewGuid().ToString()
                                                              Action =
                                                                  StepAction.Reason(ReasonOperation.Distill(NodeId nid)) } ]
                                                    | WotController.Backtrack nid ->
                                                        [ { step with
                                                              Id = Guid.NewGuid().ToString()
                                                              Action =
                                                                  StepAction.Reason(
                                                                      ReasonOperation.Backtrack(NodeId nid)
                                                                  ) } ]
                                                    | WotController.Refine nid ->
                                                        [ { step with
                                                              Id = Guid.NewGuid().ToString()
                                                              Action =
                                                                  StepAction.Reason(ReasonOperation.Refine(NodeId nid)) } ]
                                                    | _ -> []

                                                (st, newSteps)

                                            | ReasonOperation.Refine t ->
                                                // Create supporting evidence
                                                let ev =
                                                    { Id = $"ev_{step.Id}"
                                                      Source = ReasonerThought step.Id
                                                      Content = res.Content
                                                      Confidence = 0.8
                                                      Weight = 0.6
                                                      IsContradiction = false
                                                      ParentIds =
                                                        match
                                                            FeedbackLoop.findEvidenceBySourceId (string t) fbState
                                                        with
                                                        | Some p -> [ p.Id ]
                                                        | None -> []
                                                      Timestamp = DateTime.UtcNow }

                                                let s2, _ = FeedbackLoop.updateHypothesis (string t) ev fbState
                                                (s2, [])

                                            | ReasonOperation.Contradict t ->
                                                // Create contradicting evidence
                                                let ev =
                                                    { Id = $"ev_con_{step.Id}"
                                                      Source = ReasonerThought step.Id
                                                      Content = res.Content
                                                      Confidence = 0.9
                                                      Weight = 0.7
                                                      IsContradiction = true
                                                      ParentIds =
                                                        match
                                                            FeedbackLoop.findEvidenceBySourceId (string t) fbState
                                                        with
                                                        | Some p -> [ p.Id ]
                                                        | None -> []
                                                      Timestamp = DateTime.UtcNow }

                                                let s2, _ = FeedbackLoop.updateHypothesis (string t) ev fbState
                                                (s2, [])

                                            | ReasonOperation.Backtrack t ->
                                                let tid =
                                                    match t with
                                                    | NodeId s -> s

                                                let s2 =
                                                    FeedbackLoop.invalidateHypothesis
                                                        tid
                                                        "Reasoner signaled backtrack"
                                                        fbState

                                                (s2, [])

                                            | ReasonOperation.Distill t ->
                                                let tid =
                                                    match t with
                                                    | NodeId s -> s

                                                let ev =
                                                    { Id = $"distill_{step.Id}"
                                                      Source = ReasonerThought step.Id
                                                      Content = res.Content
                                                      Confidence = 0.95
                                                      Weight = 0.8
                                                      IsContradiction = false
                                                      ParentIds =
                                                        match FeedbackLoop.findEvidenceBySourceId tid fbState with
                                                        | Some p -> [ p.Id ]
                                                        | None -> []
                                                      Timestamp = DateTime.UtcNow }

                                                let s2, _ = FeedbackLoop.updateHypothesis tid ev fbState
                                                (s2, [])

                                            | ReasonOperation.Aggregate src ->
                                                let ev =
                                                    { Id = $"agg_{step.Id}"
                                                      Source = ReasonerThought step.Id
                                                      Content = res.Content
                                                      Confidence = 0.9
                                                      Weight = 0.5
                                                      IsContradiction = false
                                                      ParentIds =
                                                        src
                                                        |> List.choose (function
                                                            | NodeId s ->
                                                                FeedbackLoop.findEvidenceBySourceId s fbState
                                                                |> Option.map (fun e -> e.Id))
                                                      Timestamp = DateTime.UtcNow }

                                                let s2 = FeedbackLoop.addEvidence ev fbState
                                                (s2, [])

                                            | _ -> (fbState, [])

                                        // Phase 15.2: Log to symbolic memory
                                        do! SymbolicMemory.logFact runId (Some step.Id) res.Content 1.0

                                        do!
                                            succeed
                                                "reason"
                                                None
                                                (None: Map<string, string> option)
                                                res.Content
                                                res.Usage
                                                step.Metadata


                                        return! loop (injectedSteps @ rest) ctx2 fbState2

                        | _ ->
                            return fail $"Action not implemented in v0: {step.Action}" "unknown" None None step.Metadata
                }

            let initialFbState = FeedbackLoop.create runId
            let! finalResult = loop planSteps ctx initialFbState


            let tracesResult =
                match finalResult with
                | Result.Ok(_, _, t) -> t
                | Result.Error(_, t) -> t

            let canonicalTraces = tracesResult |> List.map TraceEvent.toCanonical

            // Perform symbolic reflection if available
            match reflector with
            | Some r ->
                try
                    let! reflectionRes = r.ReflectOnRunAsync(runId, canonicalTraces) |> Async.AwaitTask

                    match reflectionRes with
                    | Microsoft.FSharp.Core.Ok reflection ->
                        printfn
                            "[WoT] Reflection completed. Found %d observations for run %A"
                            reflection.Observations.Length
                            runId
                    | Microsoft.FSharp.Core.Error err -> printfn "[WoT] Reflection failed: %s" err
                with ex ->
                    printfn "[WoT] Reflection error: %s" ex.Message
            | None -> ()

            return finalResult
        }
