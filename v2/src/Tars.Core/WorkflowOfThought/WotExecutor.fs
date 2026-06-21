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
                                | ToolOutcome.NotFound -> return Some $"ToolResult failed: Tool '{toolName}' not found"
                                | ToolOutcome.CircuitOpen ->
                                    return Some $"ToolResult failed: Tool '{toolName}' circuit breaker open"
                                | ToolOutcome.Failed(_, e) ->
                                    return Some $"ToolResult failed: Tool '{toolName}' error: {e}"
                                | ToolOutcome.Succeeded resStr ->
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
        (sink: ISymbolicSink)
        : Async<Result<ExecContext * VerifyResult option * TraceEvent list, string * TraceEvent list>> =
        async {
            let runId = Guid.NewGuid()
            let feedback = ReasonFeedback(runId) :> IReasonFeedback
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
            let rec loop steps currentCtx =
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
                                        | ToolOutcome.NotFound ->
                                            return
                                                fail
                                                    $"Tool '{toolName}' not found"
                                                    "tool"
                                                    (Some toolName)
                                                    (Some resolvedArgs)
                                                    step.Metadata
                                        | ToolOutcome.CircuitOpen ->
                                            return
                                                fail
                                                    $"Tool '{toolName}' circuit breaker open"
                                                    "tool"
                                                    (Some toolName)
                                                    (Some resolvedArgs)
                                                    step.Metadata
                                        | ToolOutcome.Failed(_, e) ->
                                            return
                                                fail
                                                    $"Tool '{toolName}' failed: {e}"
                                                    "tool"
                                                    (Some toolName)
                                                    (Some resolvedArgs)
                                                    step.Metadata
                                        | ToolOutcome.Succeeded value ->
                                            match storeSingleOutput currentCtx step.Outputs (box value) with
                                            | Result.Error e ->
                                                return
                                                    fail
                                                        $"Step '{step.Id}' output failed: {e}"
                                                        "tool"
                                                        (Some toolName)
                                                        (Some resolvedArgs)
                                                        step.Metadata
                                            | Result.Ok ctx2 ->
                                                // Phase 17.4: feed the tool result into the feedback policy
                                                feedback.Observe(ToolObserved(step, toolName))

                                                do!
                                                    succeed
                                                        "tool"
                                                        (Some toolName)
                                                        (Some resolvedArgs)
                                                        value
                                                        None
                                                        step.Metadata

                                                return! loop rest ctx2

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
                                    do! sink.LogFailure(runId, Some step.Id, combinedErrors, Map.empty)

                                // Verification failure is NOT execution failure in v0
                                do! succeed "verify" None None vr None step.Metadata
                                return! loop rest ctx2

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

                                    let content = feedback.Aggregate ids
                                    Some $"Aggregate evidence from: %A{src}", Some $"Context from sources:\n{content}"
                                | ReasonOperation.Refine t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    let context = feedback.Summarize tid

                                    Some $"Refine thought: %A{t}",
                                    Some $"Previous evidence for this thought:\n{context}"
                                | ReasonOperation.Contradict t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    let context = feedback.Summarize tid

                                    Some $"Find contradictions for: %A{t}",
                                    Some $"Current evidence to challenge:\n{context}"
                                | ReasonOperation.Distill t ->
                                    let tid =
                                        match t with
                                        | NodeId s -> s

                                    let context = feedback.Summarize tid
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
                                    let reached, explain = feedback.Consensus p
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
                                    // Phase 17.4: feed the stub reason result into the feedback policy
                                    feedback.Observe(ReasonObserved(step, op, stubValue.Content, true))

                                    do! succeed "reason" None None stubValue.Content None step.Metadata

                                    return! loop rest ctx2
                            | ReasonStepMode.Llm
                            | ReasonStepMode.Replay ->
                                // Call reasoner with stepId - Llm calls LLM, Replay reads journal
                                let! r = reasoner.Reason(step.Id, currentCtx, goal, instruction, step.Agent)

                                match r with
                                | Result.Error e ->
                                    // Phase 15.2: Log to symbolic memory
                                    do! sink.LogFailure(runId, Some step.Id, e, Map.empty)
                                    return fail $"Reason step '{step.Id}' failed: {e}" "reason" None None step.Metadata
                                | Result.Ok(res: ReasoningResult) ->
                                    match storeSingleOutput currentCtx step.Outputs res.Content with
                                    | Result.Error e ->
                                        return
                                            fail $"Step '{step.Id}' output failed: {e}" "reason" None None step.Metadata
                                    | Result.Ok ctx2 ->
                                        // Phase 17.4/17.5: feed the reason result into the feedback policy,
                                        // then let it route any scored hypothesis into new steps.
                                        feedback.Observe(ReasonObserved(step, op, res.Content, false))

                                        let injectedSteps =
                                            match op with
                                            | ReasonOperation.Score t ->
                                                match feedback.Decide(string t) with
                                                | WotController.Expand(nid, k) ->
                                                    [ 1..k ]
                                                    |> List.map (fun i ->
                                                        { step with
                                                            Id = Guid.NewGuid().ToString()
                                                            Action =
                                                                StepAction.Reason(
                                                                    ReasonOperation.Generate($"Expansion {i} of {nid}")
                                                                ) })
                                                | WotController.Finalize nid ->
                                                    [ { step with
                                                          Id = Guid.NewGuid().ToString()
                                                          Action = StepAction.Reason(ReasonOperation.Distill(NodeId nid)) } ]
                                                | WotController.Backtrack nid ->
                                                    [ { step with
                                                          Id = Guid.NewGuid().ToString()
                                                          Action = StepAction.Reason(ReasonOperation.Backtrack(NodeId nid)) } ]
                                                | WotController.Refine nid ->
                                                    [ { step with
                                                          Id = Guid.NewGuid().ToString()
                                                          Action = StepAction.Reason(ReasonOperation.Refine(NodeId nid)) } ]
                                                | _ -> []
                                            | _ -> []

                                        // Phase 15.2: Log to symbolic memory
                                        do! sink.LogFact(runId, Some step.Id, res.Content, 1.0)

                                        do!
                                            succeed
                                                "reason"
                                                None
                                                (None: Map<string, string> option)
                                                res.Content
                                                res.Usage
                                                step.Metadata

                                        return! loop (injectedSteps @ rest) ctx2

                        | _ ->
                            return fail $"Action not implemented in v0: {step.Action}" "unknown" None None step.Metadata
                }

            let! finalResult = loop planSteps ctx


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
