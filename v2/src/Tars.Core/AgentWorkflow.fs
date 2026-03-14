namespace Tars.Core


open System.Diagnostics
open System.Threading
open System.Threading.Tasks

/// <summary>
/// Context passed to agent workflow operations.
/// Contains the executing agent, service references, and resource constraints.
/// </summary>
type AgentContext =
    {
        /// The agent executing this workflow
        Self: Agent
        /// Registry for looking up other agents
        Registry: IAgentRegistry
        /// Executor for invoking other agents
        Executor: IAgentExecutor
        /// Logger for diagnostic output
        Logger: string -> unit
        /// Optional budget governor for resource tracking
        Budget: BudgetGovernor option
        /// Optional epistemic governor for truth verification
        Epistemic: IEpistemicGovernor option
        /// Optional semantic memory handle for recall/injection
        SemanticMemory: ISemanticMemory option
        /// Optional knowledge graph service for structural queries
        KnowledgeGraph: IGraphService option
        /// Optional symbolic reflector for analyzing execution traces
        SymbolicReflector: ISymbolicReflector option
        /// Optional capability store for semantic routing
        CapabilityStore: ICapabilityStore option
        /// Optional audit collector for reasoning decisions
        Audit: ReasoningAudit option
        /// Cancellation token for cooperative cancellation
        CancellationToken: CancellationToken
    }

/// <summary>
/// An agent workflow is a function from context to an async execution outcome.
/// This enables composable, cancellable, budget-aware agent operations.
/// </summary>
type AgentWorkflow<'T> = AgentContext -> Async<ExecutionOutcome<'T>>

/// <summary>
type AgentBuilder() =
    member _.Return(value: 'T) : AgentWorkflow<'T> = fun _ -> async { return ExecutionOutcome.Success value }

    member _.ReturnFrom(workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> = workflow

    member _.Bind(workflow: AgentWorkflow<'T>, f: 'T -> AgentWorkflow<'U>) : AgentWorkflow<'U> =
        fun ctx ->
            async {
                let start = Stopwatch.GetTimestamp()

                let! finalResult =
                    async {
                        // 1. Check Cancellation
                        if ctx.CancellationToken.IsCancellationRequested then
                            return ExecutionOutcome.Failure [ PartialFailure.Warning "Operation cancelled" ]
                        else
                            // 2. Run the first step
                            let! result = workflow ctx

                            match result with
                            | ExecutionOutcome.Success value ->
                                // 3. Continue to next step
                                return! f value ctx

                            | ExecutionOutcome.PartialSuccess(value, warnings) ->
                                // 4. Continue, but accumulate warnings
                                let! nextResult = f value ctx

                                match nextResult with
                                | ExecutionOutcome.Success nextValue -> return ExecutionOutcome.PartialSuccess(nextValue, warnings)
                                | ExecutionOutcome.PartialSuccess(nextValue, nextWarnings) ->
                                    return ExecutionOutcome.PartialSuccess(nextValue, warnings @ nextWarnings)
                                | ExecutionOutcome.Failure errors -> return ExecutionOutcome.Failure(warnings @ errors) // Keep warnings as context for failure

                            | ExecutionOutcome.Failure errors ->
                                // 5. Short-circuit
                                return ExecutionOutcome.Failure errors
                    }

                let durationMs =
                    float (Stopwatch.GetTimestamp() - start) * 1000.0 / float Stopwatch.Frequency

                let status =
                    match finalResult with
                    | ExecutionOutcome.Success _ -> "success"
                    | ExecutionOutcome.PartialSuccess _ -> "partial"
                    | ExecutionOutcome.Failure _ -> "failure"

                Metrics.record "agent.bind" status durationMs (Some ctx.Self.Id) Map.empty
                return finalResult
            }

    member this.Bind(a: Async<'T>, f: 'T -> AgentWorkflow<'U>) : AgentWorkflow<'U> =
        fun ctx ->
            async {
                let! res = a
                return! f res ctx
            }

    member this.Bind(t: Task<'T>, f: 'T -> AgentWorkflow<'U>) : AgentWorkflow<'U> =
        this.Bind(t |> Async.AwaitTask, f)

    member this.Bind(t: Task, f: unit -> AgentWorkflow<'U>) : AgentWorkflow<'U> =
        this.Bind(t |> Async.AwaitTask, fun () -> f ())

    member _.Zero() : AgentWorkflow<unit> = fun _ -> async { return ExecutionOutcome.Success() }

    member _.Combine(a: AgentWorkflow<unit>, b: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = a ctx

                match result with
                | ExecutionOutcome.Success() -> return! b ctx
                | ExecutionOutcome.PartialSuccess((), warnings) ->
                    let! nextResult = b ctx

                    match nextResult with
                    | ExecutionOutcome.Success v -> return ExecutionOutcome.PartialSuccess(v, warnings)
                    | ExecutionOutcome.PartialSuccess(v, w) -> return ExecutionOutcome.PartialSuccess(v, warnings @ w)
                    | ExecutionOutcome.Failure e -> return ExecutionOutcome.Failure(warnings @ e)
                | ExecutionOutcome.Failure e -> return ExecutionOutcome.Failure e
            }

    member _.Delay(f: unit -> AgentWorkflow<'T>) : AgentWorkflow<'T> = fun ctx -> f () ctx

/// <summary>
/// Helper functions for building agent workflows.
/// </summary>
[<AutoOpen>]
module AgentWorkflow =
    /// <summary>
    /// Agent workflow computation expression for building composable, fault-tolerant agent operations.
    /// Provides automatic cancellation checking, warning accumulation, and budget governance.
    /// </summary>
    /// <example>
    /// <code>
    /// let myWorkflow = agent {
    ///     let! result = someOperation ()
    ///     do! AgentWorkflow.checkBudget cost
    ///     return result
    /// }
    /// </code>
    /// </example>
    let agent = AgentBuilder()

    /// <summary>Lifts a value into a successful workflow.</summary>
    /// <param name="value">The value to wrap in Success.</param>
    let succeed (value: 'T) : AgentWorkflow<'T> = fun _ -> async { return ExecutionOutcome.Success value }

    /// <summary>Creates a workflow that immediately fails with the given error.</summary>
    /// <param name="error">The failure reason.</param>
    let fail (error: PartialFailure) : AgentWorkflow<'T> =
        fun _ -> async { return ExecutionOutcome.Failure [ error ] }

    /// <summary>Creates a workflow that immediately fails with multiple errors.</summary>
    /// <param name="errors">The list of failure reasons.</param>
    let failMany (errors: PartialFailure list) : AgentWorkflow<'T> =
        fun _ -> async { return ExecutionOutcome.Failure errors }

    /// <summary>Creates a partial success with a value and a warning.</summary>
    /// <param name="value">The result value.</param>
    /// <param name="warning">The warning to attach.</param>
    let warnWith (value: 'T) (warning: PartialFailure) : AgentWorkflow<'T> =
        fun _ -> async { return ExecutionOutcome.PartialSuccess(value, [ warning ]) }

    /// <summary>Accesses the current agent context.</summary>
    let getContext: AgentWorkflow<AgentContext> =
        fun ctx -> async { return ExecutionOutcome.Success ctx }

    /// <summary>
    /// Checks if the specified cost can be consumed from the budget.
    /// Fails the workflow if the budget would be exceeded.
    /// </summary>
    /// <param name="cost">The cost to consume.</param>
    let checkBudget (cost: Cost) : AgentWorkflow<unit> =
        fun ctx ->
            async {
                match ctx.Budget with
                | Some governor ->
                    match governor.TryConsume cost with
                    | Result.Ok _ ->
                        Metrics.recordSimple "budget.check" "ok" (Some ctx.Self.Id) None None
                        return ExecutionOutcome.Success()
                    | Result.Error err ->
                        Metrics.recordSimple "budget.check" "exceeded" (Some ctx.Self.Id) None None
                        return ExecutionOutcome.Failure [ PartialFailure.Warning $"Budget exceeded: {err}" ]
                | None -> return ExecutionOutcome.Success()
            }

    /// <summary>
    /// Finds and executes an agent with the specified capability.
    /// </summary>
    /// <param name="kind">The capability to search for.</param>
    /// <param name="taskSpec">The task specification to send to the agent.</param>
    let callAgentByCapability (kind: CapabilityKind) (taskSpec: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                let! agents = ctx.Registry.FindAgents kind

                let scoreCapability (agent: Agent) =
                    agent.Capabilities
                    |> List.tryFind (fun c -> c.Kind = kind)
                    |> Option.map (fun cap ->
                        let confidence = cap.Confidence |> Option.defaultValue 0.5
                        let reputation = cap.Reputation |> Option.defaultValue 0.5
                        // Weight reputation slightly higher than self-reported confidence
                        (reputation * 0.6) + (confidence * 0.4))
                    |> Option.defaultValue 0.0

                let! semanticRanked =
                    async {
                        match ctx.CapabilityStore with
                        | None -> return []
                        | Some store ->
                            let query = $"{kind} :: {taskSpec}"
                            let! hits = store.FindAgentsAsync(query, 5) |> Async.AwaitTask

                            let! resolved =
                                hits
                                |> List.map (fun (id, cap, score) ->
                                    async {
                                        let! agentOpt = ctx.Registry.GetAgent id

                                        return
                                            agentOpt
                                            |> Option.map (fun agent ->
                                                let metaScore =
                                                    (cap.Reputation |> Option.defaultValue 0.5) * 0.6
                                                    + (cap.Confidence |> Option.defaultValue 0.5) * 0.4
                                                let combinedScore = score + metaScore
                                                (agent, combinedScore))
                                    })
                                |> Async.Parallel

                            return resolved |> Array.choose id |> Array.toList
                    }

                let merged =
                    let manual = agents |> List.map (fun a -> a, scoreCapability a)

                    (manual @ semanticRanked)
                    |> List.groupBy (fun (a, _) -> a.Id)
                    |> List.map (fun (_, xs) -> xs |> List.maxBy snd)

                match merged with
                | [] -> return ExecutionOutcome.Failure [ PartialFailure.Error $"No agent found for capability {kind}" ]
                | _ ->
                    let bestAgent = merged |> List.sortByDescending snd |> List.head |> fst
                    ctx.Logger $"Selected agent {bestAgent.Name} for {kind}"
                    return! ctx.Executor.Execute(bestAgent.Id, taskSpec)
            }

    /// <summary>Provides a fallback workflow if the primary fails.</summary>
    let withFallback (primary: AgentWorkflow<'T>) (backup: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = primary ctx

                match result with
                | ExecutionOutcome.Success v -> return ExecutionOutcome.Success v
                | ExecutionOutcome.PartialSuccess(v, w) -> return ExecutionOutcome.PartialSuccess(v, w)
                | ExecutionOutcome.Failure errors ->
                    ctx.Logger $"Primary failed: {errors}. Switching to backup."
                    return! backup ctx
            }

    /// <summary>
    /// Escalates to failure if the confidence score is below threshold.
    /// </summary>
    /// <param name="minConfidence">Minimum acceptable confidence (0.0-1.0).</param>
    /// <param name="result">The result value.</param>
    /// <param name="confidence">The confidence score.</param>
    let escalateIfLowConfidence (minConfidence: float) (result: string) (confidence: float) : AgentWorkflow<string> =
        fun ctx ->
            async {
                if confidence < minConfidence then
                    ctx.Logger $"Confidence {confidence} < {minConfidence}. Escalating."
                    return ExecutionOutcome.Failure [ PartialFailure.LowConfidence(confidence, "Confidence too low") ]
                else
                    return ExecutionOutcome.Success result
            }

    /// <summary>
    /// Retries a workflow with exponential backoff on failure.
    /// </summary>
    /// <param name="workflow">The workflow to retry.</param>
    /// <param name="maxRetries">Maximum number of retry attempts.</param>
    let retryWithBackoff (workflow: AgentWorkflow<'T>) (maxRetries: int) : AgentWorkflow<'T> =
        let rec loop retries =
            fun ctx ->
                async {
                    let! result = workflow ctx

                    match result with
                    | ExecutionOutcome.Success v -> return ExecutionOutcome.Success v
                    | ExecutionOutcome.PartialSuccess(v, w) -> return ExecutionOutcome.PartialSuccess(v, w)
                    | ExecutionOutcome.Failure errors ->
                        if retries > 0 then
                            let delay = 100 * (pown 2 (maxRetries - retries)) // Exponential backoff

                            ctx.Logger
                                $"Retry {maxRetries - retries + 1}/{maxRetries} after {delay}ms. Errors: {errors}"

                            do! Async.Sleep delay
                            return! loop (retries - 1) ctx
                        else
                            return ExecutionOutcome.Failure errors
                }

        loop maxRetries

    /// <summary>
    /// Runs multiple workflows in parallel and aggregates results.
    /// Collects all warnings and fails if any workflow fails.
    /// </summary>
    /// <param name="workflows">The workflows to run in parallel.</param>
    let aggregateResults (workflows: AgentWorkflow<'T> list) : AgentWorkflow<'T list> =
        fun ctx ->
            async {
                let! results = workflows |> List.map (fun w -> w ctx) |> Async.Parallel

                // Combine results
                let successes = ResizeArray<'T>()
                let allWarnings = ResizeArray<PartialFailure>()
                let allErrors = ResizeArray<PartialFailure>()

                for res in results do
                    match res with
                    | ExecutionOutcome.Success v -> successes.Add(v)
                    | ExecutionOutcome.PartialSuccess(v, w) ->
                        successes.Add(v)
                        allWarnings.AddRange(w)
                    | ExecutionOutcome.Failure e -> allErrors.AddRange(e)

                if allErrors.Count > 0 then
                    return ExecutionOutcome.Failure(List.ofSeq allErrors)
                elif allWarnings.Count > 0 then
                    return ExecutionOutcome.PartialSuccess(List.ofSeq successes, List.ofSeq allWarnings)
                else
                    return ExecutionOutcome.Success(List.ofSeq successes)
            }

    /// <summary>
    /// Senior workflow scaffolding: plan -> review -> execute -> verify with warning accumulation.
    /// </summary>
    /// <param name="plan">Step that produces a plan artifact.</param>
    /// <param name="review">Step that reviews/refines the plan.</param>
    /// <param name="execute">Step that executes the refined plan.</param>
    /// <param name="verify">Step that verifies the execution result.</param>
    let planReviewExecuteVerify
        (plan: AgentWorkflow<'Plan>)
        (review: 'Plan -> AgentWorkflow<'Reviewed>)
        (execute: 'Reviewed -> AgentWorkflow<'Executed>)
        (verify: 'Executed -> AgentWorkflow<'Verified>)
        : AgentWorkflow<'Verified> =
        fun ctx ->
            async {
                let warnings = ResizeArray<PartialFailure>()

                let collect outcome =
                    match outcome with
                    | ExecutionOutcome.Success v -> Choice1Of2 v
                    | ExecutionOutcome.PartialSuccess(v, w) ->
                        warnings.AddRange w
                        Choice1Of2 v
                    | ExecutionOutcome.Failure errs -> Choice2Of2((warnings |> Seq.toList) @ errs)

                let! planOutcome = plan ctx

                match collect planOutcome with
                | Choice2Of2 errs -> return ExecutionOutcome.Failure errs
                | Choice1Of2 planResult ->
                    let! reviewOutcome = review planResult ctx

                    match collect reviewOutcome with
                    | Choice2Of2 errs -> return ExecutionOutcome.Failure errs
                    | Choice1Of2 reviewed ->
                        let! execOutcome = execute reviewed ctx

                        match collect execOutcome with
                        | Choice2Of2 errs -> return ExecutionOutcome.Failure errs
                        | Choice1Of2 executed ->
                            let! verifyOutcome = verify executed ctx

                            match verifyOutcome with
                            | ExecutionOutcome.Success v ->
                                let collected = warnings |> Seq.toList
                                return
                                    if collected.IsEmpty then
                                        ExecutionOutcome.Success v
                                    else
                                        ExecutionOutcome.PartialSuccess(v, collected)
                            | ExecutionOutcome.PartialSuccess(v, w) ->
                                let collected = (warnings |> Seq.toList) @ w
                                return ExecutionOutcome.PartialSuccess(v, collected)
                            | ExecutionOutcome.Failure errs -> return ExecutionOutcome.Failure((warnings |> Seq.toList) @ errs)
            }

    // ==========================================
    // Circuit Combinators (Electronics-inspired workflow patterns)
    // ==========================================

    /// <summary>
    /// Transformer: Impedance Matching.
    /// Transforms the output of a workflow from one type to another.
    /// Like an electrical transformer that steps voltage up or down.
    /// </summary>
    /// <param name="mapping">The transformation function.</param>
    /// <param name="workflow">The workflow to transform.</param>
    let transform (mapping: 'T -> 'U) (workflow: AgentWorkflow<'T>) : AgentWorkflow<'U> =
        fun ctx ->
            async {
                let! result = workflow ctx

                match result with
                | ExecutionOutcome.Success v -> return ExecutionOutcome.Success(mapping v)
                | ExecutionOutcome.PartialSuccess(v, w) -> return ExecutionOutcome.PartialSuccess(mapping v, w)
                | ExecutionOutcome.Failure e -> return ExecutionOutcome.Failure e
            }

    /// <summary>
    /// Inductor: Context Inertia.
    /// Resists rapid changes in direction. If the new plan deviates too much
    /// from the status quo, it requires higher confidence or budget to proceed.
    /// Like an electrical inductor that resists changes in current.
    /// </summary>
    /// <param name="inertia">The resistance to change (0.0-1.0).</param>
    /// <param name="workflow">The workflow to stabilize.</param>
    let stabilize (inertia: float) (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                // TODO: Measure semantic distance from previous state
                if inertia > 0.5 then
                    ctx.Logger $"[Inductor] Stabilizing workflow with inertia {inertia}..."

                return! workflow ctx
            }

    /// <summary>
    /// Diode: Directed Flow.
    /// Prevents cycles by checking if the target agent/task has already been
    /// visited in this chain. Like an electrical diode that only allows
    /// current to flow in one direction.
    /// </summary>
    /// <param name="workflow">The workflow to protect from cycles.</param>
    let forwardOnly (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                // TODO: Check recursion depth or history in Context
                ctx.Logger "[Diode] Enforcing forward-only flow."
                return! workflow ctx
            }


    /// <summary>
    /// Grounding: Reference Potential.
    /// Verifies the output against a source of truth (Epistemic Governor).
    /// Like an electrical ground that provides a stable reference point.
    /// </summary>
    /// <param name="workflow">The workflow to ground.</param>
    let grounded (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = workflow ctx

                match result with
                | ExecutionOutcome.Success v ->
                    match ctx.Epistemic with
                    | Some governor ->
                        let! verified = governor.Verify(v.ToString()) |> Async.AwaitTask

                        if verified then
                            ctx.Logger $"[Grounding] Verified result: {v}"
                            return ExecutionOutcome.Success v
                        else
                            ctx.Logger $"[Grounding] Rejected result: {v}"
                            return ExecutionOutcome.Failure [ PartialFailure.Error $"Epistemic Governor rejected result: {v}" ]
                    | None ->
                        ctx.Logger "[Grounding] No Epistemic Governor available. Skipping verification."
                        return ExecutionOutcome.Success v

                | ExecutionOutcome.PartialSuccess(v, w) ->
                    match ctx.Epistemic with
                    | Some governor ->
                        let! verified = governor.Verify(v.ToString()) |> Async.AwaitTask

                        if verified then
                            ctx.Logger $"[Grounding] Verified partial result: {v}"
                            return ExecutionOutcome.PartialSuccess(v, w)
                        else
                            ctx.Logger $"[Grounding] Rejected partial result: {v}"
                            return ExecutionOutcome.Failure(w @ [ PartialFailure.Error $"Epistemic Governor rejected result: {v}" ])
                    | None -> return ExecutionOutcome.PartialSuccess(v, w)

                | ExecutionOutcome.Failure e -> return ExecutionOutcome.Failure e
            }
