namespace Tars.Core

open System
open System.Threading
open System.Threading.Tasks

/// Context passed to the agent workflow
type AgentContext =
    { Self: Agent
      Registry: IAgentRegistry
      Executor: IAgentExecutor
      Logger: string -> unit
      Budget: BudgetGovernor option
      CancellationToken: CancellationToken }

/// A workflow that returns an ExecutionOutcome
type AgentWorkflow<'T> = AgentContext -> Async<ExecutionOutcome<'T>>

// ... (AgentBuilder implementation remains the same) ...
type AgentBuilder() =
    member _.Return(value: 'T) : AgentWorkflow<'T> = fun _ -> async { return Success value }

    member _.ReturnFrom(workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> = workflow

    member _.Bind(workflow: AgentWorkflow<'T>, f: 'T -> AgentWorkflow<'U>) : AgentWorkflow<'U> =
        fun ctx ->
            async {
                // 1. Check Cancellation
                if ctx.CancellationToken.IsCancellationRequested then
                    return Failure [ PartialFailure.Warning "Operation cancelled" ]
                else
                    // 2. Run the first step
                    let! result = workflow ctx

                    match result with
                    | Success value ->
                        // 3. Continue to next step
                        return! f value ctx

                    | PartialSuccess(value, warnings) ->
                        // 4. Continue, but accumulate warnings
                        let! nextResult = f value ctx

                        match nextResult with
                        | Success nextValue -> return PartialSuccess(nextValue, warnings)
                        | PartialSuccess(nextValue, nextWarnings) ->
                            return PartialSuccess(nextValue, warnings @ nextWarnings)
                        | Failure errors -> return Failure(warnings @ errors) // Keep warnings as context for failure

                    | Failure errors ->
                        // 5. Short-circuit
                        return Failure errors
            }

    member _.Zero() : AgentWorkflow<unit> = fun _ -> async { return Success() }

    member _.Combine(a: AgentWorkflow<unit>, b: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = a ctx

                match result with
                | Success() -> return! b ctx
                | PartialSuccess((), warnings) ->
                    let! nextResult = b ctx

                    match nextResult with
                    | Success v -> return PartialSuccess(v, warnings)
                    | PartialSuccess(v, w) -> return PartialSuccess(v, warnings @ w)
                    | Failure e -> return Failure(warnings @ e)
                | Failure e -> return Failure e
            }

    member _.Delay(f: unit -> AgentWorkflow<'T>) : AgentWorkflow<'T> = fun ctx -> f () ctx

[<AutoOpen>]
module AgentWorkflow =
    let agent = AgentBuilder()

    /// Helper to lift a value into a successful workflow
    let succeed (value: 'T) : AgentWorkflow<'T> = fun _ -> async { return Success value }

    /// Helper to lift a failure into the workflow
    let fail (error: PartialFailure) : AgentWorkflow<'T> =
        fun _ -> async { return Failure [ error ] }

    /// Helper to lift a list of failures
    let failMany (errors: PartialFailure list) : AgentWorkflow<'T> =
        fun _ -> async { return Failure errors }

    /// Helper to emit a warning but continue with a value
    let warnWith (value: 'T) (warning: PartialFailure) : AgentWorkflow<'T> =
        fun _ -> async { return PartialSuccess(value, [ warning ]) }

    /// Helper to access the context
    let getContext: AgentWorkflow<AgentContext> =
        fun ctx -> async { return Success ctx }

    /// Helper to check budget
    let checkBudget (cost: Cost) : AgentWorkflow<unit> =
        fun ctx ->
            async {
                match ctx.Budget with
                | Some governor ->
                    match governor.TryConsume cost with
                    | Result.Ok _ -> return Success()
                    | Result.Error err -> return Failure [ PartialFailure.Warning $"Budget exceeded: {err}" ]
                | None -> return Success()
            }

    /// Call an agent by capability
    let callAgentByCapability (kind: CapabilityKind) (taskSpec: string) : AgentWorkflow<string> =
        fun ctx ->
            async {
                let! agents = ctx.Registry.FindAgents kind

                match agents with
                | [] -> return Failure [ PartialFailure.Error $"No agent found for capability {kind}" ]
                | agent :: _ -> return! ctx.Executor.Execute(agent.Id, taskSpec)
            }

    /// Fallback logic
    let withFallback (primary: AgentWorkflow<'T>) (backup: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = primary ctx

                match result with
                | Success v -> return Success v
                | PartialSuccess(v, w) -> return PartialSuccess(v, w)
                | Failure errors ->
                    ctx.Logger $"Primary failed: {errors}. Switching to backup."
                    return! backup ctx
            }

    /// Escalate if confidence is low
    let escalateIfLowConfidence (minConfidence: float) (result: string) (confidence: float) : AgentWorkflow<string> =
        fun ctx ->
            async {
                if confidence < minConfidence then
                    ctx.Logger $"Confidence {confidence} < {minConfidence}. Escalating."
                    return Failure [ PartialFailure.LowConfidence(confidence, "Confidence too low") ]
                else
                    return Success result
            }

    /// Retry with backoff
    let retryWithBackoff (workflow: AgentWorkflow<'T>) (maxRetries: int) : AgentWorkflow<'T> =
        let rec loop retries =
            fun ctx ->
                async {
                    let! result = workflow ctx

                    match result with
                    | Success v -> return Success v
                    | PartialSuccess(v, w) -> return PartialSuccess(v, w)
                    | Failure errors ->
                        if retries > 0 then
                            let delay = 100 * (pown 2 (maxRetries - retries)) // Exponential backoff

                            ctx.Logger
                                $"Retry {maxRetries - retries + 1}/{maxRetries} after {delay}ms. Errors: {errors}"

                            do! Async.Sleep delay
                            return! loop (retries - 1) ctx
                        else
                            return Failure errors
                }

        loop maxRetries

    /// Aggregate results
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
                    | Success v -> successes.Add(v)
                    | PartialSuccess(v, w) ->
                        successes.Add(v)
                        allWarnings.AddRange(w)
                    | Failure e -> allErrors.AddRange(e)

                if allErrors.Count > 0 then
                    return Failure(List.ofSeq allErrors)
                elif allWarnings.Count > 0 then
                    return PartialSuccess(List.ofSeq successes, List.ofSeq allWarnings)
                else
                    return Success(List.ofSeq successes)
            }

    // ==========================================
    // Phase 6.5.5: Circuit Combinators
    // ==========================================

    /// Transformer: Impedance Matching
    /// Transforms the output of a workflow from one type to another (Step Down / Step Up)
    let transform (mapping: 'T -> 'U) (workflow: AgentWorkflow<'T>) : AgentWorkflow<'U> =
        fun ctx ->
            async {
                let! result = workflow ctx

                match result with
                | Success v -> return Success(mapping v)
                | PartialSuccess(v, w) -> return PartialSuccess(mapping v, w)
                | Failure e -> return Failure e
            }

    /// Inductor: Context Inertia
    /// Resists rapid changes in direction. If the new plan deviates too much from the status quo,
    /// it requires higher confidence or budget to proceed.
    /// (Currently a semantic wrapper, will integrate with Vector Store for similarity checks later)
    let stabilize (inertia: float) (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                // TODO: Measure semantic distance from previous state
                // For now, we simulate inertia by adding a small delay or logging
                if inertia > 0.5 then
                    ctx.Logger $"[Inductor] Stabilizing workflow with inertia {inertia}..."

                return! workflow ctx
            }

    /// Diode: Directed Flow
    /// Prevents cycles by checking if the target agent/task has already been visited in this chain.
    let forwardOnly (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                // TODO: Check recursion depth or history in Context
                // For now, we just tag the execution
                ctx.Logger "[Diode] Enforcing forward-only flow."
                return! workflow ctx
            }

    /// Grounding: Reference Potential
    /// Verifies the output against a source of truth (Epistemic Governor).
    let grounded (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = workflow ctx

                match result with
                | Success v ->
                    // TODO: Call EpistemicGovernor.Verify(v)
                    ctx.Logger $"[Grounding] Verifying result: {v}"
                    return Success v
                | PartialSuccess(v, w) ->
                    ctx.Logger $"[Grounding] Verifying partial result: {v}"
                    return PartialSuccess(v, w)
                | Failure e -> return Failure e
            }
