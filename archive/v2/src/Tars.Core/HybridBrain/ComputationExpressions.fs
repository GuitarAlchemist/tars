namespace Tars.Core.HybridBrain

// =============================================================================
// PHASE 17.2: COMPUTATION EXPRESSIONS FOR VALIDATED EXECUTION
// =============================================================================
//
// The `tars {}`, `verify {}`, and `evidence {}` computation expressions
// provide type-safe orchestration that only operates on validated plans.
//
// Key insight: The CE can carry Writer (logs), Result (errors), State (memory),
// and Reader (config/policy) monads - a pure, testable pipeline.

open System
open System.Threading.Tasks

// =============================================================================
// EXECUTION CONTEXT
// =============================================================================

/// Log entry for execution trace
type LogEntry =
    { Timestamp: DateTimeOffset
      Level: string
      StepId: int option
      Message: string
      Metadata: Map<string, obj> }

/// Execution state during workflow
type ExecutionState =
    { CurrentStep: int
      ExecutedSteps: int list
      WorkingMemory: Map<string, obj>
      EvidenceCollected: Evidence list
      TokensUsed: int
      ApiCallsMade: int
      StartTime: DateTimeOffset
      Logs: LogEntry list }

/// Configuration for execution
type ExecutionConfig =
    { VerboseLogging: bool
      DryRun: bool
      MaxRetries: int
      DefaultTimeout: TimeSpan
      Drives: Tars.Core.BaseDrives
      ActionExecutor: Tars.Core.HybridBrain.Action -> Task<Result<obj option, string>> }

/// Result of step execution
type StepResult =
    | StepSuccess of output: obj option * logs: LogEntry list
    | StepFailure of error: string * logs: LogEntry list
    | StepSkipped of reason: string

/// Detailed status of the run
type RunOutcome =
    | Success
    | Partial of reason: string
    | Failure of error: string

/// Result of plan execution
type ExecutionResult =
    { Outcome: RunOutcome
      Confidence: float
      FinalOutput: obj option
      StepsExecuted: int
      TotalSteps: int
      TokensUsed: int
      ApiCallsMade: int
      Duration: TimeSpan
      Logs: LogEntry list
      Evidence: Evidence list }

// =============================================================================
// TARS COMPUTATION EXPRESSION
// =============================================================================

/// The state-result-writer monad for TARS workflows (Async)
type TarsComputation<'T> =
    ExecutionConfig -> ExecutionState -> Task<Result<'T * ExecutionState, string * ExecutionState>>

module TarsComputation =

    /// Return a value in the computation
    let returnM (x: 'T) : TarsComputation<'T> =
        fun _ state -> Task.FromResult(Ok(x, state))

    /// Bind two computations
    let bind (comp: TarsComputation<'T>) (f: 'T -> TarsComputation<'U>) : TarsComputation<'U> =
        fun config state ->
            task {
                match! comp config state with
                | Ok(value, newState) -> return! (f value) config newState
                | Error(err, newState) -> return Error(err, newState)
            }

    /// Map over a computation
    let map (f: 'T -> 'U) (comp: TarsComputation<'T>) : TarsComputation<'U> = bind comp (fun x -> returnM (f x))

    /// Get current state
    let getState: TarsComputation<ExecutionState> =
        fun _ state -> Task.FromResult(Ok(state, state))

    /// Modify state
    let modifyState (f: ExecutionState -> ExecutionState) : TarsComputation<unit> =
        fun _ state -> Task.FromResult(Ok((), f state))

    /// Get config
    let getConfig: TarsComputation<ExecutionConfig> =
        fun config state -> Task.FromResult(Ok(config, state))

    /// Log a message
    let log (level: string) (message: string) : TarsComputation<unit> =
        modifyState (fun state ->
            let entry =
                { Timestamp = DateTimeOffset.UtcNow
                  Level = level
                  StepId = Some state.CurrentStep
                  Message = message
                  Metadata = Map.empty }

            { state with
                Logs = entry :: state.Logs })

    /// Log info
    let logInfo msg = log "INFO" msg

    /// Log warning
    let logWarn msg = log "WARN" msg

    /// Log error
    let logError msg = log "ERROR" msg

    /// Fail with error
    let fail (error: string) : TarsComputation<'T> =
        fun _ state -> Task.FromResult(Error(error, state))

    /// Run computation with initial state
    let run
        (config: ExecutionConfig)
        (comp: TarsComputation<'T>)
        : Task<Result<'T * ExecutionState, string * ExecutionState>> =
        let initialState =
            { CurrentStep = 0
              ExecutedSteps = []
              WorkingMemory = Map.empty
              EvidenceCollected = []
              TokensUsed = 0
              ApiCallsMade = 0
              StartTime = DateTimeOffset.UtcNow
              Logs = [] }

        comp config initialState

/// Builder for the tars computation expression
type TarsBuilder() =

    member _.Return(x: 'T) : TarsComputation<'T> = TarsComputation.returnM x

    member _.ReturnFrom(comp: TarsComputation<'T>) : TarsComputation<'T> = comp

    member _.Bind(comp: TarsComputation<'T>, f: 'T -> TarsComputation<'U>) : TarsComputation<'U> =
        TarsComputation.bind comp f

    member _.Zero() : TarsComputation<unit> = TarsComputation.returnM ()

    member _.Combine(comp1: TarsComputation<unit>, comp2: TarsComputation<'T>) : TarsComputation<'T> =
        TarsComputation.bind comp1 (fun () -> comp2)

    member _.Delay(f: unit -> TarsComputation<'T>) : TarsComputation<'T> = fun config state -> (f ()) config state

    member _.For(items: 'T seq, body: 'T -> TarsComputation<unit>) : TarsComputation<unit> =
        items
        |> Seq.fold (fun acc item -> TarsComputation.bind acc (fun () -> body item)) (TarsComputation.returnM ())

    member _.While(guard: unit -> bool, body: TarsComputation<unit>) : TarsComputation<unit> =
        let rec loop () =
            if guard () then
                TarsComputation.bind body (fun () -> loop ())
            else
                TarsComputation.returnM ()

        loop ()

    member _.TryWith(comp: TarsComputation<'T>, handler: exn -> TarsComputation<'T>) : TarsComputation<'T> =
        fun config state ->
            task {
                try
                    return! comp config state
                with ex ->
                    return! handler ex config state
            }

    member _.TryFinally(comp: TarsComputation<'T>, finalizer: unit -> unit) : TarsComputation<'T> =
        fun config state ->
            task {
                try
                    return! comp config state
                finally
                    finalizer ()
            }

// =============================================================================
// VERIFY COMPUTATION EXPRESSION
// =============================================================================

/// Builder for verification blocks
type VerifyBuilder() =

    let mutable violations: ValidationError list = []

    member _.Yield(_: unit) = ()

    member _.Run(_: unit) : ValidationError list =
        let result = violations
        violations <- []
        result

    member _.Require(condition: bool, error: ValidationError) =
        if not condition then
            violations <- error :: violations

    member _.RequireNotNull(value: obj, fieldName: string) =
        if isNull value then
            violations <- (MissingEvidence fieldName) :: violations

    member _.RequireBudget(resource: string, limit: float, actual: float) =
        if actual > limit then
            violations <- (BudgetExceeded(resource, limit, actual)) :: violations

// =============================================================================
// EVIDENCE COMPUTATION EXPRESSION
// =============================================================================

/// Evidence builder result
type EvidenceResult = Result<Evidence, string>

/// Builder for evidence collection
type EvidenceBuilder() =

    member _.Yield(_: unit) = None

    member _.Run(evidence: Evidence option) : EvidenceResult =
        match evidence with
        | Some e -> Ok e
        | None -> Error "No evidence collected"

    [<CustomOperation("source")>]
    member _.Source(_, src: Source) =
        Some
            { Id = Guid.NewGuid()
              Source = src
              Claim = ""
              Confidence = 0.5
              RetrievedAt = DateTimeOffset.UtcNow
              ExpiresAt = None }

    [<CustomOperation("claim")>]
    member _.Claim(evidence: Evidence option, claim: string) =
        evidence |> Option.map (fun e -> { e with Claim = claim })

    [<CustomOperation("confidence")>]
    member _.Confidence(evidence: Evidence option, conf: float) =
        evidence |> Option.map (fun e -> { e with Confidence = conf })

    [<CustomOperation("expires")>]
    member _.Expires(evidence: Evidence option, expires: DateTimeOffset) =
        evidence |> Option.map (fun e -> { e with ExpiresAt = Some expires })

// =============================================================================
// STEP EXECUTION
// =============================================================================

module StepExecution =

    /// Execute a single validated step
    let executeStep (step: Step) : TarsComputation<StepResult> =
        fun config state ->
            task {
                let newState = { state with CurrentStep = step.Id }

                // Check preconditions
                let preconditionsFailed =
                    step.Preconditions |> List.filter (fun c -> not (c.Predicate()))

                if not preconditionsFailed.IsEmpty then
                    let errors = preconditionsFailed |> List.map (_.Description) |> String.concat "; "

                    let logEntry =
                        { Timestamp = DateTimeOffset.UtcNow
                          Level = "ERROR"
                          StepId = Some step.Id
                          Message = $"Precondition failed: {errors}"
                          Metadata = Map.empty }

                    return
                        Ok(
                            StepFailure($"Preconditions failed: {errors}", [ logEntry ]),
                            { newState with
                                Logs = logEntry :: newState.Logs }
                        )
                else
                    // Execute the action
                    match! config.ActionExecutor step.Action with
                    | Ok output ->
                        let logEntry =
                            { Timestamp = DateTimeOffset.UtcNow
                              Level = "INFO"
                              StepId = Some step.Id
                              Message = $"Executed step: {step.Name}"
                              Metadata = Map.ofList [ ("action", box step.Action); ("output", box output) ] }

                        let finalState =
                            { newState with
                                ExecutedSteps = step.Id :: newState.ExecutedSteps
                                ApiCallsMade = newState.ApiCallsMade + 1
                                Logs = logEntry :: newState.Logs }

                        return Ok(StepSuccess(output, [ logEntry ]), finalState)
                    | Error err ->
                        let logEntry =
                            { Timestamp = DateTimeOffset.UtcNow
                              Level = "ERROR"
                              StepId = Some step.Id
                              Message = $"Action execution failed: {err}"
                              Metadata = Map.ofList [ ("action", box step.Action) ] }

                        return
                            Ok(
                                StepFailure(err, [ logEntry ]),
                                { newState with
                                    Logs = logEntry :: newState.Logs }
                            )
            }

    /// Execute all steps in a validated plan
    let executePlan (plan: Plan<Executable>) : TarsComputation<ExecutionResult> =
        fun config state ->
            task {
                let mutable currentState = state
                let startTime = DateTimeOffset.UtcNow
                let mutable success = true
                let mutable stepsExecuted = 0
                let mutable lastError = None

                for step in plan.Steps do
                    if success then
                        match! executeStep step config currentState with
                        | Ok(StepSuccess(_, logs), newState) ->
                            currentState <- newState
                            stepsExecuted <- stepsExecuted + 1
                        | Ok(StepFailure(err, logs), newState) ->
                            currentState <- newState
                            success <- false
                            lastError <- Some err
                        | Ok(StepSkipped(reason), newState) -> currentState <- newState
                        | Error(err, newState) ->
                            currentState <- newState
                            success <- false
                            lastError <- Some err

                // Calculate confidence based on evidence
                let confidence =
                    if currentState.EvidenceCollected.IsEmpty then
                        0.5
                    else
                        currentState.EvidenceCollected |> List.averageBy (fun e -> e.Confidence)

                let outcome =
                    if success then
                        Success
                    elif stepsExecuted > 0 then
                        Partial(lastError |> Option.defaultValue "Step failure")
                    else
                        Failure(lastError |> Option.defaultValue "Execution failed")

                let result =
                    { Outcome = outcome
                      Confidence = confidence
                      FinalOutput = None
                      StepsExecuted = stepsExecuted
                      TotalSteps = plan.Steps.Length
                      TokensUsed = currentState.TokensUsed
                      ApiCallsMade = currentState.ApiCallsMade
                      Duration = DateTimeOffset.UtcNow - startTime
                      Logs = currentState.Logs |> List.rev
                      Evidence = currentState.EvidenceCollected }

                return Ok(result, currentState)
            }

// =============================================================================
// HIGH-LEVEL API
// =============================================================================

module HybridBrain =

    /// Process a draft plan through the full pipeline
    let processAndExecute
        (ctx: ValidationContext)
        (draft: Plan<Draft>)
        (config: ExecutionConfig)
        : Task<Result<ExecutionResult, FormalCritique>> =
        task {
            match StateTransitions.fullPipeline ctx draft with
            | Error critique -> return Error critique
            | Ok executable ->
                let initialState =
                    { CurrentStep = 0
                      ExecutedSteps = []
                      WorkingMemory = Map.empty
                      EvidenceCollected = []
                      TokensUsed = 0
                      ApiCallsMade = 0
                      StartTime = DateTimeOffset.UtcNow
                      Logs = [] }

                match! StepExecution.executePlan executable config initialState with
                | Ok(result, _) -> return Ok result
                | Error(err, state) ->
                    // Convert execution error to critique
                    return
                        Error
                            { ParseErrors = []
                              ValidationErrors = [ InvariantViolated("EXEC", err) ]
                              Suggestions = []
                              MinimalCounterExample = None }
        }

    /// Default execution config
    let defaultConfig =
        { VerboseLogging = false
          DryRun = false
          MaxRetries = 3
          DefaultTimeout = TimeSpan.FromMinutes(5.0)
          Drives =
            { Accuracy = 0.8
              Speed = 0.5
              Creativity = 0.5
              Safety = 0.9 }
          ActionExecutor = fun action -> Task.FromResult(Ok(Some(box $"Executed {action} (Dry Run)"): obj option)) }

// =============================================================================
// COMPUTATION EXPRESSION INSTANCES
// =============================================================================

/// Module containing computation expression instances
module Builders =
    /// The tars computation expression instance
    let tars = TarsBuilder()
    /// The verify computation expression instance
    let verify = VerifyBuilder()
    /// The evidence computation expression instance
    let evidence = EvidenceBuilder()
