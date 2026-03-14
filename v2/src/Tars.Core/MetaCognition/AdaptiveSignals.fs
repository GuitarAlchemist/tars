namespace Tars.Core.MetaCognition

/// Pure-logic mid-execution monitoring.
/// Evaluates step progress and produces adaptation signals/decisions.
module AdaptiveSignals =

    /// Configuration for adaptation thresholds.
    type AdaptationConfig =
        { MinConfidence: float
          MaxConsecutiveFailures: int
          BudgetWarningPercent: float }

    let defaultConfig =
        { MinConfidence = 0.3
          MaxConsecutiveFailures = 3
          BudgetWarningPercent = 0.8 }

    let fromMetaConfig (mc: MetaCognitionConfig) =
        { MinConfidence = mc.AdaptiveMinConfidence
          MaxConsecutiveFailures = mc.AdaptiveMaxConsecutiveFailures
          BudgetWarningPercent = mc.AdaptiveBudgetWarningPercent }

    // =====================================================================
    // Signal evaluation
    // =====================================================================

    /// Evaluate completed steps and produce adaptation signals.
    let evaluateProgress
        (config: AdaptationConfig)
        (steps: StepProgress list)
        (totalBudget: int option)
        : AdaptationSignal list =

        if steps.IsEmpty then []
        else
            let signals = ResizeArray<AdaptationSignal>()

            // Check confidence dropping
            let recentConfidences =
                steps
                |> List.choose (fun s -> s.Confidence)
                |> List.rev
                |> List.truncate 3

            match recentConfidences with
            | latest :: _ when latest < config.MinConfidence ->
                signals.Add(AdaptationSignal.ConfidenceDropping(latest, config.MinConfidence))
            | _ -> ()

            // Check consecutive failures
            let recentFailures =
                steps
                |> List.rev
                |> List.takeWhile (fun s -> not s.Succeeded)

            if recentFailures.Length >= config.MaxConsecutiveFailures then
                signals.Add(AdaptationSignal.ConsecutiveFailures recentFailures.Length)

            // Check individual step failures (latest only)
            match steps |> List.tryLast with
            | Some s when not s.Succeeded ->
                let err = s.ErrorMessage |> Option.defaultValue "unknown error"
                signals.Add(AdaptationSignal.StepFailing(s.StepId, err))
            | _ -> ()

            // Check budget
            match totalBudget with
            | Some budget when budget > 0 ->
                let used = float steps.Length / float budget
                if used >= config.BudgetWarningPercent then
                    signals.Add(AdaptationSignal.BudgetExhausting used)
            | _ -> ()

            signals |> Seq.toList

    // =====================================================================
    // Decision logic
    // =====================================================================

    /// Decide adaptation action from signals.
    /// Pure rules — no LLM involved.
    let decideAction
        (signals: AdaptationSignal list)
        (availablePatterns: string list)
        : AdaptationAction =

        if signals.IsEmpty then
            AdaptationAction.ContinueCurrent
        else
            // Priority: Abort > Switch > Recovery > Continue
            let hasConsecutiveFailures =
                signals |> List.tryPick (fun s ->
                    match s with
                    | AdaptationSignal.ConsecutiveFailures n when n >= 5 -> Some n
                    | _ -> None)

            let hasBudgetExhaustion =
                signals |> List.tryPick (fun s ->
                    match s with
                    | AdaptationSignal.BudgetExhausting pct when pct >= 0.95 -> Some pct
                    | _ -> None)

            // Abort if too many failures or budget gone
            match hasConsecutiveFailures with
            | Some n ->
                AdaptationAction.Abort(sprintf "Aborting after %d consecutive failures" n)
            | None ->

            match hasBudgetExhaustion with
            | Some pct ->
                AdaptationAction.Abort(sprintf "Budget exhausted (%.0f%% used)" (pct * 100.0))
            | None ->

            // Switch pattern if moderate consecutive failures and alternatives exist
            let shouldSwitch =
                signals |> List.exists (fun s ->
                    match s with
                    | AdaptationSignal.ConsecutiveFailures n when n >= 3 -> true
                    | AdaptationSignal.ConfidenceDropping _ -> true
                    | _ -> false)

            if shouldSwitch && not availablePatterns.IsEmpty then
                let suggestion = availablePatterns |> List.head
                AdaptationAction.SwitchPattern(suggestion, "Consecutive failures or low confidence")
            else

            // Insert recovery step on single failure
            let hasStepFailure =
                signals |> List.exists (fun s ->
                    match s with
                    | AdaptationSignal.StepFailing _ -> true
                    | _ -> false)

            if hasStepFailure then
                AdaptationAction.InsertRecoveryStep "Retry with simplified input or gather more context"
            else
                AdaptationAction.ContinueCurrent
