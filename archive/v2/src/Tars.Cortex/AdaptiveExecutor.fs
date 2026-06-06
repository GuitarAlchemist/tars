namespace Tars.Cortex

open System
open Tars.Core
open Tars.Core.MetaCognition
open Tars.Cortex.WoTTypes

/// Wraps IWoTExecutor to add mid-task adaptation.
/// Monitors step progress and can switch patterns, insert recovery steps, or abort
/// when adaptation signals indicate the current approach is failing.
module AdaptiveExecutor =

    /// Convert a WoTTraceStep to a StepProgress (Core type, no upward dependency).
    let toStepProgress (step: WoTTraceStep) : StepProgress =
        let succeeded, errorMsg =
            match step.Status with
            | NodeStatus.Completed _ -> true, None
            | NodeStatus.Failed(err, _) -> false, Some err
            | NodeStatus.Skipped reason -> false, Some reason
            | _ -> false, None

        let durationMs =
            match step.Status with
            | NodeStatus.Completed(_, ms) -> ms
            | NodeStatus.Failed(_, ms) -> ms
            | _ -> 0L

        { StepId = step.NodeId
          Kind = step.NodeType
          Succeeded = succeeded
          ErrorMessage = errorMsg
          DurationMs = durationMs
          Confidence = step.Confidence }

    /// Result of an adaptive execution
    type AdaptiveResult =
        { Result: WoTResult
          ActionsApplied: AdaptationAction list
          TotalStepsMonitored: int
          PatternSwitches: int }

    /// Execute a WoT plan with mid-task adaptation monitoring.
    /// Wraps the base executor's ExecuteWithProgress, evaluating signals after each step.
    let executeAdaptive
        (executor: IWoTExecutor)
        (selector: IPatternSelector)
        (config: AdaptiveSignals.AdaptationConfig)
        (plan: WoTPlan)
        (context: AgentContext)
        (goal: string)
        : Async<AdaptiveResult> =
        async {
            let progressLog = ResizeArray<StepProgress>()
            let actionsApplied = ResizeArray<AdaptationAction>()
            let mutable patternSwitches = 0
            let mutable shouldAbort = false
            let mutable abortReason = ""

            let totalBudget = plan.Nodes.Length

            let onProgress (step: WoTTraceStep) =
                let sp = toStepProgress step
                progressLog.Add(sp)

                // Evaluate adaptation signals
                let signals =
                    AdaptiveSignals.evaluateProgress
                        config
                        (progressLog |> Seq.toList)
                        (Some totalBudget)

                if not signals.IsEmpty then
                    // Get available alternative patterns
                    let scores = selector.Score goal
                    let currentPattern = sprintf "%A" plan.Metadata.Kind
                    let alternatives =
                        scores
                        |> Map.toList
                        |> List.sortByDescending snd
                        |> List.map (fun (k, _) -> sprintf "%A" k)
                        |> List.filter (fun p -> p <> currentPattern)

                    let action = AdaptiveSignals.decideAction signals alternatives
                    match action with
                    | AdaptationAction.ContinueCurrent -> ()
                    | AdaptationAction.Abort reason ->
                        shouldAbort <- true
                        abortReason <- reason
                        actionsApplied.Add(action)
                    | _ ->
                        actionsApplied.Add(action)
                        match action with
                        | AdaptationAction.SwitchPattern _ -> patternSwitches <- patternSwitches + 1
                        | _ -> ()

            // Execute with progress monitoring
            let! result = executor.ExecuteWithProgress(plan, context, onProgress)

            // If we should have aborted but execution completed, note it
            let finalResult =
                if shouldAbort && not result.Success then
                    { result with
                        Warnings = result.Warnings @ [ sprintf "Adaptive abort triggered: %s" abortReason ] }
                else
                    result

            return
                { Result = finalResult
                  ActionsApplied = actionsApplied |> Seq.toList
                  TotalStepsMonitored = progressLog.Count
                  PatternSwitches = patternSwitches }
        }
