namespace Tars.Cortex

open System
open System.Text.RegularExpressions
open Tars.Cortex.WoTTypes

/// <summary>
/// Golden trace regression checker.
/// Auto-compares WoT execution results against stored golden traces
/// and flags regressions. Optionally promotes successful results to
/// new golden baselines.
/// </summary>
module RegressionChecker =

    // =========================================================================
    // Types
    // =========================================================================

    /// A single step-level change detected during regression comparison.
    type StepChange =
        { Field: string
          Expected: string
          Actual: string }

    /// Regression severity level.
    type RegressionSeverity =
        | None
        | Minor   // drift score <= 0.2
        | Major   // drift score <= 0.5
        | Critical // drift score > 0.5

    /// Report produced by the regression checker.
    type RegressionReport =
        { GoldenName: string
          DriftScore: float
          Severity: RegressionSeverity
          StepChanges: StepChange list
          Warnings: string list
          IsRegression: bool }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Sanitize a goal string into a safe file-system friendly golden name.
    let goalToGoldenName (goal: string) : string =
        let cleaned =
            Regex.Replace(goal.ToLowerInvariant().Trim(), @"[^a-z0-9]+", "_")
        let trimmed =
            cleaned.Trim('_')
        if trimmed.Length > 80 then trimmed.Substring(0, 80).TrimEnd('_')
        else trimmed

    /// Compute a 0.0-1.0 drift score from a list of trace diffs.
    /// Each diff field contributes a weighted fraction.
    let private computeDriftScore (diffs: GoldenTraceStore.TraceDiff list) (totalSteps: int) : float =
        if diffs.IsEmpty then 0.0
        else
            let fieldWeight (field: string) =
                match field with
                | "FinalStatus"     -> 0.30
                | "FailedSteps"     -> 0.25
                | "SuccessfulSteps" -> 0.20
                | "TotalSteps"      -> 0.10
                | "StepStatuses"    -> 0.20
                | "StepKinds"       -> 0.15
                | "PatternKind"     -> 0.10
                | _                 -> 0.05
            let raw = diffs |> List.sumBy (fun d -> fieldWeight d.Field)
            min 1.0 raw

    /// Classify severity from a drift score.
    let private classifySeverity (driftScore: float) : RegressionSeverity =
        if driftScore <= 0.0 then RegressionSeverity.None
        elif driftScore <= 0.2 then Minor
        elif driftScore <= 0.5 then Major
        else Critical

    // =========================================================================
    // Core Functions
    // =========================================================================

    /// <summary>
    /// Check a WoT result for regression against a stored golden trace.
    /// Returns None if no golden trace exists for this goal.
    /// Returns Some report with drift score and changed steps otherwise.
    /// </summary>
    let checkRegression (result: WoTResult) (goal: string) : RegressionReport option =
        let goldenName = goalToGoldenName goal
        let allGoldens = GoldenTraceStore.listAll ()

        if not (allGoldens |> List.contains goldenName) then
            // No golden trace exists for this goal — nothing to compare
            Option.None
        else
            let canonical = GoldenTraceStore.toCanonical result.Trace
            let comparison = GoldenTraceStore.compareAgainstGolden goldenName canonical

            let stepChanges =
                comparison.Diffs
                |> List.map (fun d ->
                    { Field = d.Field
                      Expected = d.Expected
                      Actual = d.Actual })

            let driftScore = computeDriftScore comparison.Diffs result.Metrics.TotalSteps
            let severity = classifySeverity driftScore

            let warnings =
                [ if driftScore > 0.5 then
                      sprintf "CRITICAL regression detected (drift=%.2f) against golden '%s'" driftScore goldenName
                  if comparison.Diffs |> List.exists (fun d -> d.Field = "FinalStatus") then
                      "Final execution status changed from golden baseline"
                  if comparison.Diffs |> List.exists (fun d -> d.Field = "FailedSteps") then
                      "Number of failed steps differs from golden baseline" ]

            Some
                { GoldenName = goldenName
                  DriftScore = driftScore
                  Severity = severity
                  StepChanges = stepChanges
                  Warnings = warnings
                  IsRegression = driftScore > 0.0 }

    /// <summary>
    /// Maybe promote a successful WoT result to a golden trace.
    /// - If no golden exists for this goal, saves the result as a new baseline.
    /// - If a golden exists and the new result has a higher success rate
    ///   (more successful steps as a fraction of total), updates the baseline.
    /// Returns Some path if a golden was saved/updated, None otherwise.
    /// </summary>
    let maybePromoteToGolden (result: WoTResult) (goal: string) : Result<string, string> option =
        if not result.Success then
            // Only promote successful results
            Option.None
        else
            let goldenName = goalToGoldenName goal
            let allGoldens = GoldenTraceStore.listAll ()
            let canonical = GoldenTraceStore.toCanonical result.Trace

            if not (allGoldens |> List.contains goldenName) then
                // No golden exists — save as new baseline
                Some (GoldenTraceStore.save goldenName canonical)
            else
                // Golden exists — check if the new result is better
                match GoldenTraceStore.load goldenName with
                | Ok existing ->
                    let existingRate =
                        if existing.TotalSteps > 0 then
                            float existing.SuccessfulSteps / float existing.TotalSteps
                        else 0.0
                    let newRate =
                        if canonical.TotalSteps > 0 then
                            float canonical.SuccessfulSteps / float canonical.TotalSteps
                        else 0.0

                    if newRate > existingRate then
                        Some (GoldenTraceStore.save goldenName canonical)
                    else
                        // Existing golden is equal or better — keep it
                        Option.None
                | Error _ ->
                    // Failed to load existing golden — overwrite
                    Some (GoldenTraceStore.save goldenName canonical)

    /// Format a regression report as a human-readable string.
    let formatReport (report: RegressionReport) : string =
        let severityStr =
            match report.Severity with
            | RegressionSeverity.None -> "PASS"
            | Minor -> "MINOR"
            | Major -> "MAJOR"
            | Critical -> "CRITICAL"

        let header =
            sprintf "[Regression %s] golden='%s' drift=%.2f"
                severityStr report.GoldenName report.DriftScore

        if report.StepChanges.IsEmpty then
            header
        else
            let changes =
                report.StepChanges
                |> List.map (fun c -> sprintf "  - %s: expected '%s', got '%s'" c.Field c.Expected c.Actual)
                |> String.concat "\n"
            sprintf "%s\n%s" header changes
