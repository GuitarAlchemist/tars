namespace Tars.Core.WorkflowOfThought

open System
open System.IO

/// <summary>
/// Domain model for mutant selection and promotion (Phase 15.5)
/// </summary>
[<RequireQualifiedAccess>]
module Selection =

    /// <summary>
    /// Criteria for accepting a variant as an improvement.
    /// </summary>
    type Criteria =
        { MinPassRate: float
          MaxCostIncrease: float
          RequireZeroRegressions: bool }

    let defaultCriteria =
        { MinPassRate = 0.8
          MaxCostIncrease = 0.2
          RequireZeroRegressions = true }

    /// <summary>
    /// Performance metrics for a single run or benchmark.
    /// </summary>
    type Performance =
        { PassRate: float
          TotalCost: decimal
          DiffCount: int }

    /// <summary>
    /// Selection decision outcome.
    /// </summary>
    type Decision =
        | Promote of Rationale: string
        | Rollback of Rationale: string
        | InsufficientData of string

    /// <summary>
    /// Compares mutant performance against a baseline and selects.
    /// </summary>
    let evaluate (criteria: Criteria) (baseline: Performance option) (mutant: Performance) : Decision =
        if mutant.PassRate < criteria.MinPassRate then
            Rollback(sprintf "Pass rate %.2f below threshold %.2f" mutant.PassRate criteria.MinPassRate)
        elif criteria.RequireZeroRegressions && mutant.DiffCount > 0 then
            Rollback(sprintf "Found %d regressions" mutant.DiffCount)
        else
            match baseline with
            | Some b ->
                let costIncrease =
                    if b.TotalCost = 0.0m then
                        0.0
                    else
                        float (mutant.TotalCost - b.TotalCost) / float b.TotalCost

                if mutant.PassRate > b.PassRate then
                    Promote(sprintf "Pass rate improved: %.2f -> %.2f" b.PassRate mutant.PassRate)
                elif costIncrease > criteria.MaxCostIncrease then
                    Rollback(
                        sprintf
                            "Cost increased by %.2f%% (limit %.0f%%)"
                            (costIncrease * 100.0)
                            (criteria.MaxCostIncrease * 100.0)
                    )
                else
                    // Promotion tier: Better or equal pass rate AND cost within bounds
                    Promote "Performance within acceptable bounds"
            | None -> Promote "Promoted as initial baseline variant"

    /// <summary>
    /// Service to apply the selection decision to the filesystem.
    /// </summary>
    type SelectionService() =

        member _.ApplyDecision(decision: Decision, variantPath: string, originalPath: string) =
            match decision with
            | Promote rationale ->
                if File.Exists variantPath then
                    File.Copy(variantPath, originalPath, true)
                    File.Delete(variantPath)
                    Result.Ok(sprintf "PROMOTED: %s" rationale)
                else
                    Result.Error "Variant path not found"

            | Rollback rationale ->
                if File.Exists variantPath then
                    File.Delete(variantPath)

                Result.Ok(sprintf "REJECTED: %s" rationale)

            | InsufficientData msg -> Result.Error(sprintf "SKIPPED: %s" msg)
