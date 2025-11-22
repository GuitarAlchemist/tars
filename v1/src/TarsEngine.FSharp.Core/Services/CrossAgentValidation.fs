namespace TarsEngine.FSharp.Core.Services

open System

/// Utilities for aggregating validation results from multiple agents.
module CrossAgentValidation =

    /// Role assumed by a validating agent.
    type AgentRole =
        | Reasoner
        | Reviewer
        | Benchmarker
        | PerformanceBenchmarker
        | SafetyGovernor
        | CoordinationLead
        | MetaCritic
        | GoalSynthesizer
        | SpecGuardian
        | Custom of string

    /// Outcome reported by an agent.
    type ValidationOutcome =
        | Pass
        | Fail
        | NeedsReview

    /// Detailed validation result emitted by a single agent.
    type AgentValidationResult =
        { AgentId: string
          Role: AgentRole
          Outcome: ValidationOutcome
          Confidence: float option
          Notes: string option
          ProducedAt: DateTime }

    /// Rules for computing consensus across multiple agent results.
    type ConsensusRule =
        { MinimumPassCount: int
          RequiredRoles: AgentRole list
          AllowNeedsReview: bool
          MinimumConfidence: float option
          MaxFailureCount: int option }

    /// Aggregate outcome after applying consensus rules.
    type ConsensusOutcome =
        | ConsensusPassed of AgentValidationResult list
        | ConsensusNeedsReview of AgentValidationResult list * string
        | ConsensusFailed of AgentValidationResult list * string

    let private hasRequiredRole role results =
        results |> List.exists (fun r -> r.Role = role)

    let private filterPasses (rule: ConsensusRule) results =
        results
        |> List.filter (fun r ->
            match r.Outcome with
            | ValidationOutcome.Pass ->
                match rule.MinimumConfidence with
                | Some threshold ->
                    r.Confidence |> Option.exists (fun c -> c >= threshold)
                | None -> true
            | _ -> false)

    /// Evaluates agent results according to the provided rule set.
    let evaluate (rule: ConsensusRule) (results: AgentValidationResult list) : ConsensusOutcome =
        if results.IsEmpty then
            ConsensusNeedsReview(results, "No agent results supplied.")
        else
            let passes = filterPasses rule results
            let failures = results |> List.filter (fun r -> r.Outcome = ValidationOutcome.Fail)
            let needsReview = results |> List.filter (fun r -> r.Outcome = ValidationOutcome.NeedsReview)

            let missingRoles =
                rule.RequiredRoles
                |> List.filter (fun role -> hasRequiredRole role passes |> not)

            let failureExceeded =
                match rule.MaxFailureCount with
                | Some limit -> failures.Length > limit
                | None -> false

            if not missingRoles.IsEmpty then
                let missing =
                    missingRoles
                    |> List.map (function
                        | Custom value -> value
                        | other -> other.ToString())
                    |> String.concat ", "
                ConsensusFailed(results, $"Consensus missing required roles: {missing}.")

            elif passes.Length < rule.MinimumPassCount then
                ConsensusFailed(results, $"Consensus requires at least {rule.MinimumPassCount} passing agents (have {passes.Length}).")

            elif failureExceeded then
                ConsensusFailed(results, $"Consensus exceeded maximum failure count ({failures.Length}).")

            elif (not rule.AllowNeedsReview) && needsReview.Length > 0 then
                ConsensusNeedsReview(results, "Needs review entries present but consensus requires clear pass/fail.")

            else
                ConsensusPassed results
