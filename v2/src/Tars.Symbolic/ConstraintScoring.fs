namespace Tars.Symbolic

open System
open Tars.Core

/// Constraint scoring functions (Logic Tensor Network-style, without tensors)
/// Provides continuous (0.0-1.0) scores instead of binary pass/fail
module ConstraintScoring =

    /// Score belief consistency based on contradiction count
    /// Returns 1.0 if no contradictions, approaches 0.0 as contradictions increase
    let scoreBeliefConsistency (belief: string) (existingBeliefs: string list) : float =
        // TODO: Implement actual contradiction detection
        // For now, check for explicit "NOT" patterns as a simple heuristic
        let contradictions =
            existingBeliefs
            |> List.filter (fun existing ->
                (belief.Contains("NOT") && existing.Contains(belief.Replace("NOT ", "")))
                || (existing.Contains("NOT") && belief.Contains(existing.Replace("NOT ", ""))))
            |> List.length

        // Continuous score: 1.0 (no contradictions) to 0.0 (many contradictions)
        1.0 / (1.0 + float contradictions * 0.5)

    /// Score grammar validity based on parse success
    /// Returns 1.0 if parseable, 0.0 if not
    let scoreGrammarValidity (rule: string) (production: string) : float =
        // TODO: Integrate with actual grammar parser
        // For now, basic validation
        if String.IsNullOrWhiteSpace(production) then
            0.0
        elif production.Contains("|") || production.Contains("*") || production.Contains("+") then
            1.0
        elif production.Length > 3 then
            0.8 // Probably valid
        else
            0.5 // Uncertain

    /// Score alignment with threshold
    /// Returns proportional score if below threshold, 1.0 if above
    let scoreAlignment (metric: string) (threshold: float) (actual: float) : float =
        if actual >= threshold then 1.0 else actual / threshold // Proportional

    /// Score code complexity bound
    /// Returns 1.0 if within bound, inverse ratio if exceeded
    let scoreComplexity (maxComplexity: float) (actual: float) : float =
        if actual <= maxComplexity then
            1.0
        else
            maxComplexity / actual // Inverse: lower is better

    /// Score resource usage against quota
    /// Returns 1.0 if within quota, inverse ratio if exceeded
    let scoreResourceUsage (quota: int) (actual: int) : float =
        if actual <= quota then 1.0 else float quota / float actual // Inverse

    /// Score temporal ordering
    /// Returns 1.0 if correctly ordered, 0.0 if not, 0.5 if unknown
    let scoreTemporalOrdering (beforeTime: DateTime option) (afterTime: DateTime option) : float =
        match beforeTime, afterTime with
        | Some bt, Some at when bt < at -> 1.0
        | Some bt, Some at when bt > at -> 0.0
        | _ -> 0.5 // Unknown, neutral score

    /// Score custom invariant with exception handling
    let scoreCustomInvariant (validator: unit -> bool) : float =
        try
            if validator () then 1.0 else 0.0
        with _ ->
            0.0 // Failed validation = violated

    /// Calculate multi-factor belief stability score
    let scoreBeliefStability
        (belief: string)
        (existingBeliefs: string list)
        (evidenceCount: int)
        (ageInDays: float)
        : float =

        let consistencyScore = scoreBeliefConsistency belief existingBeliefs

        // Evidence strength: more evidence = higher score (capped at 1.0)
        let evidenceScore = min 1.0 (float evidenceCount / 5.0)

        // Temporal stability: older beliefs are more stable (capped at 1.0)
        let temporalScore = min 1.0 (ageInDays / 30.0) // Max at 30 days

        // Weighted average (consistency matters most)
        consistencyScore * 0.5 + evidenceScore * 0.3 + temporalScore * 0.2

    /// Score an action against a single invariant
    let scoreAgainstInvariant (invariant: SymbolicInvariant) (context: Map<string, obj>) : float =
        match invariant with
        | SymbolicInvariant.GrammarValidity(rule, prod) -> scoreGrammarValidity rule prod

        | SymbolicInvariant.BeliefConsistency beliefs ->
            // Score average consistency across all beliefs
            if beliefs.IsEmpty then
                1.0
            else
                beliefs
                |> List.map (fun b -> scoreBeliefConsistency b (beliefs |> List.filter ((<>) b)))
                |> List.average

        | SymbolicInvariant.AlignmentThreshold(metric, threshold) ->
            // Try to get actual value from context
            match context.TryFind metric with
            | Some(:? float as actual) -> scoreAlignment metric threshold actual
            | _ -> 0.5 // Unknown, neutral

        | SymbolicInvariant.CodeComplexityBound maxComplexity ->
            match context.TryFind "complexity" with
            | Some(:? float as actual) -> scoreComplexity maxComplexity actual
            | _ -> 1.0 // Assume OK if not measured

        | SymbolicInvariant.ResourceQuota(resource, limit) ->
            match context.TryFind resource with
            | Some(:? int as actual) -> scoreResourceUsage limit actual
            | _ -> 1.0 // Assume OK if not measured

        | SymbolicInvariant.TemporalConstraint(before, after) ->
            match context.TryFind "before_time", context.TryFind "after_time" with
            | Some(:? DateTime as bt), Some(:? DateTime as at) -> scoreTemporalOrdering (Some bt) (Some at)
            | _ -> 0.5 // Unknown

        | SymbolicInvariant.CustomInvariant(_, validator) -> scoreCustomInvariant validator

    /// Calculate normalized score (ensures 0.0-1.0 range)
    let normalize (score: float) : float = max 0.0 (min 1.0 score)

    /// Apply threshold to score (hard cutoff)
    let applyThreshold (threshold: float) (score: float) : float =
        if score >= threshold then score else 0.0

    /// Smooth score using sigmoid function (soft cutoff)
    let smoothScore (midpoint: float) (steepness: float) (score: float) : float =
        1.0 / (1.0 + exp (-steepness * (score - midpoint)))

    /// Calculate confidence interval for score
    let confidenceInterval (score: float) (sampleSize: int) : float * float =
        let z = 1.96 // 95% confidence
        let variance = score * (1.0 - score) / float sampleSize
        let margin = z * sqrt variance
        (max 0.0 (score - margin), min 1.0 (score + margin))

    /// Combine scores using different strategies
    type CombinationStrategy =
        | MinimumScore // Pessimistic: take worst score
        | MaximumScore // Optimistic: take best score
        | AverageScore // Balanced: average all scores
        | WeightedAverage of weights: float list
        | HarmonicMean // Emphasize low scores
        | GeometricMean // Balanced but sensitive to zeros

    /// Combine multiple scores using specified strategy
    let combineScores (strategy: CombinationStrategy) (scores: float list) : float =
        if scores.IsEmpty then
            0.0
        else
            match strategy with
            | MinimumScore -> scores |> List.min
            | MaximumScore -> scores |> List.max
            | AverageScore -> scores |> List.average
            | WeightedAverage weights ->
                if weights.Length <> scores.Length then
                    failwith "Weights and scores must have same length"

                List.zip scores weights
                |> List.map (fun (s, w) -> s * w)
                |> List.sum
                |> fun total -> total / (List.sum weights)
            | HarmonicMean ->
                let n = float scores.Length
                n / (scores |> List.map (fun s -> 1.0 / max 0.001 s) |> List.sum)
            | GeometricMean -> scores |> List.map (fun s -> log (max 0.001 s)) |> List.average |> exp

    /// Score multiple invariants and aggregate
    let scoreInvariants
        (invariants: SymbolicInvariant list)
        (context: Map<string, obj>)
        (strategy: CombinationStrategy)
        : float =

        invariants
        |> List.map (fun inv -> scoreAgainstInvariant inv context)
        |> combineScores strategy
        |> normalize

    /// Performance metrics for scoring
    type ScoringMetrics =
        { InvariantsChecked: int
          TotalTimeMs: float
          AverageTimePerInvariant: float
          Score: float }

    /// Score with performance tracking
    let scoreWithMetrics
        (invariants: SymbolicInvariant list)
        (context: Map<string, obj>)
        (strategy: CombinationStrategy)
        : float * ScoringMetrics =

        let sw = System.Diagnostics.Stopwatch.StartNew()
        let score = scoreInvariants invariants context strategy
        sw.Stop()

        let metrics =
            { InvariantsChecked = invariants.Length
              TotalTimeMs = sw.Elapsed.TotalMilliseconds
              AverageTimePerInvariant =
                if invariants.IsEmpty then
                    0.0
                else
                    sw.Elapsed.TotalMilliseconds / float invariants.Length
              Score = score }

        (score, metrics)
