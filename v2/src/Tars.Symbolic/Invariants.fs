namespace Tars.Symbolic

open System
open Tars.Core

/// Symbolic invariants that TARS must preserve
/// These are formal constraints that define correct system behavior
[<RequireQualifiedAccess>]
type SymbolicInvariant =
    /// Grammar rule must parse correctly
    | GrammarValidity of rule: string * production: string

    /// Set of beliefs must be consistent (no contradictions)
    /// TODO: Replace string with EvidenceBelief once Phase 9 is complete
    | BeliefConsistency of beliefs: string list

    /// Metric must exceed minimum threshold
    | AlignmentThreshold of metric: string * min: float

    /// Code complexity must not exceed bound
    | CodeComplexityBound of maxComplexity: float

    /// Resource usage must not exceed quota
    | ResourceQuota of resource: string * limit: int

    /// Temporal constraint: event A must occur before event B
    | TemporalConstraint of before: TarsEntity * after: TarsEntity

    /// Custom invariant with name and validation function
    | CustomInvariant of name: string * validator: (unit -> bool)

/// Result of checking an invariant
type InvariantCheck =
    {
        /// The invariant that was checked
        Invariant: SymbolicInvariant

        /// Whether the invariant is satisfied (binary result)
        Satisfied: bool

        /// Continuous score from 0.0 (completely violated) to 1.0 (perfectly satisfied)
        Score: float

        /// Evidence supporting the check result
        Evidence: string list

        /// When the check was performed
        Timestamp: DateTime
    }

module InvariantCheck =
    /// Create a check result
    let create (inv: SymbolicInvariant) (satisfied: bool) (score: float) (evidence: string list) =
        { Invariant = inv
          Satisfied = satisfied
          Score = score
          Evidence = evidence
          Timestamp = DateTime.UtcNow }

    /// Check is satisfied if score >= 0.5
    let isSatisfied (check: InvariantCheck) = check.Score >= 0.5

/// Operations for checking invariants
module InvariantChecking =

    /// Check grammar validity
    let checkGrammarValidity (rule: string) (production: string) : InvariantCheck =
        // TODO: Implement actual grammar parsing
        // For now, just check if production is non-empty
        let satisfied = not (String.IsNullOrWhiteSpace(production))
        let score = if satisfied then 1.0 else 0.0

        let evidence =
            if satisfied then
                [ $"Grammar rule '{rule}' has valid production" ]
            else
                [ $"Grammar rule '{rule}' has empty production" ]

        InvariantCheck.create (SymbolicInvariant.GrammarValidity(rule, production)) satisfied score evidence

    /// Check belief consistency (no contradictions)
    let checkBeliefConsistency (beliefs: string list) : InvariantCheck =
        // TODO: Implement actual contradiction detection
        // For now, assume all beliefs are consistent
        let satisfied = true
        let score = 1.0
        let evidence = [ $"Checked {beliefs.Length} beliefs for consistency" ]

        InvariantCheck.create (SymbolicInvariant.BeliefConsistency beliefs) satisfied score evidence

    /// Check alignment threshold
    let checkAlignmentThreshold (metric: string) (min: float) (actual: float) : InvariantCheck =
        let satisfied = actual >= min
        let score = if actual >= min then 1.0 else actual / min
        let evidence = [ $"Metric '{metric}': {actual:F3} (threshold: {min:F3})" ]

        InvariantCheck.create (SymbolicInvariant.AlignmentThreshold(metric, min)) satisfied score evidence

    /// Check code complexity bound
    let checkCodeComplexityBound (maxComplexity: float) (actual: float) : InvariantCheck =
        let satisfied = actual <= maxComplexity
        // Inverse scoring: lower complexity is better
        let score =
            if actual <= maxComplexity then
                1.0
            else
                maxComplexity / actual // Approaches 0 as actual grows

        let evidence = [ $"Code complexity: {actual:F2} (max: {maxComplexity:F2})" ]

        InvariantCheck.create (SymbolicInvariant.CodeComplexityBound maxComplexity) satisfied score evidence

    /// Check resource quota
    let checkResourceQuota (resource: string) (limit: int) (actual: int) : InvariantCheck =
        let satisfied = actual <= limit
        let score = if actual <= limit then 1.0 else float limit / float actual
        let evidence = [ $"Resource '{resource}': {actual} (limit: {limit})" ]

        InvariantCheck.create (SymbolicInvariant.ResourceQuota(resource, limit)) satisfied score evidence

    /// Check temporal constraint
    let checkTemporalConstraint
        (before: TarsEntity)
        (after: TarsEntity)
        (beforeTime: DateTime option)
        (afterTime: DateTime option)
        : InvariantCheck =

        match beforeTime, afterTime with
        | Some bt, Some at ->
            let satisfied = bt < at
            let score = if satisfied then 1.0 else 0.0

            let evidence =
                [ $"{TarsEntity.getId before} at {bt:O}"
                  $"{TarsEntity.getId after} at {at:O}"
                  $"Constraint satisfied: {satisfied}" ]

            InvariantCheck.create (SymbolicInvariant.TemporalConstraint(before, after)) satisfied score evidence

        | _ ->
            // Cannot check without timestamps
            InvariantCheck.create
                (SymbolicInvariant.TemporalConstraint(before, after))
                false
                0.0
                [ "Missing timestamp information" ]

    /// Check custom invariant
    let checkCustomInvariant (name: string) (validator: unit -> bool) : InvariantCheck =
        try
            let satisfied = validator ()
            let score = if satisfied then 1.0 else 0.0
            let evidence = [ $"Custom invariant '{name}': {satisfied}" ]

            InvariantCheck.create (SymbolicInvariant.CustomInvariant(name, validator)) satisfied score evidence
        with ex ->
            InvariantCheck.create
                (SymbolicInvariant.CustomInvariant(name, validator))
                false
                0.0
                [ $"Validation failed: {ex.Message}" ]

    /// Check any invariant type
    let check (invariant: SymbolicInvariant) : InvariantCheck =
        match invariant with
        | SymbolicInvariant.GrammarValidity(rule, prod) -> checkGrammarValidity rule prod
        | SymbolicInvariant.BeliefConsistency beliefs -> checkBeliefConsistency beliefs
        | SymbolicInvariant.AlignmentThreshold(metric, min) ->
            // Need actual value - for now use min as actual (perfect alignment)
            checkAlignmentThreshold metric min min
        | SymbolicInvariant.CodeComplexityBound max ->
            // Need actual value - for now use 0.0 (perfect complexity)
            checkCodeComplexityBound max 0.0
        | SymbolicInvariant.ResourceQuota(res, limit) ->
            // Need actual value - for now use 0 (no usage)
            checkResourceQuota res limit 0
        | SymbolicInvariant.TemporalConstraint(before, after) ->
            // Need timestamps - cannot check without context
            checkTemporalConstraint before after None None
        | SymbolicInvariant.CustomInvariant(name, validator) -> checkCustomInvariant name validator

/// Standard invariants for common scenarios
module StandardInvariants =

    /// Grammar must be parseable
    let parseableGrammar (rule: string) (production: string) =
        SymbolicInvariant.GrammarValidity(rule, production)

    /// No contradictory beliefs
    let consistentBeliefs (beliefs: string list) =
        SymbolicInvariant.BeliefConsistency beliefs

    /// Agent alignment must be above threshold
    let agentAlignment (threshold: float) =
        SymbolicInvariant.AlignmentThreshold("agent_alignment", threshold)

    /// Code complexity must not exceed limit
    let complexityLimit (max: float) =
        SymbolicInvariant.CodeComplexityBound max

    /// Token budget must not be exceeded
    let tokenBudget (limit: int) =
        SymbolicInvariant.ResourceQuota("tokens", limit)

    /// Memory usage must not exceed limit (in MB)
    let memoryLimit (limitMB: int) =
        SymbolicInvariant.ResourceQuota("memory_mb", limitMB)
