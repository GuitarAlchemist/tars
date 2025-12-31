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
