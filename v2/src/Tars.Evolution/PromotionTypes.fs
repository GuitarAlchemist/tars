namespace Tars.Evolution

open System

/// The promotion staircase levels — each level represents higher abstraction
type PromotionLevel =
    | Implementation  // Raw code pattern observed in traces
    | Helper          // Extracted into a reusable function/module
    | Builder         // Computation expression / composable builder
    | DslClause       // WoT DSL construct
    | GrammarRule     // Full grammar production rule with compiler support

module PromotionLevel =
    let rank = function
        | Implementation -> 0
        | Helper -> 1
        | Builder -> 2
        | DslClause -> 3
        | GrammarRule -> 4

    let next = function
        | Implementation -> Some Helper
        | Helper -> Some Builder
        | Builder -> Some DslClause
        | DslClause -> Some GrammarRule
        | GrammarRule -> None

    let label = function
        | Implementation -> "implementation"
        | Helper -> "helper"
        | Builder -> "builder"
        | DslClause -> "dsl_clause"
        | GrammarRule -> "grammar_rule"

/// Tracks how many times a pattern has been observed across tasks
type RecurrenceRecord = {
    PatternId: string
    PatternName: string
    FirstSeen: DateTime
    LastSeen: DateTime
    OccurrenceCount: int
    TaskIds: string list
    Contexts: string list
    CurrentLevel: PromotionLevel
    PromotionHistory: (PromotionLevel * DateTime) list
    AverageScore: float
}

/// The 8 promotion criteria from Compound Engineering
type PromotionCriteria = {
    MinOccurrences: bool        // Appeared in >= 3 real tasks
    RemovesComplexity: bool     // Removes meaningful incidental complexity
    MoreReadable: bool          // More readable than expanded form
    StableSemantics: bool       // Semantics are stable across uses
    AutoValidatable: bool       // Can be validated automatically
    NoOverlap: bool             // Doesn't overlap with existing constructs
    ComposesCleanly: bool       // Composes with existing builders/types
    ImprovesPlanning: bool      // Improves AI planning, not just typing speed
}

module PromotionCriteria =
    let score (c: PromotionCriteria) =
        [ c.MinOccurrences; c.RemovesComplexity; c.MoreReadable
          c.StableSemantics; c.AutoValidatable; c.NoOverlap
          c.ComposesCleanly; c.ImprovesPlanning ]
        |> List.filter id
        |> List.length

    let empty = {
        MinOccurrences = false; RemovesComplexity = false
        MoreReadable = false; StableSemantics = false
        AutoValidatable = false; NoOverlap = false
        ComposesCleanly = false; ImprovesPlanning = false
    }

/// Result of Grammar Governor evaluation
type GovernanceDecision =
    | Approve of reason: string
    | Reject of reason: string
    | Defer of reason: string

/// Round-trip validation result.
type RoundtripResult = {
    PatternId: string
    OriginalTemplate: string       // The promoted abstraction
    ExpandedCode: string           // The lower-level expansion
    ReconstructedTemplate: string  // Re-abstracted from expanded code
    SemanticMatch: float           // 0.0-1.0 how similar original and reconstructed are
    Passed: bool                   // SemanticMatch >= threshold
    Issues: string list            // Any detected semantic losses
}

module RoundtripResult =
    let empty patternId = {
        PatternId = patternId
        OriginalTemplate = ""
        ExpandedCode = ""
        ReconstructedTemplate = ""
        SemanticMatch = 0.0
        Passed = false
        Issues = []
    }

/// Source system that produced/evolved a rule's weight
type RuleSource =
    | Tars
    | GuitarAlchemist
    | MachinDeOuf
    | Evolved
    | Manual

/// A grammar rule annotated with probabilistic weight metadata
type WeightedRule = {
    PatternId: string
    PatternName: string
    Level: PromotionLevel
    /// Raw 0-8 criteria score from GrammarGovernor
    RawScore: int
    /// Current probability weight in [0.0, 1.0]
    Weight: float
    /// Bayesian confidence in the weight estimate (higher = more certain)
    Confidence: float
    /// Success rate from execution outcomes
    SuccessRate: float
    /// How many times this rule has been selected for use
    SelectionCount: int
    /// Who/what assigned this weight
    Source: RuleSource
    /// When the weight was last updated
    LastUpdated: DateTime
}

/// A promotion candidate submitted for governance review
type PromotionCandidate = {
    Record: RecurrenceRecord
    ProposedLevel: PromotionLevel
    Criteria: PromotionCriteria
    Evidence: string list
    PatternTemplate: string
    RollbackExpansion: string option  // How to expand back to lower level
}

/// Lineage tracking for promoted abstractions
type LineageRecord = {
    Id: string
    PatternId: string
    FromLevel: PromotionLevel
    ToLevel: PromotionLevel
    PromotedAt: DateTime
    Criteria: PromotionCriteria
    Decision: GovernanceDecision
    RollbackExpansion: string option
    DerivedFrom: string option
    PromotedBy: string
    Confidence: float
}

/// Store for promotion-related data
type IPromotionStore =
    abstract member GetRecurrenceRecords: unit -> RecurrenceRecord list
    abstract member UpsertRecurrenceRecord: RecurrenceRecord -> unit
    abstract member GetLineageRecords: unit -> LineageRecord list
    abstract member AddLineageRecord: LineageRecord -> unit
    abstract member GetWeights: unit -> WeightedRule list
    abstract member SaveWeights: WeightedRule list -> unit

/// Validates promotions by expansion and re-abstraction
type IRoundtripValidator =
    abstract member Validate: candidate: PromotionCandidate -> RoundtripResult

/// Gates pattern promotions
type IPromotionGate =
    abstract member Decide: existing: RecurrenceRecord list * candidate: PromotionCandidate -> GovernanceDecision * RoundtripResult option

/// Layer 4: Evolution metadata attached to grammar nodes
type EvolutionMetadata = {
    PromotionLevel: PromotionLevel option
    OccurrenceCount: int
    PromotedFrom: string option
    LineageId: string option
    MutationHistory: string list
    RollbackExpansion: string option
    DerivedFrom: string option
    Confidence: float
    SemanticType: string option
    Effects: string list
}

module EvolutionMetadata =
    let empty = {
        PromotionLevel = None; OccurrenceCount = 0
        PromotedFrom = None; LineageId = None
        MutationHistory = []; RollbackExpansion = None
        DerivedFrom = None; Confidence = 0.0
        SemanticType = None; Effects = []
    }
