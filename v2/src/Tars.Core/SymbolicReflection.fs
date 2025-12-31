namespace Tars.Core

open System

// =============================================================================
// PHASE 15.1: STRUCTURED REFLECTION TYPES
// =============================================================================
//
// Transforms agent reflection from text generation into structured belief updates
// with formal justification chains. Reflection is a formal knowledge update operation.
// Reference: docs/3_Roadmap/2_Phases/phase_15_symbolic_reflection.md

/// Trigger that initiates a reflection cycle
type ReflectionTrigger =
    | TaskCompleted of taskId: Guid * result: ReflectionTaskResult
    | TaskFailed of taskId: Guid * error: string
    | ContradictionDetected of belief1: string * belief2: string
    | InvariantViolated of invariant: string * action: AgentAction
    | ResourceExhausted of resource: string
    | PatternRecognized of patternName: string * confidence: float
    | AnomalyFound of description: string * severity: AnomalySeverity
    | EpistemicConflict of conflictingBeliefs: string list
    | PeriodicReview of interval: TimeSpan
    | ExternalFeedback of source: string * feedback: string

and ReflectionTaskResult =
    | TaskSuccess of output: string
    | TaskPartialSuccess of output: string * warnings: string list
    | TaskFailure of error: string

/// Trend direction for metrics
type ReflectionTrend =
    | Improving
    | Stable
    | Degrading
    | TrendUnknown

/// An observation made during reflection
type ReflectionObservation =
    | ContradictionObserved of source: string * target: string * reason: string
    | PatternObserved of patternName: string * instances: int * confidence: float
    | AnomalyObserved of description: string * severity: AnomalySeverity
    | PerformanceObserved of metric: string * value: float * trend: ReflectionTrend
    | BehaviorChangeObserved of description: string * before: string * after: string
    | ResourceTrendObserved of resource: string * usageHistory: float list * trend: ReflectionTrend
    | SuccessPatternObserved of actionType: string * successRate: float * sampleCount: int
    | FailurePatternObserved of actionType: string * failureRate: float * commonCause: string option

/// Source of evidence for a reflection belief
type ReflectionEvidenceSource =
    | DirectObservation of description: string * timestamp: DateTimeOffset
    | ExternalSource of url: string * extractedAt: DateTimeOffset
    | InferredFrom of premises: string list
    | TestResult of testName: string * passed: bool
    | UserFeedback of feedback: string * userId: string option
    | AgentExperience of agentId: AgentId * experienceType: string

/// Evidence supporting a reflection belief or update
type ReflectionEvidence =
    { Id: Guid
      Source: ReflectionEvidenceSource
      Content: string
      Confidence: float
      Timestamp: DateTimeOffset
      Metadata: Map<string, string> }

    static member Create(source, content, confidence) =
        { Id = Guid.NewGuid()
          Source = source
          Content = content
          Confidence = confidence
          Timestamp = DateTimeOffset.UtcNow
          Metadata = Map.empty }

/// A belief with its evidence chain for reflection
type ReflectionBelief =
    { Id: Guid
      Statement: string
      Confidence: float
      Evidence: ReflectionEvidence list
      CreatedAt: DateTimeOffset
      LastUpdated: DateTimeOffset
      Tags: string list }

    static member Create(statement, confidence) =
        { Id = Guid.NewGuid()
          Statement = statement
          Confidence = confidence
          Evidence = []
          CreatedAt = DateTimeOffset.UtcNow
          LastUpdated = DateTimeOffset.UtcNow
          Tags = [] }

/// Types of belief updates that can occur during reflection
type ReflectionBeliefUpdate =
    | AddBelief of belief: ReflectionBelief * evidence: ReflectionEvidence list
    | RevokeBelief of beliefId: Guid * reason: string * replacement: ReflectionBelief option
    | AdjustConfidence of beliefId: Guid * oldConfidence: float * newConfidence: float * reason: string
    | ResolveContradiction of resolution: ReflectionConflictResolution
    | MergeBeliefs of sourceIds: Guid list * merged: ReflectionBelief
    | SplitBelief of originalId: Guid * refined: ReflectionBelief list
    | AddEvidence of beliefId: Guid * evidence: ReflectionEvidence
    | RemoveEvidence of beliefId: Guid * evidenceId: Guid * reason: string

and ReflectionConflictResolution =
    { ConflictId: Guid
      ConflictingBeliefs: Guid list
      Strategy: ReflectionResolutionStrategy
      Winner: Guid option
      MergedBelief: ReflectionBelief option
      Justification: string }

and ReflectionResolutionStrategy =
    | HighestConfidenceWins
    | MostRecentWins
    | MostEvidenceWins
    | MergeCompatible
    | SplitIntoContexts
    | DeferToHuman

/// A step in the reflection reasoning chain
type ReflectionReasoningStep =
    | AssumptionMade of assumption: string * basis: ReflectionEvidence
    | InferenceMade of premises: string list * conclusion: string * rule: ReflectionInferenceRule
    | CounterevidenceConsidered of evidence: ReflectionEvidence * weight: float * accepted: bool
    | AlternativeRejected of alternative: string * reason: string
    | ConfidenceUpdated of beliefId: Guid * oldValue: float * newValue: float * reason: string

and ReflectionInferenceRule =
    | ModusPonens // If P and P→Q, then Q
    | ModusTollens // If P→Q and ¬Q, then ¬P
    | Syllogism // If P→Q and Q→R, then P→R
    | Contraposition // P→Q ≡ ¬Q→¬P
    | Generalization // P(x) for all observed x → ∀x P(x)
    | Specialization // ∀x P(x) → P(c) for specific c
    | Abduction // Q and P→Q → maybe P
    | Analogy // P similar to Q, P has property → Q might have property
    | StatisticalInference // Observed pattern → probabilistic conclusion

/// Symbolic proof for a reflection belief update
type ReflectionProof =
    | Tautology of statement: string
    | ProofContradiction of statement: string
    | ValidationSuccess of testName: string * details: string
    | ValidationFailure of testName: string * error: string
    | LogicalInference of premises: string list * conclusion: string * rule: ReflectionInferenceRule
    | StatisticalEvidence of samples: int * successRate: float * confidenceInterval: float
    | ExpertAssertion of source: string * credibility: float

/// Justification for a reflection belief update
type ReflectionJustification =
    { Id: Guid
      Update: ReflectionBeliefUpdate
      Evidence: ReflectionEvidence list
      ReasoningChain: ReflectionReasoningStep list
      Proof: ReflectionProof option
      Confidence: float
      CreatedAt: DateTimeOffset }

    static member Create(update, evidence, reasoningChain, proof, confidence) =
        { Id = Guid.NewGuid()
          Update = update
          Evidence = evidence
          ReasoningChain = reasoningChain
          Proof = proof
          Confidence = confidence
          CreatedAt = DateTimeOffset.UtcNow }

/// A complete symbolic reflection
type SymbolicReflection =
    { ReflectionId: Guid
      Timestamp: DateTimeOffset
      AgentId: AgentId
      Trigger: ReflectionTrigger
      Observations: ReflectionObservation list
      BeliefUpdates: ReflectionBeliefUpdate list
      Justifications: ReflectionJustification list
      ImpactScore: float
      Confidence: float
      ProcessingTime: TimeSpan option
      Metadata: Map<string, string> }

    /// Create a new reflection
    static member Create(agentId, trigger) =
        { ReflectionId = Guid.NewGuid()
          Timestamp = DateTimeOffset.UtcNow
          AgentId = agentId
          Trigger = trigger
          Observations = []
          BeliefUpdates = []
          Justifications = []
          ImpactScore = 0.0
          Confidence = 0.0
          ProcessingTime = None
          Metadata = Map.empty }

    /// Add an observation
    member this.WithObservation(obs: ReflectionObservation) =
        { this with
            Observations = obs :: this.Observations }

    /// Add a belief update with justification
    member this.WithBeliefUpdate(update: ReflectionBeliefUpdate, justification: ReflectionJustification) =
        { this with
            BeliefUpdates = update :: this.BeliefUpdates
            Justifications = justification :: this.Justifications }

    /// Calculate overall confidence from justifications
    member this.CalculateConfidence() =
        if this.Justifications.IsEmpty then
            0.0
        else
            this.Justifications |> List.averageBy (fun j -> j.Confidence)

    /// Calculate impact score based on update types
    member this.CalculateImpact() =
        this.BeliefUpdates
        |> List.sumBy (fun update ->
            match update with
            | AddBelief _ -> 0.3
            | RevokeBelief _ -> 0.8
            | AdjustConfidence(_, old, newC, _) -> abs (newC - old)
            | ResolveContradiction _ -> 0.9
            | MergeBeliefs _ -> 0.5
            | SplitBelief _ -> 0.6
            | AddEvidence _ -> 0.2
            | RemoveEvidence _ -> 0.4)
        |> fun total -> min 1.0 total

module SymbolicReflectionHelpers =

    /// Create a reflection from a task completion
    let fromTaskCompletion (agentId: AgentId) (taskId: Guid) (result: ReflectionTaskResult) =
        let trigger =
            match result with
            | TaskSuccess _ -> TaskCompleted(taskId, result)
            | TaskPartialSuccess _ -> TaskCompleted(taskId, result)
            | TaskFailure error -> TaskFailed(taskId, error)

        SymbolicReflection.Create(agentId, trigger)

    /// Create a reflection from a contradiction
    let fromContradiction (agentId: AgentId) (belief1: string) (belief2: string) =
        SymbolicReflection.Create(agentId, ContradictionDetected(belief1, belief2))

    /// Create a reflection from an invariant violation
    let fromInvariantViolation (agentId: AgentId) (invariant: string) (action: AgentAction) =
        SymbolicReflection.Create(agentId, InvariantViolated(invariant, action))

    /// Create a reflection from external feedback
    let fromFeedback (agentId: AgentId) (source: string) (feedback: string) =
        SymbolicReflection.Create(agentId, ExternalFeedback(source, feedback))

    /// Get a summary of the reflection
    let summarize (reflection: SymbolicReflection) =
        sprintf
            "Reflection %A: %d observations, %d updates, confidence=%.2f, impact=%.2f"
            reflection.ReflectionId
            reflection.Observations.Length
            reflection.BeliefUpdates.Length
            reflection.Confidence
            reflection.ImpactScore
