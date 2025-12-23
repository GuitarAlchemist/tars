/// TARS Knowledge Types - Core symbolic representations
/// Following the vision: "Symbols are earned, not assumed"
namespace Tars.Knowledge

open System

// =============================================================================
// ENTITY IDENTIFIERS
// =============================================================================

/// Unique identifier for a belief
[<Struct>]
type BeliefId =
    | BeliefId of Guid

    static member New() = BeliefId(Guid.NewGuid())
    static member Parse(s: string) = BeliefId(Guid.Parse(s))
    member this.Value = let (BeliefId g) = this in g

    override this.ToString() =
        $"b:{this.Value.ToString().Substring(0, 8)}"

/// Unique identifier for an entity (subject or object of beliefs)
[<Struct>]
type EntityId =
    | EntityId of string

    member this.Value = let (EntityId s) = this in s
    override this.ToString() = this.Value

/// Unique identifier for an agent
[<Struct>]
type AgentId =
    | AgentId of string

    member this.Value = let (AgentId s) = this in s
    static member System = AgentId "system"
    static member User = AgentId "user"

/// Unique identifier for a plan
[<Struct>]
type PlanId =
    | PlanId of Guid

    static member New() = PlanId(Guid.NewGuid())
    member this.Value = let (PlanId g) = this in g

    override this.ToString() =
        $"p:{this.Value.ToString().Substring(0, 8)}"

/// Unique identifier for a run/execution
[<Struct>]
type RunId =
    | RunId of Guid

    static member New() = RunId(Guid.NewGuid())
    member this.Value = let (RunId g) = this in g

// =============================================================================
// RELATION TYPES
// =============================================================================

/// Semantic relation types for beliefs
/// Keep this small and atomic - "essays become assertions"
type RelationType =
    // Definitional
    | IsA // X is_a Y (type/class membership)
    | PartOf // X part_of Y (composition)
    | HasProperty // X has_property Y (attribute)

    // Logical
    | Supports // X supports Y (evidence)
    | Contradicts // X contradicts Y (conflict)
    | DerivedFrom // X derived_from Y (inference chain)

    // Causal
    | Causes // X causes Y
    | Prevents // X prevents Y
    | Enables // X enables Y

    // Temporal
    | Precedes // X precedes Y (time order)
    | Supersedes // X supersedes Y (version)

    // Reference
    | Mentions // X mentions Y
    | Cites // X cites Y
    | Implements // X implements Y

    // Custom with explicit name
    | Custom of string

    override this.ToString() =
        match this with
        | Custom s -> s
        | _ -> sprintf "%A" this |> fun s -> s.ToLowerInvariant()

// =============================================================================
// PROVENANCE - "Every belief answers: Who? When? From what?"
// =============================================================================

/// Source type for provenance
type SourceType =
    | Run of RunId // From a TARS execution run
    | Agent of AgentId // Asserted by an agent during reasoning
    | External of Uri // Ingested from external source
    | User // Explicitly provided by user
    | Inference // Derived from other beliefs
    | System // System initialization

/// Provenance tracks the origin of every belief
/// This is non-negotiable - provenance is the soul of symbolic truth
type Provenance =
    { Source: SourceType
      SourceUri: Uri option // URL/file path if applicable
      ContentHash: string option // SHA256 of source content
      ExtractedBy: AgentId option // Agent that extracted this
      ExtractedAt: DateTime // When extraction happened
      Confidence: float // 0.0-1.0 confidence from source
      EvidenceChain: BeliefId list } // Chain of beliefs this derives from

    static member FromRun(runId: RunId, agentId: AgentId) =
        { Source = Run runId
          SourceUri = None
          ContentHash = None
          ExtractedBy = Some agentId
          ExtractedAt = DateTime.UtcNow
          Confidence = 1.0
          EvidenceChain = [] }

    static member FromExternal(uri: Uri, hash: string option, confidence: float) =
        { Source = External uri
          SourceUri = Some uri
          ContentHash = hash
          ExtractedBy = None
          ExtractedAt = DateTime.UtcNow
          Confidence = confidence
          EvidenceChain = [] }

    static member FromUser() =
        { Source = User
          SourceUri = None
          ContentHash = None
          ExtractedBy = None
          ExtractedAt = DateTime.UtcNow
          Confidence = 1.0
          EvidenceChain = [] }

// =============================================================================
// BELIEF - The atomic unit of knowledge
// =============================================================================

/// A belief is a subject-predicate-object triple with metadata
/// Beliefs are EARNED through evidence, not assumed
type Belief =
    { Id: BeliefId
      Subject: EntityId
      Predicate: RelationType
      Object: EntityId
      Provenance: Provenance
      Confidence: float // Current confidence (may change)
      ValidFrom: DateTime // When this belief became valid
      InvalidAt: DateTime option // When this belief was invalidated
      Version: int // Version number for this belief
      Tags: string list } // Optional categorization

    member this.IsValid = this.InvalidAt.IsNone

    member this.TripleString = $"({this.Subject} {this.Predicate} {this.Object})"

    override this.ToString() =
        let validity = if this.IsValid then "✓" else "✗"
        $"{validity} {this.Id}: {this.TripleString} [conf={this.Confidence:F2}]"

// =============================================================================
// BELIEF EVENTS - Event-sourced operations (never mutate, append events)
// =============================================================================

/// Events that modify the belief ledger
/// "Evolution is logged, not forgotten"
type BeliefEvent =
    /// Assert a new belief with provenance
    | Assert of Belief

    /// Retract a belief (mark as invalid)
    | Retract of beliefId: BeliefId * reason: string * retractedBy: AgentId

    /// Weaken confidence based on new evidence
    | Weaken of beliefId: BeliefId * newConfidence: float * reason: string

    /// Strengthen confidence based on supporting evidence
    | Strengthen of beliefId: BeliefId * newConfidence: float * reason: string

    /// Link two beliefs with a relation
    | Link of source: BeliefId * target: BeliefId * relation: RelationType

    /// Mark contradiction between beliefs
    | Contradict of belief1: BeliefId * belief2: BeliefId * explanation: string

    /// Schema/ontology evolution
    | SchemaEvolve of change: string * affectedBeliefs: BeliefId list

/// Wrapper for a belief event with metadata
type BeliefEventEntry =
    { EventId: Guid
      Event: BeliefEvent
      Timestamp: DateTime
      AgentId: AgentId
      RunId: RunId option
      Metadata: Map<string, string> }

    static member Create(event: BeliefEvent, agentId: AgentId, ?runId: RunId) =
        { EventId = Guid.NewGuid()
          Event = event
          Timestamp = DateTime.UtcNow
          AgentId = agentId
          RunId = runId
          Metadata = Map.empty }

// =============================================================================
// EVIDENCE - Candidates before they become beliefs
// =============================================================================

/// Status of an evidence candidate
type EvidenceStatus =
    | Pending // Awaiting verification
    | Verified // Promoted to belief
    | Rejected // Did not pass verification
    | Conflicting // Conflicts with existing beliefs

/// A proposed assertion extracted from evidence
type ProposedAssertion =
    { Id: Guid
      Subject: string
      Predicate: string
      Object: string
      SourceSection: string // The text section this came from
      Confidence: float // Extraction confidence
      ExtractorAgent: AgentId
      ExtractedAt: DateTime }

/// Evidence candidate - raw content before verification
/// "The Internet never writes beliefs directly. It only produces evidence candidates."
type EvidenceCandidate =
    { Id: Guid
      SourceUrl: Uri
      ContentHash: string
      FetchedAt: DateTime
      RawContent: string
      Segments: string list
      ProposedAssertions: ProposedAssertion list
      Status: EvidenceStatus
      VerifiedAt: DateTime option
      VerifiedBy: AgentId option
      RejectionReason: string option }

// =============================================================================
// PLAN - Hypotheses about future actions
// =============================================================================

/// Status of a plan
type PlanStatus =
    | Draft // Being developed
    | Active // Currently being executed
    | Paused // Temporarily suspended
    | Completed // Successfully finished
    | Failed // Did not complete
    | Superseded // Replaced by newer version

/// Status of a plan step
type StepStatus =
    | NotStarted
    | InProgress
    | Completed
    | Failed of reason: string
    | Skipped of reason: string

/// A step in a plan
type PlanStep =
    { Order: int
      Description: string
      EstimatedEffort: TimeSpan option
      Dependencies: int list // Orders of dependent steps
      Status: StepStatus
      CompletedAt: DateTime option
      Notes: string list }

/// A plan is a hypothesis about future actions
/// "Plans are hypotheses, not beliefs. Different symbolic class."
type Plan =
    { Id: PlanId
      Goal: string
      Assumptions: BeliefId list // Beliefs this plan depends on
      Steps: PlanStep list
      SuccessMetrics: string list
      RiskFactors: string list
      Version: int
      ParentVersion: PlanId option // If forked from another plan
      Status: PlanStatus
      CreatedAt: DateTime
      UpdatedAt: DateTime
      CreatedBy: AgentId
      Tags: string list }

    member this.IsActive =
        match this.Status with
        | PlanStatus.Active -> true
        | _ -> false

    member this.IsComplete =
        match this.Status with
        | PlanStatus.Completed -> true
        | _ -> false


/// Events that modify plans
type PlanEvent =
    | PlanCreated of Plan
    | StepStarted of planId: PlanId * stepOrder: int
    | StepCompleted of planId: PlanId * stepOrder: int * evidence: string
    | StepFailed of planId: PlanId * stepOrder: int * reason: string
    | AssumptionInvalidated of planId: PlanId * beliefId: BeliefId * reason: string
    | PlanForked of original: PlanId * newPlan: Plan
    | PlanCompleted of planId: PlanId
    | PlanFailed of planId: PlanId * reason: string
    | PlanSuperseded of planId: PlanId * by: PlanId

// =============================================================================
// HELPER MODULES
// =============================================================================

module Belief =
    /// Create a new belief
    let create (subject: string) (predicate: RelationType) (obj: string) (provenance: Provenance) =
        { Id = BeliefId.New()
          Subject = EntityId subject
          Predicate = predicate
          Object = EntityId obj
          Provenance = provenance
          Confidence = provenance.Confidence
          ValidFrom = DateTime.UtcNow
          InvalidAt = None
          Version = 1
          Tags = [] }

    /// Create from a simple triple string (for testing)
    let fromTriple (subject: string) (predicate: RelationType) (obj: string) =
        create subject predicate obj (Provenance.FromUser())


module Plan =
    /// Create a new plan
    let create goal assumptions steps createdBy =
        { Id = PlanId.New()
          Goal = goal
          Assumptions = assumptions
          Steps = steps
          SuccessMetrics = []
          RiskFactors = []
          Version = 1
          ParentVersion = None
          Status = Draft
          CreatedAt = DateTime.UtcNow
          UpdatedAt = DateTime.UtcNow
          CreatedBy = createdBy
          Tags = [] }
