namespace Tars.DSL.Wot

open System
open Tars.Core.WorkflowOfThought

// =============================================================================
// PHASE 17: EXTENDED GoT DSL - Full Graph-of-Thoughts Support
// =============================================================================
// Implements the full (G, T, E, R) framework from the GoT paper:
// - G: Graph with typed edges
// - T: Transformations (Generate, Aggregate, Refine, Contradict, Distill, Backtrack)
// - E: Evaluator (multi-metric scoring)
// - R: Ranker (top-K selection, pruning)
// =============================================================================

type DslId = string

// -----------------------------------------------------------------------------
// Node Types
// -----------------------------------------------------------------------------

type NodeKind =
    | Reason
    | Work

/// GoT-style transformations
type GoTTransformation =
    | Generate // Create new thoughts
    | Aggregate // Combine multiple thoughts (Merge)
    | Refine // Improve existing thought
    | Contradict // Find contradictions
    | Distill // Summarize/compress
    | Backtrack // Undo failed path
    | Score // Evaluate quality

/// Edge relationship types (semantic edges) for workflow graphs
type EdgeRelation =
    | EdgeDependsOn // Execution dependency
    | EdgeSupports // Provides evidence for
    | EdgeContradicts // Conflicts with
    | EdgeRefines // Improves upon
    | EdgeTriggers // Causes execution of
    | EdgeAggregates // Merges from multiple sources


/// Typed edge with relationship
type DslEdge =
    { From: DslId
      To: DslId
      Relation: EdgeRelation }

// -----------------------------------------------------------------------------
// Agent/Role Definitions (Multi-Agent Support)
// -----------------------------------------------------------------------------

/// Agent role configuration
type DslAgent =
    { Id: string
      Role: string
      ModelHint: string option // e.g., "reasoning", "fast", "cloud"
      Persona: string option // Optional persona name
      SystemPrompt: string option } // Optional custom system prompt

/// Agent assignment for a node
type NodeAgent =
    | ByRole of string // Route by role name
    | ById of string // Route by agent ID
    | Default // Use default routing

// -----------------------------------------------------------------------------
// Structured Output Schema
// -----------------------------------------------------------------------------

/// Output field type
type OutputFieldType =
    | StringType
    | NumberType
    | BoolType
    | ListType of OutputFieldType
    | ObjectType of OutputSchema

/// Output field definition
and OutputField =
    { Name: string
      Type: OutputFieldType
      Required: bool
      Description: string option }

/// Full output schema
and OutputSchema = { Fields: OutputField list }

/// Typed output for a node
type NodeOutput =
    | SimpleOutput of string // Single variable name
    | StructuredOutput of string * OutputSchema // Variable with schema

// -----------------------------------------------------------------------------
// Safety & Policy
// -----------------------------------------------------------------------------

/// Safety policy configuration
type DslSafetyPolicy =
    { PiiRedaction: bool
      ExternalClaimsRequireEvidence: bool
      LegalReviewRequiredFor: string list // e.g., ["public_statement", "regulator_submission"]
      RequireApprovalAboveRisk: float option } // 0.0-1.0

/// Budget constraints
type DslBudgets =
    { MaxToolCalls: int
      MaxLlmTokens: int
      MaxRuntimeMinutes: int
      MaxRetries: int }

/// Policy for resolving conflicts between agents
type DslConsensusPolicy =
    { Protocol: ConsensusProtocol
      Participants: string list // List of agent roles/IDs involved
      MinConfidence: float // Minimum confidence to even consider consensus
      AutoResolveConflicts: bool }

/// Full policies block
type DslPolicies =
    { Safety: DslSafetyPolicy
      Budgets: DslBudgets
      Consensus: DslConsensusPolicy option }

// -----------------------------------------------------------------------------
// Success Criteria
// -----------------------------------------------------------------------------

type SuccessCriterion =
    | ConfidenceAbove of float // confidence >= threshold
    | ContainsValue of string * string // variable contains value
    | MatchesRegex of string * string // variable matches pattern
    | ClaimHasEvidence of string // claim has evidence link
    | AllChecksPass // all WotChecks pass

// -----------------------------------------------------------------------------
// Meta Block
// -----------------------------------------------------------------------------

type DslMeta =
    { Id: string
      Title: string
      Domain: string
      Objective: string
      Constraints: string list
      SuccessCriteria: SuccessCriterion list }

// -----------------------------------------------------------------------------
// Extended Node Definition
// -----------------------------------------------------------------------------

type DslNode =
    { Id: DslId
      Kind: NodeKind
      Name: string
      Title: string option // Human-readable title
      Inputs: string list
      Outputs: NodeOutput list // Extended output support

      // Work node properties
      Tool: string option
      Args: Map<string, obj> option
      Checks: WotCheck list

      // Reason node properties
      Goal: string option
      Invariants: string list
      Constraints: string list
      Verdict: string option

      // Conditional execution: expression like "${confidence} > 0.7"
      // When set, the node only executes if the condition evaluates to true.
      Condition: string option

      // GoT-specific
      Agent: NodeAgent
      Transformation: GoTTransformation option
      StructuredOutput: Map<string, obj> option // For complex output definitions

      // Evidence tracking
      RequiresEvidence: bool
      EvidenceRefs: string list
      Metadata: Meta } // Extensible metadata

// -----------------------------------------------------------------------------
// Parallel Execution Groups
// -----------------------------------------------------------------------------

/// A group of node IDs that should execute concurrently.
/// During compilation, nodes in a parallel group get fan-out edges from
/// the predecessor and fan-in edges to the successor.
type ParallelGroup =
    { GroupId: string
      NodeIds: DslId list }

// -----------------------------------------------------------------------------
// Layer 4: Evolution / Reflection Metadata
// -----------------------------------------------------------------------------

/// Tracks the evolutionary lineage and promotion status of grammar constructs.
/// Enables rollback, mutation tracking, and self-improving evaluators.
type DslEvolutionMetadata = {
    PromotionLevel: string option       // "implementation" | "helper" | "builder" | "dsl_clause" | "grammar_rule"
    OccurrenceCount: int option         // How many times this pattern appeared
    DerivedFrom: string option          // What pattern/construct this was promoted from
    LineageId: string option            // Link to lineage record for full history
    MutationHistory: string list        // Previous versions of this construct
    RollbackExpansion: string option    // Code to expand back to lower-level form
    SemanticType: string option         // e.g., "ResilienceBlock", "ValidationPipeline"
    Effects: string list                // What this construct does: ["delay", "log", "re-invoke"]
    Confidence: float option            // Governor's confidence in this promotion (0.0-1.0)
    PromotedBy: string option           // "grammar_governor" | "manual"
}

module DslEvolutionMetadata =
    let empty = {
        PromotionLevel = None; OccurrenceCount = None
        DerivedFrom = None; LineageId = None
        MutationHistory = []; RollbackExpansion = None
        SemanticType = None; Effects = []
        Confidence = None; PromotedBy = None
    }

// -----------------------------------------------------------------------------
// Legacy DslPolicy (for backward compatibility)
// -----------------------------------------------------------------------------

type DslPolicy =
    { AllowedTools: Set<string>
      MaxToolCalls: int
      MaxTokens: int
      MaxTimeMs: int }

type DslInputs = Map<string, string>

// -----------------------------------------------------------------------------
// Extended Workflow Definition
// -----------------------------------------------------------------------------

/// Full advanced workflow (Phase 17)
type DslWorkflowAdvanced =
    { Meta: DslMeta
      Policies: DslPolicies
      Agents: DslAgent list
      Tools: string list // Registered tool names
      Inputs: DslInputs
      Nodes: DslNode list
      Edges: DslEdge list // Typed edges
      ParallelGroups: ParallelGroup list
      OutputDeliverables: string list
      Evolution: DslEvolutionMetadata // Layer 4: evolution/lineage tracking
      Metadata: Meta } // Workflow-level metadata

/// Legacy workflow format (backward compatible)
type DslWorkflow =
    { Name: string
      Version: string
      Description: string option
      Domain: string option
      Difficulty: string option
      Risk: string
      Inputs: DslInputs
      Policy: DslPolicy
      Nodes: DslNode list
      Edges: (DslId * DslId) list
      ParallelGroups: ParallelGroup list }

// -----------------------------------------------------------------------------
// Conversion Helpers
// -----------------------------------------------------------------------------

module DslConvert =

    /// Convert legacy edge to typed edge
    let toLegacyEdge (e: DslEdge) : DslId * DslId = (e.From, e.To)

    /// Convert simple edge to typed edge (default: DependsOn)
    let toTypedEdge (from: DslId, to': DslId) : DslEdge =
        { From = from
          To = to'
          Relation = EdgeDependsOn }

    /// Create a simple output
    let simpleOutput name : NodeOutput = SimpleOutput name

    /// Create a structured output with schema
    let structuredOutput name fields : NodeOutput =
        let schema = { Fields = fields }
        StructuredOutput(name, schema)

    /// Create default node with minimal settings
    let defaultNode id kind : DslNode =
        { Id = id
          Kind = kind
          Name = id
          Title = None
          Inputs = []
          Outputs = []
          Tool = None
          Args = None
          Checks = []
          Goal = None
          Invariants = []
          Constraints = []
          Verdict = None
          Condition = None
          Agent = Default
          Transformation = None
          StructuredOutput = None
          RequiresEvidence = false
          EvidenceRefs = []
          Metadata = Map.empty }

    /// Create a REASON node with goal
    let reasonNode id goal agent : DslNode =
        { defaultNode id Reason with
            Goal = Some goal
            Agent = agent
            Transformation = Some Generate }

    /// Create a WORK node with tool
    let workNode id tool args : DslNode =
        { defaultNode id Work with
            Tool = Some tool
            Args = Some args }

    /// Default safety policy
    let defaultSafetyPolicy: DslSafetyPolicy =
        { PiiRedaction = false
          ExternalClaimsRequireEvidence = false
          LegalReviewRequiredFor = []
          RequireApprovalAboveRisk = None }

    /// Default budgets
    let defaultBudgets: DslBudgets =
        { MaxToolCalls = 20
          MaxLlmTokens = 50000
          MaxRuntimeMinutes = 30
          MaxRetries = 3 }

    /// Default policies
    let defaultPolicies: DslPolicies =
        { Safety = defaultSafetyPolicy
          Budgets = defaultBudgets
          Consensus = None }
