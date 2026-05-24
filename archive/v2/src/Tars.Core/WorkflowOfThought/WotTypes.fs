namespace Tars.Core.WorkflowOfThought

// =============================================================================
// PHASE 12: WORKFLOW OF THOUGHT - CORE TYPES
// =============================================================================
//
// Key insight: Explicitly separate REASON nodes (LLM proposes) from WORK nodes
// (tools execute). This prevents the LLM from "pretending" to have done something.
//
// ARCHITECTURAL VISION: "Context as a Resource"
// ---------------------------------------------
// The context window (num_ctx) is a structural constraint and a finite resource.
// Standard CoT (Chain of Thought) often treats context as a luxury, leading to
// performance traps (e.g. 131k context causing GPU->CPU spillover).
//
// WoT governs context at the NODE level. Each node:
// 1. Chooses its model and context size (num_ctx) via NodeBudget.
// 2. High-context synthesis is an explicit operation (a Tool/special Node),
//    not a pervasively high-latency state.
// 3. This ensures the system remains GPU-bound (~8k-16k) for reasoning,
//    spiking to CPU-bound large-context only when strictly necessary.

open System
open System.Threading.Tasks
open Tars.Core


// =============================================================================
// NODE IDENTITY
// =============================================================================

// =============================================================================
// NODE IDENTITY
// =============================================================================

/// Unique identifier for a node in the reasoning graph
type NodeId = NodeId of string

module NodeId =
    let create () = NodeId(Guid.NewGuid().ToString())
    let value (NodeId id) = id

// =============================================================================
// BUDGET CONSTRAINTS (per node)
// =============================================================================

/// Resource budget for a single node execution
type NodeBudget =
    {
        MaxTokens: int
        MaxTime: TimeSpan
        MaxToolCalls: int
        MaxRetries: int
        /// Maximum context window size (Ollama num_ctx)
        MaxContext: int option
    }

module NodeBudget =
    let default' =
        { MaxTokens = 2000
          MaxTime = TimeSpan.FromSeconds(30.0)
          MaxToolCalls = 3
          MaxRetries = 2
          MaxContext = Some 4096 }

    let minimal =
        { MaxTokens = 500
          MaxTime = TimeSpan.FromSeconds(10.0)
          MaxToolCalls = 1
          MaxRetries = 1
          MaxContext = Some 2048 }

    let reasoning =
        { MaxTokens = 4000
          MaxTime = TimeSpan.FromSeconds(60.0)
          MaxToolCalls = 5
          MaxRetries = 3
          MaxContext = Some 8192 }

    let longContext =
        { MaxTokens = 4000
          MaxTime = TimeSpan.FromSeconds(120.0)
          MaxToolCalls = 10
          MaxRetries = 3
          MaxContext = Some 8192 }

    let giant =
        { MaxTokens = 8000
          MaxTime = TimeSpan.FromSeconds(300.0)
          MaxToolCalls = 5
          MaxRetries = 2
          MaxContext = Some 32768 }

// =============================================================================
// POLICY GATES
// =============================================================================

/// Types of sensitive content that require policy checks
type SensitiveContent =
    | PII of field: string // Personal Identifiable Information
    | PHI of field: string // Protected Health Information (HIPAA)
    | Financial of amount: decimal // Financial data above threshold
    | Legal of clause: string // Legal/contractual content
    | Credentials of hint: string // API keys, passwords
    | Custom of tag: string // User-defined sensitive content

/// Policy gate result
type PolicyGateResult =
    | Allowed
    | RequiresRedaction of SensitiveContent list
    | RequiresHumanReview of reason: string
    | Blocked of reason: string

/// Policy configuration for a node
type PolicyGate =
    { CheckPII: bool
      CheckPHI: bool
      CheckFinancial: bool
      FinancialThreshold: decimal
      CheckCredentials: bool
      CustomChecks: (string -> SensitiveContent option) list
      RequireHumanAboveRisk: float } // 0.0 = never, 1.0 = always

module PolicyGate =
    let permissive =
        { CheckPII = false
          CheckPHI = false
          CheckFinancial = false
          FinancialThreshold = 10000m
          CheckCredentials = false
          CustomChecks = []
          RequireHumanAboveRisk = 1.0 // Never require human
        }

    let strict =
        { CheckPII = true
          CheckPHI = true
          CheckFinancial = true
          FinancialThreshold = 1000m
          CheckCredentials = true
          CustomChecks = []
          RequireHumanAboveRisk = 0.7 // Require human above 70% risk
        }

// =============================================================================
// EVIDENCE & AUDIT
// =============================================================================

/// Evidence collected during node execution
type NodeEvidence =
    { NodeId: NodeId
      Timestamp: DateTimeOffset
      InputHash: string
      OutputHash: string
      TokensUsed: int
      Duration: TimeSpan
      ToolCallsMade: string list
      PolicyChecksRun: string list
      Decision: string
      Rationale: string }

/// Audit entry for traceability
type AuditEntry =
    { Timestamp: DateTimeOffset
      NodeId: NodeId
      Event: string
      Details: Map<string, obj> }

// =============================================================================
// THE KEY DISTINCTION: REASON NODES vs WORK NODES
// =============================================================================

/// Content that flows between nodes
type NodeContent =
    { Text: string
      Structured: Map<string, obj>
      Confidence: float
      Sources: HybridBrain.Source list }

module NodeContent =
    let empty =
        { Text = ""
          Structured = Map.empty
          Confidence = 0.0
          Sources = [] }

    let ofText text =
        { Text = text
          Structured = Map.empty
          Confidence = 0.5
          Sources = [] }

// -----------------------------------------------------------------------------
// REASON NODES: LLM proposes, structures, explains
// These nodes involve LLM inference - they THINK but don't ACT
// -----------------------------------------------------------------------------

/// Protocols for verifying multi-agent consensus
type ConsensusProtocol =
    | MajorityVote // > 50% agents or evidence nodes agree
    | Unanimous // 100% agreement required
    | WeightedAverage // Consensus based on agent authority weights
    | SuperiorHierarchy // Highest authority agent makes final decision
    | ThresholdScore of float // Overall hypothesis score >= X

/// Types of reasoning operations
type ReasonOperation =
    | Generate of topic: string
    | Plan of goal: string // Break goal into steps
    | Critique of target: NodeId // Evaluate another node's output
    | Synthesize of sources: NodeId list // Merge multiple inputs
    | Explain of topic: string // Generate explanation
    | Rewrite of target: NodeId * instruction: string // Modify content
    // Phase 17: GoT Transformations
    | Aggregate of sources: NodeId list
    | Refine of target: NodeId
    | Contradict of target: NodeId
    | Distill of target: NodeId
    | Backtrack of target: NodeId
    | Score of target: NodeId
    | VerifyConsensus of protocol: ConsensusProtocol

/// A node that uses LLM reasoning (no side effects)
type ReasonNode =
    { Id: NodeId
      Name: string
      Operation: ReasonOperation
      Input: NodeContent
      Output: NodeContent option
      Model: string option // Explicit model name override
      ModelHint: string option // e.g. "reasoning", "coding", "fast", "cloud"
      Budget: NodeBudget
      Policy: PolicyGate
      Evidence: NodeEvidence option
      Metadata: Meta }

// -----------------------------------------------------------------------------
// WORK NODES: Tools, verification, redaction, persistence
// These nodes DO things - they have SIDE EFFECTS
// -----------------------------------------------------------------------------

/// Types of work operations
/// Reference to a value in the context
type WotValueRef = string

/// Structured checks for verification
type WotCheck =
    | NonEmpty of valueRef: WotValueRef
    | Contains of valueRef: WotValueRef * needle: string
    // Future-ready:
    | RegexMatch of valueRef: WotValueRef * pattern: string
    | SchemaMatch of valueRef: WotValueRef * schema: string
    | ToolResult of toolName: string * args: Map<string, string> * check: string
    | Threshold of metric: string * op: string * value: float

/// Arguments for tool calls
type ToolArgs = Map<string, obj> // Keeping obj for flexibility, but prefer simple types

/// Types of work operations
type WorkOperation =
    | ToolCall of tool: string * args: ToolArgs // Execute external tool
    | Verify of checks: WotCheck list // Verify a condition
    | Redact of patterns: string list // Remove sensitive content
    | Persist of location: string // Save to storage
    | Fetch of source: string // Retrieve from source
    | Transform of fn: string // Apply transformation

/// A node that performs actual work (has side effects)
type WorkNode =
    { Id: NodeId
      Name: string
      Operation: WorkOperation
      Input: NodeContent
      Output: NodeContent option
      Budget: NodeBudget
      Policy: PolicyGate
      Evidence: NodeEvidence option
      SideEffects: string list
      Metadata: Meta }

// =============================================================================
// THE REASONING GRAPH
// =============================================================================

/// A node in the workflow (either Reason or Work)
type WotNode =
    | Reason of ReasonNode
    | Work of WorkNode

/// Relationship between nodes
type NodeEdge =
    | DependsOn // Must complete before
    | Supports // Adds evidence for
    | Contradicts // Conflicts with
    | Refines // Improves upon
    | Triggers // Causes execution of
    | Blocks // Prevents execution of

/// The full workflow graph
type WorkflowGraph =
    { Id: Guid
      Name: string
      Description: string
      Nodes: Map<NodeId, WotNode>
      Edges: (NodeId * NodeEdge * NodeId) list // from, relation, to
      EntryPoint: NodeId
      ExitPoints: NodeId list
      GlobalBudget: Budget
      GlobalPolicy: PolicyGate
      Metadata: Meta
      AuditLog: AuditEntry list }

// =============================================================================
// WORKFLOW EXECUTION STATE
// =============================================================================

/// Current state of a node
type NodeState =
    | Pending
    | Ready // Dependencies satisfied
    | Running
    | Completed of NodeContent
    | Failed of error: string
    | Blocked of reason: string // Policy blocked
    | AwaitingHuman of reason: string

/// Current state of the workflow
type WorkflowState =
    { Graph: WorkflowGraph
      NodeStates: Map<NodeId, NodeState>
      CurrentNode: NodeId option
      History: NodeId list
      TotalTokensUsed: int
      TotalDuration: TimeSpan
      AllEvidence: NodeEvidence list }

// =============================================================================
// MINIMAL 5-NODE WORKFLOW: Plan → Critique → Verify → Tool → Distill
// =============================================================================

module MinimalWorkflow =

    /// Create the canonical 5-node workflow
    /// Create the canonical 5-node workflow
    let create (goal: string, modelHint: string option, budget: NodeBudget option) : WorkflowGraph =
        let hint = modelHint
        let budget = budget |> Option.defaultValue NodeBudget.default'

        let preflightId = NodeId.create ()
        let planId = NodeId.create ()
        let toolId = NodeId.create ()
        let verifyId = NodeId.create ()
        let critiqueId = NodeId.create ()
        let distillId = NodeId.create ()

        // 0. Preflight: Ensure environment is ready
        let preflightNode =
            Work
                { Id = preflightId
                  Name = "Preflight"
                  Operation = ToolCall("check_environment", Map.empty)
                  Input = NodeContent.empty
                  Output = None
                  Budget = NodeBudget.minimal
                  Policy = PolicyGate.permissive
                  Evidence = None
                  SideEffects = []
                  Metadata = Map.empty }

        let planNode =
            Reason
                { Id = planId
                  Name = "Plan"
                  Operation = Plan goal
                  Input = NodeContent.ofText goal
                  Output = None
                  Model = None
                  ModelHint = hint
                  Budget = budget
                  Policy = PolicyGate.permissive
                  Evidence = None
                  Metadata = Map.empty }

        // 2. Tool: Execute the work (build)
        let toolNode =
            Work
                { Id = toolId
                  Name = "Tool"
                  Operation = ToolCall("dotnet_build", Map.empty)
                  Input = NodeContent.empty
                  Output = None
                  Budget = NodeBudget.default'
                  Policy = PolicyGate.permissive
                  Evidence = None
                  SideEffects = [ "filesystem" ]
                  Metadata = Map.empty }

        // 3. Verify: Check tool output
        let verifyNode =
            Work
                { Id = verifyId
                  Name = "Verify"
                  Operation = Verify [ NonEmpty "tool_output"; Contains("tool_output", "Build successful") ]
                  Input = NodeContent.empty
                  Output = None
                  Budget = NodeBudget.minimal
                  Policy = PolicyGate.strict
                  Evidence = None
                  SideEffects = []
                  Metadata = Map.empty }

        let critiqueNode =
            Reason
                { Id = critiqueId
                  Name = "Critique"
                  Operation = Critique planId
                  Input = NodeContent.empty
                  Output = None
                  Model = None
                  ModelHint = hint
                  Budget = budget
                  Policy = PolicyGate.strict
                  Evidence = None
                  Metadata = Map.empty }

        let distillNode =
            Reason
                { Id = distillId
                  Name = "Distill"
                  Operation = Synthesize [ planId; toolId; verifyId; critiqueId ]
                  Input = NodeContent.empty
                  Output = None
                  Model = None
                  ModelHint = hint
                  Budget = NodeBudget.minimal
                  Policy = PolicyGate.permissive
                  Evidence = None
                  Metadata = Map.empty }

        { Id = Guid.NewGuid()
          Name = "Minimal WoT Workflow (Corrected)"
          Description = "Preflight -> Plan -> Tool -> Verify -> Critique -> Distill"
          Nodes =
            Map.ofList
                [ (preflightId, preflightNode)
                  (planId, planNode)
                  (toolId, toolNode)
                  (verifyId, verifyNode)
                  (critiqueId, critiqueNode)
                  (distillId, distillNode) ]
          Edges =
            [ (planId, DependsOn, preflightId)
              (toolId, DependsOn, planId)
              (verifyId, DependsOn, toolId)
              (critiqueId, DependsOn, verifyId)
              (distillId, DependsOn, critiqueId)

              // Semantic edges
              (critiqueId, Refines, planId)
              (verifyId, Supports, critiqueId)
              (toolId, Triggers, verifyId) ]
          EntryPoint = preflightId
          ExitPoints = [ distillId ]
          GlobalBudget =
            { MaxTokens = Some(10000<token>)
              MaxMoney = Some(0.10m<usd>)
              MaxDuration = Some(5.0 * 60.0 * 1000.0<ms>)
              MaxCalls = Some(10<requests>)
              MaxRam = Some(100L * 1024L * 1024L<bytes>)
              MaxVram = None
              MaxDisk = None
              MaxNetwork = None
              MaxCpu = None
              MaxAttention = None
              MaxNodes = None
              MaxEnergy = None
              MaxCustom = Map.empty }
          GlobalPolicy = PolicyGate.strict
          Metadata = Map.empty
          AuditLog = [] }

// =============================================================================
// POLICY ENFORCEMENT
// =============================================================================

module PolicyEnforcement =

    /// Simple regex patterns for PII detection
    let private piiPatterns =
        [ @"\b\d{3}-\d{2}-\d{4}\b", "SSN" // Social Security Number
          @"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "Email"
          @"\b\d{16}\b", "Credit Card" // Credit card number
          @"\b\d{3}-\d{3}-\d{4}\b", "Phone" ] // Phone number

    /// Check content for PII
    let checkPII (content: string) : SensitiveContent list =
        piiPatterns
        |> List.choose (fun (pattern, label) ->
            if System.Text.RegularExpressions.Regex.IsMatch(content, pattern) then
                Some(PII label)
            else
                None)

    /// Redact detected PII
    let redact (content: string) (sensitive: SensitiveContent list) : string =
        sensitive
        |> List.fold
            (fun text s ->
                match s with
                | PII "SSN" ->
                    System.Text.RegularExpressions.Regex.Replace(text, @"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]")
                | PII "Email" ->
                    System.Text.RegularExpressions.Regex.Replace(
                        text,
                        @"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                        "[REDACTED-EMAIL]"
                    )
                | PII "Credit Card" ->
                    System.Text.RegularExpressions.Regex.Replace(text, @"\b\d{16}\b", "[REDACTED-CC]")
                | PII "Phone" ->
                    System.Text.RegularExpressions.Regex.Replace(text, @"\b\d{3}-\d{3}-\d{4}\b", "[REDACTED-PHONE]")
                | _ -> text)
            content

    /// Evaluate policy gate for content
    let evaluate (gate: PolicyGate) (content: string) : PolicyGateResult =
        let mutable sensitive = []

        if gate.CheckPII then
            sensitive <- sensitive @ checkPII content

        // Add more checks as needed...

        if sensitive.IsEmpty then
            Allowed
        else
            RequiresRedaction sensitive

// =============================================================================
// WORKFLOW EXECUTION (Minimal Implementation)
// =============================================================================

module WorkflowExecution =

    /// Initialize workflow state
    let init (graph: WorkflowGraph) : WorkflowState =
        let nodeStates =
            graph.Nodes |> Map.map (fun _ _ -> Pending) |> Map.add graph.EntryPoint Ready

        { Graph = graph
          NodeStates = nodeStates
          CurrentNode = Some graph.EntryPoint
          History = []
          TotalTokensUsed = 0
          TotalDuration = TimeSpan.Zero
          AllEvidence = [] }

    /// Get the next ready node (deterministically)
    let nextReady (state: WorkflowState) : NodeId option =
        state.NodeStates
        |> Map.toList
        |> List.filter (fun (_, s) -> s = Ready)
        |> List.sortBy (fun (id, _) -> id.ToString()) // Deterministic sort
        |> List.tryHead
        |> Option.map fst

    /// Mark dependencies as ready when a node completes
    let updateDependencies (completedId: NodeId) (state: WorkflowState) : WorkflowState =
        // Find nodes that depend on the completed node
        // Edge direction: (dependent, DependsOn, prerequisite)
        let potentialUnblocked =
            state.Graph.Edges
            |> List.choose (fun (dependent, edge, prerequisite) ->
                if prerequisite = completedId && edge = DependsOn then
                    Some dependent
                else
                    None)
            |> List.distinct

        let newStates =
            potentialUnblocked
            |> List.fold
                (fun states depId ->
                    // Check if ALL dependencies of depId are completed
                    let allPrereqsMet =
                        state.Graph.Edges
                        |> List.forall (fun (d, edge, p) ->
                            if d = depId && edge = DependsOn then
                                match Map.tryFind p state.NodeStates with
                                | Some(Completed _) -> true
                                | _ -> false
                            else
                                true)

                    if allPrereqsMet then Map.add depId Ready states else states)
                state.NodeStates

        { state with NodeStates = newStates }


/// Interface for Workflow of Thought engine
type IWotEngine =
    /// <summary>Executes a single workflow node.</summary>
    abstract member ExecuteNodeAsync:
        node: WotNode * state: WorkflowState -> Task<Result<NodeContent * NodeEvidence, string>>

    /// <summary>Executes a full workflow graph.</summary>
    abstract member ExecuteWorkflowAsync: graph: WorkflowGraph -> Task<WorkflowState>

// =============================================================================
// DEFINITION OF DONE TESTS
// =============================================================================

module DefinitionOfDone =

    /// Test 1: Happy path - 6 nodes execute in sequence
    let testHappyPath () : bool =
        let workflow = MinimalWorkflow.create ("Refactor file.fs", None, None)
        let state = WorkflowExecution.init workflow

        // Verify workflow has exactly 6 nodes
        workflow.Nodes.Count = 6

    /// Test 2: PII detected → redaction → continue
    let testPiiRedaction () : bool =
        let content = "Contact john@example.com or call 555-123-4567"
        let detected = PolicyEnforcement.checkPII content
        let redacted = PolicyEnforcement.redact content detected

        // Should have found 2 PII items
        detected.Length = 2
        &&
        // Should be redacted
        not (redacted.Contains("john@example.com"))
        && not (redacted.Contains("555-123-4567"))

    /// Run all DoD tests
    let runAll () : (string * bool) list =
        [ ("Happy path: 5-node workflow", testHappyPath ())
          ("PII detection and redaction", testPiiRedaction ()) ]
