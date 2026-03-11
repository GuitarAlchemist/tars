namespace Tars.Cortex

open System
open Tars.Core

/// <summary>
/// Unified WoT node types that all reasoning patterns compile to.
/// This is the universal execution vocabulary for TARS cognitive operations.
/// </summary>
module WoTTypes =

    // =========================================================================
    // Core WoT Node Types
    // =========================================================================

    /// <summary>
    /// Memory operations for knowledge graph interactions.
    /// </summary>
    type MemoryOp =
        /// Query the knowledge graph
        | Query of sparql: string
        /// Assert a new triple
        | Assert of subject: string * predicate: string * object_: string
        /// Retract an existing triple
        | Retract of subject: string * predicate: string * object_: string
        /// Search vector store for similar items
        | Search of embedding: float array * topK: int

    /// <summary>
    /// Operation for structured verification.
    /// </summary>
    type VerificationOp =
        | Contains of substring: string
        | Regex of pattern: string
        | JsonPath of path: string
        | Schema of schemaRef: string
        | ToolCheck of toolName: string * args: Map<string, obj>
        | CustomOp of name: string

    /// <summary>
    /// A symbolic invariant for validation nodes.
    /// </summary>
    type WoTInvariant =
        { Name: string
          Op: VerificationOp
          Weight: float }

    /// <summary>
    /// Hints for model selection in Think nodes.
    /// </summary>
    type ModelHint =
        | Fast // Speed-optimized
        | Smart // Quality-optimized
        | Reasoning // Extended thinking
        | Specific of model: string

    /// <summary>
    /// The kind of cognitive operation a node performs.
    /// </summary>
    type WoTNodeKind =
        | Reason // Think / LLM
        | Tool // Act / Side-effect
        | Validate // Symbolic check
        | Memory // Store / Retrieve
        | Control // Loop / Branch

    /// <summary>
    /// Metadata for a WoT node.
    /// </summary>
    type NodeMetadata =
        { Label: string option
          Tags: string list }

    /// <summary>
    /// Payload for Reason nodes.
    /// </summary>
    type ReasonPayload =
        { Prompt: string
          Hint: ModelHint option }

    /// <summary>
    /// Payload for Tool nodes.
    /// </summary>
    type ToolPayload =
        { Tool: string; Args: Map<string, obj> }

    /// <summary>
    /// Payload for Validate nodes.
    /// </summary>
    type ValidatePayload = { Invariants: WoTInvariant list }

    /// <summary>
    /// Payload for Memory nodes.
    /// </summary>
    type MemoryPayload = { Operation: MemoryOp }

    /// <summary>
    /// Payload for Control nodes.
    /// </summary>
    type ControlPayload =
        | Branch of condition: string * ifTrue: string * ifFalse: string option
        | Loop of bodyEntry: string * until: string * maxIterations: int
        | Parallel of children: string list
        | Decide of candidates: string list * criteria: string list
        | Observe of input: string * transform: (string -> string) option

    /// <summary>
    /// Universal WoT node that all patterns compile to.
    /// </summary>
    type WoTNode =
        { Id: string
          Kind: WoTNodeKind
          Payload: obj // See specific *Payload types above
          Metadata: NodeMetadata }

    /// <summary>
    /// Edge connecting two WoT nodes.
    /// </summary>
    type WoTEdge =
        { From: string
          To: string
          Label: string option
          Confidence: float option }

    // =========================================================================
    // Pattern Metadata
    // =========================================================================

    /// <summary>
    /// The kind of agentic pattern.
    /// </summary>
    type PatternKind =
        | ChainOfThought
        | ReAct
        | PlanAndExecute
        | GraphOfThoughts
        | TreeOfThoughts
        | WorkflowOfThought
        | Custom of name: string

    /// <summary>
    /// Metadata about the source pattern.
    /// </summary>
    type PatternMetadata =
        { Kind: PatternKind
          SourceGoal: string
          CompiledAt: DateTime
          EstimatedTokens: int option
          EstimatedSteps: int option }

    // =========================================================================
    // WoT Plan (Compiled Pattern)
    // =========================================================================

    /// <summary>
    /// A compiled WoT plan ready for execution.
    /// All reasoning patterns compile to this representation.
    /// </summary>
    type WoTPlan =
        { Id: Guid
          Nodes: WoTNode list
          Edges: WoTEdge list
          EntryNode: string
          Metadata: PatternMetadata
          Policy: string list }

    // =========================================================================
    // Execution Types
    // =========================================================================

    /// <summary>
    /// Status of a WoT node execution.
    /// </summary>
    type NodeStatus =
        | Pending
        | Running
        | Completed of output: string * durationMs: int64
        | Failed of error: string * durationMs: int64
        | Skipped of reason: string

    /// <summary>
    /// A single step in the execution trace.
    /// </summary>
    type WoTTraceStep =
        { NodeId: string
          NodeType: string
          StartedAt: DateTime
          Status: NodeStatus
          Input: string option
          Output: string option
          Confidence: float option
          TokensUsed: int option }

    /// <summary>
    /// Complete execution trace for a WoT run.
    /// </summary>
    type WoTTrace =
        { RunId: Guid
          Plan: WoTPlan
          Steps: WoTTraceStep list
          StartedAt: DateTime
          CompletedAt: DateTime option
          FinalStatus: string }

    /// <summary>
    /// A knowledge triple produced during execution.
    /// </summary>
    type WoTTriple =
        { Subject: string
          Predicate: string
          Object: string
          Confidence: float
          ProducedBy: string // Node ID
          ProducedAt: DateTime }

    /// <summary>
    /// Execution metrics for a WoT run.
    /// </summary>
    type ExecutionMetrics =
        { TotalSteps: int
          SuccessfulSteps: int
          FailedSteps: int
          TotalTokens: int
          TotalDurationMs: int64
          BranchingFactor: float
          ConstraintScore: float option }

    // =========================================================================
    // Enhanced Cognitive State
    // =========================================================================

    /// <summary>
    /// Cognitive mode for WoT state (mirrors CognitiveAnalyzer.CognitiveMode)
    /// </summary>
    type WoTCognitiveMode =
        | Exploratory
        | Convergent
        | Critical

    /// <summary>
    /// Enhanced cognitive state that integrates with WoT execution.
    /// </summary>
    type WoTCognitiveState =
        {
            /// Current cognitive mode
            Mode: WoTCognitiveMode
            /// System stability (0.0 - 1.0)
            Eigenvalue: float
            /// Information diversity (0.0 - 1.0)
            Entropy: float
            /// Reasoning graph complexity
            BranchingFactor: float
            /// Currently executing pattern
            ActivePattern: PatternKind option
            /// Current WoT run ID
            WoTRunId: Guid option
            /// Steps in current execution
            StepCount: int
            /// Remaining token budget
            TokenBudget: int option
            /// When mode last changed
            LastTransition: DateTime
            /// Phase 13 constraint score
            ConstraintScore: float option
            /// Recent execution success rate
            SuccessRate: float
        }

    /// <summary>
    /// Complete result of a WoT execution.
    /// </summary>
    type WoTResult =
        { Output: string
          Success: bool
          Trace: WoTTrace
          TriplesDelta: WoTTriple list
          ToolsUsed: string list
          Metrics: ExecutionMetrics
          Warnings: string list
          Errors: string list
          CognitiveStateAfter: WoTCognitiveState option }



    // =========================================================================
    // Pattern Compiler Interface
    // =========================================================================

    /// <summary>
    /// Interface for compiling reasoning patterns to WoT plans.
    /// </summary>
    type IPatternCompiler =
        /// Compile a CoT workflow to WoT
        abstract CompileChainOfThought: steps: int * goal: string -> WoTPlan
        /// Compile a ReAct workflow to WoT
        abstract CompileReAct: tools: string list * maxSteps: int * goal: string -> WoTPlan
        /// Compile a GoT workflow to WoT
        abstract CompileGraphOfThoughts: branchingFactor: int * maxDepth: int * goal: string -> WoTPlan
        /// Compile a ToT workflow to WoT
        abstract CompileTreeOfThoughts: beamWidth: int * searchDepth: int * goal: string -> WoTPlan
        /// Compile a general ReasoningPattern to WoT
        abstract CompilePattern: pattern: ReasoningPattern.ReasoningPattern * goal: string -> WoTPlan

    /// <summary>
    /// Interface for executing WoT plans.
    /// </summary>
    type IWoTExecutor =
        /// Execute a WoT plan and return the result
        abstract Execute: plan: WoTPlan * context: AgentContext -> Async<WoTResult>

        /// Execute with streaming progress updates
        abstract ExecuteWithProgress:
            plan: WoTPlan * context: AgentContext * onProgress: (WoTTraceStep -> unit) -> Async<WoTResult>

    /// <summary>
    /// Interface for selecting the optimal pattern for a goal.
    /// </summary>
    type IPatternSelector =
        /// Analyze a goal and recommend a pattern
        abstract Recommend: goal: string * state: WoTCognitiveState -> PatternKind
        /// Get pattern suitability scores for a goal
        abstract Score: goal: string -> Map<PatternKind, float>
