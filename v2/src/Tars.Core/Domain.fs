namespace Tars.Core

open System

/// Unique identifier for an agent
type AgentId = AgentId of Guid

/// Unique identifier for a correlation/conversation
type CorrelationId = CorrelationId of Guid

/// Represents the source or target of a message
type MessageEndpoint =
    | System
    | User
    | Agent of AgentId
    | Alias of string

/// A message flowing through the system
/// The "Verb" of the message (Speech Act)
type Performative =
    | Request // "Do this"
    | Inform // "Here is information"
    | Query // "What is X?"
    | Propose // "I can do this for cost Y"
    | Refuse // "I cannot do this (violates constraints)"
    | Failure // "Execution failed"
    | NotUnderstood // "I don't know what you mean"
    | Event // "Something happened"

/// <summary>
/// Represents the detected intent of a user prompt or agent message.
/// </summary>
/// <summary>
/// Represents the semantic domain or topic of a message.
/// </summary>
type AgentDomain =
    | Coding
    | Planning
    | Reasoning
    | Chat

/// The "Guardrails" for the request
type SemanticConstraints =
    { MaxTokens: int<token> option
      MaxComplexity: string option // e.g., "O(n)"
      Timeout: TimeSpan option
      KnowledgeBoundary: string list }

    static member Default =
        { MaxTokens = None
          MaxComplexity = None
          Timeout = None
          KnowledgeBoundary = [] }

/// The Semantic Envelope
type SemanticMessage<'T> =
    { Id: Guid
      CorrelationId: CorrelationId
      Sender: MessageEndpoint
      Receiver: MessageEndpoint option // None = Broadcast
      Performative: Performative
      Intent: AgentDomain option
      Constraints: SemanticConstraints
      Ontology: string option // Domain context (e.g., "coding", "finance")
      Language: string // Content type (e.g., "json", "fsharp", "natural")
      Content: 'T
      Timestamp: DateTime
      Metadata: Map<string, string> }

/// Alias for text-based messages (common case)
type Message = SemanticMessage<string>

/// Represents a non-fatal issue encountered during execution
type PartialFailure =
    | Warning of message: string
    | Error of message: string
    | Degradation of feature: string * reason: string
    | Timeout of operation: string * duration: TimeSpan
    | SubAgentTimeout of agentId: AgentId * taskId: Guid
    | ToolError of tool: string * error: string
    | LowConfidence of score: float * details: string
    | ProtocolViolation of message: string
    | ConstraintViolation of violation: string

/// Represents the outcome of an agentic operation, supporting partial success
type ExecutionOutcome<'T> =
    | Success of value: 'T
    | PartialSuccess of value: 'T * warnings: PartialFailure list
    | Failure of errors: PartialFailure list

type CapabilityKind =
    | Summarization
    | WebSearch
    | CodeGeneration
    | DataAnalysis
    | Planning
    | TaskExecution
    | Reasoning
    | Custom of string

type Capability =
    {
        Kind: CapabilityKind
        Description: string
        InputSchema: string option
        OutputSchema: string option
        /// Model-reported confidence for this capability (0.0-1.0)
        Confidence: float option
        /// Rolling reputation score from observed outcomes (0.0-1.0)
        Reputation: float option
    }

/// Represents a tool that an agent can execute
type Tool =
    {
        Name: string
        Description: string
        Version: string
        ParentVersion: string option
        CreatedAt: DateTime
        /// Function that takes an input string and returns an async result string
        Execute: string -> Async<Result<string, string>>
        /// Web of Things (WoT) compliant Thing Description metadata.
        ThingDescription: Map<string, obj> option
    }

    /// Helper to create a tool with a task-based execute function
    static member Create
        (name: string, description: string, execute: string -> System.Threading.Tasks.Task<Result<string, string>>)
        =
        { Name = name
          Description = description
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute = fun input -> execute input |> Async.AwaitTask
          ThingDescription = None }

    /// Internal helper to create a minimal tool record for migration/tests
    static member InternalCreateMinimal(name, description, execute) =
        { Name = name
          Description = description
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute = execute
          ThingDescription = None }

/// Represents a single unit of reasoning in a Graph of Thoughts (GoT).
type ThoughtNode =
    { Id: Guid
      Content: string
      NodeType: string // "Hypothesis", "Observation", "Action", "Synthesis"
      Confidence: float
      Metadata: Map<string, obj>
      Timestamp: DateTime }

/// Represents the relationship between thoughts.
type ThoughtEdge =
    { SourceId: Guid
      TargetId: Guid
      Relation: string // "Supports", "Contradicts", "Refines", "Transforms"
      Weight: float }

/// A non-linear reasoning structure (Graph of Thoughts).
type ThoughtGraph =
    { Nodes: Map<Guid, ThoughtNode>
      Edges: ThoughtEdge list
      ContextId: Guid }

/// The current state of an agent in its lifecycle
type AgentState =
    | Idle
    | Thinking of history: Message list
    | Acting of tool: Tool * input: string
    | Observing of tool: Tool * output: string
    | WaitingForUser of prompt: string
    | Error of error: string

/// Fundamental drivers that influence agent behavior decisions
type BaseDrives =
    { Accuracy: float // Preference for correctness/verification (0.0-1.0)
      Speed: float // Preference for low latency/quick answers (0.0-1.0)
      Creativity: float // Preference for novel/divergent thinking (0.0-1.0)
      Safety: float } // Preference for risk aversion/constraints (0.0-1.0)

// ===================================
// START OF CONSTITUTION TYPES
// ===================================

/// What specific role/function the agent's neural net performs
type NeuralRole =
    | Generate of domain: AgentDomain
    | Explore of searchSpace: string
    | Summarize of contentType: string
    | Mutate of target: string
    | Review of aspect: string
    | Coordinate of agents: AgentId list
    | GeneralReasoning

/// A symbolic invariant that must hold true in a constitution
type ConstitutionInvariant =
    | ParseCompleteness
    | BackwardCompatibility
    | GrammarValidity
    | TestPassing
    | CoverageMaintained
    | CustomInvariant of name: string * predicate: string

/// A hard limit on resources
type ResourceLimit =
    | MaxIterations of int
    | MaxTokens of int
    | MaxTimeMinutes of int
    | MaxMemoryMB of int64
    | MaxCpuPercent of int
    | MaxDiskWritesMB of int
    | MaxCost of decimal

/// A permission granted to the agent
type Permission =
    | ReadKnowledgeGraph
    | ModifyKnowledgeGraph
    | ReadCode of pattern: string
    | ModifyCode of pattern: string
    | SpawnAgent of agentType: string
    | CallTool of toolName: string
    | AccessSecret of secretName: string
    | ExecuteShellCommand of pattern: string
    | All // Administrator/God mode

/// A prohibition forcing the agent to avoid certain actions
type Prohibition =
    | CannotModifyCore
    | CannotDeleteData
    | CannotAccessNetwork
    | CannotSpawnUnlimited
    | CannotExceedBudget
    | CannotViolateInvariant of ConstitutionInvariant
    | CannotUseTool of toolName: string
    | CannotAccessPath of path: string

/// A goal the agent must achieve
type AchievementGoal =
    | ReduceComplexity of percent: int
    | MaintainCoverage
    | CompleteWithin of time: TimeSpan
    | CustomGoal of description: string

/// Time constraint for execution
type TimeConstraint =
    | MustCompleteWithin of TimeSpan
    | MustStartAfter of DateTimeOffset

/// The symbolic contract defining the agent's constraints and obligations
type SymbolicContract =
    { MustPreserve: ConstitutionInvariant list
      MustAchieve: AchievementGoal list
      ResourceBounds: ResourceLimit list
      Dependencies: AgentId list
      ConflictsWith: AgentId list
      TimeConstraints: TimeConstraint list }

    static member Empty =
        { MustPreserve = []
          MustAchieve = []
          ResourceBounds = []
          Dependencies = []
          ConflictsWith = []
          TimeConstraints = [] }

/// The full constitution combining role and contract
type AgentConstitution =
    { AgentId: AgentId
      NeuralRole: NeuralRole
      SymbolicContract: SymbolicContract
      // Explicit lists for easier access/overrides
      Invariants: ConstitutionInvariant list
      Permissions: Permission list
      Prohibitions: Prohibition list
      // Additional direct resource bounds if needed outside contract
      HardResourceBounds: ResourceLimit list }

    /// Create a new constitution with default empty values
    static member Create(id: AgentId, role: NeuralRole) =
        { AgentId = id
          NeuralRole = role
          SymbolicContract = SymbolicContract.Empty
          Invariants = []
          Permissions = []
          Prohibitions = []
          HardResourceBounds = [] }

// ===================================
// RUNTIME TYPES FOR ENFORCEMENT
// ===================================

/// An action the agent attempts to perform that requires checking
type AgentAction =
    | ExecuteTool of toolName: string * args: string
    | ReadFile of path: string
    | WriteFile of path: string
    | SpawnChild of role: NeuralRole
    | NetworkRequest of url: string
    | ModifyLedger of operation: string
    | GenericAction of name: string * details: string

/// Represents a violation of the agent's constitution
type Violation =
    | ProhibitionViolated of rule: Prohibition * details: string
    | InvariantBroken of invariant: ConstitutionInvariant * details: string
    | ResourceQuotaExceeded of limit: ResourceLimit * current: obj * max: obj
    | PermissionDenied of action: AgentAction * reason: string
    | DependencyMissing of missingAgent: AgentId
    | TimeConstraintViolated of timeLimit: TimeConstraint * elapsed: TimeSpan

/// An autonomous agent definition
type Agent =
    { Id: AgentId
      Name: string
      Version: string
      ParentVersion: string option
      CreatedAt: DateTime
      Model: string
      SystemPrompt: string
      Tools: Tool list
      Capabilities: Capability list
      State: AgentState
      Memory: Message list
      Fitness: float // Fitness Score (0.0 - 1.0)
      Drives: BaseDrives
      Constitution: AgentConstitution }

    member this.ReceiveMessage(msg: Message) =
        // Truncate message content if too large to prevent HTTP 400 errors
        // Increased to 64KB to support long reasoning outputs
        let maxMessageLength = 65536

        let truncatedMsg =
            if msg.Content.Length > maxMessageLength then
                { msg with
                    Content = msg.Content.Substring(0, maxMessageLength) + "\n... [truncated]" }
            else
                msg

        // Truncate old messages in memory if they're too large
        let truncateMessage (m: Message) =
            if m.Content.Length > maxMessageLength then
                { m with
                    Content = m.Content.Substring(0, maxMessageLength) + "\n... [truncated]" }
            else
                m

        let truncatedMemory = this.Memory |> List.map truncateMessage

        // Always transition to Idle when receiving a new message
        // This ensures pending thoughts (Thinking state) are discarded in favor of new information
        // and enables the agent to process the new message immediately.
        { this with
            Memory = truncatedMemory @ [ truncatedMsg ]
            State = Idle }

/// The result of a kernel operation
type KernelResult<'T> = Result<'T, string>

/// Strategies for routing traffic to different agent versions
type RoutingStrategy =
    /// Always route to a specific agent instance
    | Pinned of AgentId
    /// Route a percentage of traffic to a canary instance
    /// weight: 0.0 to 1.0 (e.g., 0.1 means 10% to canary)
    | Canary of primary: AgentId * canary: AgentId * weight: float
    /// Split traffic evenly between multiple instances
    | RoundRobin of AgentId list

/// Represents a node in the internal knowledge graph
type GraphNode =
    | Concept of name: string
    | AgentNode of AgentId
    | FileNode of path: string
    | TaskNode of taskId: Guid
    | BeliefNode of beliefId: Guid * content: string
    | ModuleNode of name: string
    | TypeNode of name: string
    | FunctionNode of name: string

/// Represents an edge/relationship in the internal knowledge graph
type GraphEdge =
    | RelatesTo of weight: float
    | CreatedBy
    | DependsOn
    | Solves
    | HasBelief
    | IsA
    | Contains



/// Represents the epistemic status of a belief
type EpistemicStatus =
    /// Proposed solution, untested generalization
    | Hypothesis
    /// Passed generalization tests on variants
    | VerifiedFact
    /// Abstracted and reused successfully multiple times
    | UniversalPrinciple
    /// Useful but known to be brittle
    | Heuristic
    /// Proven false through testing
    | Fallacy

/// A belief held by the agent, with tracking of its epistemic status
type EpistemicBelief =
    {
        /// Unique identifier
        Id: Guid
        /// The belief statement
        Statement: string
        /// When/where this belief applies
        Context: string
        /// Current epistemic status
        Status: EpistemicStatus
        /// Confidence score (0.0-1.0)
        Confidence: float
        /// Task IDs this was derived from
        DerivedFrom: Guid list
        /// When the belief was created
        CreatedAt: DateTime
        /// Last verification timestamp
        LastVerified: DateTime
    }

/// Result of a generalization verification
type VerificationResult =
    {
        /// Whether the generalization holds
        IsVerified: bool
        /// Verification score (0.0-1.0)
        Score: float
        /// Feedback message
        Feedback: string
        /// Variants that failed verification
        FailedVariants: string list
    }
