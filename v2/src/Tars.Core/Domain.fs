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
type AgentIntent =
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
      Intent: AgentIntent option
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
    { Kind: CapabilityKind
      Description: string
      InputSchema: string option
      OutputSchema: string option }

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
    }

/// The current state of an agent in its lifecycle
type AgentState =
    | Idle
    | Thinking of history: Message list
    | Acting of tool: Tool * input: string
    | Observing of tool: Tool * output: string
    | WaitingForUser of prompt: string
    | Error of error: string

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
      Memory: Message list } // Short-term memory/context

    member this.ReceiveMessage(msg: Message) =
        { this with
            Memory = this.Memory @ [ msg ] }

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
type Belief =
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
