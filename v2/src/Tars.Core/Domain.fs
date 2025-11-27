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
    | Event // "Something happened"

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
      Constraints: SemanticConstraints
      Content: 'T
      Timestamp: DateTime
      Metadata: Map<string, string> }

/// Alias for text-based messages (common case)
type Message = SemanticMessage<string>

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
      State: AgentState
      Memory: Message list } // Short-term memory/context

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

/// Represents an edge/relationship in the internal knowledge graph
type GraphEdge =
    | RelatesTo of weight: float
    | CreatedBy
    | DependsOn
    | Solves
