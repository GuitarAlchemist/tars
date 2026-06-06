namespace Tars.Core

open System

/// <summary>
/// Represents the illocutionary force of an agent message.
/// These are the "Speech Acts" that agents use to coordinate.
/// (Phase 5.2 Implementation)
/// </summary>
type AgentIntent =
    | Ask of query: string // Request computation or answer
    | Tell of fact: string // Share a fact or belief
    | Propose of plan: string // Suggest a plan or action
    | Accept of refId: Guid // Agree to a proposal (referenced by ID)
    | Reject of refId: Guid * reason: string // Decline a proposal
    | Act of tool: string * args: string // Execute a concrete action (Tool Use)
    | Event of topic: string * payload: string // Broadcast an observation/event
    | Error of msg: string // Report a failure or error

/// <summary>
/// A standardized message envelope for agent communication.
/// </summary>
type AgentMessage =
    {
        Id: Guid
        CorrelationId: CorrelationId
        From: MessageEndpoint
        To: MessageEndpoint option // None implies broadcast
        Intent: AgentIntent
        /// The semantic domain or ontology (e.g., "coding", "planning")
        /// Replaces the old 'AgentIntent' (Coding, Planning, etc.)
        Domain: string option
        Content: string
        Timestamp: DateTime
        Metadata: Map<string, string>
    }

    /// Create a new message
    static member Create
        (
            sender: MessageEndpoint,
            intent: AgentIntent,
            content: string,
            ?receiver: MessageEndpoint,
            ?correlationId: CorrelationId
        ) =
        { Id = Guid.NewGuid()
          CorrelationId = defaultArg correlationId (CorrelationId(Guid.NewGuid()))
          From = sender
          To = receiver
          Intent = intent
          Domain = None
          Content = content
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

/// <summary>
/// Interface for an agent communication bus.
/// </summary>
type IAgentBus =
    abstract member Publish: message: AgentMessage -> Async<unit>
    abstract member Subscribe: agentId: AgentId * handler: (AgentMessage -> Async<unit>) -> IDisposable
    abstract member Broadcast: message: AgentMessage -> Async<unit>
