namespace Tars.Core

open System

/// Unique identifier for an agent
type AgentId = AgentId of Guid

/// Unique identifier for a correlation/conversation
type CorrelationId = CorrelationId of Guid

/// <summary>
/// Represents the semantic domain or topic of a message.
/// </summary>
type AgentDomain =
    | Coding
    | Planning
    | Reasoning
    | Chat
