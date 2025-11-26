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

/// A message flowing through the system
type Message =
    { Id: Guid
      CorrelationId: CorrelationId
      Source: MessageEndpoint
      Target: MessageEndpoint
      Content: string
      Timestamp: DateTime
      Metadata: Map<string, string> }

/// Represents a tool that an agent can execute
type Tool =
    {
        Name: string
        Description: string
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
      Model: string
      SystemPrompt: string
      Tools: Tool list
      State: AgentState
      Memory: Message list } // Short-term memory/context

/// The result of a kernel operation
type KernelResult<'T> = Result<'T, string>
