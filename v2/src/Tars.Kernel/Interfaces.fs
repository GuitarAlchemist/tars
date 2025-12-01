namespace Tars.Kernel

open System
open System.Threading.Tasks
open Tars.Core

/// Represents an autonomous agent
type IAgent =
    abstract member Id: string
    abstract member Name: string
    /// Handles a semantic message. The agent must respect the Performative and Constraints.
    abstract member HandleAsync: SemanticMessage<obj> -> Task<ExecutionOutcome<unit>>

/// Represents the central nervous system (Semantic Bus)
type IEventBus =
    /// Publishes a semantic message to the bus
    abstract member PublishAsync: SemanticMessage<obj> -> Task
    /// Subscribes to messages.
    /// topic: Can be a specific AgentId, or a broadcast topic like "system.events"
    abstract member Subscribe: topic: string * handler: (SemanticMessage<obj> -> Task) -> IDisposable

/// Represents a provider for cognitive services (LLM)
type ICognitiveProvider =
    abstract member AskAsync: prompt: string -> Task<string>
    abstract member GetEmbeddingsAsync: texts: string list -> Task<float32[][]>

/// Context holding kernel services
type KernelContext = {
    SemanticMemory : ISemanticMemory
}
