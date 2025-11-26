namespace Tars.Kernel

open System
open System.Threading.Tasks

/// Represents a message flowing through the system
type IMessage =
    abstract member Id: Guid
    abstract member CorrelationId: Guid
    abstract member Source: string
    abstract member Target: string option
    abstract member Content: obj
    abstract member Timestamp: DateTime

/// Represents an autonomous agent
type IAgent =
    abstract member Id: string
    abstract member Name: string
    abstract member HandleAsync: IMessage -> Task

/// Represents the central nervous system
type IEventBus =
    abstract member PublishAsync: IMessage -> Task
    abstract member Subscribe: string * (IMessage -> Task) -> IDisposable

/// Represents a provider for cognitive services (LLM)
type ICognitiveProvider =
    abstract member AskAsync: prompt: string -> Task<string>
    abstract member GetEmbeddingsAsync: texts: string list -> Task<float32[][]>

