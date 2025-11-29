namespace Tars.Core

open System.Threading.Tasks
open System.Collections.Generic

/// Represents a vector database for long-term memory
type IVectorStore =
    abstract member SaveAsync:
        collection: string * id: string * vector: float32[] * payload: Map<string, string> -> Task

    abstract member SearchAsync:
        collection: string * vector: float32[] * limit: int -> Task<(string * float32 * Map<string, string>) list>

/// Registry for looking up agents
type IAgentRegistry =
    abstract member GetAgent: AgentId -> Async<Agent option>
    abstract member FindAgents: CapabilityKind -> Async<Agent list>

/// Interface for executing agents
type IAgentExecutor =
    abstract member Execute: agentId: AgentId * task: string -> Async<ExecutionOutcome<string>>
