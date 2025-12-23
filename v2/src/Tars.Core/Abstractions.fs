namespace Tars.Core

open System
open System.Threading.Tasks
open System.Collections.Generic

/// Represents a vector database for long-term memory
type IVectorStore =
    abstract member SaveAsync:
        collection: string * id: string * vector: float32[] * payload: Map<string, string> -> Task

    abstract member SearchAsync:
        collection: string * vector: float32[] * limit: int -> Task<(string * float32 * Map<string, string>) list>

/// Semantic capability index for routing to agents by capability description.
type ICapabilityStore =
    /// Finds agents matching a natural-language capability query.
    abstract member FindAgentsAsync: query: string * limit: int -> Task<(AgentId * Capability * float) list>

/// Registry for looking up agents
type IAgentRegistry =
    abstract member GetAgent: AgentId -> Async<Agent option>
    abstract member FindAgents: CapabilityKind -> Async<Agent list>
    abstract member GetAllAgents: unit -> Async<Agent list>

/// Interface for executing agents
type IAgentExecutor =
    abstract member Execute: agentId: AgentId * task: string -> Async<ExecutionOutcome<string>>

/// Interface for epistemic governance operations
type IEpistemicGovernor =
    /// <summary>Generates task variants for testing generalization.</summary>
    abstract member GenerateVariants: taskDescription: string * count: int -> Task<string list>

    /// <summary>Verifies that a solution generalizes to variants.</summary>
    abstract member VerifyGeneralization:
        taskDescription: string * solution: string * variants: string list -> Task<VerificationResult>

    /// <summary>Extracts a reusable principle from a task solution.</summary>
    abstract member ExtractPrinciple: taskDescription: string * solution: string -> Task<Belief>

    /// <summary>Suggests next learning tasks based on history.</summary>
    abstract member SuggestCurriculum:
        completedTasks: string list * activeBeliefs: string list * isCritical: bool -> Task<string>

    /// <summary>Verifies a statement against established beliefs.</summary>
    abstract member Verify: statement: string -> Task<bool>

    /// <summary>Retrieves relevant code context from the knowledge graph.</summary>
    abstract member GetRelatedCodeContext: query: string -> Task<string>

/// Registry for managing tools
type IToolRegistry =
    /// Register a tool
    abstract member Register: tool: Tool -> unit
    /// Get a tool by name
    abstract member Get: name: string -> Tool option
    /// Get all registered tools
    /// Get all registered tools
    abstract member GetAll: unit -> Tool list

/// Represents the Knowledge Graph service
type IGraphService =
    abstract member AddNodeAsync: TarsEntity -> Task<string>
    abstract member AddFactAsync: TarsFact -> Task<Guid>
    abstract member AddEpisodeAsync: Episode -> Task<string>
    abstract member QueryAsync: query: string -> Task<TarsFact list>
    abstract member PersistAsync: unit -> Task<unit>
