namespace Tars.Core

open System

/// Categorization of errors encountered during an episode.
type ErrorKind =
    | Perceptual   // Mis-reading environment, files, schemas, APIs
    | Logical      // Reasoning or planning mistakes
    | Mixed        // Combination of both

/// Function to generate embeddings for text.
type Embedder = string -> Async<float32[]>

/// Specific error label with description.
type ErrorTag =
    { Name        : string       // e.g., "hallucination", "schema-mismatch"
      Description : string }

/// Represents the logical/reasoning stream of a memory.
type LogicalMemory = {
    ProblemSummary  : string          // 1–3 lines, LLM-generated summary of the task
    StrategySummary : string          // Summary of the approach taken
    ErrorKinds      : ErrorKind list  // List of error categories encountered
    ErrorTags       : ErrorTag list   // Specific error tags
    OutcomeLabel    : string          // "success" | "failure" | "partial"
    Score           : float option    // Optional numeric score (0.0 - 1.0)
    CostTokens      : int option      // Token cost of the episode
    Embedding       : float32 array   // Vector representation of the logical context
    Tags            : string list     // e.g., "task:refactor", "tool:git"
}

/// Represents the structure of the codebase involved in the task.
type CodeStructure = {
    Modules      : string list
    Types        : string list
    Functions    : string list
    Dependencies : string list
}

/// Represents the perceptual/environmental stream of a memory.
type PerceptualMemory = {
    TouchedResources : string list    // File paths, MCP URIs, HTTP endpoints accessed
    EnvFingerprint   : string         // Hash of repo state or configuration
    ToolsUsed        : string list    // e.g., "mcp:filesystem", "mcp:git"
    CodeStructure    : CodeStructure option
    Embedding        : float32 array  // Vector representation of the structural context
}

/// Trace object passed to Grow
type MemoryTrace = {
    TaskId      : string
    Variables   : Map<string, obj>
    StepOutputs : Map<string, Map<string, obj>>
}

type MemorySchemaId = string

/// The top-level Semantic Memory Schema binding logical and perceptual streams.
type MemorySchema = {
    Id          : MemorySchemaId
    Logical     : LogicalMemory option
    Perceptual  : PerceptualMemory option
    CreatedAt   : DateTime
    LastUsedAt  : DateTime option
    UsageCount  : int
}

/// Query parameters for retrieving memories.
type MemoryQuery = {
    TaskId      : string
    TaskKind    : string
    TextContext : string
    Tags        : string list
}

/// Interface for the Semantic Memory Kernel Service.
type ISemanticMemory =
    abstract member Retrieve : MemoryQuery -> Async<MemorySchema list>
    abstract member Grow     : episodeTrace:obj * verificationReport:obj -> Async<MemorySchemaId>
    abstract member Refine   : unit -> Async<unit>
