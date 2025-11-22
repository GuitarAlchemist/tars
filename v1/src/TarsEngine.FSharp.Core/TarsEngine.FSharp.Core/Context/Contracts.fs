namespace TarsEngine.FSharp.Core.Context

open System
open System.Threading.Tasks

/// Core context engineering types for TARS
module Types =
    
    /// Represents a span of context with metadata
    type ContextSpan = {
        Id: string
        Text: string
        Tokens: int
        Salience: float // 0.0 - 1.0, higher = more important
        Source: string
        Timestamp: DateTime
        Intent: string option
        Metadata: Map<string, string>
    }
    
    /// Task intent classification for context-aware retrieval
    type Intent =
        | Plan
        | CodeGen
        | Eval
        | Refactor
        | Reasoning
        | MetascriptExecution
        | AutonomousImprovement
        | Unknown of string
    
    /// Context compression result
    type CompressionResult = {
        OriginalSpans: ContextSpan list
        CompressedSpans: ContextSpan list
        CompressionRatio: float
        QualityEstimate: float
        Notes: string
    }
    
    /// Retrieval profile for different intents
    type RetrievalProfile = {
        Intent: Intent
        Retrievers: string list
        ChunkStrategy: string
        MaxSpans: int
        SalienceThreshold: float
    }
    
    /// Context policy configuration
    type ContextPolicy = {
        StepTokenBudget: int
        CompressionEnabled: bool
        CompressionStrategy: string
        CompressionTargets: string list
        MaxCompressionRatio: float
        RetrievalProfiles: RetrievalProfile list
        FewShotMaxExamples: int
        FewShotPolicy: string
    }
    
    /// Memory consolidation result
    type ConsolidationResult = {
        RunId: string
        SpansProcessed: int
        SpansPromoted: int
        SpansArchived: int
        ConflictsDetected: string list
        Summary: string
    }

/// Interface for context budgeting and prioritization
type IContextBudget =
    /// Score context spans by relevance and importance
    abstract member ScoreSpans: intent:Types.Intent * spans:Types.ContextSpan list -> Types.ContextSpan list
    
    /// Enforce token budget by selecting top spans
    abstract member EnforceTokenBudget: maxTokens:int * spans:Types.ContextSpan list -> Types.ContextSpan list
    
    /// Calculate salience score for a span
    abstract member CalculateSalience: span:Types.ContextSpan * intent:Types.Intent -> float

/// Interface for context compression
type IContextCompressor =
    /// Compress context spans while preserving quality
    abstract member CompressSpans: spans:Types.ContextSpan list -> Task<Types.CompressionResult>
    
    /// Estimate compression quality
    abstract member EstimateQuality: original:string * compressed:string -> float

/// Interface for intent-aware retrieval
type IContextRetriever =
    /// Retrieve context spans based on intent and query
    abstract member RetrieveAsync: intent:Types.Intent * query:string -> Task<Types.ContextSpan list>
    
    /// Get retrieval profile for intent
    abstract member GetProfile: intent:Types.Intent -> Types.RetrievalProfile option

/// Interface for intent classification
type IIntentRouter =
    /// Classify the intent of a step
    abstract member ClassifyIntent: stepName:string * input:string -> Types.Intent
    
    /// Get confidence score for intent classification
    abstract member GetConfidence: stepName:string * input:string * intent:Types.Intent -> float

/// Interface for security and sanitization
type IContextGuard =
    /// Sanitize incoming context for security
    abstract member SanitizeContext: context:string -> string
    
    /// Approve tool call for security
    abstract member ApproveToolCall: tool:string * args:string -> bool
    
    /// Detect potential prompt injection
    abstract member DetectInjection: text:string -> bool

/// Interface for tiered memory management
type IContextMemory =
    /// Load ephemeral memory (current session)
    abstract member LoadEphemeralAsync: unit -> Task<Types.ContextSpan list>
    
    /// Load working set memory (recent important spans)
    abstract member LoadWorkingSetAsync: unit -> Task<Types.ContextSpan list>
    
    /// Load long-term memory (consolidated knowledge)
    abstract member LoadLongTermAsync: unit -> Task<Types.ContextSpan list>
    
    /// Store spans in ephemeral memory
    abstract member StoreEphemeralAsync: spans:Types.ContextSpan list -> Task<unit>
    
    /// Promote spans to working set
    abstract member PromoteToWorkingSetAsync: spans:Types.ContextSpan list -> Task<unit>
    
    /// Consolidate working set to long-term memory
    abstract member ConsolidateAsync: runId:string -> Task<Types.ConsolidationResult>
    
    /// Clear ephemeral memory
    abstract member ClearEphemeralAsync: unit -> Task<unit>

/// Interface for few-shot example management
type IFewShotManager =
    /// Get few-shot examples for intent
    abstract member GetExamplesAsync: intent:Types.Intent * maxExamples:int -> Task<Types.ContextSpan list>
    
    /// Add successful example
    abstract member AddExampleAsync: intent:Types.Intent * example:Types.ContextSpan * metrics:Map<string, float> -> Task<unit>
    
    /// Update example salience based on performance
    abstract member UpdateExampleSalienceAsync: exampleId:string * performance:float -> Task<unit>

/// Interface for structured output validation
type IOutputValidator =
    /// Validate output against JSON schema
    abstract member ValidateAsync: output:string * schemaName:string -> Task<bool * string list>
    
    /// Get schema for output type
    abstract member GetSchema: outputType:string -> string option
    
    /// Repair invalid output
    abstract member RepairOutputAsync: output:string * schemaName:string * errors:string list -> Task<string>

/// Main context assembly interface
type IContextAssembly =
    /// Assemble context for a step
    abstract member AssembleContextAsync: stepName:string * input:string * policy:Types.ContextPolicy -> Task<Types.ContextSpan list>
    
    /// Get context statistics
    abstract member GetContextStats: unit -> Map<string, float>
    
    /// Update context policy
    abstract member UpdatePolicy: policy:Types.ContextPolicy -> unit
