namespace TarsEngine.FSharp.Cli.Core

open System
open System.Collections.Concurrent
open System.Threading
open System.IO
open Microsoft.Extensions.Logging

/// TARS Unified Core System - Central foundation for all TARS operations
/// Provides unified state management, error handling, logging, and configuration
module UnifiedCore =
    
    /// Unified error type for all TARS operations
    type TarsError =
        | ValidationError of message: string * details: Map<string, string>
        | ExecutionError of message: string * innerException: Exception option
        | ConfigurationError of message: string * configPath: string
        | NetworkError of message: string * endpoint: string
        | FileSystemError of message: string * path: string
        | AgentError of agentId: string * message: string
        | CudaError of message: string * deviceId: int option
        | ProofError of message: string * proofId: string
        | UnknownError of message: string * context: Map<string, obj>

    /// Unified result type for all TARS operations
    type TarsResult<'T> =
        | Success of value: 'T * metadata: Map<string, obj>
        | Failure of error: TarsError * correlationId: string

    /// Unified operation context for tracking and correlation
    type TarsOperationContext = {
        CorrelationId: string
        OperationType: string
        StartTime: DateTime
        UserId: string option
        AgentId: string option
        ParentContext: string option
        Metadata: Map<string, obj>
    }

    /// Unified system metrics for monitoring and analytics
    type TarsSystemMetrics = {
        CpuUsage: float
        MemoryUsage: int64
        GpuUsage: float option
        ActiveAgents: int
        QueuedOperations: int
        CompletedOperations: int64
        ErrorCount: int64
        LastUpdate: DateTime
    }

    /// Unified configuration for all TARS components
    type TarsConfiguration = {
        // Core settings
        LogLevel: LogLevel
        MaxConcurrentOperations: int
        OperationTimeoutMs: int
        EnableCuda: bool
        EnableProofGeneration: bool
        
        // Agent settings
        MaxAgents: int
        AgentTimeoutMs: int
        AgentRetryCount: int
        
        // Data settings
        CacheSize: int64
        CacheExpirationMs: int
        MaxQueryResults: int
        
        // Security settings
        EnableEncryption: bool
        ProofValidationLevel: string
        AuditLevel: string
        
        // Performance settings
        CudaDeviceId: int option
        ThreadPoolSize: int
        BatchSize: int
        
        // Storage settings
        DataDirectory: string
        BackupDirectory: string
        LogDirectory: string
    }

    /// Unified state container for all TARS operations
    type TarsUnifiedState = {
        // System state
        Configuration: TarsConfiguration
        Metrics: TarsSystemMetrics
        StartTime: DateTime
        
        // Operation tracking
        ActiveOperations: ConcurrentDictionary<string, TarsOperationContext>
        CompletedOperations: ConcurrentQueue<TarsOperationContext>
        
        // Component state
        FluxVariables: ConcurrentDictionary<string, obj>
        AgentStates: ConcurrentDictionary<string, obj>
        CacheEntries: ConcurrentDictionary<string, obj * DateTime>
        ProofChain: ConcurrentQueue<obj>
        
        // Synchronization
        StateLock: ReaderWriterLockSlim
        LastStateUpdate: DateTime
    }

    /// Unified logger interface for all TARS components
    type ITarsLogger =
        abstract member LogTrace: correlationId: string * message: string * ?parameters: obj[] -> unit
        abstract member LogDebug: correlationId: string * message: string * ?parameters: obj[] -> unit
        abstract member LogInformation: correlationId: string * message: string * ?parameters: obj[] -> unit
        abstract member LogWarning: correlationId: string * message: string * ?parameters: obj[] -> unit
        abstract member LogError: correlationId: string * error: TarsError * ?ex: Exception -> unit
        abstract member LogCritical: correlationId: string * message: string * ?ex: Exception -> unit

    /// Unified component interface for all TARS modules
    type ITarsComponent =
        abstract member Name: string
        abstract member Version: string
        abstract member Initialize: TarsConfiguration -> TarsResult<unit>
        abstract member Shutdown: unit -> TarsResult<unit>
        abstract member GetHealth: unit -> TarsResult<Map<string, obj>>
        abstract member GetMetrics: unit -> TarsResult<Map<string, obj>>

    /// Unified operation interface for all TARS operations
    type ITarsOperation<'TInput, 'TOutput> =
        abstract member Name: string
        abstract member Execute: context: TarsOperationContext * input: 'TInput -> Async<TarsResult<'TOutput>>
        abstract member Validate: input: 'TInput -> TarsResult<unit>
        abstract member GetEstimatedDuration: input: 'TInput -> TimeSpan

    /// Default TARS configuration
    let defaultConfiguration = {
        // Core settings
        LogLevel = LogLevel.Information
        MaxConcurrentOperations = 100
        OperationTimeoutMs = 30000
        EnableCuda = true
        EnableProofGeneration = true
        
        // Agent settings
        MaxAgents = 50
        AgentTimeoutMs = 10000
        AgentRetryCount = 3
        
        // Data settings
        CacheSize = 1024L * 1024L * 1024L // 1GB
        CacheExpirationMs = 3600000 // 1 hour
        MaxQueryResults = 10000
        
        // Security settings
        EnableEncryption = true
        ProofValidationLevel = "Standard"
        AuditLevel = "Full"
        
        // Performance settings
        CudaDeviceId = Some 0
        ThreadPoolSize = Environment.ProcessorCount * 2
        BatchSize = 1000
        
        // Storage settings
        DataDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "data")
        BackupDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "backup")
        LogDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "logs")
    }

    /// Create initial unified state
    let createInitialState (config: TarsConfiguration) : TarsUnifiedState = {
        Configuration = config
        Metrics = {
            CpuUsage = 0.0
            MemoryUsage = 0L
            GpuUsage = None
            ActiveAgents = 0
            QueuedOperations = 0
            CompletedOperations = 0L
            ErrorCount = 0L
            LastUpdate = DateTime.Now
        }
        StartTime = DateTime.Now
        ActiveOperations = ConcurrentDictionary<string, TarsOperationContext>()
        CompletedOperations = ConcurrentQueue<TarsOperationContext>()
        FluxVariables = ConcurrentDictionary<string, obj>()
        AgentStates = ConcurrentDictionary<string, obj>()
        CacheEntries = ConcurrentDictionary<string, obj * DateTime>()
        ProofChain = ConcurrentQueue<obj>()
        StateLock = new ReaderWriterLockSlim()
        LastStateUpdate = DateTime.Now
    }

    /// Generate correlation ID for operation tracking
    let generateCorrelationId () : string =
        let timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss")
        let guid = Guid.NewGuid().ToString("N").Substring(0, 8)
        $"tars-{timestamp}-{guid}"

    /// Create operation context
    let createOperationContext (operationType: string) (userId: string option) (agentId: string option) (parentContext: string option) : TarsOperationContext = {
        CorrelationId = generateCorrelationId()
        OperationType = operationType
        StartTime = DateTime.Now
        UserId = userId
        AgentId = agentId
        ParentContext = parentContext
        Metadata = Map.empty
    }

    /// TarsResult helper functions
    module TarsResult =
        let map (f: 'T -> 'U) (result: TarsResult<'T>) : TarsResult<'U> =
            match result with
            | Success (value, metadata) -> Success (f value, metadata)
            | Failure (error, correlationId) -> Failure (error, correlationId)

        let bind (f: 'T -> TarsResult<'U>) (result: TarsResult<'T>) : TarsResult<'U> =
            match result with
            | Success (value, metadata) -> 
                match f value with
                | Success (newValue, newMetadata) -> Success (newValue, Map.fold (fun acc k v -> Map.add k v acc) metadata newMetadata)
                | Failure (error, correlationId) -> Failure (error, correlationId)
            | Failure (error, correlationId) -> Failure (error, correlationId)

        let ofOption (error: TarsError) (correlationId: string) (option: 'T option) : TarsResult<'T> =
            match option with
            | Some value -> Success (value, Map.empty)
            | None -> Failure (error, correlationId)

        let ofResult (correlationId: string) (result: Result<'T, string>) : TarsResult<'T> =
            match result with
            | Ok value -> Success (value, Map.empty)
            | Error message -> Failure (UnknownError (message, Map.empty), correlationId)

        let toOption (result: TarsResult<'T>) : 'T option =
            match result with
            | Success (value, _) -> Some value
            | Failure _ -> None

        let isSuccess (result: TarsResult<'T>) : bool =
            match result with
            | Success _ -> true
            | Failure _ -> false

        let getError (result: TarsResult<'T>) : TarsError option =
            match result with
            | Success _ -> None
            | Failure (error, _) -> Some error

    /// Error helper functions
    module TarsError =
        let toString (error: TarsError) : string =
            match error with
            | ValidationError (message, details) -> 
                let detailsStr = details |> Map.toSeq |> Seq.map (fun (k, v) -> $"{k}: {v}") |> String.concat ", "
                $"Validation Error: {message} ({detailsStr})"
            | ExecutionError (message, innerEx) -> 
                match innerEx with
                | Some ex -> $"Execution Error: {message} (Inner: {ex.Message})"
                | None -> $"Execution Error: {message}"
            | ConfigurationError (message, configPath) -> $"Configuration Error: {message} (Path: {configPath})"
            | NetworkError (message, endpoint) -> $"Network Error: {message} (Endpoint: {endpoint})"
            | FileSystemError (message, path) -> $"File System Error: {message} (Path: {path})"
            | AgentError (agentId, message) -> $"Agent Error [{agentId}]: {message}"
            | CudaError (message, deviceId) -> 
                match deviceId with
                | Some id -> $"CUDA Error: {message} (Device: {id})"
                | None -> $"CUDA Error: {message}"
            | ProofError (message, proofId) -> $"Proof Error: {message} (Proof: {proofId})"
            | UnknownError (message, context) -> 
                let contextStr = context |> Map.toSeq |> Seq.map (fun (k, v) -> $"{k}: {v}") |> String.concat ", "
                $"Unknown Error: {message} (Context: {contextStr})"

        let getCategory (error: TarsError) : string =
            match error with
            | ValidationError _ -> "Validation"
            | ExecutionError _ -> "Execution"
            | ConfigurationError _ -> "Configuration"
            | NetworkError _ -> "Network"
            | FileSystemError _ -> "FileSystem"
            | AgentError _ -> "Agent"
            | CudaError _ -> "CUDA"
            | ProofError _ -> "Proof"
            | UnknownError _ -> "Unknown"

        let isRecoverable (error: TarsError) : bool =
            match error with
            | ValidationError _ -> false
            | ExecutionError _ -> true
            | ConfigurationError _ -> false
            | NetworkError _ -> true
            | FileSystemError _ -> true
            | AgentError _ -> true
            | CudaError _ -> true
            | ProofError _ -> false
            | UnknownError _ -> false
