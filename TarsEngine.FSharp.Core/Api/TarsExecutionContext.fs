namespace TarsEngine.FSharp.Core.Api

open System
open System.Collections.Concurrent
open System.Threading
open TarsEngine.FSharp.Core.Api

/// <summary>
/// Execution context for TARS metascripts with security, tracing, and resource management
/// </summary>
type TarsExecutionContext = {
    /// Unique execution identifier
    ExecutionId: string
    
    /// Path to the current metascript being executed
    CurrentMetascript: string
    
    /// TARS Engine API instance
    EngineApi: ITarsEngineApi
    
    /// Security context for permission checking
    SecurityContext: ISecurityContext
    
    /// Resource limits and monitoring
    ResourceLimits: IResourceLimits
    
    /// Trace collector for debugging and audit
    TraceCollector: ITraceCollector
    
    /// Execution start time
    StartTime: DateTime
    
    /// Cancellation token for execution control
    CancellationToken: CancellationToken
    
    /// Execution variables and state
    Variables: ConcurrentDictionary<string, obj>
    
    /// Parent execution context (for nested executions)
    ParentContext: TarsExecutionContext option
}

/// Security context interface for permission and access control
and ISecurityContext =
    /// Check if an API is allowed to be called
    abstract member CheckApiPermission: apiName: string -> bool
    
    /// Check if a file path can be accessed
    abstract member CheckFileAccess: path: string * accessType: FileAccessType -> bool
    
    /// Check if a network request is allowed
    abstract member CheckNetworkAccess: url: string -> bool
    
    /// Get list of allowed APIs
    abstract member GetAllowedApis: unit -> string[]
    
    /// Get security policy
    abstract member GetSecurityPolicy: unit -> SecurityPolicy
    
    /// Log security violation
    abstract member LogSecurityViolation: violation: SecurityViolation -> unit

/// Resource limits interface for monitoring and enforcement
and IResourceLimits =
    /// Check if memory limit is exceeded
    abstract member CheckMemoryLimit: unit -> ResourceCheckResult
    
    /// Check if CPU time limit is exceeded
    abstract member CheckCpuLimit: unit -> ResourceCheckResult
    
    /// Check if network request limit is exceeded
    abstract member CheckNetworkLimit: unit -> ResourceCheckResult
    
    /// Check if file operation limit is exceeded
    abstract member CheckFileOperationLimit: unit -> ResourceCheckResult
    
    /// Record resource usage
    abstract member RecordResourceUsage: resourceType: ResourceType * amount: int64 -> unit
    
    /// Get current resource usage
    abstract member GetResourceUsage: unit -> ResourceUsage

/// Trace collector interface for execution tracing and debugging
and ITraceCollector =
    /// Log an API call with parameters and result
    abstract member LogApiCall: apiName: string * parameters: obj[] * result: obj option -> unit
    
    /// Log an execution event
    abstract member LogEvent: level: LogLevel * message: string * metadata: Map<string, obj> -> unit
    
    /// Start a trace span
    abstract member StartSpan: name: string -> TraceSpan
    
    /// End a trace span
    abstract member EndSpan: span: TraceSpan -> unit
    
    /// Get complete execution trace
    abstract member GetTrace: unit -> ExecutionTrace
    
    /// Export trace to file
    abstract member ExportTrace: format: TraceFormat * path: string -> bool

/// Security policy configuration
and SecurityPolicy = {
    /// Set of allowed API names
    AllowedApis: Set<string>
    
    /// Resource limits configuration
    ResourceLimits: ResourceLimitsConfig
    
    /// Network access policy
    NetworkAccess: NetworkPolicy
    
    /// File system access policy
    FileSystemAccess: FileSystemPolicy
    
    /// Maximum execution timeout
    ExecutionTimeout: TimeSpan
    
    /// Whether to allow nested metascript execution
    AllowNestedExecution: bool
    
    /// Whether to allow agent spawning
    AllowAgentSpawning: bool
}

/// Resource limits configuration
and ResourceLimitsConfig = {
    /// Maximum memory usage in MB
    MaxMemoryMB: int
    
    /// Maximum CPU time in milliseconds
    MaxCpuTimeMs: int
    
    /// Maximum number of network requests
    MaxNetworkRequests: int
    
    /// Maximum number of file operations
    MaxFileOperations: int
    
    /// Maximum number of LLM requests
    MaxLlmRequests: int
    
    /// Maximum number of vector store operations
    MaxVectorStoreOperations: int
}

/// Network access policy
and NetworkPolicy =
    | Denied
    | AllowedDomains of string list
    | AllowedUrls of string list
    | Unrestricted

/// File system access policy
and FileSystemPolicy =
    | Denied
    | ReadOnly of string list
    | ReadWrite of string list
    | Sandboxed of string

/// File access type enumeration
and FileAccessType = Read | Write | Execute | Delete

/// Security violation record
and SecurityViolation = {
    ViolationType: SecurityViolationType
    ApiName: string option
    Resource: string option
    Timestamp: DateTime
    Message: string
}

/// Security violation types
and SecurityViolationType =
    | UnauthorizedApiCall
    | UnauthorizedFileAccess
    | UnauthorizedNetworkAccess
    | ResourceLimitExceeded
    | ExecutionTimeoutExceeded

/// Resource check result
and ResourceCheckResult = {
    IsAllowed: bool
    CurrentUsage: int64
    Limit: int64
    Message: string option
}

/// Resource type enumeration
and ResourceType = Memory | CpuTime | NetworkRequests | FileOperations | LlmRequests | VectorStoreOperations

/// Resource usage tracking
and ResourceUsage = {
    MemoryMB: int64
    CpuTimeMs: int64
    NetworkRequests: int
    FileOperations: int
    LlmRequests: int
    VectorStoreOperations: int
    StartTime: DateTime
    LastUpdated: DateTime
}

/// Trace span for execution tracing
and TraceSpan = {
    SpanId: string
    Name: string
    StartTime: DateTime
    EndTime: DateTime option
    Metadata: Map<string, obj>
    ParentSpanId: string option
}

/// Complete execution trace
and ExecutionTrace = {
    ExecutionId: string
    StartTime: DateTime
    EndTime: DateTime option
    ApiCalls: ApiCallTrace[]
    Events: TraceEvent[]
    Spans: TraceSpan[]
    ResourceUsage: ResourceUsage
    SecurityViolations: SecurityViolation[]
}

/// API call trace record
and ApiCallTrace = {
    Timestamp: DateTime
    ApiName: string
    Parameters: obj[]
    Result: obj option
    ExecutionTime: TimeSpan
    Success: bool
    ErrorMessage: string option
}

/// Trace export format
and TraceFormat = Yaml | Json | Xml | Binary

/// Implementation classes for security, resource limits, and tracing
and private SecurityContextImpl(policy: SecurityPolicy) =
    interface ISecurityContext with
        member _.CheckApiPermission(apiName: string) = policy.AllowedApis.Contains(apiName)
        member _.CheckFileAccess(path: string, accessType: FileAccessType) =
            match policy.FileSystemAccess with
            | FileSystemPolicy.Denied -> false
            | FileSystemPolicy.ReadOnly paths -> accessType = FileAccessType.Read && paths |> List.exists (fun p -> path.StartsWith(p))
            | FileSystemPolicy.ReadWrite paths -> paths |> List.exists (fun p -> path.StartsWith(p))
            | FileSystemPolicy.Sandboxed root -> path.StartsWith(root)
        member _.CheckNetworkAccess(url: string) =
            match policy.NetworkAccess with
            | NetworkPolicy.Denied -> false
            | NetworkPolicy.AllowedDomains domains -> domains |> List.exists (fun d -> url.Contains(d))
            | NetworkPolicy.AllowedUrls urls -> urls |> List.contains url
            | NetworkPolicy.Unrestricted -> true
        member _.GetAllowedApis() = policy.AllowedApis |> Set.toArray
        member _.GetSecurityPolicy() = policy
        member _.LogSecurityViolation(violation: SecurityViolation) =
            // Implementation would log to security audit system
            ()

and private ResourceLimitsImpl(limits: ResourceLimitsConfig) =
    let mutable usage = {
        MemoryMB = 0L
        CpuTimeMs = 0L
        NetworkRequests = 0
        FileOperations = 0
        LlmRequests = 0
        VectorStoreOperations = 0
        StartTime = DateTime.UtcNow
        LastUpdated = DateTime.UtcNow
    }

    interface IResourceLimits with
        member _.CheckMemoryLimit() =
            { IsAllowed = usage.MemoryMB < int64 limits.MaxMemoryMB; CurrentUsage = usage.MemoryMB; Limit = int64 limits.MaxMemoryMB; Message = None }
        member _.CheckCpuLimit() =
            { IsAllowed = usage.CpuTimeMs < int64 limits.MaxCpuTimeMs; CurrentUsage = usage.CpuTimeMs; Limit = int64 limits.MaxCpuTimeMs; Message = None }
        member _.CheckNetworkLimit() =
            { IsAllowed = usage.NetworkRequests < limits.MaxNetworkRequests; CurrentUsage = int64 usage.NetworkRequests; Limit = int64 limits.MaxNetworkRequests; Message = None }
        member _.CheckFileOperationLimit() =
            { IsAllowed = usage.FileOperations < limits.MaxFileOperations; CurrentUsage = int64 usage.FileOperations; Limit = int64 limits.MaxFileOperations; Message = None }
        member _.RecordResourceUsage(resourceType: ResourceType, amount: int64) =
            usage <- { usage with LastUpdated = DateTime.UtcNow }
            match resourceType with
            | Memory -> usage <- { usage with MemoryMB = usage.MemoryMB + amount }
            | CpuTime -> usage <- { usage with CpuTimeMs = usage.CpuTimeMs + amount }
            | NetworkRequests -> usage <- { usage with NetworkRequests = usage.NetworkRequests + int amount }
            | FileOperations -> usage <- { usage with FileOperations = usage.FileOperations + int amount }
            | LlmRequests -> usage <- { usage with LlmRequests = usage.LlmRequests + int amount }
            | VectorStoreOperations -> usage <- { usage with VectorStoreOperations = usage.VectorStoreOperations + int amount }
        member _.GetResourceUsage() = usage

and private TraceCollectorImpl(executionId: string) =
    let mutable trace = {
        ExecutionId = executionId
        StartTime = DateTime.UtcNow
        EndTime = None
        ApiCalls = [||]
        Events = [||]
        Spans = [||]
        ResourceUsage = {
            MemoryMB = 0L; CpuTimeMs = 0L; NetworkRequests = 0; FileOperations = 0
            LlmRequests = 0; VectorStoreOperations = 0
            StartTime = DateTime.UtcNow; LastUpdated = DateTime.UtcNow
        }
        SecurityViolations = [||]
    }

    interface ITraceCollector with
        member _.LogApiCall(apiName: string, parameters: obj[], result: obj option) =
            // Implementation would add to trace.ApiCalls
            ()
        member _.LogEvent(level: LogLevel, message: string, metadata: Map<string, obj>) =
            // Implementation would add to trace.Events
            ()
        member _.StartSpan(name: string) =
            { SpanId = Guid.NewGuid().ToString(); Name = name; StartTime = DateTime.UtcNow; EndTime = None; Metadata = Map.empty; ParentSpanId = None }
        member _.EndSpan(span: TraceSpan) =
            // Implementation would update span and add to trace.Spans
            ()
        member _.GetTrace() = trace
        member _.ExportTrace(format: TraceFormat, path: string) =
            // Implementation would export trace to specified format and path
            true

/// <summary>
/// Factory for creating TARS execution contexts with proper security and resource management
/// </summary>
type TarsExecutionContextFactory() =
    
    /// Create a new execution context with specified security policy
    static member Create(metascriptPath: string, engineApi: ITarsEngineApi, securityPolicy: SecurityPolicy) =
        let executionId = "exec_" + Guid.NewGuid().ToString("N").[..7]
        let cancellationTokenSource = new CancellationTokenSource(securityPolicy.ExecutionTimeout)
        
        {
            ExecutionId = executionId
            CurrentMetascript = metascriptPath
            EngineApi = engineApi
            SecurityContext = SecurityContextImpl(securityPolicy) :> ISecurityContext
            ResourceLimits = ResourceLimitsImpl(securityPolicy.ResourceLimits) :> IResourceLimits
            TraceCollector = TraceCollectorImpl(executionId) :> ITraceCollector
            StartTime = DateTime.UtcNow
            CancellationToken = cancellationTokenSource.Token
            Variables = ConcurrentDictionary<string, obj>()
            ParentContext = None
        }
    
    /// Create a child execution context for nested execution
    static member CreateChild(parentContext: TarsExecutionContext, metascriptPath: string) =
        let executionId = parentContext.ExecutionId + "_child_" + Guid.NewGuid().ToString("N").[..3]
        
        { parentContext with
            ExecutionId = executionId
            CurrentMetascript = metascriptPath
            StartTime = DateTime.UtcNow
            Variables = ConcurrentDictionary<string, obj>()
            ParentContext = Some parentContext
        }

/// Default security policy for safe metascript execution
type DefaultSecurityPolicies() =

    /// Restrictive policy for untrusted metascripts
    static member Restrictive = {
        AllowedApis = Set.ofList ["Tracing"; "ExecutionContext"]
        ResourceLimits = {
            MaxMemoryMB = 128
            MaxCpuTimeMs = 10000
            MaxNetworkRequests = 0
            MaxFileOperations = 0
            MaxLlmRequests = 0
            MaxVectorStoreOperations = 0
        }
        NetworkAccess = NetworkPolicy.Denied
        FileSystemAccess = FileSystemPolicy.Denied
        ExecutionTimeout = TimeSpan.FromMinutes(1.0)
        AllowNestedExecution = false
        AllowAgentSpawning = false
    }

    /// Standard policy for trusted metascripts
    static member Standard = {
        AllowedApis = Set.ofList ["VectorStore"; "LLM"; "Tracing"; "ExecutionContext"; "FileSystem"]
        ResourceLimits = {
            MaxMemoryMB = 512
            MaxCpuTimeMs = 30000
            MaxNetworkRequests = 50
            MaxFileOperations = 100
            MaxLlmRequests = 20
            MaxVectorStoreOperations = 100
        }
        NetworkAccess = NetworkPolicy.AllowedDomains ["api.openai.com"; "api.mistral.ai"]
        FileSystemAccess = FileSystemPolicy.ReadWrite [".tars"; "output"; "temp"]
        ExecutionTimeout = TimeSpan.FromMinutes(5.0)
        AllowNestedExecution = true
        AllowAgentSpawning = false
    }

    /// Unrestricted policy for system metascripts
    static member Unrestricted = {
        AllowedApis = Set.ofList ["VectorStore"; "LLM"; "MetascriptRunner"; "AgentCoordinator"; "CudaEngine"; "FileSystem"; "WebSearch"; "GitHubApi"; "Tracing"; "ExecutionContext"]
        ResourceLimits = {
            MaxMemoryMB = 2048
            MaxCpuTimeMs = 300000
            MaxNetworkRequests = 1000
            MaxFileOperations = 1000
            MaxLlmRequests = 100
            MaxVectorStoreOperations = 1000
        }
        NetworkAccess = NetworkPolicy.Unrestricted
        FileSystemAccess = FileSystemPolicy.ReadWrite ["/"; "C:\\"]
        ExecutionTimeout = TimeSpan.FromMinutes(30.0)
        AllowNestedExecution = true
        AllowAgentSpawning = true
    }


