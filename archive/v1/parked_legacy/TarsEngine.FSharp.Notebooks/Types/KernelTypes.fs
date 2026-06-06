namespace TarsEngine.FSharp.Notebooks.Types

open System
open System.Threading
open System.Threading.Tasks

/// <summary>
/// Types for kernel management and execution
/// </summary>

/// Unique kernel identifier
type KernelId = KernelId of Guid

/// Kernel instance status
type KernelStatus = 
    | Starting
    | Ready
    | Busy
    | Idle
    | Restarting
    | Stopping
    | Stopped
    | Error of string

/// Kernel instance
type KernelInstance = {
    Id: KernelId
    Specification: KernelSpecification
    Status: KernelStatus
    ProcessId: int option
    StartTime: DateTime
    LastActivity: DateTime
    ExecutionCount: int
    WorkingDirectory: string
    Environment: Map<string, string>
}

/// Kernel execution request
type KernelExecutionRequest = {
    Id: string
    Code: string
    Silent: bool
    StoreHistory: bool
    UserExpressions: Map<string, string>
    AllowStdin: bool
    StopOnError: bool
}

/// Kernel execution result
type KernelExecutionResult = {
    RequestId: string
    Status: ExecutionStatus
    ExecutionCount: int
    Outputs: NotebookOutput list
    UserExpressions: Map<string, obj>
    ExecutionTime: TimeSpan
    Error: string option
}

/// Execution status
and ExecutionStatus = 
    | Success
    | Error
    | Aborted

/// Kernel manager configuration
type KernelManagerConfig = {
    MaxKernels: int
    KernelTimeout: TimeSpan
    RestartOnError: bool
    LogLevel: string
    WorkingDirectory: string
    DefaultKernel: string
}

/// Kernel pool for managing multiple kernels
type KernelPool = {
    AvailableKernels: Map<KernelId, KernelInstance>
    BusyKernels: Map<KernelId, KernelInstance>
    MaxPoolSize: int
    IdleTimeout: TimeSpan
}

/// Kernel communication message
type KernelMessage = 
    | ExecuteRequest of KernelExecutionRequest
    | InterruptRequest of string
    | RestartRequest of string
    | ShutdownRequest of string
    | StatusRequest of string
    | CompleteRequest of CompleteRequest
    | InspectRequest of InspectRequest

/// Code completion request
and CompleteRequest = {
    Code: string
    CursorPos: int
}

/// Code inspection request
and InspectRequest = {
    Code: string
    CursorPos: int
    DetailLevel: int
}

/// Kernel response message
type KernelResponse = 
    | ExecuteReply of KernelExecutionResult
    | StatusReply of KernelStatus
    | CompleteReply of CompleteResponse
    | InspectReply of InspectResponse
    | ErrorReply of string

/// Code completion response
and CompleteResponse = {
    Matches: string list
    CursorStart: int
    CursorEnd: int
    Metadata: Map<string, obj>
}

/// Code inspection response
and InspectResponse = {
    Found: bool
    Data: Map<string, obj>
    Metadata: Map<string, obj>
}

/// Kernel lifecycle events
type KernelEvent = 
    | KernelStarted of KernelId
    | KernelStopped of KernelId
    | KernelRestarted of KernelId
    | KernelError of KernelId * string
    | ExecutionStarted of KernelId * string
    | ExecutionCompleted of KernelId * string
    | OutputReceived of KernelId * NotebookOutput

/// Kernel manager interface
type IKernelManager =
    /// Start a new kernel instance
    abstract member StartKernelAsync: KernelSpecification -> Async<KernelInstance>
    
    /// Stop a kernel instance
    abstract member StopKernelAsync: KernelId -> Async<unit>
    
    /// Execute code in a kernel
    abstract member ExecuteCodeAsync: KernelId -> string -> Async<KernelExecutionResult>
    
    /// Get kernel status
    abstract member GetKernelStatusAsync: KernelId -> Async<KernelStatus>
    
    /// Restart a kernel
    abstract member RestartKernelAsync: KernelId -> Async<KernelInstance>
    
    /// List all kernels
    abstract member ListKernelsAsync: unit -> Async<KernelInstance list>
    
    /// Get available kernel specifications
    abstract member GetAvailableKernelSpecsAsync: unit -> Async<KernelSpecification list>

/// Polyglot notebook support
type PolyglotNotebook = {
    Cells: PolyglotCell list
    SharedVariables: Map<string, obj>
    KernelMappings: Map<string, KernelId>
    ExecutionOrder: string list
}

/// Polyglot cell with language specification
and PolyglotCell = {
    Id: string
    Language: string
    Source: string list
    Outputs: NotebookOutput list
    Dependencies: string list
    SharedVariables: string list
}

/// Cross-language data sharing
type DataSharingProtocol = {
    SourceLanguage: string
    TargetLanguage: string
    Serializer: obj -> string
    Deserializer: string -> obj
    SupportedTypes: Type list
}

/// Kernel performance metrics
type KernelMetrics = {
    KernelId: KernelId
    TotalExecutions: int
    AverageExecutionTime: TimeSpan
    MemoryUsage: int64
    CpuUsage: float
    ErrorRate: float
    LastMetricsUpdate: DateTime
}

/// Kernel resource limits
type KernelResourceLimits = {
    MaxMemoryMB: int
    MaxCpuPercent: float
    MaxExecutionTime: TimeSpan
    MaxOutputSize: int
    MaxVariables: int
}

/// Kernel security configuration
type KernelSecurityConfig = {
    AllowNetworkAccess: bool
    AllowFileSystemAccess: bool
    AllowedDirectories: string list
    BlockedCommands: string list
    SandboxMode: bool
    TrustedPackages: string list
}

/// Kernel debugging support
type KernelDebugger = {
    KernelId: KernelId
    BreakpointSupport: bool
    VariableInspection: bool
    StepExecution: bool
    CallStackSupport: bool
}

/// Debugging session
type DebugSession = {
    Id: string
    KernelId: KernelId
    Breakpoints: Breakpoint list
    CurrentFrame: StackFrame option
    Variables: Map<string, obj>
    Status: DebugStatus
}

/// Breakpoint definition
and Breakpoint = {
    Id: string
    File: string
    Line: int
    Condition: string option
    Enabled: bool
}

/// Stack frame information
and StackFrame = {
    Id: string
    Name: string
    File: string
    Line: int
    Variables: Map<string, obj>
}

/// Debug session status
and DebugStatus = 
    | Running
    | Paused
    | Stopped
    | StepInto
    | StepOver
    | StepOut

/// Kernel extension interface
type IKernelExtension =
    /// Extension name
    abstract member Name: string
    
    /// Extension version
    abstract member Version: string
    
    /// Initialize extension
    abstract member InitializeAsync: KernelInstance -> Async<unit>
    
    /// Handle custom messages
    abstract member HandleMessageAsync: KernelMessage -> Async<KernelResponse option>
    
    /// Cleanup extension
    abstract member CleanupAsync: unit -> Async<unit>

/// Kernel extension registry
type KernelExtensionRegistry = {
    Extensions: Map<string, IKernelExtension>
    LoadedExtensions: Map<KernelId, string list>
}

/// Kernel configuration provider
type IKernelConfigProvider =
    /// Get kernel specifications
    abstract member GetKernelSpecificationsAsync: unit -> Async<KernelSpecification list>
    
    /// Get default kernel for language
    abstract member GetDefaultKernelAsync: string -> Async<KernelSpecification option>
    
    /// Validate kernel configuration
    abstract member ValidateKernelConfigAsync: KernelSpecification -> Async<ValidationResult>

/// Kernel event handler
type KernelEventHandler = KernelEvent -> Async<unit>

/// Kernel manager with events
type IKernelManagerWithEvents =
    inherit IKernelManager
    
    /// Subscribe to kernel events
    abstract member SubscribeToEvents: KernelEventHandler -> unit
    
    /// Unsubscribe from kernel events
    abstract member UnsubscribeFromEvents: KernelEventHandler -> unit
