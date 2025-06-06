namespace TarsEngine.FSharp.Core.Api

open System
open System.Threading.Tasks
open System.Collections.Generic
open System.Collections.Concurrent

/// <summary>
/// Core TARS Engine API interface that provides access to all TARS capabilities
/// from within metascript execution contexts
/// </summary>
type ITarsEngineApi =
    /// Vector store operations for semantic search and knowledge management
    abstract member VectorStore: IVectorStoreApi
    
    /// LLM service for text generation, completion, and embedding
    abstract member LlmService: ILlmServiceApi
    
    /// Metascript execution and management
    abstract member MetascriptRunner: IMetascriptRunnerApi
    
    /// Multi-agent coordination and communication
    abstract member AgentCoordinator: IAgentCoordinatorApi
    
    /// CUDA-accelerated computation engine
    abstract member CudaEngine: ICudaEngineApi
    
    /// Secure file system operations
    abstract member FileSystem: IFileSystemApi
    
    /// Web search and content retrieval
    abstract member WebSearch: IWebSearchApi
    
    /// GitHub API integration
    abstract member GitHubApi: IGitHubApiService
    
    /// Execution context and tracing
    abstract member ExecutionContext: IExecutionContextApi

/// Vector store API for semantic search and knowledge management
and IVectorStoreApi =
    /// Search for similar content using semantic similarity
    abstract member SearchAsync: query: string * limit: int -> Task<SearchResult[]>
    
    /// Add content to the vector store with metadata
    abstract member AddAsync: content: string * metadata: Map<string, string> -> Task<string>
    
    /// Delete content by vector ID
    abstract member DeleteAsync: vectorId: string -> Task<bool>
    
    /// Get similar vectors to a given vector ID
    abstract member GetSimilarAsync: vectorId: string * limit: int -> Task<SearchResult[]>
    
    /// Create a new index with specified dimensions
    abstract member CreateIndexAsync: name: string * dimensions: int -> Task<string>
    
    /// Get index statistics and information
    abstract member GetIndexInfoAsync: indexName: string -> Task<IndexInfo>

/// LLM service API for text generation and processing
and ILlmServiceApi =
    /// Complete text using specified model
    abstract member CompleteAsync: prompt: string * model: string -> Task<string>
    
    /// Chat completion with message history
    abstract member ChatAsync: messages: ChatMessage[] * model: string -> Task<string>
    
    /// Generate embeddings for text
    abstract member EmbedAsync: text: string -> Task<float[]>
    
    /// List available models
    abstract member ListModelsAsync: unit -> Task<ModelInfo[]>
    
    /// Set generation parameters
    abstract member SetTemperature: temperature: float -> unit
    abstract member SetMaxTokens: maxTokens: int -> unit

/// Metascript execution API
and IMetascriptRunnerApi =
    /// Execute a metascript file
    abstract member ExecuteAsync: scriptPath: string -> Task<ExecutionResult>
    
    /// Execute metascript content directly
    abstract member ExecuteContentAsync: content: string -> Task<ExecutionResult>
    
    /// Parse metascript content into AST
    abstract member ParseAsync: content: string -> Task<MetascriptAst>
    
    /// Validate metascript syntax and semantics
    abstract member ValidateAsync: content: string -> Task<ValidationResult[]>
    
    /// Get current execution variables
    abstract member GetVariables: unit -> Map<string, obj>
    
    /// Set execution variable
    abstract member SetVariable: name: string * value: obj -> unit

/// Agent coordination API
and IAgentCoordinatorApi =
    /// Spawn a new agent with configuration
    abstract member SpawnAsync: agentType: string * config: AgentConfig -> Task<string>
    
    /// Send message to an agent
    abstract member SendMessageAsync: agentId: string * message: string -> Task<string>
    
    /// Get agent status and information
    abstract member GetStatusAsync: agentId: string -> Task<AgentStatus>
    
    /// Terminate an agent
    abstract member TerminateAsync: agentId: string -> Task<bool>
    
    /// List all active agents
    abstract member ListActiveAsync: unit -> Task<AgentInfo[]>
    
    /// Broadcast message to all agents of a type
    abstract member BroadcastAsync: agentType: string * message: string -> Task<string[]>

/// CUDA engine API for GPU-accelerated computations
and ICudaEngineApi =
    /// Execute CUDA kernel with parameters
    abstract member ExecuteKernelAsync: kernelName: string * parameters: obj[] -> Task<obj>
    
    /// Get CUDA device information
    abstract member GetDeviceInfoAsync: unit -> Task<CudaDeviceInfo>
    
    /// Allocate GPU memory
    abstract member AllocateMemoryAsync: sizeBytes: int -> Task<IntPtr>
    
    /// Free GPU memory
    abstract member FreeMemoryAsync: pointer: IntPtr -> Task<bool>
    
    /// Check if CUDA is available
    abstract member IsAvailable: bool

/// File system API with security constraints
and IFileSystemApi =
    /// Read file content (subject to security policy)
    abstract member ReadFileAsync: path: string -> Task<string>
    
    /// Write file content (subject to security policy)
    abstract member WriteFileAsync: path: string * content: string -> Task<bool>
    
    /// List files in directory (subject to security policy)
    abstract member ListFilesAsync: directory: string -> Task<FileInfo[]>
    
    /// Create directory (subject to security policy)
    abstract member CreateDirectoryAsync: path: string -> Task<bool>
    
    /// Get file metadata
    abstract member GetMetadataAsync: path: string -> Task<FileMetadata>
    
    /// Check if file exists
    abstract member ExistsAsync: path: string -> Task<bool>

/// Web search API
and IWebSearchApi =
    /// Search the web for content
    abstract member SearchAsync: query: string * limit: int -> Task<SearchResult[]>
    
    /// Fetch content from URL
    abstract member FetchAsync: url: string -> Task<WebContent>
    
    /// Post data to URL
    abstract member PostAsync: url: string * data: string -> Task<WebResponse>
    
    /// Get HTTP headers for URL
    abstract member GetHeadersAsync: url: string -> Task<Map<string, string>>

/// GitHub API service
and IGitHubApiService =
    /// Get repository information
    abstract member GetRepositoryAsync: owner: string * repo: string -> Task<RepositoryInfo>
    
    /// Create an issue
    abstract member CreateIssueAsync: repo: string * title: string * body: string -> Task<string>
    
    /// List pull requests
    abstract member ListPullRequestsAsync: repo: string -> Task<PullRequest[]>
    
    /// Get file content from repository
    abstract member GetFileContentAsync: repo: string * path: string -> Task<string>

/// Execution context API for tracing and metadata
and IExecutionContextApi =
    /// Log an event with specified level
    abstract member LogEvent: level: LogLevel * message: string -> unit
    
    /// Start a trace span
    abstract member StartTrace: name: string -> TraceId
    
    /// End a trace span
    abstract member EndTrace: traceId: TraceId -> TraceResult
    
    /// Add metadata to current execution
    abstract member AddMetadata: key: string * value: obj -> unit
    
    /// Get current execution ID
    abstract member ExecutionId: string
    
    /// Get current metascript path
    abstract member CurrentMetascript: string

/// Supporting types and data structures
and SearchResult = {
    Title: string
    Content: string
    Score: float
    Metadata: Map<string, string>
    VectorId: string option
}

and IndexInfo = {
    Name: string
    Dimensions: int
    VectorCount: int
    CreatedAt: DateTime
}

and ChatMessage = {
    Role: string
    Content: string
    Metadata: Map<string, string> option
}

and ModelInfo = {
    Name: string
    Provider: string
    MaxTokens: int
    SupportsChat: bool
    SupportsCompletion: bool
}

and ExecutionResult = {
    Success: bool
    Output: string
    Errors: string[]
    ExecutionTime: TimeSpan
    Metadata: Map<string, obj>
}

and MetascriptAst = {
    Blocks: MetascriptBlock[]
    Variables: Map<string, obj>
    Metadata: Map<string, string>
}

and MetascriptBlock = {
    BlockType: string
    Content: string
    LineNumber: int
}

and ValidationResult = {
    IsValid: bool
    Message: string
    LineNumber: int option
    Severity: ValidationSeverity
}

and ValidationSeverity = Error | Warning | Info

and AgentConfig = {
    Type: string
    Parameters: Map<string, obj>
    ResourceLimits: ResourceLimits option
}

and AgentStatus = {
    Id: string
    Type: string
    State: AgentState
    LastActivity: DateTime
    MessageCount: int
}

and AgentState = Starting | Running | Idle | Stopping | Stopped | Error

and AgentInfo = {
    Id: string
    Type: string
    Status: AgentStatus
    CreatedAt: DateTime
}

and CudaDeviceInfo = {
    DeviceId: int
    Name: string
    TotalMemory: int64
    FreeMemory: int64
    ComputeCapability: string
}

and FileInfo = {
    Name: string
    Path: string
    Size: int64
    CreatedAt: DateTime
    ModifiedAt: DateTime
    IsDirectory: bool
}

and FileMetadata = {
    Path: string
    Size: int64
    CreatedAt: DateTime
    ModifiedAt: DateTime
    Permissions: string
    Checksum: string
}

and WebContent = {
    Url: string
    Content: string
    Headers: Map<string, string>
    StatusCode: int
}

and WebResponse = {
    StatusCode: int
    Content: string
    Headers: Map<string, string>
}

and RepositoryInfo = {
    Name: string
    Owner: string
    Description: string
    Stars: int
    Forks: int
    Language: string
}

and PullRequest = {
    Id: int
    Title: string
    Author: string
    State: string
    CreatedAt: DateTime
}

and LogLevel = Debug | Info | Warning | Error | Critical

and TraceId = string

and TraceResult = {
    TraceId: TraceId
    Duration: TimeSpan
    Events: TraceEvent[]
}

and TraceEvent = {
    Timestamp: DateTime
    Level: LogLevel
    Message: string
    Metadata: Map<string, obj>
}

and ResourceLimits = {
    MaxMemoryMB: int
    MaxCpuTimeMs: int
    MaxNetworkRequests: int
    MaxFileOperations: int
}
