namespace TarsEngine.FSharp.Cli.Integration

open System
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedTypes

/// Unified Agent Interfaces - Common contracts for all TARS agent types
module UnifiedAgentInterfaces =
    
    /// Unified agent identifier
    type UnifiedAgentId = UnifiedAgentId of Guid
    
    /// Unified agent status
    type UnifiedAgentStatus =
        | Initializing
        | Ready
        | Busy of taskId: string
        | Paused
        | Stopping
        | Stopped
        | Failed of error: TarsError
        | Maintenance
    
    /// Unified agent capability
    type UnifiedAgentCapability = {
        Name: string
        Description: string
        InputTypes: string list
        OutputTypes: string list
        RequiredResources: string list
        EstimatedComplexity: ComputationComplexity
        CanBatch: bool
        MaxConcurrency: int
    }
    
    /// Unified agent message
    type UnifiedAgentMessage = {
        MessageId: string
        FromAgent: UnifiedAgentId option
        ToAgent: UnifiedAgentId option
        MessageType: string
        Payload: obj
        Priority: MessagePriority
        CreatedAt: DateTime
        ExpiresAt: DateTime option
        CorrelationId: string
        ReplyTo: string option
        RequiresResponse: bool
        Context: Map<string, obj>
    }
    
    /// Unified agent task
    type UnifiedAgentTask = {
        TaskId: string
        TaskType: string
        Description: string
        Input: obj
        Priority: MessagePriority
        CreatedAt: DateTime
        Deadline: DateTime option
        RequiredCapabilities: string list
        Context: TarsOperationContext
        Dependencies: string list
        ExpectedOutput: string option
    }
    
    /// Unified agent metrics
    type UnifiedAgentMetrics = {
        TasksCompleted: int64
        TasksFailed: int64
        AverageResponseTime: TimeSpan
        CurrentLoad: float
        MemoryUsage: int64
        CpuUsage: float
        LastActivity: DateTime
        Uptime: TimeSpan
        SuccessRate: float
        ThroughputPerHour: float
    }
    
    /// Unified agent configuration
    type UnifiedAgentConfig = {
        AgentId: UnifiedAgentId
        Name: string
        Description: string
        AgentType: string
        Version: string
        Capabilities: UnifiedAgentCapability list
        MaxConcurrentTasks: int
        TimeoutMs: int
        RetryPolicy: RetryPolicy
        HealthCheckInterval: TimeSpan
        LogLevel: Microsoft.Extensions.Logging.LogLevel
        CustomSettings: Map<string, obj>
    }
    
    /// Retry policy for agent operations
    and RetryPolicy = {
        MaxRetries: int
        BaseDelayMs: int
        BackoffMultiplier: float
        MaxDelayMs: int
        RetryableErrors: string list
    }
    
    /// Unified agent interface - All TARS agents must implement this
    type IUnifiedAgent =
        inherit ITarsComponent
        
        /// Agent configuration
        abstract member Config: UnifiedAgentConfig
        
        /// Current agent status
        abstract member Status: UnifiedAgentStatus
        
        /// Agent capabilities
        abstract member Capabilities: UnifiedAgentCapability list
        
        /// Current metrics
        abstract member Metrics: UnifiedAgentMetrics
        
        /// Initialize the agent
        abstract member InitializeAsync: cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Start the agent
        abstract member StartAsync: cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Stop the agent gracefully
        abstract member StopAsync: cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Pause the agent
        abstract member PauseAsync: cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Resume the agent
        abstract member ResumeAsync: cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Process a task
        abstract member ProcessTaskAsync: task: UnifiedAgentTask * cancellationToken: CancellationToken -> Task<TarsResult<obj>>
        
        /// Send a message to this agent
        abstract member SendMessageAsync: message: UnifiedAgentMessage * cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Check if agent can handle a specific task type
        abstract member CanHandle: taskType: string -> bool
        
        /// Get estimated processing time for a task
        abstract member EstimateProcessingTime: task: UnifiedAgentTask -> TimeSpan
        
        /// Perform health check
        abstract member HealthCheckAsync: cancellationToken: CancellationToken -> Task<TarsResult<Map<string, obj>>>
    
    /// Unified agent team interface
    type IUnifiedAgentTeam =
        inherit ITarsComponent
        
        /// Team name
        abstract member Name: string
        
        /// Team members
        abstract member Members: IUnifiedAgent list
        
        /// Team leader (optional)
        abstract member Leader: IUnifiedAgent option
        
        /// Team capabilities (combined from all members)
        abstract member TeamCapabilities: UnifiedAgentCapability list
        
        /// Add agent to team
        abstract member AddMemberAsync: agent: IUnifiedAgent * cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Remove agent from team
        abstract member RemoveMemberAsync: agentId: UnifiedAgentId * cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Assign task to best suited team member
        abstract member AssignTaskAsync: task: UnifiedAgentTask * cancellationToken: CancellationToken -> Task<TarsResult<IUnifiedAgent>>
        
        /// Execute task collaboratively
        abstract member ExecuteTeamTaskAsync: task: UnifiedAgentTask * cancellationToken: CancellationToken -> Task<TarsResult<obj>>
        
        /// Get team metrics
        abstract member GetTeamMetrics: unit -> UnifiedAgentMetrics
    
    /// Unified agent coordinator interface
    type IUnifiedAgentCoordinator =
        inherit ITarsComponent
        
        /// Register an agent
        abstract member RegisterAgentAsync: agent: IUnifiedAgent * cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Unregister an agent
        abstract member UnregisterAgentAsync: agentId: UnifiedAgentId * cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Find best agent for a task
        abstract member FindBestAgentAsync: task: UnifiedAgentTask * cancellationToken: CancellationToken -> Task<TarsResult<IUnifiedAgent>>
        
        /// Route message to appropriate agent
        abstract member RouteMessageAsync: message: UnifiedAgentMessage * cancellationToken: CancellationToken -> Task<TarsResult<unit>>
        
        /// Execute task with automatic agent selection
        abstract member ExecuteTaskAsync: task: UnifiedAgentTask * cancellationToken: CancellationToken -> Task<TarsResult<obj>>
        
        /// Get all registered agents
        abstract member GetRegisteredAgents: unit -> IUnifiedAgent list
        
        /// Get system-wide agent metrics
        abstract member GetSystemMetrics: unit -> Map<string, obj>
        
        /// Perform health check on all agents
        abstract member HealthCheckAllAsync: cancellationToken: CancellationToken -> Task<TarsResult<Map<UnifiedAgentId, Map<string, obj>>>>
    
    /// Agent factory interface for creating different agent types
    type IUnifiedAgentFactory =
        /// Create agent from configuration
        abstract member CreateAgentAsync: config: UnifiedAgentConfig * cancellationToken: CancellationToken -> Task<TarsResult<IUnifiedAgent>>
        
        /// Create team from configuration
        abstract member CreateTeamAsync: name: string * members: IUnifiedAgent list * leader: IUnifiedAgent option * cancellationToken: CancellationToken -> Task<TarsResult<IUnifiedAgentTeam>>
        
        /// Get supported agent types
        abstract member GetSupportedAgentTypes: unit -> string list
        
        /// Validate agent configuration
        abstract member ValidateConfig: config: UnifiedAgentConfig -> TarsResult<unit>
    
    /// Utility functions for unified agents
    module UnifiedAgentUtils =
        
        /// Generate unique agent ID
        let generateAgentId() = UnifiedAgentId (Guid.NewGuid())
        
        /// Create default retry policy
        let defaultRetryPolicy = {
            MaxRetries = 3
            BaseDelayMs = 1000
            BackoffMultiplier = 2.0
            MaxDelayMs = 30000
            RetryableErrors = ["NetworkError"; "TimeoutError"; "TemporaryFailure"]
        }
        
        /// Create default agent metrics
        let createDefaultMetrics() = {
            TasksCompleted = 0L
            TasksFailed = 0L
            AverageResponseTime = TimeSpan.Zero
            CurrentLoad = 0.0
            MemoryUsage = 0L
            CpuUsage = 0.0
            LastActivity = DateTime.Now
            Uptime = TimeSpan.Zero
            SuccessRate = 1.0
            ThroughputPerHour = 0.0
        }
        
        /// Check if agent can handle task based on capabilities
        let canAgentHandleTask (agent: IUnifiedAgent) (task: UnifiedAgentTask) =
            task.RequiredCapabilities
            |> List.forall (fun required ->
                agent.Capabilities
                |> List.exists (fun cap -> cap.Name = required))
        
        /// Calculate agent load score for task assignment
        let calculateAgentLoadScore (agent: IUnifiedAgent) (task: UnifiedAgentTask) =
            let metrics = agent.Metrics
            let baseScore = 1.0 - metrics.CurrentLoad
            let successRateBonus = metrics.SuccessRate * 0.2
            let responseTimeBonus = 
                if metrics.AverageResponseTime.TotalSeconds > 0.0 then
                    1.0 / metrics.AverageResponseTime.TotalSeconds * 0.1
                else 0.1
            
            baseScore + successRateBonus + responseTimeBonus
