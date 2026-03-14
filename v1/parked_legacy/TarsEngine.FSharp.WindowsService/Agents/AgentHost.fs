namespace TarsEngine.FSharp.WindowsService.Agents

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Agent execution context for isolation and resource management
/// </summary>
type AgentExecutionContext = {
    AgentId: string
    AgentName: string
    AgentType: string
    StartTime: DateTime
    CancellationToken: CancellationToken
    ServiceProvider: IServiceProvider
    Logger: ILogger
    Configuration: Map<string, obj>
    ResourceLimits: AgentResourceLimits
}

/// <summary>
/// Resource limits for agent execution
/// </summary>
and AgentResourceLimits = {
    MaxMemoryMB: int
    MaxCpuPercent: float
    MaxExecutionTime: TimeSpan
    MaxFileHandles: int
    MaxNetworkConnections: int
}

/// <summary>
/// Agent execution statistics
/// </summary>
type AgentExecutionStats = {
    AgentId: string
    StartTime: DateTime
    LastActivity: DateTime
    TotalExecutionTime: TimeSpan
    MemoryUsageMB: float
    CpuUsagePercent: float
    TasksProcessed: int64
    ErrorsEncountered: int64
    RestartCount: int
}

/// <summary>
/// Agent host for managing agent lifecycle and isolation
/// </summary>
type AgentHost(logger: ILogger<AgentHost>, serviceProvider: IServiceProvider) =
    
    let hostedAgents = ConcurrentDictionary<string, AgentExecutionContext>()
    let agentStats = ConcurrentDictionary<string, AgentExecutionStats>()
    let agentTasks = ConcurrentDictionary<string, Task>()
    let agentCancellationTokens = ConcurrentDictionary<string, CancellationTokenSource>()
    
    let defaultResourceLimits = {
        MaxMemoryMB = 512
        MaxCpuPercent = 25.0
        MaxExecutionTime = TimeSpan.FromHours(24.0)
        MaxFileHandles = 100
        MaxNetworkConnections = 50
    }
    
    /// Start hosting an agent
    member this.StartAgentAsync(agent: IAgent, config: AgentConfig) = task {
        try
            logger.LogInformation($"Starting agent host for: {agent.Name} ({agent.Id})")
            
            // Check if agent is already hosted
            if hostedAgents.ContainsKey(agent.Id) then
                let error = $"Agent {agent.Id} is already being hosted"
                logger.LogWarning(error)
                return Error error
            else
                // Create cancellation token for this agent
                let cancellationTokenSource = new CancellationTokenSource()
                agentCancellationTokens.[agent.Id] <- cancellationTokenSource
                
                // Create execution context
                let context = {
                    AgentId = agent.Id
                    AgentName = agent.Name
                    AgentType = agent.Type
                    StartTime = DateTime.UtcNow
                    CancellationToken = cancellationTokenSource.Token
                    ServiceProvider = serviceProvider
                    Logger = logger.CreateLogger($"Agent.{agent.Name}")
                    Configuration = config.Configuration
                    ResourceLimits = this.CreateResourceLimits(config)
                }
                
                // Initialize agent statistics
                let stats = {
                    AgentId = agent.Id
                    StartTime = DateTime.UtcNow
                    LastActivity = DateTime.UtcNow
                    TotalExecutionTime = TimeSpan.Zero
                    MemoryUsageMB = 0.0
                    CpuUsagePercent = 0.0
                    TasksProcessed = 0L
                    ErrorsEncountered = 0L
                    RestartCount = 0
                }
                
                hostedAgents.[agent.Id] <- context
                agentStats.[agent.Id] <- stats
                
                // Start agent execution task
                let agentTask = this.ExecuteAgentAsync(agent, context)
                agentTasks.[agent.Id] <- agentTask
                
                logger.LogInformation($"Agent host started successfully for: {agent.Name} ({agent.Id})")
                return Ok context
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to start agent host for: {agent.Name}")
            return Error ex.Message
    }
    
    /// Stop hosting an agent
    member this.StopAgentAsync(agentId: string) = task {
        try
            logger.LogInformation($"Stopping agent host for: {agentId}")
            
            match hostedAgents.TryGetValue(agentId) with
            | true, context ->
                // Cancel the agent
                match agentCancellationTokens.TryGetValue(agentId) with
                | true, cancellationTokenSource ->
                    cancellationTokenSource.Cancel()
                    
                    // Wait for agent task to complete (with timeout)
                    match agentTasks.TryGetValue(agentId) with
                    | true, agentTask ->
                        try
                            do! agentTask.WaitAsync(TimeSpan.FromSeconds(30.0))
                        with
                        | :? TimeoutException ->
                            logger.LogWarning($"Agent {agentId} did not stop gracefully within timeout")
                        | ex ->
                            logger.LogWarning(ex, $"Error waiting for agent {agentId} to stop")
                    | false, _ -> ()
                    
                    // Cleanup resources
                    cancellationTokenSource.Dispose()
                    agentCancellationTokens.TryRemove(agentId) |> ignore
                    agentTasks.TryRemove(agentId) |> ignore
                    hostedAgents.TryRemove(agentId) |> ignore
                    
                    logger.LogInformation($"Agent host stopped successfully for: {agentId}")
                    return Ok ()
                    
                | false, _ ->
                    let error = $"Cancellation token not found for agent: {agentId}"
                    logger.LogWarning(error)
                    return Error error
            
            | false, _ ->
                let error = $"Agent {agentId} is not currently hosted"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to stop agent host for: {agentId}")
            return Error ex.Message
    }
    
    /// Execute agent with monitoring and resource management
    member private this.ExecuteAgentAsync(agent: IAgent, context: AgentExecutionContext) = task {
        try
            logger.LogInformation($"Starting agent execution: {agent.Name} ({agent.Id})")
            
            // Configure agent
            do! agent.ConfigureAsync(context.Configuration, context.CancellationToken)
            
            // Start agent
            do! agent.StartAsync(context.CancellationToken)
            
            // Monitor agent execution
            do! this.MonitorAgentExecutionAsync(agent, context)
            
        with
        | :? OperationCanceledException ->
            logger.LogInformation($"Agent execution cancelled: {agent.Name} ({agent.Id})")
        | ex ->
            logger.LogError(ex, $"Agent execution failed: {agent.Name} ({agent.Id})")
            
            // Update error statistics
            match agentStats.TryGetValue(agent.Id) with
            | true, stats ->
                let updatedStats = { stats with ErrorsEncountered = stats.ErrorsEncountered + 1L }
                agentStats.[agent.Id] <- updatedStats
            | false, _ -> ()
        
        finally
            try
                // Stop agent gracefully
                do! agent.StopAsync(CancellationToken.None)
                logger.LogInformation($"Agent stopped gracefully: {agent.Name} ({agent.Id})")
            with
            | ex ->
                logger.LogWarning(ex, $"Error stopping agent: {agent.Name} ({agent.Id})")
    }
    
    /// Monitor agent execution and collect statistics
    member private this.MonitorAgentExecutionAsync(agent: IAgent, context: AgentExecutionContext) = task {
        let monitoringInterval = TimeSpan.FromSeconds(30.0)
        
        while not context.CancellationToken.IsCancellationRequested do
            try
                // Collect agent health and metrics
                let! health = agent.GetHealthAsync(context.CancellationToken)
                let! metrics = agent.GetMetricsAsync(context.CancellationToken)
                
                // Update statistics
                this.UpdateAgentStatistics(agent.Id, health, metrics)
                
                // Check resource limits
                this.CheckResourceLimits(agent.Id, context.ResourceLimits)
                
                // Wait for next monitoring cycle
                do! Task.Delay(monitoringInterval, context.CancellationToken)
                
            with
            | :? OperationCanceledException -> 
                break
            | ex ->
                logger.LogWarning(ex, $"Error monitoring agent: {agent.Name} ({agent.Id})")
                do! Task.Delay(monitoringInterval, context.CancellationToken)
    }
    
    /// Update agent statistics
    member private this.UpdateAgentStatistics(agentId: string, health: AgentHealthStatus, metrics: Map<string, obj>) =
        match agentStats.TryGetValue(agentId) with
        | true, stats ->
            let memoryUsage = 
                metrics.TryFind("MemoryUsageMB") 
                |> Option.map (fun v -> v :?> float) 
                |> Option.defaultValue stats.MemoryUsageMB
            
            let cpuUsage = 
                metrics.TryFind("CpuUsagePercent") 
                |> Option.map (fun v -> v :?> float) 
                |> Option.defaultValue stats.CpuUsagePercent
            
            let tasksProcessed = 
                metrics.TryFind("TasksProcessed") 
                |> Option.map (fun v -> v :?> int64) 
                |> Option.defaultValue stats.TasksProcessed
            
            let updatedStats = {
                stats with
                    LastActivity = DateTime.UtcNow
                    TotalExecutionTime = DateTime.UtcNow - stats.StartTime
                    MemoryUsageMB = memoryUsage
                    CpuUsagePercent = cpuUsage
                    TasksProcessed = tasksProcessed
            }
            
            agentStats.[agentId] <- updatedStats
            
        | false, _ ->
            logger.LogWarning($"Statistics not found for agent: {agentId}")
    
    /// Check resource limits and take action if exceeded
    member private this.CheckResourceLimits(agentId: string, limits: AgentResourceLimits) =
        match agentStats.TryGetValue(agentId) with
        | true, stats ->
            let violations = ResizeArray<string>()
            
            if stats.MemoryUsageMB > float limits.MaxMemoryMB then
                violations.Add($"Memory usage ({stats.MemoryUsageMB:F1}MB) exceeds limit ({limits.MaxMemoryMB}MB)")
            
            if stats.CpuUsagePercent > limits.MaxCpuPercent then
                violations.Add($"CPU usage ({stats.CpuUsagePercent:F1}%) exceeds limit ({limits.MaxCpuPercent}%)")
            
            if stats.TotalExecutionTime > limits.MaxExecutionTime then
                violations.Add($"Execution time ({stats.TotalExecutionTime}) exceeds limit ({limits.MaxExecutionTime})")
            
            if violations.Count > 0 then
                logger.LogWarning($"Agent {agentId} resource limit violations: {String.Join("; ", violations)}")
                // In a production system, we might take corrective action here
        
        | false, _ -> ()
    
    /// Create resource limits from configuration
    member private this.CreateResourceLimits(config: AgentConfig) =
        let getConfigValue<'T> (key: string) (defaultValue: 'T) =
            config.Configuration.TryFind(key)
            |> Option.map (fun v -> v :?> 'T)
            |> Option.defaultValue defaultValue
        
        {
            MaxMemoryMB = getConfigValue "MaxMemoryMB" defaultResourceLimits.MaxMemoryMB
            MaxCpuPercent = getConfigValue "MaxCpuPercent" defaultResourceLimits.MaxCpuPercent
            MaxExecutionTime = getConfigValue "MaxExecutionTime" defaultResourceLimits.MaxExecutionTime
            MaxFileHandles = getConfigValue "MaxFileHandles" defaultResourceLimits.MaxFileHandles
            MaxNetworkConnections = getConfigValue "MaxNetworkConnections" defaultResourceLimits.MaxNetworkConnections
        }
    
    /// Get hosted agent context
    member this.GetAgentContext(agentId: string) =
        match hostedAgents.TryGetValue(agentId) with
        | true, context -> Some context
        | false, _ -> None
    
    /// Get agent statistics
    member this.GetAgentStatistics(agentId: string) =
        match agentStats.TryGetValue(agentId) with
        | true, stats -> Some stats
        | false, _ -> None
    
    /// Get all hosted agents
    member this.GetHostedAgents() =
        hostedAgents.Keys |> List.ofSeq
    
    /// Get host statistics
    member this.GetHostStatistics() =
        let totalAgents = hostedAgents.Count
        let totalMemoryUsage = agentStats.Values |> Seq.sumBy (fun s -> s.MemoryUsageMB)
        let averageCpuUsage = 
            if agentStats.Count > 0 then
                agentStats.Values |> Seq.averageBy (fun s -> s.CpuUsagePercent)
            else 0.0
        
        let totalTasksProcessed = agentStats.Values |> Seq.sumBy (fun s -> s.TasksProcessed)
        let totalErrors = agentStats.Values |> Seq.sumBy (fun s -> s.ErrorsEncountered)
        
        {|
            TotalHostedAgents = totalAgents
            TotalMemoryUsageMB = totalMemoryUsage
            AverageCpuUsagePercent = averageCpuUsage
            TotalTasksProcessed = totalTasksProcessed
            TotalErrorsEncountered = totalErrors
            AgentsByType = 
                hostedAgents.Values 
                |> Seq.groupBy (fun c -> c.AgentType)
                |> Seq.map (fun (t, agents) -> (t, agents |> Seq.length))
                |> Map.ofSeq
        |}
    
    /// Restart an agent
    member this.RestartAgentAsync(agentId: string, agent: IAgent, config: AgentConfig) = task {
        logger.LogInformation($"Restarting agent: {agentId}")
        
        // Stop the current agent
        let! stopResult = this.StopAgentAsync(agentId)
        match stopResult with
        | Ok () ->
            // Update restart count
            match agentStats.TryGetValue(agentId) with
            | true, stats ->
                let updatedStats = { stats with RestartCount = stats.RestartCount + 1 }
                agentStats.[agentId] <- updatedStats
            | false, _ -> ()
            
            // Start the agent again
            return! this.StartAgentAsync(agent, config)
        | Error error ->
            logger.LogError($"Failed to stop agent {agentId} for restart: {error}")
            return Error error
    }
    
    /// Dispose resources
    interface IDisposable with
        member this.Dispose() =
            // Cancel all agents
            for kvp in agentCancellationTokens do
                try
                    kvp.Value.Cancel()
                    kvp.Value.Dispose()
                with
                | ex -> logger.LogWarning(ex, $"Error disposing cancellation token for agent: {kvp.Key}")
            
            agentCancellationTokens.Clear()
            hostedAgents.Clear()
            agentStats.Clear()
            agentTasks.Clear()
