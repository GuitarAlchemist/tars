namespace TarsEngine.FSharp.Cli.Integration

open System
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedTypes
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentInterfaces
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentRegistry

// Ensure task computation expression is available
open Microsoft.FSharp.Control

/// Unified Agent Coordination System - Main orchestrator for all TARS agents
module UnifiedAgentSystem =
    
    /// Agent coordination configuration
    type AgentCoordinationConfig = {
        MaxConcurrentTasks: int
        TaskTimeoutMs: int
        LoadBalancingStrategy: LoadBalancingStrategy
        MessageRetentionHours: int
        HealthCheckIntervalMs: int
        AutoRetryFailedTasks: bool
        MaxRetryAttempts: int
        EnableTaskPrioritization: bool
    }
    
    /// Load balancing strategies
    and LoadBalancingStrategy =
        | RoundRobin
        | LeastLoaded
        | CapabilityBased
        | PerformanceBased
        | Random
    
    /// Task execution context
    type TaskExecutionContext = {
        Task: UnifiedAgentTask
        AssignedAgent: IUnifiedAgent option
        StartTime: DateTime option
        EndTime: DateTime option
        RetryCount: int
        Status: TaskExecutionStatus
        Result: obj option
        Error: TarsError option
    }
    
    /// Task execution status
    and TaskExecutionStatus =
        | Queued
        | Assigned
        | Running
        | Completed
        | Failed
        | Cancelled
        | Retrying
    
    /// Main unified agent coordinator implementation
    type UnifiedAgentCoordinator(config: AgentCoordinationConfig, registry: UnifiedAgentRegistry, logger: ITarsLogger) =
        let activeTaskContexts = ConcurrentDictionary<string, TaskExecutionContext>()
        let taskQueue = ConcurrentQueue<UnifiedAgentTask>()
        let messageQueue = ConcurrentQueue<UnifiedAgentMessage>()
        let mutable isRunning = false
        let mutable isDisposed = false
        let cancellationTokenSource = new CancellationTokenSource()
        
        /// Task processing loop (initialized later)
        let mutable taskProcessingTask : Task option = None

        /// Message processing loop (initialized later)
        let mutable messageProcessingTask : Task option = None
        
        /// Start the processing loops
        member private this.StartProcessingLoops() =
            // Task processing loop
            let taskLoop =
                task {
                    while not cancellationTokenSource.Token.IsCancellationRequested && isRunning do
                        try
                            match taskQueue.TryDequeue() with
                            | true, task ->
                                let! _ = this.ProcessQueuedTaskAsync(task, cancellationTokenSource.Token)
                                ()
                            | false, _ ->
                                do! Task.Delay(100, cancellationTokenSource.Token)
                        with
                        | :? OperationCanceledException -> ()
                        | ex ->
                            logger.LogError(generateCorrelationId(), ExecutionError ("Task processing loop error", Some ex), ex)
                            do! Task.Delay(1000, cancellationTokenSource.Token)
                }

            // Message processing loop
            let messageLoop =
                task {
                    while not cancellationTokenSource.Token.IsCancellationRequested && isRunning do
                        try
                            match messageQueue.TryDequeue() with
                            | true, message ->
                                let! _ = this.ProcessQueuedMessageAsync(message, cancellationTokenSource.Token)
                                ()
                            | false, _ ->
                                do! Task.Delay(100, cancellationTokenSource.Token)
                        with
                        | :? OperationCanceledException -> ()
                        | ex ->
                            logger.LogError(generateCorrelationId(), ExecutionError ("Message processing loop error", Some ex), ex)
                            do! Task.Delay(1000, cancellationTokenSource.Token)
                }

            taskProcessingTask <- Some (Task.Run(fun () -> taskLoop :> Task))
            messageProcessingTask <- Some (Task.Run(fun () -> messageLoop :> Task))

        /// Start the coordinator
        member this.StartAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, "Starting Unified Agent Coordinator")

                    isRunning <- true

                    // Start processing loops
                    this.StartProcessingLoops()

                    logger.LogInformation(correlationId, "Unified Agent Coordinator started successfully")
                    return Success ((), Map [("correlationId", box correlationId); ("timestamp", box DateTime.Now)])

                with
                | ex ->
                    let error = ExecutionError ("Failed to start agent coordinator", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Stop the coordinator
        member this.StopAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, "Stopping Unified Agent Coordinator")
                    
                    isRunning <- false
                    cancellationTokenSource.Cancel()
                    
                    // Wait for processing loops to complete
                    do! Task.Delay(1000, cancellationToken)
                    
                    logger.LogInformation(correlationId, "Unified Agent Coordinator stopped successfully")
                    return Success ((), Map [("correlationId", box correlationId); ("timestamp", box DateTime.Now)])
                
                with
                | ex ->
                    let error = ExecutionError ("Failed to stop agent coordinator", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Find best agent for a task using the configured load balancing strategy
        member this.FindBestAgentAsync(task: UnifiedAgentTask, cancellationToken: CancellationToken) =
            async {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogDebug(correlationId, $"Finding best agent for task: {task.TaskType}")

                    // Find agents with required capabilities
                    let filter = {
                        createDefaultFilter() with
                            Capabilities = Some task.RequiredCapabilities
                            Status = Some [Ready]
                    }

                    let! agentsResult = registry.FindAgentsAsync(filter, cancellationToken) |> Async.AwaitTask

                    match agentsResult with
                    | Success (agents, _) when agents.Length > 0 ->
                        let bestAgent =
                            match config.LoadBalancingStrategy with
                            | LeastLoaded ->
                                agents
                                |> List.minBy (fun agent -> agent.Metrics.CurrentLoad)
                            | PerformanceBased ->
                                agents
                                |> List.maxBy (fun agent ->
                                    UnifiedAgentUtils.calculateAgentLoadScore agent task)
                            | CapabilityBased ->
                                agents
                                |> List.maxBy (fun agent ->
                                    let matchingCaps =
                                        task.RequiredCapabilities
                                        |> List.filter (fun req ->
                                            agent.Capabilities |> List.exists (fun cap -> cap.Name = req))
                                    float matchingCaps.Length / float task.RequiredCapabilities.Length)
                            | RoundRobin ->
                                // Simple round-robin based on task ID hash
                                let index = Math.Abs(task.TaskId.GetHashCode()) % agents.Length
                                agents.[index]
                            | Random ->
                                let random = System.Random()
                                agents.[0 // HONEST: Cannot generate without real measurement]

                        let taskType = if isNull task.TaskType then "unknown" else task.TaskType
                        logger.LogDebug(correlationId, $"Selected agent: {bestAgent.Config.Name} for task: {taskType}")
                        return Success (bestAgent, Map [("agentId", box bestAgent.Config.AgentId); ("taskId", box task.TaskId)])

                    | Success ([], _) ->
                        let taskType: string = if isNull task.TaskType then "unknown" else task.TaskType
                        let error = ValidationError ($"No agents available for task type: {taskType}", Map [("taskType", box taskType)])
                        return Failure (error, correlationId)

                    | Failure (error, corrId) ->
                        return Failure (error, corrId)

                with
                | ex ->
                    let taskType: string = if isNull task.TaskType then "unknown" else task.TaskType
                    let error = ExecutionError ($"Failed to find best agent for task: {taskType}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            } |> Async.StartAsTask
        
        /// Execute task with automatic agent selection and retry logic
        member this.ExecuteTaskAsync(task: UnifiedAgentTask, cancellationToken: CancellationToken) =
            try
                async {
                    let correlationId = generateCorrelationId()
                    let taskType = if isNull task.TaskType then "unknown" else task.TaskType
                    logger.LogInformation(correlationId, $"Executing task: {task.TaskId} ({taskType})")

                    // Create task execution context
                    let context = {
                        Task = task
                        AssignedAgent = None
                        StartTime = Some DateTime.Now
                        EndTime = None
                        RetryCount = 0
                        Status = Queued
                        Result = None
                        Error = None
                    }

                    activeTaskContexts.[task.TaskId] <- context

                    let rec executeWithRetry (currentContext: TaskExecutionContext) =
                        async {
                            // Find best agent
                            let! agentResult = this.FindBestAgentAsync(task, cancellationToken) |> Async.AwaitTask

                            match agentResult with
                            | Success (agent, _) ->
                                let updatedContext = {
                                    currentContext with
                                        AssignedAgent = Some agent
                                        Status = Assigned
                                }
                                activeTaskContexts.[task.TaskId] <- updatedContext

                                // Execute task on agent
                                let! taskResult = agent.ProcessTaskAsync(task, cancellationToken) |> Async.AwaitTask

                                match taskResult with
                                | Success (result, _) ->
                                    let finalContext = {
                                        updatedContext with
                                            Status = Completed
                                            EndTime = Some DateTime.Now
                                            Result = Some result
                                    }
                                    activeTaskContexts.[task.TaskId] <- finalContext

                                    logger.LogInformation(correlationId, $"Task completed successfully: {task.TaskId}")
                                    return Success (result, Map [("taskId", box task.TaskId); ("agentId", box agent.Config.AgentId)])

                                | Failure (error, corrId) ->
                                    if config.AutoRetryFailedTasks && currentContext.RetryCount < config.MaxRetryAttempts then
                                        logger.LogWarning(corrId, $"Task failed, retrying: {task.TaskId} (attempt {currentContext.RetryCount + 1})")

                                        let retryContext = {
                                            currentContext with
                                                RetryCount = currentContext.RetryCount + 1
                                                Status = Retrying
                                                Error = Some error
                                        }
                                        activeTaskContexts.[task.TaskId] <- retryContext

                                        // Wait before retry
                                        do! Task.Delay(1000 * (currentContext.RetryCount + 1), cancellationToken) |> Async.AwaitTask
                                        let! retryResult = executeWithRetry retryContext
                                        return retryResult
                                    else
                                        let failedContext = {
                                            currentContext with
                                                Status = Failed
                                                EndTime = Some DateTime.Now
                                                Error = Some error
                                        }
                                        activeTaskContexts.[task.TaskId] <- failedContext

                                        let logError = ExecutionError ($"Task failed permanently: {task.TaskId}", None)
                                        logger.LogError(corrId, logError)
                                        return Failure (error, corrId)

                            | Failure (error, corrId) ->
                                let failedContext = {
                                    currentContext with
                                        Status = Failed
                                        EndTime = Some DateTime.Now
                                        Error = Some error
                                }
                                activeTaskContexts.[task.TaskId] <- failedContext
                                return Failure (error, corrId)
                        }

                    let! result = executeWithRetry context
                    return result
                } |> Async.StartAsTask
            with
            | ex ->
                let error = ExecutionError ($"Failed to execute task: {task.TaskId}", Some ex)
                logger.LogError(generateCorrelationId(), error, ex)
                Task.FromResult(Failure (error, generateCorrelationId()))
        
        /// Process queued task (internal)
        member private this.ProcessQueuedTaskAsync(task: UnifiedAgentTask, cancellationToken: CancellationToken) =
            this.ExecuteTaskAsync(task, cancellationToken)
        
        /// Process queued message (internal)
        member private this.ProcessQueuedMessageAsync(message: UnifiedAgentMessage, cancellationToken: CancellationToken) =
            async {
                try
                    match message.ToAgent with
                    | Some agentId ->
                        let! agentResult = registry.GetAgentAsync(agentId, cancellationToken) |> Async.AwaitTask
                        match agentResult with
                        | Success (Some agent, _) ->
                            let! result = agent.SendMessageAsync(message, cancellationToken) |> Async.AwaitTask
                            return result
                        | Success (None, _) ->
                            let agentIdStr = match agentId with UnifiedAgentId guid -> guid.ToString()
                            let error = ValidationError ($"Target agent not found: {agentIdStr}", Map [("agentId", agentIdStr)])
                            return Failure (error, generateCorrelationId())
                        | Failure (error, corrId) ->
                            return Failure (error, corrId)
                    | None ->
                        // Broadcast message - not implemented in this version
                        let error = ValidationError ("Broadcast messages not yet implemented", Map.empty)
                        return Failure (error, generateCorrelationId())
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to process message: {message.MessageId}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            } |> Async.StartAsTask
        
        /// Get system-wide metrics
        member this.GetSystemMetrics() =
            let totalTasks = activeTaskContexts.Count
            let completedTasks = 
                activeTaskContexts.Values
                |> Seq.filter (fun ctx -> ctx.Status = Completed)
                |> Seq.length
            let failedTasks =
                activeTaskContexts.Values
                |> Seq.filter (fun ctx -> ctx.Status = Failed)
                |> Seq.length
            let runningTasks =
                activeTaskContexts.Values
                |> Seq.filter (fun ctx -> ctx.Status = Running)
                |> Seq.length
            
            let registryStats = registry.GetRegistryStatistics()
            
            Map [
                ("totalTasks", box totalTasks)
                ("completedTasks", box completedTasks)
                ("failedTasks", box failedTasks)
                ("runningTasks", box runningTasks)
                ("registryStats", box registryStats)
                ("timestamp", box DateTime.Now)
            ]
        
        interface IUnifiedAgentCoordinator with
            member this.RegisterAgentAsync(agent, cancellationToken) = 
                registry.RegisterAgentAsync(agent, cancellationToken)
            
            member this.UnregisterAgentAsync(agentId, cancellationToken) = 
                registry.UnregisterAgentAsync(agentId, cancellationToken)
            
            member this.FindBestAgentAsync(task, cancellationToken) = 
                this.FindBestAgentAsync(task, cancellationToken)
            
            member this.RouteMessageAsync(message, cancellationToken) = 
                messageQueue.Enqueue(message)
                Success ((), Map [("messageId", box message.MessageId)]) |> Task.FromResult
            
            member this.ExecuteTaskAsync(task, cancellationToken) = 
                this.ExecuteTaskAsync(task, cancellationToken)
            
            member this.GetRegisteredAgents() = 
                let result = registry.GetAllAgentsAsync(CancellationToken.None).Result
                match result with
                | Success (agents, _) -> agents
                | Failure _ -> []
            
            member this.GetSystemMetrics() = 
                this.GetSystemMetrics()
            
            member this.HealthCheckAllAsync(cancellationToken) = 
                registry.PerformHealthChecksAsync(cancellationToken)
        
        interface ITarsComponent with
            member this.Name = "UnifiedAgentCoordinator"
            member this.Version = "1.0.0"

            member this.Initialize(config) =
                // Initialize with configuration
                Success ((), Map.empty)

            member this.Shutdown() =
                // Shutdown the component
                if isRunning then
                    cancellationTokenSource.Cancel()
                    isRunning <- false
                Success ((), Map.empty)

            member this.GetHealth() =
                let health = Map [
                    ("status", box (if isRunning then "running" else "stopped"))
                    ("activeTasks", box activeTaskContexts.Count)
                    ("queuedTasks", box taskQueue.Count)
                    ("queuedMessages", box messageQueue.Count)
                ]
                Success (health, Map.empty)

            member this.GetMetrics() =
                Success (this.GetSystemMetrics(), Map.empty)
        
        interface IDisposable with
            member this.Dispose() =
                if not isDisposed then
                    isDisposed <- true
                    cancellationTokenSource.Cancel()
                    cancellationTokenSource.Dispose()
                    activeTaskContexts.Clear()
    
    /// Default coordination configuration
    let defaultCoordinationConfig = {
        MaxConcurrentTasks = 100
        TaskTimeoutMs = 300000 // 5 minutes
        LoadBalancingStrategy = PerformanceBased
        MessageRetentionHours = 24
        HealthCheckIntervalMs = 60000 // 1 minute
        AutoRetryFailedTasks = true
        MaxRetryAttempts = 3
        EnableTaskPrioritization = true
    }
    
    /// Create unified agent coordinator
    let createAgentCoordinator (config: AgentCoordinationConfig option) (registry: UnifiedAgentRegistry) (logger: ITarsLogger) =
        let finalConfig = config |> Option.defaultValue defaultCoordinationConfig
        new UnifiedAgentCoordinator(finalConfig, registry, logger)
