namespace TarsEngine.FSharp.WindowsService.Tasks

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open System.Threading.Channels
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Worker thread information
/// </summary>
type WorkerInfo = {
    Id: string
    Name: string
    Status: WorkerStatus
    CurrentTask: string option
    TasksCompleted: int64
    TasksFailed: int64
    TotalExecutionTime: TimeSpan
    AverageExecutionTime: TimeSpan
    LastActivity: DateTime
    StartTime: DateTime
}

/// <summary>
/// Worker status enumeration
/// </summary>
and WorkerStatus =
    | Idle
    | Busy
    | Stopping
    | Stopped
    | Failed

/// <summary>
/// Task execution worker
/// </summary>
type TaskWorker = {
    Info: WorkerInfo
    CancellationTokenSource: CancellationTokenSource
    WorkerTask: Task
}

/// <summary>
/// Execution performance metrics
/// </summary>
type ExecutionMetrics = {
    TotalTasksExecuted: int64
    TotalTasksSucceeded: int64
    TotalTasksFailed: int64
    TotalTasksCancelled: int64
    AverageExecutionTimeMs: float
    ThroughputTasksPerSecond: float
    ActiveWorkers: int
    IdleWorkers: int
    QueueUtilization: float
    ResourceUtilization: float
}

/// <summary>
/// High-performance parallel task executor with resource management
/// </summary>
type TaskExecutor(
    logger: ILogger<TaskExecutor>,
    taskQueue: TaskQueue,
    taskScheduler: TaskScheduler,
    taskMonitor: TaskMonitor) =
    
    let workers = ConcurrentDictionary<string, TaskWorker>()
    let executionMetrics = ConcurrentDictionary<string, int64>()
    let executionTimes = ConcurrentQueue<float>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable config: TaskConfig option = None
    
    let maxExecutionTimeHistory = 1000
    
    /// Configure the task executor
    member this.ConfigureAsync(taskConfig: TaskConfig, cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Configuring task executor...")
            
            config <- Some taskConfig
            
            logger.LogInformation($"Task executor configured with {taskConfig.MaxConcurrentTasks} max concurrent tasks")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to configure task executor")
            raise
    }
    
    /// Start the task executor
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting task executor...")
            
            match config with
            | Some taskConfig ->
                cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                isRunning <- true
                
                // Start worker threads
                do! this.StartWorkersAsync(taskConfig, cancellationTokenSource.Value.Token)
                
                // Start metrics collection
                do! this.StartMetricsCollectionAsync(cancellationTokenSource.Value.Token)
                
                logger.LogInformation($"Task executor started with {workers.Count} workers")
            
            | None ->
                let error = "Task executor not configured"
                logger.LogError(error)
                raise (InvalidOperationException(error))
                
        with
        | ex ->
            logger.LogError(ex, "Failed to start task executor")
            isRunning <- false
            raise
    }
    
    /// Stop the task executor
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping task executor...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Stop all workers
            do! this.StopWorkersAsync(cancellationToken)
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            workers.Clear()
            
            logger.LogInformation("Task executor stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping task executor")
    }
    
    /// Start worker threads
    member private this.StartWorkersAsync(taskConfig: TaskConfig, cancellationToken: CancellationToken) = task {
        logger.LogInformation($"Starting {taskConfig.MaxConcurrentTasks} worker threads...")
        
        for i in 1 .. taskConfig.MaxConcurrentTasks do
            let workerId = $"Worker-{i}"
            let workerName = $"TaskWorker-{i}"
            
            let workerInfo = {
                Id = workerId
                Name = workerName
                Status = Idle
                CurrentTask = None
                TasksCompleted = 0L
                TasksFailed = 0L
                TotalExecutionTime = TimeSpan.Zero
                AverageExecutionTime = TimeSpan.Zero
                LastActivity = DateTime.UtcNow
                StartTime = DateTime.UtcNow
            }
            
            let workerCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken)
            let workerTask = this.WorkerLoopAsync(workerInfo, taskConfig, workerCts.Token)
            
            let worker = {
                Info = workerInfo
                CancellationTokenSource = workerCts
                WorkerTask = workerTask
            }
            
            workers.[workerId] <- worker
            logger.LogDebug($"Started worker: {workerId}")
    }
    
    /// Stop all workers
    member private this.StopWorkersAsync(cancellationToken: CancellationToken) = task {
        logger.LogInformation("Stopping all workers...")
        
        // Cancel all workers
        for kvp in workers do
            kvp.Value.CancellationTokenSource.Cancel()
        
        // Wait for workers to complete
        let workerTasks = workers.Values |> Seq.map (fun w -> w.WorkerTask) |> Array.ofSeq
        
        try
            do! Task.WhenAll(workerTasks).WaitAsync(TimeSpan.FromSeconds(30.0), cancellationToken)
        with
        | :? TimeoutException ->
            logger.LogWarning("Some workers did not complete within timeout")
        | ex ->
            logger.LogWarning(ex, "Error waiting for workers to complete")
        
        // Dispose worker resources
        for kvp in workers do
            try
                kvp.Value.CancellationTokenSource.Dispose()
            with
            | ex -> logger.LogWarning(ex, $"Error disposing worker {kvp.Key}")
        
        logger.LogInformation("All workers stopped")
    }
    
    /// Main worker loop
    member private this.WorkerLoopAsync(initialInfo: WorkerInfo, taskConfig: TaskConfig, cancellationToken: CancellationToken) = task {
        let mutable workerInfo = initialInfo
        
        try
            logger.LogDebug($"Worker {workerInfo.Id} started")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Update worker status to idle
                    workerInfo <- { workerInfo with Status = Idle; CurrentTask = None; LastActivity = DateTime.UtcNow }
                    this.UpdateWorkerInfo(workerInfo)
                    
                    // Get next task from scheduler
                    let! nextTask = taskScheduler.GetNextTaskAsync(workerInfo.Id, cancellationToken)
                    
                    match nextTask with
                    | Some task ->
                        // Execute the task
                        let! executionResult = this.ExecuteTaskAsync(task, workerInfo, taskConfig, cancellationToken)
                        
                        // Update worker statistics
                        workerInfo <- this.UpdateWorkerStatistics(workerInfo, executionResult)
                        this.UpdateWorkerInfo(workerInfo)
                        
                        // Record task completion
                        taskScheduler.RecordTaskCompletion(executionResult)
                        taskMonitor.RecordTaskExecution(executionResult)
                        
                        // Update global metrics
                        this.UpdateExecutionMetrics(executionResult)
                        
                    | None ->
                        // No tasks available, wait a bit
                        do! Task.Delay(TimeSpan.FromMilliseconds(100.0), cancellationToken)
                        
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, $"Error in worker {workerInfo.Id}")
                    workerInfo <- { workerInfo with Status = Failed; LastActivity = DateTime.UtcNow }
                    this.UpdateWorkerInfo(workerInfo)
                    do! Task.Delay(TimeSpan.FromSeconds(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Worker {workerInfo.Id} cancelled")
        | ex ->
            logger.LogError(ex, $"Worker {workerInfo.Id} failed")
        finally
            workerInfo <- { workerInfo with Status = Stopped; LastActivity = DateTime.UtcNow }
            this.UpdateWorkerInfo(workerInfo)
            logger.LogDebug($"Worker {workerInfo.Id} stopped")
    }
    
    /// Execute a single task
    member private this.ExecuteTaskAsync(task: TaskDefinition, workerInfo: WorkerInfo, taskConfig: TaskConfig, cancellationToken: CancellationToken) = task {
        let startTime = DateTime.UtcNow
        let executionContext = {
            TaskId = task.Id
            CreatedAt = task.CreatedAt
            StartedAt = Some startTime
            CompletedAt = None
            ExecutionTime = None
            RetryCount = 0
            MaxRetries = task.MaxRetries
            LastError = None
            Metadata = task.Metadata
        }
        
        try
            logger.LogInformation($"Worker {workerInfo.Id} executing task: {task.Name} ({task.Id})")
            
            // Update worker status
            let busyWorkerInfo = { workerInfo with Status = Busy; CurrentTask = Some task.Id; LastActivity = DateTime.UtcNow }
            this.UpdateWorkerInfo(busyWorkerInfo)
            
            // Mark task as running in queue
            taskQueue.MarkTaskAsRunning(task.Id) |> ignore
            
            // Create timeout cancellation token
            use timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken)
            timeoutCts.CancelAfter(task.Timeout)
            
            // Execute the task
            let! result = task.ExecuteAsync(executionContext, timeoutCts.Token)
            
            let completedAt = DateTime.UtcNow
            let executionTime = completedAt - startTime
            
            // Create execution result
            let executionResult = {
                TaskId = task.Id
                Status = if result.IsOk then TaskStatus.Completed else TaskStatus.Failed
                Result = result
                ExecutionTime = executionTime
                RetryCount = 0
                CompletedAt = completedAt
                Metadata = task.Metadata
            }
            
            // Complete task in queue
            taskQueue.CompleteTask(task.Id, result) |> ignore
            
            logger.LogInformation($"Worker {workerInfo.Id} completed task {task.Id} in {executionTime.TotalMilliseconds:F0}ms")
            return executionResult
            
        with
        | :? OperationCanceledException when cancellationToken.IsCancellationRequested ->
            logger.LogInformation($"Task {task.Id} cancelled by shutdown")
            let executionResult = {
                TaskId = task.Id
                Status = TaskStatus.Cancelled
                Result = Error "Task cancelled by shutdown"
                ExecutionTime = DateTime.UtcNow - startTime
                RetryCount = 0
                CompletedAt = DateTime.UtcNow
                Metadata = task.Metadata
            }
            taskQueue.CancelTask(task.Id) |> ignore
            return executionResult
            
        | :? OperationCanceledException ->
            logger.LogWarning($"Task {task.Id} timed out after {task.Timeout}")
            let executionResult = {
                TaskId = task.Id
                Status = TaskStatus.Timeout
                Result = Error $"Task timed out after {task.Timeout}"
                ExecutionTime = DateTime.UtcNow - startTime
                RetryCount = 0
                CompletedAt = DateTime.UtcNow
                Metadata = task.Metadata
            }
            taskQueue.CompleteTask(task.Id, Error "Task timed out") |> ignore
            return executionResult
            
        | ex ->
            logger.LogError(ex, $"Task {task.Id} failed with exception")
            let executionResult = {
                TaskId = task.Id
                Status = TaskStatus.Failed
                Result = Error ex.Message
                ExecutionTime = DateTime.UtcNow - startTime
                RetryCount = 0
                CompletedAt = DateTime.UtcNow
                Metadata = task.Metadata
            }
            taskQueue.CompleteTask(task.Id, Error ex.Message) |> ignore
            return executionResult
    }
    
    /// Update worker information
    member private this.UpdateWorkerInfo(workerInfo: WorkerInfo) =
        match workers.TryGetValue(workerInfo.Id) with
        | true, worker ->
            let updatedWorker = { worker with Info = workerInfo }
            workers.[workerInfo.Id] <- updatedWorker
        | false, _ -> ()
    
    /// Update worker statistics after task execution
    member private this.UpdateWorkerStatistics(workerInfo: WorkerInfo, executionResult: TaskExecutionResult) =
        let newTasksCompleted = 
            if executionResult.Status = TaskStatus.Completed then 
                workerInfo.TasksCompleted + 1L 
            else 
                workerInfo.TasksCompleted
        
        let newTasksFailed = 
            if executionResult.Status = TaskStatus.Failed then 
                workerInfo.TasksFailed + 1L 
            else 
                workerInfo.TasksFailed
        
        let newTotalExecutionTime = workerInfo.TotalExecutionTime.Add(executionResult.ExecutionTime)
        let totalTasks = newTasksCompleted + newTasksFailed
        let newAverageExecutionTime = 
            if totalTasks > 0L then
                TimeSpan.FromMilliseconds(newTotalExecutionTime.TotalMilliseconds / float totalTasks)
            else
                TimeSpan.Zero
        
        {
            workerInfo with
                TasksCompleted = newTasksCompleted
                TasksFailed = newTasksFailed
                TotalExecutionTime = newTotalExecutionTime
                AverageExecutionTime = newAverageExecutionTime
                LastActivity = DateTime.UtcNow
        }
    
    /// Update global execution metrics
    member private this.UpdateExecutionMetrics(executionResult: TaskExecutionResult) =
        // Update counters
        executionMetrics.AddOrUpdate("TotalTasksExecuted", 1L, fun _ current -> current + 1L) |> ignore
        
        match executionResult.Status with
        | TaskStatus.Completed ->
            executionMetrics.AddOrUpdate("TotalTasksSucceeded", 1L, fun _ current -> current + 1L) |> ignore
        | TaskStatus.Failed ->
            executionMetrics.AddOrUpdate("TotalTasksFailed", 1L, fun _ current -> current + 1L) |> ignore
        | TaskStatus.Cancelled ->
            executionMetrics.AddOrUpdate("TotalTasksCancelled", 1L, fun _ current -> current + 1L) |> ignore
        | _ -> ()
        
        // Record execution time
        executionTimes.Enqueue(executionResult.ExecutionTime.TotalMilliseconds)
        
        // Keep execution time history manageable
        while executionTimes.Count > maxExecutionTimeHistory do
            executionTimes.TryDequeue() |> ignore
    
    /// Start metrics collection
    member private this.StartMetricsCollectionAsync(cancellationToken: CancellationToken) = task {
        let metricsTask = task {
            try
                while not cancellationToken.IsCancellationRequested && isRunning do
                    try
                        // Collect and log metrics periodically
                        let metrics = this.GetExecutionMetrics()
                        logger.LogInformation($"Execution Metrics - Active: {metrics.ActiveWorkers}, Throughput: {metrics.ThroughputTasksPerSecond:F2} tasks/sec, Avg Time: {metrics.AverageExecutionTimeMs:F0}ms")
                        
                        do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                        
                    with
                    | :? OperationCanceledException ->
                        break
                    | ex ->
                        logger.LogWarning(ex, "Error collecting metrics")
                        do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                        
            with
            | :? OperationCanceledException ->
                logger.LogDebug("Metrics collection cancelled")
            | ex ->
                logger.LogError(ex, "Metrics collection failed")
        }
        
        // Don't await - let it run in background
        metricsTask |> ignore
    }
    
    /// Get execution metrics
    member this.GetExecutionMetrics() =
        let totalExecuted = executionMetrics.GetOrAdd("TotalTasksExecuted", 0L)
        let totalSucceeded = executionMetrics.GetOrAdd("TotalTasksSucceeded", 0L)
        let totalFailed = executionMetrics.GetOrAdd("TotalTasksFailed", 0L)
        let totalCancelled = executionMetrics.GetOrAdd("TotalTasksCancelled", 0L)
        
        let averageExecutionTime = 
            if executionTimes.Count > 0 then
                executionTimes |> Seq.average
            else 0.0
        
        let activeWorkers = workers.Values |> Seq.filter (fun w -> w.Info.Status = Busy) |> Seq.length
        let idleWorkers = workers.Values |> Seq.filter (fun w -> w.Info.Status = Idle) |> Seq.length
        
        // Calculate throughput (tasks per second over last minute)
        let recentTasks = executionTimes |> Seq.length
        let throughput = float recentTasks / 60.0 // Approximate
        
        {
            TotalTasksExecuted = totalExecuted
            TotalTasksSucceeded = totalSucceeded
            TotalTasksFailed = totalFailed
            TotalTasksCancelled = totalCancelled
            AverageExecutionTimeMs = averageExecutionTime
            ThroughputTasksPerSecond = throughput
            ActiveWorkers = activeWorkers
            IdleWorkers = idleWorkers
            QueueUtilization = 0.0 // Would be calculated from queue
            ResourceUtilization = 0.0 // Would be calculated from system resources
        }
    
    /// Reconfigure the task executor
    member this.ReconfigureAsync(taskConfig: TaskConfig, cancellationToken: CancellationToken) = task {
        logger.LogInformation("Reconfiguring task executor...")
        
        // Stop current execution
        do! this.StopAsync(cancellationToken)
        
        // Apply new configuration
        do! this.ConfigureAsync(taskConfig, cancellationToken)
        
        // Start with new configuration
        do! this.StartAsync(cancellationToken)
        
        logger.LogInformation("Task executor reconfiguration completed")
    }
    
    /// Get executor status
    member this.GetStatus() =
        if isRunning then
            let activeWorkers = workers.Values |> Seq.filter (fun w -> w.Info.Status = Busy) |> Seq.length
            let totalWorkers = workers.Count
            $"Running with {activeWorkers}/{totalWorkers} active workers"
        else
            "Stopped"
    
    /// Get executor metrics
    member this.GetMetrics() =
        let metrics = this.GetExecutionMetrics()
        Map.ofList [
            ("TotalTasksExecuted", metrics.TotalTasksExecuted :> obj)
            ("TotalTasksSucceeded", metrics.TotalTasksSucceeded :> obj)
            ("TotalTasksFailed", metrics.TotalTasksFailed :> obj)
            ("ActiveWorkers", metrics.ActiveWorkers :> obj)
            ("IdleWorkers", metrics.IdleWorkers :> obj)
            ("AverageExecutionTimeMs", metrics.AverageExecutionTimeMs :> obj)
            ("ThroughputTasksPerSecond", metrics.ThroughputTasksPerSecond :> obj)
        ]
    
    /// Get worker information
    member this.GetWorkers() =
        workers.Values |> Seq.map (fun w -> w.Info) |> List.ofSeq
