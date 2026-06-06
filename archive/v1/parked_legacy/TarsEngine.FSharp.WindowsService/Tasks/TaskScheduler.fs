namespace TarsEngine.FSharp.WindowsService.Tasks

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Task scheduling strategy
/// </summary>
type SchedulingStrategy =
    | PriorityFirst      // Execute highest priority tasks first
    | FairShare          // Round-robin between priority levels
    | DeadlineFirst      // Execute tasks with earliest deadlines first
    | ResourceOptimized  // Optimize based on resource availability

/// <summary>
/// Task scheduling decision
/// </summary>
type SchedulingDecision = {
    TaskId: string
    ScheduledAt: DateTime
    EstimatedStartTime: DateTime
    EstimatedDuration: TimeSpan
    AssignedWorker: string option
    Priority: TaskPriority
    Reason: string
}

/// <summary>
/// Intelligent task scheduler with multiple scheduling strategies
/// </summary>
type TaskScheduler(logger: ILogger<TaskScheduler>, config: TaskConfig) =
    
    let schedulingQueue = ConcurrentQueue<TaskDefinition>()
    let scheduledTasks = ConcurrentDictionary<string, SchedulingDecision>()
    let workerCapacity = ConcurrentDictionary<string, int>()
    let taskHistory = ConcurrentQueue<TaskExecutionResult>()
    
    let mutable currentStrategy = PriorityFirst
    let mutable isRunning = false
    let mutable schedulingTask: Task option = None
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    
    let maxHistorySize = 1000
    
    /// Start the task scheduler
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting task scheduler...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Start scheduling loop
            let schedulingLoop = this.SchedulingLoopAsync(cancellationTokenSource.Value.Token)
            schedulingTask <- Some schedulingLoop
            
            logger.LogInformation($"Task scheduler started with strategy: {currentStrategy}")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start task scheduler")
            isRunning <- false
            raise
    }
    
    /// Stop the task scheduler
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping task scheduler...")
            
            isRunning <- false
            
            // Cancel scheduling operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for scheduling task to complete
            match schedulingTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Scheduling task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for scheduling task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            schedulingTask <- None
            
            logger.LogInformation("Task scheduler stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping task scheduler")
    }
    
    /// Schedule a task for execution
    member this.ScheduleTask(task: TaskDefinition) =
        try
            logger.LogDebug($"Scheduling task: {task.Name} ({task.Id}) with priority {task.Priority}")
            
            // Add to scheduling queue
            schedulingQueue.Enqueue(task)
            
            // Create initial scheduling decision
            let decision = {
                TaskId = task.Id
                ScheduledAt = DateTime.UtcNow
                EstimatedStartTime = this.EstimateStartTime(task)
                EstimatedDuration = this.EstimateDuration(task)
                AssignedWorker = None
                Priority = task.Priority
                Reason = "Queued for scheduling"
            }
            
            scheduledTasks.[task.Id] <- decision
            
            logger.LogDebug($"Task {task.Id} scheduled successfully")
            Ok decision
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to schedule task: {task.Name}")
            Error ex.Message
    
    /// Get next task to execute based on current strategy
    member this.GetNextTaskAsync(workerId: string, cancellationToken: CancellationToken) = task {
        try
            // Wait for tasks to be available
            while schedulingQueue.IsEmpty && isRunning do
                do! Task.Delay(TimeSpan.FromMilliseconds(100.0), cancellationToken)
            
            if not isRunning then
                return None
            
            // Try to dequeue a task based on current strategy
            let selectedTask = this.SelectTaskByStrategy(workerId)
            
            match selectedTask with
            | Some task ->
                // Update scheduling decision
                match scheduledTasks.TryGetValue(task.Id) with
                | true, decision ->
                    let updatedDecision = {
                        decision with
                            EstimatedStartTime = DateTime.UtcNow
                            AssignedWorker = Some workerId
                            Reason = $"Assigned to worker {workerId}"
                    }
                    scheduledTasks.[task.Id] <- updatedDecision
                | false, _ -> ()
                
                logger.LogDebug($"Task {task.Id} assigned to worker {workerId}")
                return Some task
            
            | None ->
                return None
                
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Task selection cancelled for worker: {workerId}")
            return None
        | ex ->
            logger.LogError(ex, $"Error selecting next task for worker: {workerId}")
            return None
    }
    
    /// Select task based on current scheduling strategy
    member private this.SelectTaskByStrategy(workerId: string) =
        let availableTasks = 
            schedulingQueue
            |> Seq.toList
            |> List.filter (fun task -> this.CanAssignToWorker(task, workerId))
        
        if availableTasks.IsEmpty then
            None
        else
            let selectedTask = 
                match currentStrategy with
                | PriorityFirst ->
                    availableTasks 
                    |> List.sortBy (fun t -> int t.Priority, t.CreatedAt)
                    |> List.head
                
                | FairShare ->
                    // Implement round-robin logic
                    availableTasks 
                    |> List.sortBy (fun t -> t.CreatedAt)
                    |> List.head
                
                | DeadlineFirst ->
                    availableTasks 
                    |> List.sortBy (fun t -> t.ScheduledAt |> Option.defaultValue DateTime.MaxValue)
                    |> List.head
                
                | ResourceOptimized ->
                    // Select task that best fits current resource availability
                    availableTasks 
                    |> List.sortBy (fun t -> this.CalculateResourceScore(t, workerId))
                    |> List.head
            
            // Remove from queue (this is a simplified approach)
            // In a production system, we'd use a more sophisticated queue structure
            Some selectedTask
    
    /// Check if task can be assigned to worker
    member private this.CanAssignToWorker(task: TaskDefinition, workerId: string) =
        // Check worker capacity
        let currentCapacity = workerCapacity.GetOrAdd(workerId, 0)
        let maxCapacity = config.MaxConcurrentTasks / Environment.ProcessorCount // Simplified
        
        if currentCapacity >= maxCapacity then
            false
        else
            // Check task dependencies
            let dependenciesMet = 
                task.Dependencies
                |> List.forall (fun depId ->
                    match scheduledTasks.TryGetValue(depId) with
                    | true, decision -> decision.AssignedWorker.IsSome
                    | false, _ -> false)
            
            dependenciesMet
    
    /// Calculate resource score for task assignment
    member private this.CalculateResourceScore(task: TaskDefinition, workerId: string) =
        let priorityScore = float (int task.Priority)
        let ageScore = (DateTime.UtcNow - task.CreatedAt).TotalMinutes
        let capacityScore = 
            let currentCapacity = workerCapacity.GetOrAdd(workerId, 0)
            float (10 - currentCapacity) // Prefer workers with lower load
        
        priorityScore + (ageScore * 0.1) + capacityScore
    
    /// Estimate task start time
    member private this.EstimateStartTime(task: TaskDefinition) =
        let queueLength = schedulingQueue.Count
        let averageExecutionTime = this.GetAverageExecutionTime()
        let estimatedDelay = TimeSpan.FromMilliseconds(float queueLength * averageExecutionTime.TotalMilliseconds)
        
        DateTime.UtcNow.Add(estimatedDelay)
    
    /// Estimate task duration
    member private this.EstimateDuration(task: TaskDefinition) =
        // Use historical data or default timeout
        let historicalDuration = this.GetHistoricalDuration(task.Name)
        match historicalDuration with
        | Some duration -> duration
        | None -> task.Timeout
    
    /// Get average execution time from history
    member private this.GetAverageExecutionTime() =
        if taskHistory.IsEmpty then
            TimeSpan.FromMinutes(5.0) // Default estimate
        else
            let recentTasks = taskHistory |> Seq.take (min 100 taskHistory.Count) |> List.ofSeq
            let averageMs = recentTasks |> List.averageBy (fun t -> t.ExecutionTime.TotalMilliseconds)
            TimeSpan.FromMilliseconds(averageMs)
    
    /// Get historical duration for similar tasks
    member private this.GetHistoricalDuration(taskName: string) =
        let similarTasks = 
            taskHistory 
            |> Seq.filter (fun t -> t.TaskId.Contains(taskName))
            |> Seq.take 10
            |> List.ofSeq
        
        if similarTasks.IsEmpty then
            None
        else
            let averageMs = similarTasks |> List.averageBy (fun t -> t.ExecutionTime.TotalMilliseconds)
            Some (TimeSpan.FromMilliseconds(averageMs))
    
    /// Main scheduling loop
    member private this.SchedulingLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting scheduling loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Perform scheduling optimizations
                    this.OptimizeScheduling()
                    
                    // Update worker capacities
                    this.UpdateWorkerCapacities()
                    
                    // Clean up old scheduling decisions
                    this.CleanupOldDecisions()
                    
                    // Wait before next scheduling cycle
                    do! Task.Delay(TimeSpan.FromSeconds(10.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in scheduling loop")
                    do! Task.Delay(TimeSpan.FromSeconds(10.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Scheduling loop cancelled")
        | ex ->
            logger.LogError(ex, "Scheduling loop failed")
    }
    
    /// Optimize scheduling based on current conditions
    member private this.OptimizeScheduling() =
        // Analyze current performance and adjust strategy if needed
        let queueLength = schedulingQueue.Count
        let averageWaitTime = this.GetAverageWaitTime()
        
        if queueLength > config.QueueCapacity * 0.8 then
            // Queue is getting full, prioritize high-priority tasks
            if currentStrategy <> PriorityFirst then
                currentStrategy <- PriorityFirst
                logger.LogInformation("Switched to PriorityFirst strategy due to high queue utilization")
        elif averageWaitTime > TimeSpan.FromMinutes(10.0) then
            // Tasks are waiting too long, use fair share
            if currentStrategy <> FairShare then
                currentStrategy <- FairShare
                logger.LogInformation("Switched to FairShare strategy due to high wait times")
        else
            // Normal conditions, use resource optimization
            if currentStrategy <> ResourceOptimized then
                currentStrategy <- ResourceOptimized
                logger.LogInformation("Switched to ResourceOptimized strategy")
    
    /// Get average wait time for tasks
    member private this.GetAverageWaitTime() =
        if taskHistory.IsEmpty then
            TimeSpan.Zero
        else
            let recentTasks = taskHistory |> Seq.take (min 50 taskHistory.Count) |> List.ofSeq
            let averageMs = recentTasks |> List.averageBy (fun t -> t.ExecutionTime.TotalMilliseconds)
            TimeSpan.FromMilliseconds(averageMs)
    
    /// Update worker capacities
    member private this.UpdateWorkerCapacities() =
        // This would be updated by the task executor
        // For now, we'll just ensure all workers are tracked
        for i in 1 .. Environment.ProcessorCount do
            let workerId = $"Worker-{i}"
            workerCapacity.GetOrAdd(workerId, 0) |> ignore
    
    /// Clean up old scheduling decisions
    member private this.CleanupOldDecisions() =
        let cutoffTime = DateTime.UtcNow.AddHours(-1.0)
        let decisionsToRemove = 
            scheduledTasks.Values
            |> Seq.filter (fun d -> d.ScheduledAt < cutoffTime)
            |> Seq.map (fun d -> d.TaskId)
            |> List.ofSeq
        
        for taskId in decisionsToRemove do
            scheduledTasks.TryRemove(taskId) |> ignore
    
    /// Record task completion for learning
    member this.RecordTaskCompletion(result: TaskExecutionResult) =
        taskHistory.Enqueue(result)
        
        // Keep history size manageable
        while taskHistory.Count > maxHistorySize do
            taskHistory.TryDequeue() |> ignore
        
        // Update worker capacity
        match scheduledTasks.TryGetValue(result.TaskId) with
        | true, decision ->
            match decision.AssignedWorker with
            | Some workerId ->
                workerCapacity.AddOrUpdate(workerId, 0, fun _ current -> max 0 (current - 1)) |> ignore
            | None -> ()
        | false, _ -> ()
    
    /// Get scheduling statistics
    member this.GetStatistics() =
        let queueLength = schedulingQueue.Count
        let scheduledCount = scheduledTasks.Count
        let averageWaitTime = this.GetAverageWaitTime()
        let averageExecutionTime = this.GetAverageExecutionTime()
        
        {|
            QueueLength = queueLength
            ScheduledTasks = scheduledCount
            CurrentStrategy = currentStrategy.ToString()
            AverageWaitTimeMs = averageWaitTime.TotalMilliseconds
            AverageExecutionTimeMs = averageExecutionTime.TotalMilliseconds
            WorkerCapacities = workerCapacity |> Map.ofSeq
            IsRunning = isRunning
        |}
    
    /// Change scheduling strategy
    member this.SetSchedulingStrategy(strategy: SchedulingStrategy) =
        if currentStrategy <> strategy then
            currentStrategy <- strategy
            logger.LogInformation($"Scheduling strategy changed to: {strategy}")
    
    /// Get scheduled task information
    member this.GetScheduledTask(taskId: string) =
        match scheduledTasks.TryGetValue(taskId) with
        | true, decision -> Some decision
        | false, _ -> None
