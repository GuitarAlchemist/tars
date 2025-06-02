namespace TarsEngine.FSharp.WindowsService.Tasks

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open System.Threading.Channels
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Task priority levels
/// </summary>
type TaskPriority =
    | Critical = 1
    | High = 2
    | Normal = 3
    | Low = 4
    | Background = 5

/// <summary>
/// Task status enumeration
/// </summary>
type TaskStatus =
    | Queued
    | Running
    | Completed
    | Failed
    | Cancelled
    | Timeout

/// <summary>
/// Task execution context
/// </summary>
type TaskExecutionContext = {
    TaskId: string
    CreatedAt: DateTime
    StartedAt: DateTime option
    CompletedAt: DateTime option
    ExecutionTime: TimeSpan option
    RetryCount: int
    MaxRetries: int
    LastError: string option
    Metadata: Map<string, obj>
}

/// <summary>
/// Task definition
/// </summary>
type TaskDefinition = {
    Id: string
    Name: string
    Description: string
    Priority: TaskPriority
    Timeout: TimeSpan
    MaxRetries: int
    CreatedBy: string
    CreatedAt: DateTime
    ScheduledAt: DateTime option
    Dependencies: string list
    Metadata: Map<string, obj>
    ExecuteAsync: TaskExecutionContext -> CancellationToken -> Task<Result<obj, string>>
}

/// <summary>
/// Task execution result
/// </summary>
type TaskExecutionResult = {
    TaskId: string
    Status: TaskStatus
    Result: Result<obj, string>
    ExecutionTime: TimeSpan
    RetryCount: int
    CompletedAt: DateTime
    Metadata: Map<string, obj>
}

/// <summary>
/// High-performance task queue with priority support
/// </summary>
type TaskQueue(logger: ILogger<TaskQueue>, config: TaskConfig) =
    
    let priorityQueues = 
        [1..5] 
        |> List.map (fun priority -> 
            (enum<TaskPriority>(priority), Channel.CreateUnbounded<TaskDefinition>()))
        |> Map.ofList
    
    let runningTasks = ConcurrentDictionary<string, TaskDefinition>()
    let completedTasks = ConcurrentDictionary<string, TaskExecutionResult>()
    let taskContexts = ConcurrentDictionary<string, TaskExecutionContext>()
    
    let mutable isProcessing = false
    let mutable processingCancellationToken: CancellationTokenSource option = None
    
    /// Enqueue a task
    member this.EnqueueTask(task: TaskDefinition) =
        try
            logger.LogInformation($"Enqueueing task: {task.Name} ({task.Id}) with priority {task.Priority}")
            
            // Check queue capacity
            let totalQueuedTasks = this.GetQueuedTaskCount()
            if totalQueuedTasks >= config.QueueCapacity then
                let error = $"Task queue is at capacity ({config.QueueCapacity})"
                logger.LogWarning(error)
                Error error
            else
                // Create execution context
                let context = {
                    TaskId = task.Id
                    CreatedAt = task.CreatedAt
                    StartedAt = None
                    CompletedAt = None
                    ExecutionTime = None
                    RetryCount = 0
                    MaxRetries = task.MaxRetries
                    LastError = None
                    Metadata = task.Metadata
                }
                
                taskContexts.[task.Id] <- context
                
                // Add to appropriate priority queue
                match priorityQueues.TryGetValue(task.Priority) with
                | true, (channel, _) ->
                    if channel.Writer.TryWrite(task) then
                        logger.LogDebug($"Task {task.Name} ({task.Id}) enqueued successfully")
                        Ok ()
                    else
                        let error = "Failed to write task to channel"
                        logger.LogError(error)
                        Error error
                | false, _ ->
                    let error = $"Invalid priority level: {task.Priority}"
                    logger.LogError(error)
                    Error error
                    
        with
        | ex ->
            logger.LogError(ex, $"Failed to enqueue task: {task.Name}")
            Error ex.Message
    
    /// Dequeue the next task (priority-based)
    member this.DequeueTaskAsync(cancellationToken: CancellationToken) = task {
        try
            // Check queues in priority order (Critical -> Background)
            for priority in [TaskPriority.Critical; TaskPriority.High; TaskPriority.Normal; TaskPriority.Low; TaskPriority.Background] do
                match priorityQueues.TryGetValue(priority) with
                | true, (channel, _) ->
                    let! hasTask = channel.Reader.WaitToReadAsync(cancellationToken).AsTask()
                    if hasTask && channel.Reader.TryRead() then
                        let task = channel.Reader.TryRead() |> snd
                        logger.LogDebug($"Dequeued task: {task.Name} ({task.Id}) with priority {task.Priority}")
                        return Some task
                | false, _ -> ()
            
            return None
            
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Task dequeue operation was cancelled")
            return None
        | ex ->
            logger.LogError(ex, "Error during task dequeue")
            return None
    }
    
    /// Mark task as running
    member this.MarkTaskAsRunning(taskId: string) =
        match taskContexts.TryGetValue(taskId) with
        | true, context ->
            let updatedContext = { context with StartedAt = Some DateTime.UtcNow }
            taskContexts.[taskId] <- updatedContext
            logger.LogDebug($"Task {taskId} marked as running")
            Ok ()
        | false, _ ->
            let error = $"Task context not found: {taskId}"
            logger.LogWarning(error)
            Error error
    
    /// Complete a task
    member this.CompleteTask(taskId: string, result: Result<obj, string>) =
        try
            match taskContexts.TryGetValue(taskId) with
            | true, context ->
                let completedAt = DateTime.UtcNow
                let executionTime = 
                    match context.StartedAt with
                    | Some startTime -> completedAt - startTime
                    | None -> TimeSpan.Zero
                
                let status = 
                    match result with
                    | Ok _ -> TaskStatus.Completed
                    | Error _ -> TaskStatus.Failed
                
                let executionResult = {
                    TaskId = taskId
                    Status = status
                    Result = result
                    ExecutionTime = executionTime
                    RetryCount = context.RetryCount
                    CompletedAt = completedAt
                    Metadata = context.Metadata
                }
                
                completedTasks.[taskId] <- executionResult
                runningTasks.TryRemove(taskId) |> ignore
                
                let updatedContext = { 
                    context with 
                        CompletedAt = Some completedAt
                        ExecutionTime = Some executionTime 
                }
                taskContexts.[taskId] <- updatedContext
                
                logger.LogInformation($"Task {taskId} completed with status {status} in {executionTime}")
                Ok executionResult
                
            | false, _ ->
                let error = $"Task context not found: {taskId}"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to complete task: {taskId}")
            Error ex.Message
    
    /// Retry a failed task
    member this.RetryTask(taskId: string, error: string) =
        try
            match taskContexts.TryGetValue(taskId) with
            | true, context ->
                if context.RetryCount < context.MaxRetries then
                    let updatedContext = { 
                        context with 
                            RetryCount = context.RetryCount + 1
                            LastError = Some error
                            StartedAt = None
                    }
                    taskContexts.[taskId] <- updatedContext
                    
                    // Re-enqueue the task (would need the original task definition)
                    logger.LogInformation($"Task {taskId} scheduled for retry {updatedContext.RetryCount}/{context.MaxRetries}")
                    Ok true
                else
                    logger.LogWarning($"Task {taskId} exceeded maximum retries ({context.MaxRetries})")
                    Ok false
            | false, _ ->
                let error = $"Task context not found: {taskId}"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to retry task: {taskId}")
            Error ex.Message
    
    /// Cancel a task
    member this.CancelTask(taskId: string) =
        try
            // Remove from running tasks
            runningTasks.TryRemove(taskId) |> ignore
            
            // Update context
            match taskContexts.TryGetValue(taskId) with
            | true, context ->
                let completedAt = DateTime.UtcNow
                let executionTime = 
                    match context.StartedAt with
                    | Some startTime -> completedAt - startTime
                    | None -> TimeSpan.Zero
                
                let executionResult = {
                    TaskId = taskId
                    Status = TaskStatus.Cancelled
                    Result = Error "Task was cancelled"
                    ExecutionTime = executionTime
                    RetryCount = context.RetryCount
                    CompletedAt = completedAt
                    Metadata = context.Metadata
                }
                
                completedTasks.[taskId] <- executionResult
                
                let updatedContext = { 
                    context with 
                        CompletedAt = Some completedAt
                        ExecutionTime = Some executionTime 
                }
                taskContexts.[taskId] <- updatedContext
                
                logger.LogInformation($"Task {taskId} cancelled")
                Ok ()
            | false, _ ->
                let error = $"Task context not found: {taskId}"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to cancel task: {taskId}")
            Error ex.Message
    
    /// Get queued task count
    member this.GetQueuedTaskCount() =
        priorityQueues.Values
        |> Seq.sumBy (fun (channel, _) -> 
            if channel.Reader.CanCount then channel.Reader.Count else 0)
    
    /// Get running task count
    member this.GetRunningTaskCount() =
        runningTasks.Count
    
    /// Get completed task count
    member this.GetCompletedTaskCount() =
        completedTasks.Count
    
    /// Get task statistics
    member this.GetTaskStatistics() =
        let queuedCount = this.GetQueuedTaskCount()
        let runningCount = this.GetRunningTaskCount()
        let completedCount = this.GetCompletedTaskCount()
        
        let completedTasks = completedTasks.Values |> List.ofSeq
        let successfulTasks = completedTasks |> List.filter (fun t -> t.Status = TaskStatus.Completed) |> List.length
        let failedTasks = completedTasks |> List.filter (fun t -> t.Status = TaskStatus.Failed) |> List.length
        let cancelledTasks = completedTasks |> List.filter (fun t -> t.Status = TaskStatus.Cancelled) |> List.length
        
        let averageExecutionTime = 
            if completedTasks.Length > 0 then
                completedTasks 
                |> List.map (fun t -> t.ExecutionTime.TotalMilliseconds)
                |> List.average
            else 0.0
        
        {|
            QueuedTasks = queuedCount
            RunningTasks = runningCount
            CompletedTasks = completedCount
            SuccessfulTasks = successfulTasks
            FailedTasks = failedTasks
            CancelledTasks = cancelledTasks
            AverageExecutionTimeMs = averageExecutionTime
            QueueUtilization = (float queuedCount / float config.QueueCapacity) * 100.0
            TasksByPriority = 
                priorityQueues 
                |> Map.map (fun priority (channel, _) -> 
                    if channel.Reader.CanCount then channel.Reader.Count else 0)
        |}
    
    /// Get task by ID
    member this.GetTask(taskId: string) =
        match taskContexts.TryGetValue(taskId) with
        | true, context -> Some context
        | false, _ -> None
    
    /// Get task result
    member this.GetTaskResult(taskId: string) =
        match completedTasks.TryGetValue(taskId) with
        | true, result -> Some result
        | false, _ -> None
    
    /// Clear completed tasks (for memory management)
    member this.ClearCompletedTasks(olderThan: TimeSpan) =
        let cutoffTime = DateTime.UtcNow - olderThan
        let tasksToRemove = 
            completedTasks.Values
            |> Seq.filter (fun t -> t.CompletedAt < cutoffTime)
            |> Seq.map (fun t -> t.TaskId)
            |> List.ofSeq
        
        for taskId in tasksToRemove do
            completedTasks.TryRemove(taskId) |> ignore
            taskContexts.TryRemove(taskId) |> ignore
        
        logger.LogInformation($"Cleared {tasksToRemove.Length} completed tasks older than {olderThan}")
        tasksToRemove.Length
