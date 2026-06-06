namespace TarsEngine.FSharp.WindowsService.Tasks

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Task monitoring event types
/// </summary>
type TaskMonitoringEvent =
    | TaskStarted of TaskDefinition
    | TaskCompleted of TaskExecutionResult
    | TaskFailed of TaskExecutionResult
    | TaskCancelled of TaskExecutionResult
    | TaskTimeout of TaskExecutionResult
    | QueueSizeChanged of int
    | WorkerStatusChanged of string * WorkerStatus

/// <summary>
/// Performance trend data
/// </summary>
type PerformanceTrend = {
    Timestamp: DateTime
    TotalTasks: int64
    CompletedTasks: int64
    FailedTasks: int64
    AverageExecutionTimeMs: float
    ThroughputTasksPerSecond: float
    QueueSize: int
    ActiveWorkers: int
}

/// <summary>
/// Task monitoring alert
/// </summary>
type TaskMonitoringAlert = {
    Id: string
    Type: AlertType
    Severity: AlertSeverity
    Message: string
    Timestamp: DateTime
    TaskId: string option
    WorkerId: string option
    Metadata: Map<string, obj>
}

/// <summary>
/// Alert types
/// </summary>
and AlertType =
    | HighFailureRate
    | LongExecutionTime
    | QueueBacklog
    | WorkerFailure
    | ResourceExhaustion
    | PerformanceDegradation

/// <summary>
/// Alert severity levels
/// </summary>
and AlertSeverity =
    | Info
    | Warning
    | Error
    | Critical

/// <summary>
/// Task monitoring statistics
/// </summary>
type TaskMonitoringStats = {
    TotalTasksMonitored: int64
    TasksInProgress: int
    CompletedTasks: int64
    FailedTasks: int64
    CancelledTasks: int64
    AverageExecutionTimeMs: float
    MedianExecutionTimeMs: float
    P95ExecutionTimeMs: float
    P99ExecutionTimeMs: float
    ThroughputTasksPerSecond: float
    FailureRate: float
    ActiveAlerts: int
    TotalAlerts: int64
}

/// <summary>
/// Real-time task monitoring and analytics system
/// </summary>
type TaskMonitor(logger: ILogger<TaskMonitor>) =
    
    let monitoringEvents = ConcurrentQueue<TaskMonitoringEvent>()
    let performanceTrends = ConcurrentQueue<PerformanceTrend>()
    let activeAlerts = ConcurrentDictionary<string, TaskMonitoringAlert>()
    let executionResults = ConcurrentQueue<TaskExecutionResult>()
    let monitoringStats = ConcurrentDictionary<string, int64>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable monitoringTask: Task option = None
    
    let maxEventHistory = 10000
    let maxTrendHistory = 1440 // 24 hours of minute-by-minute data
    let maxExecutionHistory = 10000
    
    /// Start the task monitor
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting task monitor...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Start monitoring loop
            let monitoringLoop = this.MonitoringLoopAsync(cancellationTokenSource.Value.Token)
            monitoringTask <- Some monitoringLoop
            
            logger.LogInformation("Task monitor started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start task monitor")
            isRunning <- false
            raise
    }
    
    /// Stop the task monitor
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping task monitor...")
            
            isRunning <- false
            
            // Cancel monitoring operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for monitoring task to complete
            match monitoringTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Monitoring task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for monitoring task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            monitoringTask <- None
            
            logger.LogInformation("Task monitor stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping task monitor")
    }
    
    /// Record task execution
    member this.RecordTaskExecution(result: TaskExecutionResult) =
        try
            // Add to execution history
            executionResults.Enqueue(result)
            
            // Keep history size manageable
            while executionResults.Count > maxExecutionHistory do
                executionResults.TryDequeue() |> ignore
            
            // Update statistics
            this.UpdateStatistics(result)
            
            // Create monitoring event
            let event = 
                match result.Status with
                | TaskStatus.Completed -> TaskCompleted result
                | TaskStatus.Failed -> TaskFailed result
                | TaskStatus.Cancelled -> TaskCancelled result
                | TaskStatus.Timeout -> TaskTimeout result
                | _ -> TaskCompleted result
            
            this.RecordEvent(event)
            
            // Check for alerts
            this.CheckForAlerts(result)
            
            logger.LogDebug($"Recorded task execution: {result.TaskId} - {result.Status}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Error recording task execution: {result.TaskId}")
    
    /// Record monitoring event
    member this.RecordEvent(event: TaskMonitoringEvent) =
        try
            monitoringEvents.Enqueue(event)
            
            // Keep event history manageable
            while monitoringEvents.Count > maxEventHistory do
                monitoringEvents.TryDequeue() |> ignore
            
            logger.LogDebug($"Recorded monitoring event: {event}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Error recording monitoring event: {event}")
    
    /// Update monitoring statistics
    member private this.UpdateStatistics(result: TaskExecutionResult) =
        monitoringStats.AddOrUpdate("TotalTasksMonitored", 1L, fun _ current -> current + 1L) |> ignore
        
        match result.Status with
        | TaskStatus.Completed ->
            monitoringStats.AddOrUpdate("CompletedTasks", 1L, fun _ current -> current + 1L) |> ignore
        | TaskStatus.Failed ->
            monitoringStats.AddOrUpdate("FailedTasks", 1L, fun _ current -> current + 1L) |> ignore
        | TaskStatus.Cancelled ->
            monitoringStats.AddOrUpdate("CancelledTasks", 1L, fun _ current -> current + 1L) |> ignore
        | _ -> ()
    
    /// Check for alerts based on task execution
    member private this.CheckForAlerts(result: TaskExecutionResult) =
        // Check for long execution time
        if result.ExecutionTime > TimeSpan.FromMinutes(10.0) then
            let alert = {
                Id = Guid.NewGuid().ToString()
                Type = LongExecutionTime
                Severity = Warning
                Message = $"Task {result.TaskId} took {result.ExecutionTime.TotalMinutes:F1} minutes to execute"
                Timestamp = DateTime.UtcNow
                TaskId = Some result.TaskId
                WorkerId = None
                Metadata = Map.ofList [("ExecutionTimeMs", result.ExecutionTime.TotalMilliseconds :> obj)]
            }
            this.RaiseAlert(alert)
        
        // Check failure rate
        let recentResults = executionResults |> Seq.take (min 100 executionResults.Count) |> List.ofSeq
        if recentResults.Length >= 10 then
            let failureRate = 
                recentResults 
                |> List.filter (fun r -> r.Status = TaskStatus.Failed) 
                |> List.length 
                |> fun count -> float count / float recentResults.Length
            
            if failureRate > 0.2 then // 20% failure rate
                let alert = {
                    Id = Guid.NewGuid().ToString()
                    Type = HighFailureRate
                    Severity = Error
                    Message = $"High failure rate detected: {failureRate * 100.0:F1}% of recent tasks failed"
                    Timestamp = DateTime.UtcNow
                    TaskId = None
                    WorkerId = None
                    Metadata = Map.ofList [("FailureRate", failureRate :> obj)]
                }
                this.RaiseAlert(alert)
    
    /// Raise an alert
    member private this.RaiseAlert(alert: TaskMonitoringAlert) =
        try
            activeAlerts.[alert.Id] <- alert
            monitoringStats.AddOrUpdate("TotalAlerts", 1L, fun _ current -> current + 1L) |> ignore
            
            logger.LogWarning($"Alert raised: {alert.Type} - {alert.Message}")
            
            // In a production system, we might send notifications here
            
        with
        | ex ->
            logger.LogError(ex, $"Error raising alert: {alert.Message}")
    
    /// Main monitoring loop
    member private this.MonitoringLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting monitoring loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Collect performance trends
                    this.CollectPerformanceTrend()
                    
                    // Clean up old alerts
                    this.CleanupOldAlerts()
                    
                    // Generate periodic reports
                    this.GeneratePeriodicReport()
                    
                    // Wait for next monitoring cycle
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in monitoring loop")
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Monitoring loop cancelled")
        | ex ->
            logger.LogError(ex, "Monitoring loop failed")
    }
    
    /// Collect performance trend data
    member private this.CollectPerformanceTrend() =
        try
            let totalTasks = monitoringStats.GetOrAdd("TotalTasksMonitored", 0L)
            let completedTasks = monitoringStats.GetOrAdd("CompletedTasks", 0L)
            let failedTasks = monitoringStats.GetOrAdd("FailedTasks", 0L)
            
            let recentResults = executionResults |> Seq.take (min 100 executionResults.Count) |> List.ofSeq
            let averageExecutionTime = 
                if recentResults.Length > 0 then
                    recentResults |> List.averageBy (fun r -> r.ExecutionTime.TotalMilliseconds)
                else 0.0
            
            let throughput = float recentResults.Length / 60.0 // Approximate tasks per second
            
            let trend = {
                Timestamp = DateTime.UtcNow
                TotalTasks = totalTasks
                CompletedTasks = completedTasks
                FailedTasks = failedTasks
                AverageExecutionTimeMs = averageExecutionTime
                ThroughputTasksPerSecond = throughput
                QueueSize = 0 // Would be provided by queue
                ActiveWorkers = 0 // Would be provided by executor
            }
            
            performanceTrends.Enqueue(trend)
            
            // Keep trend history manageable
            while performanceTrends.Count > maxTrendHistory do
                performanceTrends.TryDequeue() |> ignore
                
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting performance trend")
    
    /// Clean up old alerts
    member private this.CleanupOldAlerts() =
        let cutoffTime = DateTime.UtcNow.AddHours(-24.0)
        let alertsToRemove = 
            activeAlerts.Values
            |> Seq.filter (fun a -> a.Timestamp < cutoffTime)
            |> Seq.map (fun a -> a.Id)
            |> List.ofSeq
        
        for alertId in alertsToRemove do
            activeAlerts.TryRemove(alertId) |> ignore
    
    /// Generate periodic report
    member private this.GeneratePeriodicReport() =
        let stats = this.GetMonitoringStatistics()
        
        if stats.TotalTasksMonitored > 0L then
            logger.LogInformation($"Task Monitor Report - Total: {stats.TotalTasksMonitored}, Completed: {stats.CompletedTasks}, Failed: {stats.FailedTasks}, Avg Time: {stats.AverageExecutionTimeMs:F0}ms, Throughput: {stats.ThroughputTasksPerSecond:F2}/sec")
            
            if stats.ActiveAlerts > 0 then
                logger.LogWarning($"Active Alerts: {stats.ActiveAlerts}")
    
    /// Get monitoring statistics
    member this.GetMonitoringStatistics() =
        let totalTasks = monitoringStats.GetOrAdd("TotalTasksMonitored", 0L)
        let completedTasks = monitoringStats.GetOrAdd("CompletedTasks", 0L)
        let failedTasks = monitoringStats.GetOrAdd("FailedTasks", 0L)
        let cancelledTasks = monitoringStats.GetOrAdd("CancelledTasks", 0L)
        let totalAlerts = monitoringStats.GetOrAdd("TotalAlerts", 0L)
        
        let recentResults = executionResults |> Seq.take (min 1000 executionResults.Count) |> List.ofSeq
        
        let averageExecutionTime = 
            if recentResults.Length > 0 then
                recentResults |> List.averageBy (fun r -> r.ExecutionTime.TotalMilliseconds)
            else 0.0
        
        let executionTimes = recentResults |> List.map (fun r -> r.ExecutionTime.TotalMilliseconds) |> List.sort
        let medianExecutionTime = 
            if executionTimes.Length > 0 then
                let mid = executionTimes.Length / 2
                if executionTimes.Length % 2 = 0 then
                    (executionTimes.[mid - 1] + executionTimes.[mid]) / 2.0
                else
                    executionTimes.[mid]
            else 0.0
        
        let p95ExecutionTime = 
            if executionTimes.Length > 0 then
                let index = int (float executionTimes.Length * 0.95)
                executionTimes.[min index (executionTimes.Length - 1)]
            else 0.0
        
        let p99ExecutionTime = 
            if executionTimes.Length > 0 then
                let index = int (float executionTimes.Length * 0.99)
                executionTimes.[min index (executionTimes.Length - 1)]
            else 0.0
        
        let throughput = float recentResults.Length / 60.0 // Approximate
        
        let failureRate = 
            if totalTasks > 0L then
                float failedTasks / float totalTasks
            else 0.0
        
        {
            TotalTasksMonitored = totalTasks
            TasksInProgress = 0 // Would be provided by executor
            CompletedTasks = completedTasks
            FailedTasks = failedTasks
            CancelledTasks = cancelledTasks
            AverageExecutionTimeMs = averageExecutionTime
            MedianExecutionTimeMs = medianExecutionTime
            P95ExecutionTimeMs = p95ExecutionTime
            P99ExecutionTimeMs = p99ExecutionTime
            ThroughputTasksPerSecond = throughput
            FailureRate = failureRate
            ActiveAlerts = activeAlerts.Count
            TotalAlerts = totalAlerts
        }
    
    /// Get performance trends
    member this.GetPerformanceTrends(hours: int) =
        let cutoffTime = DateTime.UtcNow.AddHours(-float hours)
        performanceTrends
        |> Seq.filter (fun t -> t.Timestamp >= cutoffTime)
        |> Seq.sortBy (fun t -> t.Timestamp)
        |> List.ofSeq
    
    /// Get active alerts
    member this.GetActiveAlerts() =
        activeAlerts.Values |> List.ofSeq
    
    /// Get alerts by severity
    member this.GetAlertsBySeverity(severity: AlertSeverity) =
        activeAlerts.Values 
        |> Seq.filter (fun a -> a.Severity = severity)
        |> List.ofSeq
    
    /// Clear alert
    member this.ClearAlert(alertId: string) =
        match activeAlerts.TryRemove(alertId) with
        | true, alert ->
            logger.LogInformation($"Alert cleared: {alert.Type} - {alert.Message}")
            Ok ()
        | false, _ ->
            Error $"Alert not found: {alertId}"
    
    /// Get recent task executions
    member this.GetRecentTaskExecutions(count: int) =
        executionResults 
        |> Seq.take (min count executionResults.Count)
        |> List.ofSeq
    
    /// Get task execution by ID
    member this.GetTaskExecution(taskId: string) =
        executionResults
        |> Seq.tryFind (fun r -> r.TaskId = taskId)
