namespace TarsEngine.FSharp.WindowsService.Monitoring

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// System health status levels
/// </summary>
type HealthStatus =
    | Excellent
    | Good
    | Fair
    | Poor
    | Critical

/// <summary>
/// Health check result
/// </summary>
type HealthCheckResult = {
    ComponentName: string
    Status: HealthStatus
    Message: string
    Timestamp: DateTime
    ResponseTime: TimeSpan
    Metrics: Map<string, obj>
    Issues: string list
}

/// <summary>
/// System health summary
/// </summary>
type SystemHealthSummary = {
    OverallStatus: HealthStatus
    LastCheck: DateTime
    ComponentCount: int
    HealthyComponents: int
    UnhealthyComponents: int
    CriticalIssues: int
    Warnings: int
    AverageResponseTime: TimeSpan
    SystemUptime: TimeSpan
    PerformanceScore: float
}

/// <summary>
/// Health trend data point
/// </summary>
type HealthTrendPoint = {
    Timestamp: DateTime
    OverallHealth: float
    ComponentHealth: Map<string, float>
    PerformanceMetrics: Map<string, float>
    IssueCount: int
}

/// <summary>
/// Predictive health analysis
/// </summary>
type PredictiveHealthAnalysis = {
    PredictedStatus: HealthStatus
    Confidence: float
    TimeHorizon: TimeSpan
    RiskFactors: string list
    Recommendations: string list
    TrendDirection: TrendDirection
}

/// <summary>
/// Trend direction enumeration
/// </summary>
and TrendDirection =
    | Improving
    | Stable
    | Declining
    | Critical

/// <summary>
/// Comprehensive system health monitoring with predictive analytics
/// </summary>
type HealthMonitor(
    logger: ILogger<HealthMonitor>,
    performanceCollector: PerformanceCollector,
    alertManager: AlertManager,
    diagnosticsCollector: DiagnosticsCollector) =
    
    let healthChecks = ConcurrentDictionary<string, Func<CancellationToken, Task<HealthCheckResult>>>()
    let healthHistory = ConcurrentQueue<HealthCheckResult>()
    let healthTrends = ConcurrentQueue<HealthTrendPoint>()
    let componentStatus = ConcurrentDictionary<string, HealthCheckResult>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable monitoringTask: Task option = None
    let mutable config: MonitoringConfig option = None
    
    let maxHealthHistory = 10000
    let maxTrendHistory = 2880 // 48 hours of minute-by-minute data
    
    /// Configure the health monitor
    member this.ConfigureAsync(monitoringConfig: MonitoringConfig, cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Configuring health monitor...")
            
            config <- Some monitoringConfig
            
            // Register default health checks
            this.RegisterDefaultHealthChecks()
            
            logger.LogInformation("Health monitor configured successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to configure health monitor")
            raise
    }
    
    /// Start the health monitor
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting health monitor...")
            
            match config with
            | Some monitoringConfig ->
                cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                isRunning <- true
                
                // Start monitoring loop
                let monitoringLoop = this.MonitoringLoopAsync(monitoringConfig, cancellationTokenSource.Value.Token)
                monitoringTask <- Some monitoringLoop
                
                // Start predictive analysis
                do! this.StartPredictiveAnalysisAsync(cancellationTokenSource.Value.Token)
                
                logger.LogInformation("Health monitor started successfully")
            
            | None ->
                let error = "Health monitor not configured"
                logger.LogError(error)
                raise (InvalidOperationException(error))
                
        with
        | ex ->
            logger.LogError(ex, "Failed to start health monitor")
            isRunning <- false
            raise
    }
    
    /// Stop the health monitor
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping health monitor...")
            
            isRunning <- false
            
            // Cancel all operations
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
                    logger.LogWarning("Health monitoring task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for health monitoring task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            monitoringTask <- None
            
            logger.LogInformation("Health monitor stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping health monitor")
    }
    
    /// Register a health check
    member this.RegisterHealthCheck(name: string, healthCheck: Func<CancellationToken, Task<HealthCheckResult>>) =
        try
            logger.LogInformation($"Registering health check: {name}")
            
            healthChecks.[name] <- healthCheck
            
            logger.LogDebug($"Health check {name} registered successfully")
            Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to register health check: {name}")
            Error ex.Message
    
    /// Unregister a health check
    member this.UnregisterHealthCheck(name: string) =
        try
            match healthChecks.TryRemove(name) with
            | true, _ ->
                componentStatus.TryRemove(name) |> ignore
                logger.LogInformation($"Health check {name} unregistered successfully")
                Ok ()
            | false, _ ->
                let error = $"Health check {name} not found"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to unregister health check: {name}")
            Error ex.Message
    
    /// Register default health checks
    member private this.RegisterDefaultHealthChecks() =
        // System resource health check
        let systemResourceCheck = Func<CancellationToken, Task<HealthCheckResult>>(fun ct -> task {
            let startTime = DateTime.UtcNow
            
            try
                // Get system metrics from performance collector
                let! metrics = performanceCollector.GetCurrentMetricsAsync(ct)
                
                let cpuUsage = metrics.GetValueOrDefault("CpuUsagePercent", 0.0 :> obj) :?> float
                let memoryUsage = metrics.GetValueOrDefault("MemoryUsagePercent", 0.0 :> obj) :?> float
                let diskUsage = metrics.GetValueOrDefault("DiskUsagePercent", 0.0 :> obj) :?> float
                
                let status = 
                    if cpuUsage > 90.0 || memoryUsage > 90.0 || diskUsage > 95.0 then Critical
                    elif cpuUsage > 80.0 || memoryUsage > 80.0 || diskUsage > 90.0 then Poor
                    elif cpuUsage > 70.0 || memoryUsage > 70.0 || diskUsage > 85.0 then Fair
                    elif cpuUsage > 50.0 || memoryUsage > 50.0 || diskUsage > 75.0 then Good
                    else Excellent
                
                let issues = ResizeArray<string>()
                if cpuUsage > 80.0 then issues.Add($"High CPU usage: {cpuUsage:F1}%")
                if memoryUsage > 80.0 then issues.Add($"High memory usage: {memoryUsage:F1}%")
                if diskUsage > 90.0 then issues.Add($"High disk usage: {diskUsage:F1}%")
                
                return {
                    ComponentName = "SystemResources"
                    Status = status
                    Message = $"CPU: {cpuUsage:F1}%, Memory: {memoryUsage:F1}%, Disk: {diskUsage:F1}%"
                    Timestamp = DateTime.UtcNow
                    ResponseTime = DateTime.UtcNow - startTime
                    Metrics = metrics
                    Issues = issues |> List.ofSeq
                }
                
            with
            | ex ->
                return {
                    ComponentName = "SystemResources"
                    Status = Critical
                    Message = $"Health check failed: {ex.Message}"
                    Timestamp = DateTime.UtcNow
                    ResponseTime = DateTime.UtcNow - startTime
                    Metrics = Map.empty
                    Issues = [ex.Message]
                }
        })
        
        this.RegisterHealthCheck("SystemResources", systemResourceCheck) |> ignore
        
        // Service availability health check
        let serviceAvailabilityCheck = Func<CancellationToken, Task<HealthCheckResult>>(fun ct -> task {
            let startTime = DateTime.UtcNow
            
            try
                // Check if core services are running
                let issues = ResizeArray<string>()
                let mutable overallHealthy = true
                
                // This would check actual service status in a real implementation
                let serviceStatus = Map.ofList [
                    ("AgentManager", true)
                    ("TaskExecutor", true)
                    ("TaskScheduler", true)
                    ("TaskMonitor", true)
                ]
                
                for kvp in serviceStatus do
                    if not kvp.Value then
                        issues.Add($"Service {kvp.Key} is not running")
                        overallHealthy <- false
                
                let status = if overallHealthy then Excellent else Critical
                
                return {
                    ComponentName = "ServiceAvailability"
                    Status = status
                    Message = if overallHealthy then "All services running" else "Some services unavailable"
                    Timestamp = DateTime.UtcNow
                    ResponseTime = DateTime.UtcNow - startTime
                    Metrics = serviceStatus |> Map.map (fun _ v -> v :> obj)
                    Issues = issues |> List.ofSeq
                }
                
            with
            | ex ->
                return {
                    ComponentName = "ServiceAvailability"
                    Status = Critical
                    Message = $"Health check failed: {ex.Message}"
                    Timestamp = DateTime.UtcNow
                    ResponseTime = DateTime.UtcNow - startTime
                    Metrics = Map.empty
                    Issues = [ex.Message]
                }
        })
        
        this.RegisterHealthCheck("ServiceAvailability", serviceAvailabilityCheck) |> ignore
    
    /// Main monitoring loop
    member private this.MonitoringLoopAsync(monitoringConfig: MonitoringConfig, cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting health monitoring loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Run all health checks
                    do! this.RunHealthChecksAsync(cancellationToken)
                    
                    // Analyze overall system health
                    let healthSummary = this.AnalyzeSystemHealth()
                    
                    // Record health trend
                    this.RecordHealthTrend(healthSummary)
                    
                    // Check for alerts
                    this.CheckHealthAlerts(healthSummary)
                    
                    // Log health summary
                    this.LogHealthSummary(healthSummary)
                    
                    // Wait for next monitoring cycle
                    do! Task.Delay(monitoringConfig.HealthCheckInterval, cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in health monitoring loop")
                    do! Task.Delay(monitoringConfig.HealthCheckInterval, cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Health monitoring loop cancelled")
        | ex ->
            logger.LogError(ex, "Health monitoring loop failed")
    }
    
    /// Run all registered health checks
    member private this.RunHealthChecksAsync(cancellationToken: CancellationToken) = task {
        let healthCheckTasks = 
            healthChecks
            |> Seq.map (fun kvp -> task {
                try
                    let! result = kvp.Value.Invoke(cancellationToken)
                    componentStatus.[kvp.Key] <- result
                    
                    // Add to history
                    healthHistory.Enqueue(result)
                    
                    // Keep history size manageable
                    while healthHistory.Count > maxHealthHistory do
                        healthHistory.TryDequeue() |> ignore
                    
                    logger.LogDebug($"Health check completed: {kvp.Key} - {result.Status}")
                    
                with
                | ex ->
                    logger.LogWarning(ex, $"Health check failed: {kvp.Key}")
                    
                    let failedResult = {
                        ComponentName = kvp.Key
                        Status = Critical
                        Message = $"Health check failed: {ex.Message}"
                        Timestamp = DateTime.UtcNow
                        ResponseTime = TimeSpan.Zero
                        Metrics = Map.empty
                        Issues = [ex.Message]
                    }
                    
                    componentStatus.[kvp.Key] <- failedResult
                    healthHistory.Enqueue(failedResult)
            })
            |> Array.ofSeq
        
        do! Task.WhenAll(healthCheckTasks)
    }
    
    /// Analyze overall system health
    member private this.AnalyzeSystemHealth() =
        let components = componentStatus.Values |> List.ofSeq
        let componentCount = components.Length
        
        if componentCount = 0 then
            {
                OverallStatus = Critical
                LastCheck = DateTime.UtcNow
                ComponentCount = 0
                HealthyComponents = 0
                UnhealthyComponents = 0
                CriticalIssues = 0
                Warnings = 0
                AverageResponseTime = TimeSpan.Zero
                SystemUptime = TimeSpan.Zero
                PerformanceScore = 0.0
            }
        else
            let healthyComponents = components |> List.filter (fun c -> c.Status = Excellent || c.Status = Good) |> List.length
            let unhealthyComponents = componentCount - healthyComponents
            let criticalIssues = components |> List.filter (fun c -> c.Status = Critical) |> List.length
            let warnings = components |> List.filter (fun c -> c.Status = Poor || c.Status = Fair) |> List.length
            
            let averageResponseTime = 
                if componentCount > 0 then
                    let totalMs = components |> List.sumBy (fun c -> c.ResponseTime.TotalMilliseconds)
                    TimeSpan.FromMilliseconds(totalMs / float componentCount)
                else
                    TimeSpan.Zero
            
            let overallStatus = 
                if criticalIssues > 0 then Critical
                elif warnings > componentCount / 2 then Poor
                elif warnings > 0 then Fair
                elif healthyComponents = componentCount then Excellent
                else Good
            
            let performanceScore = 
                let healthScore = float healthyComponents / float componentCount
                let responseScore = if averageResponseTime.TotalMilliseconds < 100.0 then 1.0 else 0.5
                (healthScore + responseScore) / 2.0 * 100.0
            
            {
                OverallStatus = overallStatus
                LastCheck = DateTime.UtcNow
                ComponentCount = componentCount
                HealthyComponents = healthyComponents
                UnhealthyComponents = unhealthyComponents
                CriticalIssues = criticalIssues
                Warnings = warnings
                AverageResponseTime = averageResponseTime
                SystemUptime = TimeSpan.Zero // Would be calculated from service start time
                PerformanceScore = performanceScore
            }
    
    /// Record health trend data
    member private this.RecordHealthTrend(healthSummary: SystemHealthSummary) =
        let overallHealth = 
            match healthSummary.OverallStatus with
            | Excellent -> 100.0
            | Good -> 80.0
            | Fair -> 60.0
            | Poor -> 40.0
            | Critical -> 20.0
        
        let componentHealth = 
            componentStatus
            |> Seq.map (fun kvp -> 
                let health = 
                    match kvp.Value.Status with
                    | Excellent -> 100.0
                    | Good -> 80.0
                    | Fair -> 60.0
                    | Poor -> 40.0
                    | Critical -> 20.0
                (kvp.Key, health))
            |> Map.ofSeq
        
        let trendPoint = {
            Timestamp = DateTime.UtcNow
            OverallHealth = overallHealth
            ComponentHealth = componentHealth
            PerformanceMetrics = Map.empty // Would be populated from performance collector
            IssueCount = healthSummary.CriticalIssues + healthSummary.Warnings
        }
        
        healthTrends.Enqueue(trendPoint)
        
        // Keep trend history manageable
        while healthTrends.Count > maxTrendHistory do
            healthTrends.TryDequeue() |> ignore
    
    /// Check for health alerts
    member private this.CheckHealthAlerts(healthSummary: SystemHealthSummary) =
        // Critical system health
        if healthSummary.OverallStatus = Critical then
            alertManager.RaiseAlert("SystemHealth", "Critical", $"System health is critical with {healthSummary.CriticalIssues} critical issues", Map.empty) |> ignore
        
        // High response times
        if healthSummary.AverageResponseTime > TimeSpan.FromSeconds(5.0) then
            alertManager.RaiseAlert("HealthResponseTime", "Warning", $"Health check response time is high: {healthSummary.AverageResponseTime.TotalSeconds:F1}s", Map.empty) |> ignore
        
        // Component failures
        for kvp in componentStatus do
            if kvp.Value.Status = Critical then
                alertManager.RaiseAlert($"Component_{kvp.Key}", "Error", $"Component {kvp.Key} is in critical state: {kvp.Value.Message}", Map.empty) |> ignore
    
    /// Log health summary
    member private this.LogHealthSummary(healthSummary: SystemHealthSummary) =
        logger.LogInformation($"System Health: {healthSummary.OverallStatus} - {healthSummary.HealthyComponents}/{healthSummary.ComponentCount} healthy, Score: {healthSummary.PerformanceScore:F1}%")
        
        if healthSummary.CriticalIssues > 0 then
            logger.LogWarning($"Critical issues detected: {healthSummary.CriticalIssues}")
        
        if healthSummary.Warnings > 0 then
            logger.LogWarning($"Warnings detected: {healthSummary.Warnings}")
    
    /// Start predictive analysis
    member private this.StartPredictiveAnalysisAsync(cancellationToken: CancellationToken) = task {
        let analysisTask = task {
            try
                while not cancellationToken.IsCancellationRequested && isRunning do
                    try
                        // Perform predictive analysis every 15 minutes
                        let analysis = this.PerformPredictiveAnalysis()
                        
                        if analysis.PredictedStatus = Critical || analysis.TrendDirection = Critical then
                            logger.LogWarning($"Predictive analysis indicates potential issues: {String.Join(", ", analysis.RiskFactors)}")
                            
                            // Raise predictive alert
                            alertManager.RaiseAlert("PredictiveHealth", "Warning", $"Predictive analysis indicates declining health trend", Map.empty) |> ignore
                        
                        do! Task.Delay(TimeSpan.FromMinutes(15.0), cancellationToken)
                        
                    with
                    | :? OperationCanceledException ->
                        break
                    | ex ->
                        logger.LogWarning(ex, "Error in predictive analysis")
                        do! Task.Delay(TimeSpan.FromMinutes(15.0), cancellationToken)
                        
            with
            | :? OperationCanceledException ->
                logger.LogDebug("Predictive analysis cancelled")
            | ex ->
                logger.LogError(ex, "Predictive analysis failed")
        }
        
        // Don't await - let it run in background
        analysisTask |> ignore
    }
    
    /// Perform predictive health analysis
    member private this.PerformPredictiveAnalysis() =
        let recentTrends = 
            healthTrends 
            |> Seq.take (min 60 healthTrends.Count) // Last hour
            |> List.ofSeq
            |> List.rev
        
        if recentTrends.Length < 10 then
            // Not enough data for prediction
            {
                PredictedStatus = Good
                Confidence = 0.0
                TimeHorizon = TimeSpan.FromHours(1.0)
                RiskFactors = ["Insufficient data for prediction"]
                Recommendations = ["Continue monitoring"]
                TrendDirection = Stable
            }
        else
            // Simple trend analysis (in production, this would use more sophisticated ML)
            let healthValues = recentTrends |> List.map (fun t -> t.OverallHealth)
            let firstHalf = healthValues |> List.take (healthValues.Length / 2) |> List.average
            let secondHalf = healthValues |> List.skip (healthValues.Length / 2) |> List.average
            
            let trendDirection = 
                if secondHalf > firstHalf + 5.0 then Improving
                elif secondHalf < firstHalf - 10.0 then Declining
                elif secondHalf < firstHalf - 20.0 then Critical
                else Stable
            
            let predictedHealth = secondHalf + (secondHalf - firstHalf) // Simple linear extrapolation
            let predictedStatus = 
                if predictedHealth >= 90.0 then Excellent
                elif predictedHealth >= 70.0 then Good
                elif predictedHealth >= 50.0 then Fair
                elif predictedHealth >= 30.0 then Poor
                else Critical
            
            let riskFactors = ResizeArray<string>()
            let recommendations = ResizeArray<string>()
            
            if trendDirection = Declining || trendDirection = Critical then
                riskFactors.Add("Declining health trend detected")
                recommendations.Add("Investigate recent changes")
                recommendations.Add("Check resource utilization")
            
            if predictedHealth < 50.0 then
                riskFactors.Add("Predicted health below acceptable threshold")
                recommendations.Add("Consider scaling resources")
                recommendations.Add("Review system configuration")
            
            {
                PredictedStatus = predictedStatus
                Confidence = min 1.0 (float recentTrends.Length / 60.0) // Confidence based on data availability
                TimeHorizon = TimeSpan.FromHours(1.0)
                RiskFactors = riskFactors |> List.ofSeq
                Recommendations = recommendations |> List.ofSeq
                TrendDirection = trendDirection
            }
    
    /// Get current system health
    member this.GetSystemHealth() =
        this.AnalyzeSystemHealth()
    
    /// Get component health status
    member this.GetComponentHealth(componentName: string) =
        match componentStatus.TryGetValue(componentName) with
        | true, result -> Some result
        | false, _ -> None
    
    /// Get all component health statuses
    member this.GetAllComponentHealth() =
        componentStatus.Values |> List.ofSeq
    
    /// Get health trends
    member this.GetHealthTrends(hours: int) =
        let cutoffTime = DateTime.UtcNow.AddHours(-float hours)
        healthTrends
        |> Seq.filter (fun t -> t.Timestamp >= cutoffTime)
        |> Seq.sortBy (fun t -> t.Timestamp)
        |> List.ofSeq
    
    /// Get predictive analysis
    member this.GetPredictiveAnalysis() =
        this.PerformPredictiveAnalysis()
    
    /// Reconfigure the health monitor
    member this.ReconfigureAsync(monitoringConfig: MonitoringConfig, cancellationToken: CancellationToken) = task {
        logger.LogInformation("Reconfiguring health monitor...")
        
        // Stop current monitoring
        do! this.StopAsync(cancellationToken)
        
        // Apply new configuration
        do! this.ConfigureAsync(monitoringConfig, cancellationToken)
        
        // Start with new configuration
        do! this.StartAsync(cancellationToken)
        
        logger.LogInformation("Health monitor reconfiguration completed")
    }
    
    /// Get monitor status
    member this.GetStatus() =
        if isRunning then
            let healthSummary = this.GetSystemHealth()
            $"Running - {healthSummary.OverallStatus} health with {healthSummary.ComponentCount} components"
        else
            "Stopped"
    
    /// Get monitor metrics
    member this.GetMetrics() =
        let healthSummary = this.GetSystemHealth()
        Map.ofList [
            ("OverallStatus", healthSummary.OverallStatus.ToString() :> obj)
            ("ComponentCount", healthSummary.ComponentCount :> obj)
            ("HealthyComponents", healthSummary.HealthyComponents :> obj)
            ("UnhealthyComponents", healthSummary.UnhealthyComponents :> obj)
            ("CriticalIssues", healthSummary.CriticalIssues :> obj)
            ("PerformanceScore", healthSummary.PerformanceScore :> obj)
        ]
