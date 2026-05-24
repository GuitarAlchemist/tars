namespace TarsEngine.FSharp.Cli.Monitoring

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem

/// Unified Monitoring System - Real-time system monitoring using unified architecture
module UnifiedMonitoring =
    
    /// Monitoring metric types
    type MetricType =
        | Counter
        | Gauge
        | Histogram
        | Timer
        | Rate
    
    /// Monitoring alert severity
    type AlertSeverity =
        | Info
        | Warning
        | Error
        | Critical
    
    /// Monitoring metric
    type MonitoringMetric = {
        Name: string
        MetricType: MetricType
        Value: float
        Unit: string
        Timestamp: DateTime
        Tags: Map<string, string>
        CorrelationId: string
    }
    
    /// Monitoring alert
    type MonitoringAlert = {
        AlertId: string
        Name: string
        Severity: AlertSeverity
        Message: string
        Threshold: float
        CurrentValue: float
        Timestamp: DateTime
        IsActive: bool
        Tags: Map<string, string>
        CorrelationId: string
    }
    
    /// System health status
    type SystemHealthStatus = {
        OverallHealth: float
        ComponentHealth: Map<string, float>
        ActiveAlerts: MonitoringAlert list
        LastUpdate: DateTime
        Uptime: TimeSpan
        PerformanceMetrics: Map<string, float>
        ResourceUtilization: Map<string, float>
        ProofId: string option
        CorrelationId: string
    }
    
    /// Performance analytics
    type PerformanceAnalytics = {
        AverageResponseTime: float
        ThroughputPerSecond: float
        ErrorRate: float
        MemoryUsage: float
        CpuUsage: float
        GpuUsage: float option
        DiskUsage: float
        NetworkUsage: float
        CacheHitRatio: float
        ActiveConnections: int
        QueueDepth: int
        LastAnalysis: DateTime
    }
    
    /// Monitoring configuration
    type MonitoringConfiguration = {
        MetricRetentionDays: int
        AlertRetentionDays: int
        HealthCheckInterval: TimeSpan
        MetricCollectionInterval: TimeSpan
        AlertEvaluationInterval: TimeSpan
        PerformanceAnalysisInterval: TimeSpan
        EnableRealTimeMonitoring: bool
        EnableAlerting: bool
        EnablePerformanceAnalytics: bool
        MaxMetricsInMemory: int
        MaxAlertsInMemory: int
    }
    
    /// Monitoring context
    type MonitoringContext = {
        ConfigManager: UnifiedConfigurationManager
        ProofGenerator: UnifiedProofGenerator
        Logger: ITarsLogger
        Configuration: MonitoringConfiguration
        CorrelationId: string
        StartTime: DateTime
    }
    
    /// Create monitoring context
    let createMonitoringContext (logger: ITarsLogger) (configManager: UnifiedConfigurationManager) (proofGenerator: UnifiedProofGenerator) =
        let config = {
            MetricRetentionDays = ConfigurationExtensions.getInt configManager "tars.monitoring.metricRetentionDays" 30
            AlertRetentionDays = ConfigurationExtensions.getInt configManager "tars.monitoring.alertRetentionDays" 7
            HealthCheckInterval = TimeSpan.FromSeconds(ConfigurationExtensions.getFloat configManager "tars.monitoring.healthCheckIntervalSeconds" 30.0)
            MetricCollectionInterval = TimeSpan.FromSeconds(ConfigurationExtensions.getFloat configManager "tars.monitoring.metricCollectionIntervalSeconds" 10.0)
            AlertEvaluationInterval = TimeSpan.FromSeconds(ConfigurationExtensions.getFloat configManager "tars.monitoring.alertEvaluationIntervalSeconds" 15.0)
            PerformanceAnalysisInterval = TimeSpan.FromMinutes(ConfigurationExtensions.getFloat configManager "tars.monitoring.performanceAnalysisIntervalMinutes" 5.0)
            EnableRealTimeMonitoring = ConfigurationExtensions.getBool configManager "tars.monitoring.enableRealTime" true
            EnableAlerting = ConfigurationExtensions.getBool configManager "tars.monitoring.enableAlerting" true
            EnablePerformanceAnalytics = ConfigurationExtensions.getBool configManager "tars.monitoring.enablePerformanceAnalytics" true
            MaxMetricsInMemory = ConfigurationExtensions.getInt configManager "tars.monitoring.maxMetricsInMemory" 100000
            MaxAlertsInMemory = ConfigurationExtensions.getInt configManager "tars.monitoring.maxAlertsInMemory" 10000
        }
        
        {
            ConfigManager = configManager
            ProofGenerator = proofGenerator
            Logger = logger
            Configuration = config
            CorrelationId = generateCorrelationId()
            StartTime = DateTime.UtcNow
        }
    
    /// Collect system performance metrics
    let collectSystemMetrics (context: MonitoringContext) =
        task {
            try
                let process = Process.GetCurrentProcess()
                let timestamp = DateTime.UtcNow
                
                // Memory metrics
                let memoryUsage = float process.WorkingSet64 / (1024.0 * 1024.0) // MB
                let gcMemory = float (GC.GetTotalMemory(false)) / (1024.0 * 1024.0) // MB
                
                // CPU metrics (simplified)
                let cpuUsage = float process.TotalProcessorTime.TotalMilliseconds / Environment.TickCount64 * 100.0
                
                // Thread metrics
                let threadCount = float process.Threads.Count
                
                let metrics = [
                    {
                        Name = "system.memory.working_set"
                        MetricType = Gauge
                        Value = memoryUsage
                        Unit = "MB"
                        Timestamp = timestamp
                        Tags = Map [("process", process.ProcessName)]
                        CorrelationId = context.CorrelationId
                    }
                    {
                        Name = "system.memory.gc_total"
                        MetricType = Gauge
                        Value = gcMemory
                        Unit = "MB"
                        Timestamp = timestamp
                        Tags = Map [("type", "managed")]
                        CorrelationId = context.CorrelationId
                    }
                    {
                        Name = "system.cpu.usage"
                        MetricType = Gauge
                        Value = Math.Min(cpuUsage, 100.0)
                        Unit = "percent"
                        Timestamp = timestamp
                        Tags = Map [("process", process.ProcessName)]
                        CorrelationId = context.CorrelationId
                    }
                    {
                        Name = "system.threads.count"
                        MetricType = Gauge
                        Value = threadCount
                        Unit = "count"
                        Timestamp = timestamp
                        Tags = Map [("process", process.ProcessName)]
                        CorrelationId = context.CorrelationId
                    }
                ]
                
                return Success (metrics, Map [("metricsCollected", box metrics.Length)])
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "MetricCollectionError" "Failed to collect system metrics" (Some ex), ex)
                let error = ExecutionError ($"Metric collection failed: {ex.Message}", Some ex)
                return Failure (error, context.CorrelationId)
        }
    
    /// Evaluate alert conditions
    let evaluateAlerts (context: MonitoringContext) (metrics: MonitoringMetric list) (existingAlerts: MonitoringAlert list) =
        task {
            try
                let mutable newAlerts = []
                let mutable updatedAlerts = []
                
                // Define alert thresholds
                let alertRules = [
                    ("system.memory.working_set", 500.0, Critical, "High memory usage")
                    ("system.memory.working_set", 300.0, Warning, "Elevated memory usage")
                    ("system.cpu.usage", 90.0, Critical, "High CPU usage")
                    ("system.cpu.usage", 70.0, Warning, "Elevated CPU usage")
                    ("system.threads.count", 100.0, Warning, "High thread count")
                ]
                
                for (metricName, threshold, severity, message) in alertRules do
                    let relevantMetrics = metrics |> List.filter (fun m -> m.Name = metricName)
                    
                    for metric in relevantMetrics do
                        let existingAlert = existingAlerts |> List.tryFind (fun a -> a.Name = metricName && a.Severity = severity)
                        
                        if metric.Value > threshold then
                            match existingAlert with
                            | Some alert when not alert.IsActive ->
                                // Reactivate existing alert
                                let updatedAlert =
                                    { alert with
                                        IsActive = true
                                        CurrentValue = metric.Value
                                        Timestamp = DateTime.UtcNow
                                        CorrelationId = context.CorrelationId }
                                updatedAlerts <- updatedAlert :: updatedAlerts
                            
                            | None ->
                                // Create new alert
                                let valueStr = metric.Value.ToString("F2")
                                let newAlert = {
                                    AlertId = generateCorrelationId()
                                    Name = metricName
                                    Severity = severity
                                    Message = $"{message}: {valueStr} {metric.Unit}"
                                    Threshold = threshold
                                    CurrentValue = metric.Value
                                    Timestamp = DateTime.UtcNow
                                    IsActive = true
                                    Tags = metric.Tags
                                    CorrelationId = context.CorrelationId
                                }
                                newAlerts <- newAlert :: newAlerts
                            
                            | Some _ -> () // Alert already active
                        
                        else
                            match existingAlert with
                            | Some alert when alert.IsActive ->
                                // Deactivate alert
                                let deactivatedAlert =
                                    { alert with
                                        IsActive = false
                                        CurrentValue = metric.Value
                                        Timestamp = DateTime.UtcNow
                                        CorrelationId = context.CorrelationId }
                                updatedAlerts <- deactivatedAlert :: updatedAlerts
                            
                            | _ -> () // No action needed
                
                return Success ((newAlerts, updatedAlerts), Map [
                    ("newAlerts", box newAlerts.Length)
                    ("updatedAlerts", box updatedAlerts.Length)
                ])
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "AlertEvaluationError" "Failed to evaluate alerts" (Some ex), ex)
                let error = ExecutionError ($"Alert evaluation failed: {ex.Message}", Some ex)
                return Failure (error, context.CorrelationId)
        }
    
    /// Calculate system health score
    let calculateSystemHealth (context: MonitoringContext) (metrics: MonitoringMetric list) (alerts: MonitoringAlert list) =
        task {
            try
                let activeAlerts = alerts |> List.filter (fun a -> a.IsActive)
                
                // Base health score
                let mutable healthScore = 1.0
                
                // Reduce health based on active alerts
                for alert in activeAlerts do
                    match alert.Severity with
                    | Critical -> healthScore <- healthScore - 0.3
                    | Error -> healthScore <- healthScore - 0.2
                    | Warning -> healthScore <- healthScore - 0.1
                    | Info -> healthScore <- healthScore - 0.05
                
                // Reduce health based on metric values
                for metric in metrics do
                    match metric.Name with
                    | "system.memory.working_set" when metric.Value > 400.0 ->
                        healthScore <- healthScore - 0.1
                    | "system.cpu.usage" when metric.Value > 80.0 ->
                        healthScore <- healthScore - 0.15
                    | _ -> ()
                
                let finalHealthScore = Math.Max(0.0, Math.Min(1.0, healthScore))
                
                // Generate proof for health calculation
                let! healthProof =
                    ProofExtensions.generateSystemHealthProof
                        context.ProofGenerator
                        (Map [
                            ("healthScore", box finalHealthScore)
                            ("activeAlerts", box activeAlerts.Length)
                            ("metricsCount", box metrics.Length)
                        ])
                        context.CorrelationId
                
                let proofId = match healthProof with
                              | Success (proof, _) -> Some proof.ProofId
                              | Failure _ -> None
                
                let componentHealth = Map [
                    ("Memory", if metrics |> List.exists (fun m -> m.Name = "system.memory.working_set" && m.Value > 400.0) then 0.7 else 1.0)
                    ("CPU", if metrics |> List.exists (fun m -> m.Name = "system.cpu.usage" && m.Value > 80.0) then 0.6 else 1.0)
                    ("Threads", if metrics |> List.exists (fun m -> m.Name = "system.threads.count" && m.Value > 80.0) then 0.8 else 1.0)
                ]
                
                let performanceMetrics = 
                    metrics 
                    |> List.map (fun m -> (m.Name, m.Value))
                    |> Map.ofList
                
                let resourceUtilization = Map [
                    ("memory", metrics |> List.tryFind (fun m -> m.Name = "system.memory.working_set") |> Option.map (fun m -> m.Value / 1024.0) |> Option.defaultValue 0.0)
                    ("cpu", metrics |> List.tryFind (fun m -> m.Name = "system.cpu.usage") |> Option.map (fun m -> m.Value) |> Option.defaultValue 0.0)
                ]
                
                let healthStatus = {
                    OverallHealth = finalHealthScore
                    ComponentHealth = componentHealth
                    ActiveAlerts = activeAlerts
                    LastUpdate = DateTime.UtcNow
                    Uptime = DateTime.UtcNow - context.StartTime
                    PerformanceMetrics = performanceMetrics
                    ResourceUtilization = resourceUtilization
                    ProofId = proofId
                    CorrelationId = context.CorrelationId
                }
                
                return Success (healthStatus, Map [("healthScore", box finalHealthScore)])
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "HealthCalculationError" "Failed to calculate system health" (Some ex), ex)
                let error = ExecutionError ($"Health calculation failed: {ex.Message}", Some ex)
                return Failure (error, context.CorrelationId)
        }
    
    /// Unified Monitoring Manager implementation
    type UnifiedMonitoringManager(logger: ITarsLogger, configManager: UnifiedConfigurationManager, proofGenerator: UnifiedProofGenerator) =
        
        let context = createMonitoringContext logger configManager proofGenerator
        let metrics = ConcurrentQueue<MonitoringMetric>()
        let alerts = ConcurrentQueue<MonitoringAlert>()
        let mutable lastHealthStatus = None
        let mutable isMonitoring = false
        let cancellationTokenSource = new CancellationTokenSource()
        
        /// Start monitoring
        member this.StartMonitoringAsync() : Task<TarsResult<unit, TarsError>> =
            task {
                try
                    if isMonitoring then
                        return Success ((), Map [("status", box "Already monitoring")])
                    
                    isMonitoring <- true
                    context.Logger.LogInformation(context.CorrelationId, "Starting unified monitoring system")
                    
                    // Start background monitoring tasks
                    let monitoringTask = this.RunMonitoringLoopAsync(cancellationTokenSource.Token)
                    
                    return Success ((), Map [("status", box "Monitoring started")])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to start monitoring: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Stop monitoring
        member this.StopMonitoringAsync() : Task<TarsResult<unit, TarsError>> =
            task {
                try
                    if not isMonitoring then
                        return Success ((), Map [("status", box "Not monitoring")])
                    
                    isMonitoring <- false
                    cancellationTokenSource.Cancel()
                    context.Logger.LogInformation(context.CorrelationId, "Stopping unified monitoring system")
                    
                    return Success ((), Map [("status", box "Monitoring stopped")])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to stop monitoring: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Run monitoring loop
        member private this.RunMonitoringLoopAsync(cancellationToken: CancellationToken) : Task =
            task {
                try
                    while not cancellationToken.IsCancellationRequested && isMonitoring do
                        // Collect metrics
                        let! metricsResult = collectSystemMetrics context
                        match metricsResult with
                        | Success (newMetrics, _) ->
                            for metric in newMetrics do
                                metrics.Enqueue(metric)
                            
                            // Limit metrics in memory
                            while metrics.Count > context.Configuration.MaxMetricsInMemory do
                                metrics.TryDequeue() |> ignore
                            
                            // Evaluate alerts
                            let currentAlerts = alerts.ToArray() |> Array.toList
                            let! alertResult = evaluateAlerts context newMetrics currentAlerts
                            match alertResult with
                            | Success ((newAlerts, updatedAlerts), _) ->
                                for alert in newAlerts @ updatedAlerts do
                                    alerts.Enqueue(alert)
                                
                                // Limit alerts in memory
                                while alerts.Count > context.Configuration.MaxAlertsInMemory do
                                    alerts.TryDequeue() |> ignore
                            
                            | Failure (error, _) ->
                                context.Logger.LogError(context.CorrelationId, error, Exception("Alert evaluation failed"))
                            
                            // Calculate health status
                            let allAlerts = alerts.ToArray() |> Array.toList
                            let! healthResult = calculateSystemHealth context newMetrics allAlerts
                            match healthResult with
                            | Success (healthStatus, _) ->
                                lastHealthStatus <- Some healthStatus
                            | Failure (error, _) ->
                                context.Logger.LogError(context.CorrelationId, error, Exception("Health calculation failed"))
                        
                        | Failure (error, _) ->
                            context.Logger.LogError(context.CorrelationId, error, Exception("Metric collection failed"))
                        
                        // Wait for next collection interval
                        do! Task.Delay(context.Configuration.MetricCollectionInterval, cancellationToken)
                
                with
                | :? OperationCanceledException -> ()
                | ex ->
                    context.Logger.LogError(context.CorrelationId, TarsError.create "MonitoringLoopError" "Monitoring loop failed" (Some ex), ex)
            }
        
        /// Get current system health
        member this.GetSystemHealthAsync() : Task<TarsResult<SystemHealthStatus, TarsError>> =
            task {
                try
                    match lastHealthStatus with
                    | Some health -> 
                        return Success (health, Map [("lastUpdate", box health.LastUpdate)])
                    | None ->
                        // Generate health status on demand
                        let currentMetrics = metrics.ToArray() |> Array.toList
                        let currentAlerts = alerts.ToArray() |> Array.toList
                        let! healthResult = calculateSystemHealth context currentMetrics currentAlerts
                        return healthResult
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to get system health: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get recent metrics
        member this.GetRecentMetricsAsync(count: int) : Task<TarsResult<MonitoringMetric list, TarsError>> =
            task {
                try
                    let recentMetrics = 
                        metrics.ToArray() 
                        |> Array.sortByDescending (fun m -> m.Timestamp)
                        |> Array.take (Math.Min(count, metrics.Count))
                        |> Array.toList
                    
                    return Success (recentMetrics, Map [("count", box recentMetrics.Length)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to get recent metrics: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get active alerts
        member this.GetActiveAlertsAsync() : Task<TarsResult<MonitoringAlert list, TarsError>> =
            task {
                try
                    let activeAlerts = 
                        alerts.ToArray() 
                        |> Array.filter (fun a -> a.IsActive)
                        |> Array.sortByDescending (fun a -> a.Timestamp)
                        |> Array.toList
                    
                    return Success (activeAlerts, Map [("count", box activeAlerts.Length)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to get active alerts: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get monitoring statistics
        member this.GetStatistics() : Map<string, obj> =
            Map [
                ("isMonitoring", box isMonitoring)
                ("totalMetrics", box metrics.Count)
                ("totalAlerts", box alerts.Count)
                ("activeAlerts", box (alerts.ToArray() |> Array.filter (fun a -> a.IsActive) |> Array.length))
                ("uptime", box (DateTime.UtcNow - context.StartTime).TotalMinutes)
                ("lastHealthUpdate", box (lastHealthStatus |> Option.map (fun h -> h.LastUpdate) |> Option.defaultValue DateTime.MinValue))
                ("correlationId", box context.CorrelationId)
                ("configurationValid", box true)
            ]
        
        /// Dispose resources
        interface IDisposable with
            member this.Dispose() =
                cancellationTokenSource.Cancel()
                cancellationTokenSource.Dispose()

