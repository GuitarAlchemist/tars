namespace TarsEngine.FSharp.WindowsService.Monitoring

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Performance metric data point
/// </summary>
type PerformanceMetric = {
    Name: string
    Value: float
    Unit: string
    Timestamp: DateTime
    Category: string
    Tags: Map<string, string>
}

/// <summary>
/// Performance counter configuration
/// </summary>
type PerformanceCounterConfig = {
    CategoryName: string
    CounterName: string
    InstanceName: string option
    MetricName: string
    Unit: string
    SampleInterval: TimeSpan
}

/// <summary>
/// System performance snapshot
/// </summary>
type SystemPerformanceSnapshot = {
    Timestamp: DateTime
    CpuUsagePercent: float
    MemoryUsagePercent: float
    MemoryUsageMB: float
    DiskUsagePercent: float
    DiskReadBytesPerSec: float
    DiskWriteBytesPerSec: float
    NetworkBytesPerSec: float
    ProcessCount: int
    ThreadCount: int
    HandleCount: int
    GCCollections: Map<int, int>
    GCMemory: int64
}

/// <summary>
/// Application performance metrics
/// </summary>
type ApplicationPerformanceMetrics = {
    Timestamp: DateTime
    RequestsPerSecond: float
    AverageResponseTimeMs: float
    ErrorRate: float
    ActiveConnections: int
    QueueLength: int
    WorkerUtilization: float
    CacheHitRate: float
    DatabaseConnectionsActive: int
    DatabaseQueryTimeMs: float
}

/// <summary>
/// Performance trend analysis
/// </summary>
type PerformanceTrendAnalysis = {
    MetricName: string
    TimeWindow: TimeSpan
    TrendDirection: TrendDirection
    ChangePercent: float
    CurrentValue: float
    PreviousValue: float
    Volatility: float
    Prediction: float option
}

/// <summary>
/// Advanced performance metrics collection and analysis system
/// </summary>
type PerformanceCollector(logger: ILogger<PerformanceCollector>) =
    
    let performanceCounters = ConcurrentDictionary<string, PerformanceCounter>()
    let metricHistory = ConcurrentDictionary<string, ConcurrentQueue<PerformanceMetric>>()
    let systemSnapshots = ConcurrentQueue<SystemPerformanceSnapshot>()
    let applicationMetrics = ConcurrentQueue<ApplicationPerformanceMetrics>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable collectionTask: Task option = None
    let mutable config: MonitoringConfig option = None
    
    let maxMetricHistory = 10000
    let maxSnapshotHistory = 2880 // 48 hours of minute-by-minute data
    
    // Performance counter configurations
    let defaultCounterConfigs = [
        { CategoryName = "Processor"; CounterName = "% Processor Time"; InstanceName = Some "_Total"; MetricName = "CpuUsagePercent"; Unit = "percent"; SampleInterval = TimeSpan.FromSeconds(5.0) }
        { CategoryName = "Memory"; CounterName = "Available MBytes"; InstanceName = None; MetricName = "AvailableMemoryMB"; Unit = "megabytes"; SampleInterval = TimeSpan.FromSeconds(5.0) }
        { CategoryName = "Memory"; CounterName = "% Committed Bytes In Use"; InstanceName = None; MetricName = "MemoryUsagePercent"; Unit = "percent"; SampleInterval = TimeSpan.FromSeconds(5.0) }
        { CategoryName = "LogicalDisk"; CounterName = "% Free Space"; InstanceName = Some "_Total"; MetricName = "DiskFreeSpacePercent"; Unit = "percent"; SampleInterval = TimeSpan.FromSeconds(30.0) }
        { CategoryName = "LogicalDisk"; CounterName = "Disk Read Bytes/sec"; InstanceName = Some "_Total"; MetricName = "DiskReadBytesPerSec"; Unit = "bytes/sec"; SampleInterval = TimeSpan.FromSeconds(5.0) }
        { CategoryName = "LogicalDisk"; CounterName = "Disk Write Bytes/sec"; InstanceName = Some "_Total"; MetricName = "DiskWriteBytesPerSec"; Unit = "bytes/sec"; SampleInterval = TimeSpan.FromSeconds(5.0) }
        { CategoryName = "Network Interface"; CounterName = "Bytes Total/sec"; InstanceName = None; MetricName = "NetworkBytesPerSec"; Unit = "bytes/sec"; SampleInterval = TimeSpan.FromSeconds(5.0) }
    ]
    
    /// Start the performance collector
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting performance collector...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Initialize performance counters
            this.InitializePerformanceCounters()
            
            // Start collection loop
            let collectionLoop = this.CollectionLoopAsync(cancellationTokenSource.Value.Token)
            collectionTask <- Some collectionLoop
            
            logger.LogInformation("Performance collector started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start performance collector")
            isRunning <- false
            raise
    }
    
    /// Stop the performance collector
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping performance collector...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for collection task to complete
            match collectionTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Performance collection task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for performance collection task to complete")
            | None -> ()
            
            // Dispose performance counters
            this.DisposePerformanceCounters()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            collectionTask <- None
            
            logger.LogInformation("Performance collector stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping performance collector")
    }
    
    /// Initialize performance counters
    member private this.InitializePerformanceCounters() =
        try
            logger.LogInformation("Initializing performance counters...")
            
            for counterConfig in defaultCounterConfigs do
                try
                    let counter = 
                        match counterConfig.InstanceName with
                        | Some instanceName -> new PerformanceCounter(counterConfig.CategoryName, counterConfig.CounterName, instanceName)
                        | None -> new PerformanceCounter(counterConfig.CategoryName, counterConfig.CounterName)
                    
                    // Initialize the counter (first call often returns 0)
                    counter.NextValue() |> ignore
                    
                    performanceCounters.[counterConfig.MetricName] <- counter
                    metricHistory.[counterConfig.MetricName] <- ConcurrentQueue<PerformanceMetric>()
                    
                    logger.LogDebug($"Initialized performance counter: {counterConfig.MetricName}")
                    
                with
                | ex ->
                    logger.LogWarning(ex, $"Failed to initialize performance counter: {counterConfig.MetricName}")
            
            logger.LogInformation($"Initialized {performanceCounters.Count} performance counters")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize performance counters")
    
    /// Dispose performance counters
    member private this.DisposePerformanceCounters() =
        for kvp in performanceCounters do
            try
                kvp.Value.Dispose()
            with
            | ex -> logger.LogWarning(ex, $"Error disposing performance counter: {kvp.Key}")
        
        performanceCounters.Clear()
    
    /// Main collection loop
    member private this.CollectionLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting performance collection loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Collect system performance snapshot
                    let! systemSnapshot = this.CollectSystemPerformanceAsync()
                    systemSnapshots.Enqueue(systemSnapshot)
                    
                    // Keep snapshot history manageable
                    while systemSnapshots.Count > maxSnapshotHistory do
                        systemSnapshots.TryDequeue() |> ignore
                    
                    // Collect individual metrics
                    this.CollectIndividualMetrics()
                    
                    // Collect application metrics
                    let! appMetrics = this.CollectApplicationMetricsAsync()
                    applicationMetrics.Enqueue(appMetrics)
                    
                    // Keep application metrics history manageable
                    while applicationMetrics.Count > maxSnapshotHistory do
                        applicationMetrics.TryDequeue() |> ignore
                    
                    // Wait for next collection cycle
                    do! Task.Delay(TimeSpan.FromSeconds(5.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in performance collection loop")
                    do! Task.Delay(TimeSpan.FromSeconds(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Performance collection loop cancelled")
        | ex ->
            logger.LogError(ex, "Performance collection loop failed")
    }
    
    /// Collect system performance snapshot
    member private this.CollectSystemPerformanceAsync() = task {
        try
            let timestamp = DateTime.UtcNow
            
            // Get CPU usage
            let cpuUsage = 
                match performanceCounters.TryGetValue("CpuUsagePercent") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 0.0
            
            // Get memory usage
            let availableMemoryMB = 
                match performanceCounters.TryGetValue("AvailableMemoryMB") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 0.0
            
            let memoryUsagePercent = 
                match performanceCounters.TryGetValue("MemoryUsagePercent") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 0.0
            
            // Get disk usage
            let diskFreeSpacePercent = 
                match performanceCounters.TryGetValue("DiskFreeSpacePercent") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 100.0
            
            let diskUsagePercent = 100.0 - diskFreeSpacePercent
            
            let diskReadBytesPerSec = 
                match performanceCounters.TryGetValue("DiskReadBytesPerSec") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 0.0
            
            let diskWriteBytesPerSec = 
                match performanceCounters.TryGetValue("DiskWriteBytesPerSec") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 0.0
            
            // Get network usage
            let networkBytesPerSec = 
                match performanceCounters.TryGetValue("NetworkBytesPerSec") with
                | true, counter -> float (counter.NextValue())
                | false, _ -> 0.0
            
            // Get process information
            let currentProcess = Process.GetCurrentProcess()
            let allProcesses = Process.GetProcesses()
            
            let processCount = allProcesses.Length
            let threadCount = currentProcess.Threads.Count
            let handleCount = currentProcess.HandleCount
            
            // Get GC information
            let gcCollections = 
                [0..2] 
                |> List.map (fun gen -> (gen, GC.CollectionCount(gen)))
                |> Map.ofList
            
            let gcMemory = GC.GetTotalMemory(false)
            
            // Calculate memory usage in MB
            let totalMemoryMB = float (GC.GetTotalMemory(false)) / (1024.0 * 1024.0)
            
            return {
                Timestamp = timestamp
                CpuUsagePercent = cpuUsage
                MemoryUsagePercent = memoryUsagePercent
                MemoryUsageMB = totalMemoryMB
                DiskUsagePercent = diskUsagePercent
                DiskReadBytesPerSec = diskReadBytesPerSec
                DiskWriteBytesPerSec = diskWriteBytesPerSec
                NetworkBytesPerSec = networkBytesPerSec
                ProcessCount = processCount
                ThreadCount = threadCount
                HandleCount = handleCount
                GCCollections = gcCollections
                GCMemory = gcMemory
            }
            
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting system performance snapshot")
            return {
                Timestamp = DateTime.UtcNow
                CpuUsagePercent = 0.0
                MemoryUsagePercent = 0.0
                MemoryUsageMB = 0.0
                DiskUsagePercent = 0.0
                DiskReadBytesPerSec = 0.0
                DiskWriteBytesPerSec = 0.0
                NetworkBytesPerSec = 0.0
                ProcessCount = 0
                ThreadCount = 0
                HandleCount = 0
                GCCollections = Map.empty
                GCMemory = 0L
            }
    }
    
    /// Collect individual metrics
    member private this.CollectIndividualMetrics() =
        for kvp in performanceCounters do
            try
                let value = float (kvp.Value.NextValue())
                let metric = {
                    Name = kvp.Key
                    Value = value
                    Unit = "unknown" // Would be looked up from configuration
                    Timestamp = DateTime.UtcNow
                    Category = "System"
                    Tags = Map.empty
                }
                
                match metricHistory.TryGetValue(kvp.Key) with
                | true, history ->
                    history.Enqueue(metric)
                    
                    // Keep history size manageable
                    while history.Count > maxMetricHistory do
                        history.TryDequeue() |> ignore
                | false, _ -> ()
                
            with
            | ex ->
                logger.LogWarning(ex, $"Error collecting metric: {kvp.Key}")
    
    /// Collect application metrics
    member private this.CollectApplicationMetricsAsync() = task {
        try
            // These would be collected from actual application components
            // For now, we'll return default values
            return {
                Timestamp = DateTime.UtcNow
                RequestsPerSecond = 0.0
                AverageResponseTimeMs = 0.0
                ErrorRate = 0.0
                ActiveConnections = 0
                QueueLength = 0
                WorkerUtilization = 0.0
                CacheHitRate = 0.0
                DatabaseConnectionsActive = 0
                DatabaseQueryTimeMs = 0.0
            }
            
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting application metrics")
            return {
                Timestamp = DateTime.UtcNow
                RequestsPerSecond = 0.0
                AverageResponseTimeMs = 0.0
                ErrorRate = 0.0
                ActiveConnections = 0
                QueueLength = 0
                WorkerUtilization = 0.0
                CacheHitRate = 0.0
                DatabaseConnectionsActive = 0
                DatabaseQueryTimeMs = 0.0
            }
    }
    
    /// Get current metrics
    member this.GetCurrentMetricsAsync(cancellationToken: CancellationToken) = task {
        try
            let! systemSnapshot = this.CollectSystemPerformanceAsync()
            let! appMetrics = this.CollectApplicationMetricsAsync()
            
            return Map.ofList [
                ("CpuUsagePercent", systemSnapshot.CpuUsagePercent :> obj)
                ("MemoryUsagePercent", systemSnapshot.MemoryUsagePercent :> obj)
                ("MemoryUsageMB", systemSnapshot.MemoryUsageMB :> obj)
                ("DiskUsagePercent", systemSnapshot.DiskUsagePercent :> obj)
                ("DiskReadBytesPerSec", systemSnapshot.DiskReadBytesPerSec :> obj)
                ("DiskWriteBytesPerSec", systemSnapshot.DiskWriteBytesPerSec :> obj)
                ("NetworkBytesPerSec", systemSnapshot.NetworkBytesPerSec :> obj)
                ("ProcessCount", systemSnapshot.ProcessCount :> obj)
                ("ThreadCount", systemSnapshot.ThreadCount :> obj)
                ("HandleCount", systemSnapshot.HandleCount :> obj)
                ("GCMemory", systemSnapshot.GCMemory :> obj)
                ("RequestsPerSecond", appMetrics.RequestsPerSecond :> obj)
                ("AverageResponseTimeMs", appMetrics.AverageResponseTimeMs :> obj)
                ("ErrorRate", appMetrics.ErrorRate :> obj)
                ("ActiveConnections", appMetrics.ActiveConnections :> obj)
                ("QueueLength", appMetrics.QueueLength :> obj)
                ("WorkerUtilization", appMetrics.WorkerUtilization :> obj)
            ]
            
        with
        | ex ->
            logger.LogError(ex, "Error getting current metrics")
            return Map.empty
    }
    
    /// Get metric history
    member this.GetMetricHistory(metricName: string, hours: int) =
        match metricHistory.TryGetValue(metricName) with
        | true, history ->
            let cutoffTime = DateTime.UtcNow.AddHours(-float hours)
            history
            |> Seq.filter (fun m -> m.Timestamp >= cutoffTime)
            |> Seq.sortBy (fun m -> m.Timestamp)
            |> List.ofSeq
        | false, _ -> []
    
    /// Get system performance snapshots
    member this.GetSystemPerformanceSnapshots(hours: int) =
        let cutoffTime = DateTime.UtcNow.AddHours(-float hours)
        systemSnapshots
        |> Seq.filter (fun s -> s.Timestamp >= cutoffTime)
        |> Seq.sortBy (fun s -> s.Timestamp)
        |> List.ofSeq
    
    /// Get application metrics history
    member this.GetApplicationMetricsHistory(hours: int) =
        let cutoffTime = DateTime.UtcNow.AddHours(-float hours)
        applicationMetrics
        |> Seq.filter (fun m -> m.Timestamp >= cutoffTime)
        |> Seq.sortBy (fun m -> m.Timestamp)
        |> List.ofSeq
    
    /// Analyze performance trends
    member this.AnalyzePerformanceTrends(metricName: string, hours: int) =
        let metrics = this.GetMetricHistory(metricName, hours)
        
        if metrics.Length < 10 then
            {
                MetricName = metricName
                TimeWindow = TimeSpan.FromHours(float hours)
                TrendDirection = Stable
                ChangePercent = 0.0
                CurrentValue = 0.0
                PreviousValue = 0.0
                Volatility = 0.0
                Prediction = None
            }
        else
            let values = metrics |> List.map (fun m -> m.Value)
            let currentValue = values |> List.last
            let previousValue = values |> List.head
            let changePercent = if previousValue <> 0.0 then ((currentValue - previousValue) / previousValue) * 100.0 else 0.0
            
            let trendDirection = 
                if changePercent > 10.0 then Improving
                elif changePercent < -10.0 then Declining
                elif changePercent < -25.0 then Critical
                else Stable
            
            // Calculate volatility (standard deviation)
            let mean = values |> List.average
            let variance = values |> List.map (fun v -> (v - mean) ** 2.0) |> List.average
            let volatility = sqrt variance
            
            // Simple linear prediction
            let prediction = 
                if values.Length >= 5 then
                    let recentValues = values |> List.rev |> List.take 5
                    let trend = (recentValues |> List.last) - (recentValues |> List.head)
                    Some (currentValue + trend)
                else None
            
            {
                MetricName = metricName
                TimeWindow = TimeSpan.FromHours(float hours)
                TrendDirection = trendDirection
                ChangePercent = changePercent
                CurrentValue = currentValue
                PreviousValue = previousValue
                Volatility = volatility
                Prediction = prediction
            }
    
    /// Get performance summary
    member this.GetPerformanceSummary() =
        let recentSnapshots = systemSnapshots |> Seq.take (min 60 systemSnapshots.Count) |> List.ofSeq
        
        if recentSnapshots.IsEmpty then
            Map.empty
        else
            let avgCpu = recentSnapshots |> List.averageBy (fun s -> s.CpuUsagePercent)
            let avgMemory = recentSnapshots |> List.averageBy (fun s -> s.MemoryUsagePercent)
            let avgDisk = recentSnapshots |> List.averageBy (fun s -> s.DiskUsagePercent)
            let avgNetwork = recentSnapshots |> List.averageBy (fun s -> s.NetworkBytesPerSec)
            
            Map.ofList [
                ("AverageCpuUsagePercent", avgCpu :> obj)
                ("AverageMemoryUsagePercent", avgMemory :> obj)
                ("AverageDiskUsagePercent", avgDisk :> obj)
                ("AverageNetworkBytesPerSec", avgNetwork :> obj)
                ("SampleCount", recentSnapshots.Length :> obj)
                ("TimeWindow", "Last Hour" :> obj)
            ]
    
    /// Get available metrics
    member this.GetAvailableMetrics() =
        metricHistory.Keys |> List.ofSeq
