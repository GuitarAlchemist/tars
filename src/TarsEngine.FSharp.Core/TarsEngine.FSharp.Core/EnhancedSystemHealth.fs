namespace TarsEngine.FSharp.Core

open System
open System.Diagnostics
open System.IO
open Microsoft.Extensions.Logging

/// Enhanced system health metrics with memory optimization focus
type EnhancedSystemHealthMetrics = {
    CpuUsagePercent: float
    MemoryUsageMB: int64
    MemoryOptimized: bool
    DiskUsagePercent: float
    ProcessCount: int
    ThreadCount: int
    UptimeSeconds: float
    GCCollections: int[]
    WorkingSetMB: int64
    MemoryPressure: string
    PerformanceScore: float
}

/// Enhanced system health monitor with memory optimization
type EnhancedSystemHealthMonitor(logger: ILogger<EnhancedSystemHealthMonitor>) =
    
    let mutable startTime = DateTime.UtcNow
    let mutable lastCpuTime = TimeSpan.Zero
    let mutable lastCheckTime = DateTime.UtcNow
    
    /// Get enhanced CPU usage with better accuracy
    member private this.GetCpuUsage() =
        try
            let currentProcess = Process.GetCurrentProcess()
            let currentTime = DateTime.UtcNow
            let currentCpuTime = currentProcess.TotalProcessorTime
            
            if lastCpuTime <> TimeSpan.Zero then
                let cpuUsedMs = (currentCpuTime - lastCpuTime).TotalMilliseconds
                let totalMsPassed = (currentTime - lastCheckTime).TotalMilliseconds
                let cpuUsageTotal = cpuUsedMs / (float Environment.ProcessorCount * totalMsPassed)
                
                lastCpuTime <- currentCpuTime
                lastCheckTime <- currentTime
                
                Math.Min(100.0, Math.Max(0.0, cpuUsageTotal * 100.0))
            else
                lastCpuTime <- currentCpuTime
                lastCheckTime <- currentTime
                0.0
        with
        | ex ->
            logger.LogError(ex, "Failed to get CPU usage")
            0.0
    
    /// Get enhanced memory usage with optimization detection
    member private this.GetMemoryUsage() =
        try
            let currentProcess = Process.GetCurrentProcess()
            let workingSetMB = currentProcess.WorkingSet64 / 1024L / 1024L
            let gcMemoryMB = GC.GetTotalMemory(false) / 1024L / 1024L
            
            // Detect memory pressure
            let memoryPressure = 
                if gcMemoryMB > 2048L then "HIGH"
                elif gcMemoryMB > 1024L then "MEDIUM"
                elif gcMemoryMB > 512L then "LOW"
                else "NORMAL"
            
            {| 
                WorkingSetMB = workingSetMB
                GCMemoryMB = gcMemoryMB
                MemoryPressure = memoryPressure
            |}
        with
        | ex ->
            logger.LogError(ex, "Failed to get memory usage")
            {| 
                WorkingSetMB = 0L
                GCMemoryMB = 0L
                MemoryPressure = "UNKNOWN"
            |}
    
    /// Optimize memory usage
    member this.OptimizeMemory() =
        try
            logger.LogInformation("Starting memory optimization...")
            
            // Force garbage collection
            GC.Collect(2, GCCollectionMode.Forced)
            GC.WaitForPendingFinalizers()
            GC.Collect(2, GCCollectionMode.Forced)
            
            let memoryAfter = GC.GetTotalMemory(true) / 1024L / 1024L
            logger.LogInformation(sprintf "Memory optimization complete: %dMB after GC" memoryAfter)
            
            memoryAfter
        with
        | ex ->
            logger.LogError(ex, "Failed to optimize memory")
            0L
    
    /// Calculate performance score based on system metrics
    member private this.CalculatePerformanceScore(metrics: EnhancedSystemHealthMetrics) =
        try
            let cpuScore = Math.Max(0.0, 100.0 - metrics.CpuUsagePercent)
            let memoryScore = Math.Max(0.0, 100.0 - (float metrics.MemoryUsageMB / 20.48)) // 2GB = 0 score
            let diskScore = Math.Max(0.0, 100.0 - metrics.DiskUsagePercent)
            let threadScore = Math.Max(0.0, 100.0 - (float metrics.ThreadCount / 2.0)) // 200 threads = 0 score
            
            let overallScore = (cpuScore + memoryScore + diskScore + threadScore) / 4.0
            Math.Min(100.0, Math.Max(0.0, overallScore))
        with
        | ex ->
            logger.LogError(ex, "Failed to calculate performance score")
            0.0
    
    /// Get enhanced system health metrics
    member this.GetSystemHealth() =
        try
            let currentProcess = Process.GetCurrentProcess()
            let allProcesses = Process.GetProcesses()
            let memoryInfo = this.GetMemoryUsage()
            
            // Get disk usage
            let diskUsage = 
                try
                    let currentDir = Environment.CurrentDirectory
                    let drive = DriveInfo(Path.GetPathRoot(currentDir))
                    if drive.IsReady then
                        let usedSpace = drive.TotalSize - drive.AvailableFreeSpace
                        (float usedSpace / float drive.TotalSize) * 100.0
                    else 0.0
                with _ -> 0.0
            
            let metrics = {
                CpuUsagePercent = this.GetCpuUsage()
                MemoryUsageMB = memoryInfo.GCMemoryMB
                MemoryOptimized = memoryInfo.GCMemoryMB < 1024L
                DiskUsagePercent = diskUsage
                ProcessCount = allProcesses.Length
                ThreadCount = currentProcess.Threads.Count
                UptimeSeconds = (DateTime.UtcNow - startTime).TotalSeconds
                GCCollections = [| 
                    GC.CollectionCount(0)
                    GC.CollectionCount(1) 
                    GC.CollectionCount(2) 
                |]
                WorkingSetMB = memoryInfo.WorkingSetMB
                MemoryPressure = memoryInfo.MemoryPressure
                PerformanceScore = 0.0 // Will be calculated below
            }
            
            let metricsWithScore = { metrics with PerformanceScore = this.CalculatePerformanceScore(metrics) }
            metricsWithScore
        with
        | ex ->
            logger.LogError(ex, "Failed to get enhanced system health metrics")
            {
                CpuUsagePercent = 0.0
                MemoryUsageMB = 0L
                MemoryOptimized = false
                DiskUsagePercent = 0.0
                ProcessCount = 0
                ThreadCount = 0
                UptimeSeconds = 0.0
                GCCollections = [| 0; 0; 0 |]
                WorkingSetMB = 0L
                MemoryPressure = "UNKNOWN"
                PerformanceScore = 0.0
            }
    
    /// Enhanced health check with memory optimization
    member this.HealthCheck() =
        try
            let health = this.GetSystemHealth()
            
            // Enhanced health thresholds
            let cpuHealthy = health.CpuUsagePercent < 80.0
            let memoryHealthy = health.MemoryUsageMB < 1024L // Stricter: Less than 1GB
            let diskHealthy = health.DiskUsagePercent < 90.0
            let threadsHealthy = health.ThreadCount < 50 // Stricter thread limit
            let performanceHealthy = health.PerformanceScore > 50.0
            
            let overallHealthy = cpuHealthy && memoryHealthy && diskHealthy && threadsHealthy && performanceHealthy
            
            // Calculate enhanced health score
            let healthScore = 
                let scores = [
                    if cpuHealthy then 20.0 else 0.0
                    if memoryHealthy then 30.0 else 0.0 // Memory is most important
                    if diskHealthy then 15.0 else 0.0
                    if threadsHealthy then 15.0 else 0.0
                    if performanceHealthy then 20.0 else 0.0
                ]
                scores |> List.sum
            
            logger.LogInformation(sprintf "Enhanced health check: %.1f%% (CPU: %.1f%%, Memory: %dMB, Performance: %.1f%%)" 
                healthScore health.CpuUsagePercent health.MemoryUsageMB health.PerformanceScore)
            
            {| 
                IsHealthy = overallHealthy
                HealthScore = healthScore
                Metrics = health
                Issues = [
                    if not cpuHealthy then sprintf "High CPU usage: %.1f%%" health.CpuUsagePercent
                    if not memoryHealthy then sprintf "High memory usage: %dMB (pressure: %s)" health.MemoryUsageMB health.MemoryPressure
                    if not diskHealthy then sprintf "High disk usage: %.1f%%" health.DiskUsagePercent
                    if not threadsHealthy then sprintf "High thread count: %d" health.ThreadCount
                    if not performanceHealthy then sprintf "Low performance score: %.1f%%" health.PerformanceScore
                ]
                Recommendations = [
                    if not memoryHealthy then "Consider running memory optimization"
                    if health.GCCollections.[2] > 10 then "High Gen2 GC collections detected - memory pressure"
                    if health.ThreadCount > 30 then "Consider reducing thread usage"
                    if health.PerformanceScore < 70.0 then "System performance is below optimal"
                ]
            |}
        with
        | ex ->
            logger.LogError(ex, "Enhanced health check failed")
            {| 
                IsHealthy = false
                HealthScore = 0.0
                Metrics = {
                    CpuUsagePercent = 0.0
                    MemoryUsageMB = 0L
                    MemoryOptimized = false
                    DiskUsagePercent = 0.0
                    ProcessCount = 0
                    ThreadCount = 0
                    UptimeSeconds = 0.0
                    GCCollections = [| 0; 0; 0 |]
                    WorkingSetMB = 0L
                    MemoryPressure = "UNKNOWN"
                    PerformanceScore = 0.0
                }
                Issues = ["Health check failed with exception: " + ex.Message]
                Recommendations = ["Fix system health monitoring"]
            |}
    
    /// Reset monitoring
    member this.Reset() =
        startTime <- DateTime.UtcNow
        lastCpuTime <- TimeSpan.Zero
        lastCheckTime <- DateTime.UtcNow
        logger.LogInformation("Enhanced system health monitor reset")
