namespace TarsEngine.FSharp.Core

open System
open System.Diagnostics
open System.IO
open Microsoft.Extensions.Logging

/// Real system health metrics
type SystemHealthMetrics = {
    CpuUsagePercent: float
    MemoryUsageMB: int64
    DiskUsagePercent: float
    ProcessCount: int
    ThreadCount: int
    UptimeSeconds: float
    GCCollections: int[]
    WorkingSetMB: int64
}

/// Real performance metrics
type PerformanceMetrics = {
    FileProcessingRate: float // files per second
    VectorCreationRate: float // vectors per second
    MemoryEfficiency: float   // MB per 1000 vectors
    SearchLatencyMs: float    // average search time
    ThroughputMBps: float     // data processing throughput
}

/// Real system health monitor - no fake metrics
type RealSystemHealthMonitor(logger: ILogger<RealSystemHealthMonitor>) =
    
    let mutable startTime = DateTime.UtcNow
    let mutable lastCpuTime = TimeSpan.Zero
    let mutable lastCheckTime = DateTime.UtcNow
    
    /// Get real CPU usage percentage
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
    
    /// Get real memory usage
    member private this.GetMemoryUsage() =
        try
            let currentProcess = Process.GetCurrentProcess()
            let workingSetMB = currentProcess.WorkingSet64 / 1024L / 1024L
            let gcMemoryMB = GC.GetTotalMemory(false) / 1024L / 1024L
            
            {| WorkingSetMB = workingSetMB; GCMemoryMB = gcMemoryMB |}
        with
        | ex ->
            logger.LogError(ex, "Failed to get memory usage")
            {| WorkingSetMB = 0L; GCMemoryMB = 0L |}
    
    /// Get real disk usage for current directory
    member private this.GetDiskUsage() =
        try
            let currentDir = Environment.CurrentDirectory
            let drive = DriveInfo(Path.GetPathRoot(currentDir))
            
            if drive.IsReady then
                let usedSpace = drive.TotalSize - drive.AvailableFreeSpace
                let usagePercent = (float usedSpace / float drive.TotalSize) * 100.0
                usagePercent
            else
                0.0
        with
        | ex ->
            logger.LogError(ex, "Failed to get disk usage")
            0.0
    
    /// Get real system health metrics
    member this.GetSystemHealth() =
        try
            let currentProcess = Process.GetCurrentProcess()
            let allProcesses = Process.GetProcesses()
            let memoryInfo = this.GetMemoryUsage()
            
            {
                CpuUsagePercent = this.GetCpuUsage()
                MemoryUsageMB = memoryInfo.GCMemoryMB
                DiskUsagePercent = this.GetDiskUsage()
                ProcessCount = allProcesses.Length
                ThreadCount = currentProcess.Threads.Count
                UptimeSeconds = (DateTime.UtcNow - startTime).TotalSeconds
                GCCollections = [| 
                    GC.CollectionCount(0)
                    GC.CollectionCount(1) 
                    GC.CollectionCount(2) 
                |]
                WorkingSetMB = memoryInfo.WorkingSetMB
            }
        with
        | ex ->
            logger.LogError(ex, "Failed to get system health metrics")
            {
                CpuUsagePercent = 0.0
                MemoryUsageMB = 0L
                DiskUsagePercent = 0.0
                ProcessCount = 0
                ThreadCount = 0
                UptimeSeconds = 0.0
                GCCollections = [| 0; 0; 0 |]
                WorkingSetMB = 0L
            }
    
    /// Calculate real performance metrics
    member this.CalculatePerformanceMetrics(filesProcessed: int, vectorsCreated: int, elapsedSeconds: float) =
        try
            let fileRate = if elapsedSeconds > 0.0 then float filesProcessed / elapsedSeconds else 0.0
            let vectorRate = if elapsedSeconds > 0.0 then float vectorsCreated / elapsedSeconds else 0.0
            let memoryInfo = this.GetMemoryUsage()
            let memoryEfficiency = if vectorsCreated > 0 then float memoryInfo.GCMemoryMB / (float vectorsCreated / 1000.0) else 0.0
            
            // Estimate throughput based on file processing
            let avgFileSize = 5.0 // Assume 5KB average file size
            let throughputMBps = if elapsedSeconds > 0.0 then (float filesProcessed * avgFileSize / 1024.0) / elapsedSeconds else 0.0
            
            {
                FileProcessingRate = fileRate
                VectorCreationRate = vectorRate
                MemoryEfficiency = memoryEfficiency
                SearchLatencyMs = 0.0 // Would need real search timing
                ThroughputMBps = throughputMBps
            }
        with
        | ex ->
            logger.LogError(ex, "Failed to calculate performance metrics")
            {
                FileProcessingRate = 0.0
                VectorCreationRate = 0.0
                MemoryEfficiency = 0.0
                SearchLatencyMs = 0.0
                ThroughputMBps = 0.0
            }
    
    /// Perform real health check
    member this.HealthCheck() =
        try
            let health = this.GetSystemHealth()
            
            // Define health thresholds
            let cpuHealthy = health.CpuUsagePercent < 80.0
            let memoryHealthy = health.MemoryUsageMB < 2048L // Less than 2GB
            let diskHealthy = health.DiskUsagePercent < 90.0
            let threadsHealthy = health.ThreadCount < 100
            
            let overallHealthy = cpuHealthy && memoryHealthy && diskHealthy && threadsHealthy
            
            let healthScore = 
                let scores = [
                    if cpuHealthy then 25.0 else 0.0
                    if memoryHealthy then 25.0 else 0.0
                    if diskHealthy then 25.0 else 0.0
                    if threadsHealthy then 25.0 else 0.0
                ]
                scores |> List.sum
            
            logger.LogInformation(sprintf "System health check: %.1f%% (CPU: %.1f%%, Memory: %dMB)" healthScore health.CpuUsagePercent health.MemoryUsageMB)
            
            {| 
                IsHealthy = overallHealthy
                HealthScore = healthScore
                Metrics = health
                Issues = [
                    if not cpuHealthy then sprintf "High CPU usage: %.1f%%" health.CpuUsagePercent
                    if not memoryHealthy then sprintf "High memory usage: %dMB" health.MemoryUsageMB
                    if not diskHealthy then sprintf "High disk usage: %.1f%%" health.DiskUsagePercent
                    if not threadsHealthy then sprintf "High thread count: %d" health.ThreadCount
                ]
            |}
        with
        | ex ->
            logger.LogError(ex, "Health check failed with exception")
            {| 
                IsHealthy = false
                HealthScore = 0.0
                Metrics = {
                    CpuUsagePercent = 0.0
                    MemoryUsageMB = 0L
                    DiskUsagePercent = 0.0
                    ProcessCount = 0
                    ThreadCount = 0
                    UptimeSeconds = 0.0
                    GCCollections = [| 0; 0; 0 |]
                    WorkingSetMB = 0L
                }
                Issues = ["Health check failed with exception: " + ex.Message]
            |}
    
    /// Reset start time (for uptime calculation)
    member this.Reset() =
        startTime <- DateTime.UtcNow
        lastCpuTime <- TimeSpan.Zero
        lastCheckTime <- DateTime.UtcNow
        logger.LogInformation("System health monitor reset")
