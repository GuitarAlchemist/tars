namespace TarsEngine.FSharp.Core

open System
open System.Diagnostics
open Microsoft.Extensions.Logging

/// Improved system health monitor with better scoring
type ImprovedSystemHealthMonitor(logger: ILogger<ImprovedSystemHealthMonitor>) =
    let mutable lastOptimizationTime = DateTime.UtcNow

    /// Simple memory optimization
    member private this.OptimizeMemorySimple() =
        let memoryBefore = GC.GetTotalMemory(false) / 1024L / 1024L

        // Aggressive garbage collection
        for i in 1..3 do
            GC.Collect(2, GCCollectionMode.Forced, true, true)
            GC.WaitForPendingFinalizers()
            System.Threading.Thread.Sleep(50)

        let memoryAfter = GC.GetTotalMemory(true) / 1024L / 1024L
        let reduction = memoryBefore - memoryAfter
        let reductionPercent = if memoryBefore > 0L then float reduction / float memoryBefore * 100.0 else 0.0

        (memoryBefore, memoryAfter, reduction, reductionPercent)

    /// Get memory statistics
    member private this.GetMemoryStats() =
        let totalMemory = GC.GetTotalMemory(false) / 1024L / 1024L
        let gen0Collections = GC.CollectionCount(0)
        let gen1Collections = GC.CollectionCount(1)
        let gen2Collections = GC.CollectionCount(2)

        {|
            TotalMemoryMB = totalMemory
            Gen0Collections = gen0Collections
            Gen1Collections = gen1Collections
            Gen2Collections = gen2Collections
            IsWithinLimits = totalMemory < 500L
            MemoryPressure = if totalMemory > 1000L then "HIGH" elif totalMemory > 500L then "MEDIUM" else "LOW"
        |}

    /// Get improved system health with better scoring algorithm
    member this.GetImprovedSystemHealth() =
        try
            let currentProcess = Process.GetCurrentProcess()
            let memoryStats = this.GetMemoryStats()
            
            // CPU usage calculation (simplified - using process CPU time)
            let cpuUsage =
                try
                    let startTime = currentProcess.TotalProcessorTime
                    System.Threading.Thread.Sleep(100)
                    let endTime = currentProcess.TotalProcessorTime
                    let cpuTime = (endTime - startTime).TotalMilliseconds
                    Math.Min(100.0, cpuTime / 10.0) // Rough approximation
                with
                | _ -> 5.0 // Default low CPU usage
            
            // Thread count
            let threadCount = currentProcess.Threads.Count
            
            // Calculate improved health score
            let memoryScore = 
                if memoryStats.TotalMemoryMB < 500L then 100.0
                elif memoryStats.TotalMemoryMB < 1000L then 80.0
                elif memoryStats.TotalMemoryMB < 1500L then 60.0
                else 40.0
            
            let cpuScore = Math.Max(0.0, 100.0 - cpuUsage)
            let threadScore = if threadCount < 20 then 100.0 elif threadCount < 50 then 80.0 else 60.0
            
            // Weighted health score (memory is most important)
            let overallScore = (memoryScore * 0.6) + (cpuScore * 0.2) + (threadScore * 0.2)
            
            {|
                MemoryUsageMB = memoryStats.TotalMemoryMB
                CpuUsagePercent = cpuUsage
                ThreadCount = threadCount
                MemoryScore = memoryScore
                CpuScore = cpuScore
                ThreadScore = threadScore
                OverallHealthScore = overallScore
                MemoryPressure = memoryStats.MemoryPressure
                IsHealthy = overallScore >= 80.0
                Gen0Collections = memoryStats.Gen0Collections
                Gen1Collections = memoryStats.Gen1Collections
                Gen2Collections = memoryStats.Gen2Collections
            |}
        with
        | ex ->
            logger.LogError(ex, "Failed to get improved system health")
            {|
                MemoryUsageMB = 0L
                CpuUsagePercent = 0.0
                ThreadCount = 0
                MemoryScore = 0.0
                CpuScore = 0.0
                ThreadScore = 0.0
                OverallHealthScore = 0.0
                MemoryPressure = "UNKNOWN"
                IsHealthy = false
                Gen0Collections = 0
                Gen1Collections = 0
                Gen2Collections = 0
            |}
    
    /// Perform comprehensive system optimization
    member this.OptimizeSystemComprehensively() =
        try
            logger.LogInformation("Starting comprehensive system optimization")
            
            let healthBefore = this.GetImprovedSystemHealth()
            
            // Perform aggressive memory optimization
            let (memBefore, memAfter, reduction, reductionPercent) = this.OptimizeMemorySimple()
            
            // Additional optimizations
            this.OptimizeThreads()
            this.OptimizeProcessPriority()
            
            let healthAfter = this.GetImprovedSystemHealth()
            
            lastOptimizationTime <- DateTime.UtcNow
            
            logger.LogInformation(sprintf "Comprehensive optimization complete: Health %.1f%%→%.1f%%, Memory %dMB→%dMB" 
                healthBefore.OverallHealthScore healthAfter.OverallHealthScore memBefore memAfter)
            
            {|
                HealthBefore = healthBefore.OverallHealthScore
                HealthAfter = healthAfter.OverallHealthScore
                MemoryBefore = memBefore
                MemoryAfter = memAfter
                MemoryReduction = reduction
                ReductionPercent = reductionPercent
                OptimizationSuccessful = healthAfter.OverallHealthScore > healthBefore.OverallHealthScore
                MemoryWithinLimits = memAfter < 500L
            |}
        with
        | ex ->
            logger.LogError(ex, "Comprehensive system optimization failed")
            {|
                HealthBefore = 0.0
                HealthAfter = 0.0
                MemoryBefore = 0L
                MemoryAfter = 0L
                MemoryReduction = 0L
                ReductionPercent = 0.0
                OptimizationSuccessful = false
                MemoryWithinLimits = false
            |}
    
    /// Optimize thread usage
    member private this.OptimizeThreads() =
        try
            // Set thread pool limits
            let minWorker, minIO = System.Threading.ThreadPool.GetMinThreads()
            let maxWorker, maxIO = System.Threading.ThreadPool.GetMaxThreads()
            
            // Optimize thread pool settings
            System.Threading.ThreadPool.SetMinThreads(Math.Min(minWorker, 4), Math.Min(minIO, 4)) |> ignore
            System.Threading.ThreadPool.SetMaxThreads(Math.Min(maxWorker, 20), Math.Min(maxIO, 20)) |> ignore
            
            logger.LogDebug("Thread pool optimized")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to optimize threads")
    
    /// Optimize process priority
    member private this.OptimizeProcessPriority() =
        try
            let currentProcess = Process.GetCurrentProcess()
            currentProcess.PriorityClass <- ProcessPriorityClass.Normal
            logger.LogDebug("Process priority optimized")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to optimize process priority")
    
    /// Enhanced health check with comprehensive analysis
    member this.HealthCheckComprehensive() =
        try
            let health = this.GetImprovedSystemHealth()
            
            // Determine if optimization is needed
            let needsOptimization = 
                health.OverallHealthScore < 80.0 ||
                health.MemoryUsageMB > 500L ||
                (DateTime.UtcNow - lastOptimizationTime).TotalMinutes > 5.0
            
            // Auto-optimize if needed
            let optimizationResult = 
                if needsOptimization then
                    Some (this.OptimizeSystemComprehensively())
                else
                    None
            
            let finalHealth = this.GetImprovedSystemHealth()
            
            {|
                IsHealthy = finalHealth.OverallHealthScore >= 80.0
                HealthScore = finalHealth.OverallHealthScore
                MemoryUsageMB = finalHealth.MemoryUsageMB
                CpuUsagePercent = finalHealth.CpuUsagePercent
                ThreadCount = finalHealth.ThreadCount
                MemoryPressure = finalHealth.MemoryPressure
                OptimizationPerformed = optimizationResult.IsSome
                OptimizationResult = optimizationResult
                Issues = [
                    if finalHealth.MemoryUsageMB > 500L then sprintf "High memory usage: %dMB" finalHealth.MemoryUsageMB
                    if finalHealth.CpuUsagePercent > 80.0 then sprintf "High CPU usage: %.1f%%" finalHealth.CpuUsagePercent
                    if finalHealth.ThreadCount > 50 then sprintf "High thread count: %d" finalHealth.ThreadCount
                    if finalHealth.OverallHealthScore < 80.0 then sprintf "Low health score: %.1f%%" finalHealth.OverallHealthScore
                ]
                Recommendations = [
                    if finalHealth.MemoryUsageMB > 500L then "Consider reducing memory usage or increasing available memory"
                    if finalHealth.Gen2Collections > 10 then "High Gen2 GC collections - consider memory optimization"
                    if finalHealth.ThreadCount > 30 then "Consider reducing thread usage"
                    "Regular system optimization recommended"
                ]
            |}
        with
        | ex ->
            logger.LogError(ex, "Comprehensive health check failed")
            {|
                IsHealthy = false
                HealthScore = 0.0
                MemoryUsageMB = 0L
                CpuUsagePercent = 0.0
                ThreadCount = 0
                MemoryPressure = "UNKNOWN"
                OptimizationPerformed = false
                OptimizationResult = None
                Issues = ["Health check failed: " + ex.Message]
                Recommendations = ["Fix health monitoring system"]
            |}
