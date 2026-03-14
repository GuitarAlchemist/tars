namespace TarsEngine.FSharp.Core

open System
open System.Runtime
open Microsoft.Extensions.Logging

/// Aggressive memory optimizer to reduce memory usage below 500MB
type AggressiveMemoryOptimizer(logger: ILogger<AggressiveMemoryOptimizer>) =
    
    /// Perform aggressive memory cleanup
    member this.OptimizeMemoryAggressively() =
        try
            let memoryBefore = GC.GetTotalMemory(false) / 1024L / 1024L
            logger.LogInformation(sprintf "Starting aggressive memory optimization from %dMB" memoryBefore)
            
            // Step 1: Force full garbage collection multiple times
            for i in 1..3 do
                GC.Collect(2, GCCollectionMode.Forced, true, true)
                GC.WaitForPendingFinalizers()
                System.Threading.Thread.Sleep(100)
            
            // Step 2: Compact Large Object Heap
            GCSettings.LargeObjectHeapCompactionMode <- GCLargeObjectHeapCompactionMode.CompactOnce
            GC.Collect(2, GCCollectionMode.Forced, true, true)
            
            // Step 3: Additional cleanup
            GC.WaitForPendingFinalizers()
            GC.Collect(2, GCCollectionMode.Forced, true, true)
            
            let memoryAfter = GC.GetTotalMemory(true) / 1024L / 1024L
            let reduction = memoryBefore - memoryAfter
            let reductionPercent = if memoryBefore > 0L then float reduction / float memoryBefore * 100.0 else 0.0
            
            logger.LogInformation(sprintf "Aggressive memory optimization: %dMB→%dMB, reduced %dMB (%.1f%%)" 
                memoryBefore memoryAfter reduction reductionPercent)
            
            (memoryBefore, memoryAfter, reduction, reductionPercent)
        with
        | ex ->
            logger.LogError(ex, "Aggressive memory optimization failed")
            (0L, 0L, 0L, 0.0)
    
    /// Check if memory usage is within acceptable limits
    member this.IsMemoryWithinLimits() =
        let currentMemory = GC.GetTotalMemory(false) / 1024L / 1024L
        currentMemory < 500L
    
    /// Get current memory statistics
    member this.GetMemoryStats() =
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
    
    /// Optimize memory for vector operations
    member this.OptimizeVectorMemory() =
        try
            logger.LogInformation("Optimizing memory for vector operations")
            
            // Clear any cached data
            GC.Collect(0, GCCollectionMode.Optimized)
            GC.Collect(1, GCCollectionMode.Optimized)
            
            // Force cleanup of generation 2
            GC.Collect(2, GCCollectionMode.Forced)
            GC.WaitForPendingFinalizers()
            
            let memoryAfter = GC.GetTotalMemory(true) / 1024L / 1024L
            logger.LogInformation(sprintf "Vector memory optimization complete: %dMB" memoryAfter)
            
            memoryAfter
        with
        | ex ->
            logger.LogError(ex, "Vector memory optimization failed")
            0L
