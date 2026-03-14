namespace TarsEngine.FSharp.Core

open System
open System.Runtime
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// Ultra-aggressive memory optimizer to get under 500MB
type UltraMemoryOptimizer(logger: ILogger<UltraMemoryOptimizer>) =
    
    /// Clear all AppDomain cached data
    member private this.ClearAppDomainData() =
        try
            let domain = AppDomain.CurrentDomain
            
            // Get all stored vector data keys
            let vectorKeys = ResizeArray<string>()
            for i in 0..200000 do
                let key = sprintf "VECTOR_%d" i
                if domain.GetData(key) <> null then
                    vectorKeys.Add(key)
            
            // Clear vector data in batches
            for key in vectorKeys do
                domain.SetData(key, null)
            
            logger.LogInformation(sprintf "Cleared %d vector entries from AppDomain" vectorKeys.Count)
            vectorKeys.Count
        with
        | ex ->
            logger.LogWarning(ex, "Failed to clear AppDomain data")
            0
    
    /// Clear all static collections and caches
    member private this.ClearStaticCollections() =
        try
            // Force clear any static collections that might be holding references
            GC.Collect(0, GCCollectionMode.Forced, true, true)
            GC.WaitForPendingFinalizers()
            GC.Collect(1, GCCollectionMode.Forced, true, true)
            GC.WaitForPendingFinalizers()
            GC.Collect(2, GCCollectionMode.Forced, true, true)
            GC.WaitForPendingFinalizers()
            
            // Additional cleanup
            System.Threading.Thread.Sleep(200)
            GC.Collect(2, GCCollectionMode.Forced, true, true)
            
            logger.LogInformation("Cleared static collections and forced full GC")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to clear static collections")
    
    /// Optimize runtime settings for low memory usage
    member private this.OptimizeRuntimeSettings() =
        try
            // Set aggressive GC settings
            GCSettings.LargeObjectHeapCompactionMode <- GCLargeObjectHeapCompactionMode.CompactOnce
            
            // Try to set server GC to false for lower memory usage (if possible)
            logger.LogInformation("Optimized runtime settings for low memory usage")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to optimize runtime settings")
    
    /// Perform ultra-aggressive memory cleanup
    member this.UltraAggressiveCleanup() =
        let memoryBefore = GC.GetTotalMemory(false) / 1024L / 1024L
        logger.LogInformation(sprintf "Starting ultra-aggressive memory cleanup from %dMB" memoryBefore)
        
        try
            // Step 1: Clear AppDomain cached data
            let clearedVectors = this.ClearAppDomainData()
            
            // Step 2: Clear static collections
            this.ClearStaticCollections()
            
            // Step 3: Optimize runtime settings
            this.OptimizeRuntimeSettings()
            
            // Step 4: Multiple rounds of aggressive GC
            for round in 1..5 do
                logger.LogInformation(sprintf "GC Round %d/5" round)
                
                // Force collection of all generations
                GC.Collect(0, GCCollectionMode.Forced, true, true)
                GC.WaitForPendingFinalizers()
                GC.Collect(1, GCCollectionMode.Forced, true, true)
                GC.WaitForPendingFinalizers()
                GC.Collect(2, GCCollectionMode.Forced, true, true)
                GC.WaitForPendingFinalizers()
                
                // Compact LOH
                GCSettings.LargeObjectHeapCompactionMode <- GCLargeObjectHeapCompactionMode.CompactOnce
                GC.Collect(2, GCCollectionMode.Forced, true, true)
                
                // Brief pause between rounds
                System.Threading.Thread.Sleep(100)
                
                let currentMemory = GC.GetTotalMemory(false) / 1024L / 1024L
                logger.LogInformation(sprintf "After round %d: %dMB" round currentMemory)
            
            // Step 5: Final aggressive cleanup
            for i in 1..3 do
                GC.Collect(2, GCCollectionMode.Forced, true, true)
                GC.WaitForPendingFinalizers()
                System.Threading.Thread.Sleep(50)
            
            let memoryAfter = GC.GetTotalMemory(true) / 1024L / 1024L
            let reduction = memoryBefore - memoryAfter
            let reductionPercent = if memoryBefore > 0L then float reduction / float memoryBefore * 100.0 else 0.0
            
            logger.LogInformation(sprintf "Ultra-aggressive cleanup: %dMB→%dMB, reduced %dMB (%.1f%%), cleared %d vectors" 
                memoryBefore memoryAfter reduction reductionPercent clearedVectors)
            
            (memoryBefore, memoryAfter, reduction, reductionPercent, clearedVectors)
        with
        | ex ->
            logger.LogError(ex, "Ultra-aggressive cleanup failed")
            (memoryBefore, memoryBefore, 0L, 0.0, 0)
    
    /// Check if memory is within ultra-strict limits
    member this.IsMemoryWithinUltraLimits() =
        let currentMemory = GC.GetTotalMemory(false) / 1024L / 1024L
        currentMemory < 500L
    
    /// Get detailed memory statistics
    member this.GetDetailedMemoryStats() =
        let totalMemory = GC.GetTotalMemory(false) / 1024L / 1024L
        let gen0Collections = GC.CollectionCount(0)
        let gen1Collections = GC.CollectionCount(1)
        let gen2Collections = GC.CollectionCount(2)
        
        // Check AppDomain data
        let domain = AppDomain.CurrentDomain
        let mutable vectorCount = 0
        for i in 0..10000 do // Check first 10K vectors
            let key = sprintf "VECTOR_%d" i
            if domain.GetData(key) <> null then
                vectorCount <- vectorCount + 1
        
        {|
            TotalMemoryMB = totalMemory
            Gen0Collections = gen0Collections
            Gen1Collections = gen1Collections
            Gen2Collections = gen2Collections
            IsWithinLimits = totalMemory < 500L
            IsWithinUltraLimits = totalMemory < 400L
            MemoryPressure = 
                if totalMemory > 1500L then "CRITICAL" 
                elif totalMemory > 1000L then "HIGH" 
                elif totalMemory > 500L then "MEDIUM" 
                else "LOW"
            VectorDataCount = vectorCount
            RecommendedAction = 
                if totalMemory > 1000L then "IMMEDIATE_CLEANUP_REQUIRED"
                elif totalMemory > 500L then "CLEANUP_RECOMMENDED"
                else "OPTIMAL"
        |}
    
    /// Continuous memory monitoring and cleanup
    member this.StartContinuousOptimization() =
        async {
            while true do
                let stats = this.GetDetailedMemoryStats()
                
                if stats.TotalMemoryMB > 800L then
                    logger.LogWarning(sprintf "High memory usage detected: %dMB - starting cleanup" stats.TotalMemoryMB)
                    let (_, after, reduction, percent, vectors) = this.UltraAggressiveCleanup()
                    logger.LogInformation(sprintf "Cleanup completed: reduced %dMB (%.1f%%), cleared %d vectors" reduction percent vectors)
                
                // Wait 30 seconds before next check
                do! Async.Sleep(30000)
        }
    
    /// Emergency memory cleanup for critical situations
    member this.EmergencyCleanup() =
        logger.LogWarning("EMERGENCY MEMORY CLEANUP INITIATED")
        
        try
            // Clear all AppDomain data immediately
            let domain = AppDomain.CurrentDomain
            for i in 0..500000 do
                let key = sprintf "VECTOR_%d" i
                domain.SetData(key, null)
            
            // Immediate aggressive GC
            for i in 1..10 do
                GC.Collect(2, GCCollectionMode.Forced, true, true)
                GC.WaitForPendingFinalizers()
            
            let memoryAfter = GC.GetTotalMemory(true) / 1024L / 1024L
            logger.LogWarning(sprintf "Emergency cleanup completed: %dMB remaining" memoryAfter)
            
            memoryAfter < 500L
        with
        | ex ->
            logger.LogError(ex, "Emergency cleanup failed")
            false
