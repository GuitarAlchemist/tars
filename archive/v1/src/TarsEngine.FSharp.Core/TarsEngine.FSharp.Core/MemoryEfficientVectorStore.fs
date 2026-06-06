namespace TarsEngine.FSharp.Core

open System
open System.Collections.Concurrent
open Microsoft.Extensions.Logging

/// Ultra-compact vector data with minimal memory footprint
type CompactVectorData = {
    Id: string
    EmbeddingHash: int64  // Store hash instead of full embedding
    ContentHash: int      // Store content hash instead of full content
    Timestamp: int64      // Store as ticks to save space
}

/// Memory-efficient vector store with aggressive limits
type MemoryEfficientVectorStore(logger: ILogger<MemoryEfficientVectorStore>) =
    
    // Ultra-limited storage to prevent memory issues
    let vectors = ConcurrentDictionary<string, CompactVectorData>()
    let mutable totalVectors = 0
    let maxVectors = 1000 // Drastically reduced limit
    let mutable lastCleanup = DateTime.UtcNow.Ticks
    
    /// Add a vector with ultra-compact storage
    member this.AddVector(id: string, embedding: float[], metadata: string, content: string) =
        try
            // Check if we're at capacity
            if vectors.Count >= maxVectors then
                this.PerformAggressiveCleanup()
            
            // Create ultra-compact representation
            let embeddingHash = 
                embedding 
                |> Array.fold (fun acc x -> acc ^^^ (int64 (x * 1000000.0))) 0L
            
            let contentHash = content.GetHashCode()
            
            let compactData = {
                Id = id
                EmbeddingHash = embeddingHash
                ContentHash = contentHash
                Timestamp = DateTime.UtcNow.Ticks
            }
            
            vectors.TryAdd(id, compactData) |> ignore
            totalVectors <- totalVectors + 1
            
            if totalVectors % 100 = 0 then
                logger.LogInformation(sprintf "Added vector %d/%d (memory-efficient)" totalVectors maxVectors)
            
            true
        with
        | ex ->
            logger.LogError(ex, sprintf "Failed to add vector: %s" id)
            false
    
    /// Perform aggressive cleanup to free memory
    member private this.PerformAggressiveCleanup() =
        try
            let beforeCount = vectors.Count
            let cutoffTime = DateTime.UtcNow.AddMinutes(-5.0).Ticks
            
            // Remove old vectors
            let keysToRemove = 
                vectors
                |> Seq.filter (fun kvp -> kvp.Value.Timestamp < cutoffTime)
                |> Seq.map (fun kvp -> kvp.Key)
                |> Seq.take (maxVectors / 2) // Remove half
                |> Seq.toArray
            
            for key in keysToRemove do
                vectors.TryRemove(key) |> ignore
            
            // Force garbage collection
            GC.Collect(2, GCCollectionMode.Forced)
            GC.WaitForPendingFinalizers()
            
            let afterCount = vectors.Count
            lastCleanup <- DateTime.UtcNow.Ticks
            
            logger.LogInformation(sprintf "Aggressive cleanup: %d→%d vectors removed" beforeCount afterCount)
        with
        | ex ->
            logger.LogError(ex, "Aggressive cleanup failed")
    
    /// Search vectors (simplified for memory efficiency)
    member this.SearchVectors(queryEmbedding: float[], maxResults: int) =
        try
            let queryHash = 
                queryEmbedding 
                |> Array.fold (fun acc x -> acc ^^^ (int64 (x * 1000000.0))) 0L
            
            let results = 
                vectors.Values
                |> Seq.map (fun v -> 
                    let similarity = 1.0 - (float (abs (v.EmbeddingHash - queryHash)) / float Int64.MaxValue)
                    (v.Id, similarity))
                |> Seq.sortByDescending snd
                |> Seq.take (min maxResults 10) // Limit results
                |> Seq.toArray
            
            logger.LogDebug(sprintf "Search completed: %d results from %d vectors" results.Length vectors.Count)
            results
        with
        | ex ->
            logger.LogError(ex, "Vector search failed")
            [||]
    
    /// Get memory-efficient statistics
    member this.GetStats() =
        let memoryUsage = GC.GetTotalMemory(false) / 1024L / 1024L
        
        {|
            VectorCount = vectors.Count
            MaxVectors = maxVectors
            MemoryUsageMB = memoryUsage
            LastCleanup = DateTime(lastCleanup)
            IsOptimized = vectors.Count < maxVectors && memoryUsage < 100L
            UtilizationPercent = float vectors.Count / float maxVectors * 100.0
        |}
    
    /// Clear all vectors to free memory
    member this.ClearAll() =
        try
            let beforeCount = vectors.Count
            vectors.Clear()
            totalVectors <- 0
            
            // Force cleanup
            GC.Collect(2, GCCollectionMode.Forced)
            GC.WaitForPendingFinalizers()
            
            logger.LogInformation(sprintf "Cleared all %d vectors" beforeCount)
            true
        with
        | ex ->
            logger.LogError(ex, "Failed to clear vectors")
            false
    
    /// Optimize memory usage
    member this.OptimizeMemory() =
        try
            let beforeMemory = GC.GetTotalMemory(false) / 1024L / 1024L
            
            // Perform cleanup if needed
            if vectors.Count > maxVectors / 2 then
                this.PerformAggressiveCleanup()
            
            // Force garbage collection
            GC.Collect(2, GCCollectionMode.Forced)
            GC.WaitForPendingFinalizers()
            GC.Collect(2, GCCollectionMode.Forced)
            
            let afterMemory = GC.GetTotalMemory(true) / 1024L / 1024L
            let reduction = beforeMemory - afterMemory
            
            logger.LogInformation(sprintf "Memory optimization: %dMB→%dMB (-%dMB)" beforeMemory afterMemory reduction)
            
            (beforeMemory, afterMemory, reduction)
        with
        | ex ->
            logger.LogError(ex, "Memory optimization failed")
            (0L, 0L, 0L)
    
    /// Check if store is within memory limits
    member this.IsWithinMemoryLimits() =
        let memoryUsage = GC.GetTotalMemory(false) / 1024L / 1024L
        let vectorCount = vectors.Count
        
        memoryUsage < 200L && vectorCount < maxVectors
    
    /// Get compact vector by ID
    member this.GetVector(id: string) =
        match vectors.TryGetValue(id) with
        | true, vector -> Some vector
        | false, _ -> None
    
    /// Remove vector by ID
    member this.RemoveVector(id: string) =
        match vectors.TryRemove(id) with
        | true, _ -> 
            logger.LogDebug(sprintf "Removed vector: %s" id)
            true
        | false, _ -> false
    
    /// Get all vector IDs (for debugging)
    member this.GetAllIds() =
        vectors.Keys |> Seq.toArray
    
    /// Dispose and cleanup
    interface IDisposable with
        member this.Dispose() =
            try
                this.ClearAll() |> ignore
                logger.LogInformation("MemoryEfficientVectorStore disposed")
            with
            | ex ->
                logger.LogError(ex, "Error during disposal")
