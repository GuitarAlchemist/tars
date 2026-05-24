namespace TarsEngine.FSharp.Core

open System
open System.Collections.Concurrent
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// Memory-optimized vector data structure
type OptimizedVectorData = {
    Id: string
    Embedding: float32[] // Use float32 instead of float64 to save memory
    ContentHash: string  // Store hash instead of full content to save memory
    Timestamp: DateTime
}

/// Memory-optimized vector store with compression and cleanup
type OptimizedVectorStore(logger: ILogger<OptimizedVectorStore>) =
    
    // Memory-optimized storage with size limits
    let vectors = ConcurrentDictionary<string, OptimizedVectorData>()
    let mutable totalVectors = 0
    let maxVectors = 50000 // Limit to prevent memory issues
    let mutable lastCleanup = DateTime.UtcNow
    
    /// Add a vector with memory optimization
    member this.AddVector(id: string, embedding: float[], metadata: string, content: string) =
        try
            // Convert to float32 to save memory
            let optimizedEmbedding = embedding |> Array.map float32
            
            // Create content hash instead of storing full content
            let contentHash = content.GetHashCode().ToString("X8")
            
            let vectorData = {
                Id = id
                Embedding = optimizedEmbedding
                ContentHash = contentHash
                Timestamp = DateTime.UtcNow
            }
            
            // Check if we need cleanup
            if vectors.Count >= maxVectors then
                this.PerformCleanup()
            
            let success = vectors.TryAdd(id, vectorData)
            if success then
                totalVectors <- totalVectors + 1
                logger.LogDebug(sprintf "Added optimized vector %s (total: %d)" id totalVectors)
                true
            else
                logger.LogWarning(sprintf "Vector %s already exists" id)
                false
        with
        | ex ->
            logger.LogError(ex, sprintf "Failed to add vector %s" id)
            false
    
    /// Perform memory cleanup by removing oldest vectors
    member private this.PerformCleanup() =
        try
            let oldestVectors = 
                vectors.Values
                |> Seq.sortBy (fun v -> v.Timestamp)
                |> Seq.take (maxVectors / 4) // Remove 25% of vectors
                |> Seq.toArray
            
            let removedCount = 
                oldestVectors
                |> Array.map (fun v -> vectors.TryRemove(v.Id))
                |> Array.filter (fun (success, _) -> success)
                |> Array.length
            
            lastCleanup <- DateTime.UtcNow
            logger.LogInformation(sprintf "Memory cleanup: removed %d old vectors" removedCount)
        with
        | ex ->
            logger.LogError(ex, "Failed to perform memory cleanup")
    
    /// Get a vector by ID
    member this.GetVector(id: string) =
        match vectors.TryGetValue(id) with
        | true, vector -> Some vector
        | false, _ -> None
    
    /// Get all vectors (with limit for memory safety)
    member this.GetAllVectors() =
        vectors.Values |> Seq.take 1000 |> Seq.toArray // Limit to prevent memory issues
    
    /// Search vectors by similarity (optimized version)
    member this.SearchSimilar(queryEmbedding: float[], topK: int) =
        let queryEmbedding32 = queryEmbedding |> Array.map float32
        
        let calculateSimilarity (v1: float32[]) (v2: float32[]) =
            if v1.Length <> v2.Length then 0.0f
            else
                let mutable dotProduct = 0.0f
                let mutable norm1 = 0.0f
                let mutable norm2 = 0.0f
                
                for i in 0..v1.Length-1 do
                    dotProduct <- dotProduct + v1.[i] * v2.[i]
                    norm1 <- norm1 + v1.[i] * v1.[i]
                    norm2 <- norm2 + v2.[i] * v2.[i]
                
                if norm1 = 0.0f || norm2 = 0.0f then 0.0f
                else dotProduct / (sqrt(norm1) * sqrt(norm2))
        
        vectors.Values
        |> Seq.map (fun v -> (v, calculateSimilarity queryEmbedding32 v.Embedding))
        |> Seq.sortByDescending snd
        |> Seq.take (min topK 100) // Limit results for performance
        |> Seq.toArray
    
    /// Get optimized statistics
    member this.GetStats() =
        let vectorCount = vectors.Count
        let avgEmbeddingSize = 
            if vectorCount = 0 then 0.0
            else vectors.Values |> Seq.averageBy (fun v -> float v.Embedding.Length)
        
        // Calculate memory usage more accurately
        let memoryUsage = 
            vectors.Values 
            |> Seq.sumBy (fun v -> 
                v.Embedding.Length * sizeof<float32> + // Use float32 size
                v.Id.Length * sizeof<char> + 
                v.ContentHash.Length * sizeof<char> +
                sizeof<DateTime>)
        
        {| 
            VectorCount = vectorCount
            AverageEmbeddingSize = avgEmbeddingSize
            MemoryUsageBytes = memoryUsage
            MemoryUsageMB = float memoryUsage / 1024.0 / 1024.0
            MaxVectors = maxVectors
            LastCleanup = lastCleanup
            MemoryOptimized = true
        |}
    
    /// Clear all vectors and reset
    member this.Clear() =
        vectors.Clear()
        totalVectors <- 0
        lastCleanup <- DateTime.UtcNow
        logger.LogInformation("Optimized vector store cleared")
    
    /// Enhanced health check with memory monitoring
    member this.HealthCheck() =
        try
            let stats = this.GetStats()
            
            // Check memory usage
            let memoryHealthy = stats.MemoryUsageMB < 100.0 // Much stricter limit
            let countHealthy = stats.VectorCount < maxVectors
            let cleanupHealthy = (DateTime.UtcNow - lastCleanup).TotalMinutes < 60.0
            
            let isHealthy = memoryHealthy && countHealthy && cleanupHealthy
            
            if isHealthy then
                logger.LogDebug(sprintf "Optimized vector store health check passed: %d vectors, %.1fMB" stats.VectorCount stats.MemoryUsageMB)
            else
                logger.LogWarning(sprintf "Optimized vector store health issues: Memory: %.1fMB, Count: %d/%d" stats.MemoryUsageMB stats.VectorCount maxVectors)
            
            isHealthy
        with
        | ex ->
            logger.LogError(ex, "Optimized vector store health check failed")
            false
    
    /// Force garbage collection and cleanup
    member this.OptimizeMemory() =
        try
            // Perform cleanup if needed
            if vectors.Count > maxVectors * 3 / 4 then
                this.PerformCleanup()
            
            // Force garbage collection
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            
            let stats = this.GetStats()
            logger.LogInformation(sprintf "Memory optimization complete: %d vectors, %.1fMB" stats.VectorCount stats.MemoryUsageMB)
            
            stats.MemoryUsageMB
        with
        | ex ->
            logger.LogError(ex, "Failed to optimize memory")
            0.0

/// Optimized embedding generator with reduced memory footprint
type OptimizedEmbeddingGenerator(logger: ILogger<OptimizedEmbeddingGenerator>) =
    
    /// Generate optimized embeddings (smaller dimensions for memory efficiency)
    member this.GenerateEmbedding(text: string, dimensions: int) =
        try
            if String.IsNullOrEmpty(text) then
                Array.zeroCreate dimensions
            else
                // Use smaller dimension limit for memory efficiency
                let actualDimensions = min dimensions 128 // Limit to 128 dimensions
                let chars = text.ToCharArray()
                let embedding = Array.zeroCreate actualDimensions
                
                // Optimized embedding generation
                for i in 0..min (chars.Length - 1) (actualDimensions - 1) do
                    let charCode = float (int chars.[i])
                    let position = float i / float chars.Length
                    
                    // Simple but effective embedding
                    embedding.[i] <- charCode * (1.0 + position) / 256.0
                
                // Normalize
                let magnitude = sqrt (embedding |> Array.sumBy (fun x -> x * x))
                if magnitude > 0.0 then
                    for i in 0..embedding.Length-1 do
                        embedding.[i] <- embedding.[i] / magnitude
                
                logger.LogDebug(sprintf "Generated optimized %dD embedding for text of length %d" actualDimensions text.Length)
                embedding
        with
        | ex ->
            logger.LogError(ex, sprintf "Failed to generate optimized embedding for text of length %d" text.Length)
            Array.zeroCreate (min dimensions 128)
    
    /// Health check for optimized embedding generator
    member this.HealthCheck() =
        try
            let testText = "Optimized embedding test"
            let testDimensions = 128
            let embedding = this.GenerateEmbedding(testText, testDimensions)
            
            let isValid = 
                embedding.Length = testDimensions &&
                embedding |> Array.exists (fun x -> x <> 0.0)
            
            if isValid then
                logger.LogDebug("Optimized embedding generator health check passed")
                true
            else
                logger.LogError("Optimized embedding generator health check failed")
                false
        with
        | ex ->
            logger.LogError(ex, "Optimized embedding generator health check failed")
            false
