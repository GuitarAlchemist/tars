namespace TarsEngine.FSharp.Core

open System
open System.Collections.Concurrent
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// Real vector data structure
type VectorData = {
    Id: string
    Embedding: float[]
    Metadata: string
    Content: string
    Timestamp: DateTime
}

/// Real vector store implementation - no fake data
type RealVectorStore(logger: ILogger<RealVectorStore>) =
    
    // Real in-memory storage
    let vectors = ConcurrentDictionary<string, VectorData>()
    let mutable totalVectors = 0
    
    /// Add a real vector to the store
    member this.AddVector(id: string, embedding: float[], metadata: string, content: string) =
        try
            let vectorData = {
                Id = id
                Embedding = embedding
                Metadata = metadata
                Content = content
                Timestamp = DateTime.UtcNow
            }
            
            let success = vectors.TryAdd(id, vectorData)
            if success then
                totalVectors <- totalVectors + 1
                logger.LogDebug($"Added vector {id} to store (total: {totalVectors})")
                true
            else
                logger.LogWarning($"Vector {id} already exists in store")
                false
        with
        | ex ->
            logger.LogError(ex, $"Failed to add vector {id}")
            false
    
    /// Get a vector by ID
    member this.GetVector(id: string) =
        match vectors.TryGetValue(id) with
        | true, vector -> Some vector
        | false, _ -> None
    
    /// Get all vectors
    member this.GetAllVectors() =
        vectors.Values |> Seq.toArray
    
    /// Search vectors by similarity (simple implementation)
    member this.SearchSimilar(queryEmbedding: float[], topK: int) =
        let calculateSimilarity (v1: float[]) (v2: float[]) =
            if v1.Length <> v2.Length then 0.0
            else
                let mutable dotProduct = 0.0
                let mutable norm1 = 0.0
                let mutable norm2 = 0.0
                
                for i in 0..v1.Length-1 do
                    dotProduct <- dotProduct + v1.[i] * v2.[i]
                    norm1 <- norm1 + v1.[i] * v1.[i]
                    norm2 <- norm2 + v2.[i] * v2.[i]
                
                if norm1 = 0.0 || norm2 = 0.0 then 0.0
                else dotProduct / (sqrt(norm1) * sqrt(norm2))
        
        vectors.Values
        |> Seq.map (fun v -> (v, calculateSimilarity queryEmbedding v.Embedding))
        |> Seq.sortByDescending snd
        |> Seq.take (min topK vectors.Count)
        |> Seq.toArray
    
    /// Get statistics
    member this.GetStats() =
        let vectorCount = vectors.Count
        let avgEmbeddingSize = 
            if vectorCount = 0 then 0.0
            else vectors.Values |> Seq.averageBy (fun v -> float v.Embedding.Length)
        
        let memoryUsage = 
            vectors.Values 
            |> Seq.sumBy (fun v -> 
                v.Embedding.Length * sizeof<float> + 
                v.Metadata.Length * sizeof<char> + 
                v.Content.Length * sizeof<char>)
        
        {| 
            VectorCount = vectorCount
            AverageEmbeddingSize = avgEmbeddingSize
            MemoryUsageBytes = memoryUsage
            MemoryUsageMB = float memoryUsage / 1024.0 / 1024.0
        |}
    
    /// Clear all vectors
    member this.Clear() =
        vectors.Clear()
        totalVectors <- 0
        logger.LogInformation("Vector store cleared")
    
    /// Check if vector store is working
    member this.HealthCheck() =
        try
            // Test adding and retrieving a vector
            let testId = "health_check_" + Guid.NewGuid().ToString()
            let testEmbedding = [| 1.0; 2.0; 3.0 |]
            let testMetadata = "health_check"
            let testContent = "This is a health check vector"
            
            // Add test vector
            let added = this.AddVector(testId, testEmbedding, testMetadata, testContent)
            if not added then
                logger.LogError("Health check failed: Could not add test vector")
                false
            else
                // Retrieve test vector
                match this.GetVector(testId) with
                | Some retrievedVector ->
                    let isValid = 
                        retrievedVector.Id = testId &&
                        retrievedVector.Embedding.Length = testEmbedding.Length &&
                        retrievedVector.Metadata = testMetadata &&
                        retrievedVector.Content = testContent
                    
                    // Clean up test vector
                    vectors.TryRemove(testId) |> ignore
                    totalVectors <- totalVectors - 1
                    
                    if isValid then
                        logger.LogDebug("Vector store health check passed")
                        true
                    else
                        logger.LogError("Health check failed: Retrieved vector data is invalid")
                        false
                | None ->
                    logger.LogError("Health check failed: Could not retrieve test vector")
                    false
        with
        | ex ->
            logger.LogError(ex, "Health check failed with exception")
            false

/// Real embedding generator - simple but functional
type RealEmbeddingGenerator(logger: ILogger<RealEmbeddingGenerator>) =
    
    /// Generate real embeddings from text (simple but functional implementation)
    member this.GenerateEmbedding(text: string, dimensions: int) =
        try
            if String.IsNullOrEmpty(text) then
                Array.zeroCreate dimensions
            else
                // Simple but real embedding generation
                let chars = text.ToCharArray()
                let embedding = Array.zeroCreate dimensions
                
                // Use character codes and position weighting
                for i in 0..chars.Length-1 do
                    let charCode = float (int chars.[i])
                    let position = float i / float chars.Length
                    let index = i % dimensions
                    
                    // Combine character code with position weighting
                    embedding.[index] <- embedding.[index] + charCode * (1.0 + position)
                
                // Normalize the embedding
                let magnitude = sqrt (embedding |> Array.sumBy (fun x -> x * x))
                if magnitude > 0.0 then
                    for i in 0..embedding.Length-1 do
                        embedding.[i] <- embedding.[i] / magnitude
                
                logger.LogDebug($"Generated {dimensions}D embedding for text of length {text.Length}")
                embedding
        with
        | ex ->
            logger.LogError(ex, $"Failed to generate embedding for text of length {text.Length}")
            Array.zeroCreate dimensions
    
    /// Health check for embedding generator
    member this.HealthCheck() =
        try
            let testText = "This is a test for embedding generation"
            let testDimensions = 384
            let embedding = this.GenerateEmbedding(testText, testDimensions)
            
            let isValid = 
                embedding.Length = testDimensions &&
                embedding |> Array.exists (fun x -> x <> 0.0)
            
            if isValid then
                logger.LogDebug("Embedding generator health check passed")
                true
            else
                logger.LogError("Embedding generator health check failed: Invalid embedding generated")
                false
        with
        | ex ->
            logger.LogError(ex, "Embedding generator health check failed with exception")
            false
