// ================================================
// 🌐 TARS Non-Euclidean Vector Store
// ================================================
// Advanced vector storage with hyperbolic/spherical geometry

namespace TarsEngine.FSharp.Core

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging

module TarsNonEuclideanVectorStore =

    /// Geometric space types
    type GeometricSpace =
        | Euclidean
        | Hyperbolic of curvature: float
        | Spherical of radius: float
        | Projective
        | DualQuaternion
        | Minkowski
        | Pauli
        | Wavelet

    /// Vector with geometric space information
    type GeometricVector = {
        Id: string
        Values: float32[]
        Space: GeometricSpace
        Metadata: Map<string, obj>
        Timestamp: DateTime
    }

    /// Distance metrics for different spaces
    type DistanceMetric =
        | EuclideanDistance
        | HyperbolicDistance
        | SphericalDistance
        | ProjectiveDistance
        | QuaternionDistance
        | MinkowskiDistance
        | PauliDistance
        | WaveletDistance

    /// Search result with distance and space information
    type SearchResult = {
        Vector: GeometricVector
        Distance: float
        Similarity: float
        SpaceTransformed: bool
    }

    /// Non-Euclidean vector store configuration
    type VectorStoreConfig = {
        DefaultSpace: GeometricSpace
        EnableSpaceTransformation: bool
        MaxVectors: int
        DimensionSize: int
        UseCuda: bool
        PersistToDisk: bool
        StoragePath: string option
    }

    /// Non-Euclidean vector store implementation
    type TarsNonEuclideanVectorStore(config: VectorStoreConfig, logger: ILogger) =
        let vectors = ConcurrentDictionary<string, GeometricVector>()
        let spaceIndices = ConcurrentDictionary<GeometricSpace, ConcurrentDictionary<string, GeometricVector>>()
        
        /// Initialize space indices
        do
            for space in [Euclidean; Hyperbolic(1.0); Spherical(1.0); Projective; DualQuaternion; Minkowski; Pauli; Wavelet] do
                spaceIndices.[space] <- ConcurrentDictionary<string, GeometricVector>()

        /// Compute distance between vectors in different geometric spaces
        member private _.ComputeDistance (v1: float32[]) (v2: float32[]) (space: GeometricSpace) : float =
            match space with
            | Euclidean ->
                // Standard Euclidean distance
                let mutable sum = 0.0
                for i in 0 .. v1.Length - 1 do
                    let diff = float (v1.[i] - v2.[i])
                    sum <- sum + diff * diff
                sqrt sum
            
            | Hyperbolic(curvature) ->
                // Hyperbolic distance using Poincaré disk model
                let norm1 = v1 |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                let norm2 = v2 |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                let dotProduct = Array.map2 (fun x y -> float x * float y) v1 v2 |> Array.sum
                
                let u1 = v1 |> Array.map (fun x -> float x / norm1)
                let u2 = v2 |> Array.map (fun x -> float x / norm2)
                let cosTheta = Array.map2 (*) u1 u2 |> Array.sum |> max -1.0 |> min 1.0
                
                abs curvature * acos cosTheta
            
            | Spherical(radius) ->
                // Spherical distance (great circle distance)
                let norm1 = v1 |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                let norm2 = v2 |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                
                if norm1 = 0.0 || norm2 = 0.0 then 0.0
                else
                    let dotProduct = Array.map2 (fun x y -> float x * float y) v1 v2 |> Array.sum
                    let cosTheta = (dotProduct / (norm1 * norm2)) |> max -1.0 |> min 1.0
                    radius * acos cosTheta
            
            | Projective ->
                // Projective distance using cross-ratio
                let norm1 = v1 |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                let norm2 = v2 |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                let dotProduct = Array.map2 (fun x y -> float x * float y) v1 v2 |> Array.sum
                
                if norm1 = 0.0 || norm2 = 0.0 then Double.MaxValue
                else
                    let cosTheta = dotProduct / (norm1 * norm2)
                    if abs cosTheta >= 1.0 then 0.0
                    else log (abs ((1.0 + cosTheta) / (1.0 - cosTheta)))
            
            | DualQuaternion ->
                // Dual quaternion distance
                if v1.Length >= 8 && v2.Length >= 8 then
                    let q1 = v1.[0..3] |> Array.map float
                    let q2 = v2.[0..3] |> Array.map float
                    let d1 = v1.[4..7] |> Array.map float
                    let d2 = v2.[4..7] |> Array.map float
                    
                    let qDist = Array.map2 (fun x y -> (x - y) * (x - y)) q1 q2 |> Array.sum |> sqrt
                    let dDist = Array.map2 (fun x y -> (x - y) * (x - y)) d1 d2 |> Array.sum |> sqrt
                    qDist + dDist
                else
                    // Fallback to Euclidean
                    let mutable sum = 0.0
                    for i in 0 .. v1.Length - 1 do
                        let diff = float (v1.[i] - v2.[i])
                        sum <- sum + diff * diff
                    sqrt sum
            
            | Minkowski ->
                // Minkowski spacetime distance
                if v1.Length >= 4 && v2.Length >= 4 then
                    let dt = float (v1.[0] - v2.[0])
                    let dx = float (v1.[1] - v2.[1])
                    let dy = float (v1.[2] - v2.[2])
                    let dz = float (v1.[3] - v2.[3])
                    
                    let spacetime = dt * dt - dx * dx - dy * dy - dz * dz
                    if spacetime >= 0.0 then sqrt spacetime else sqrt (-spacetime)
                else
                    // Fallback to Euclidean
                    let mutable sum = 0.0
                    for i in 0 .. v1.Length - 1 do
                        let diff = float (v1.[i] - v2.[i])
                        sum <- sum + diff * diff
                    sqrt sum
            
            | Pauli ->
                // Pauli matrix-based distance
                let mutable sum = 0.0
                for i in 0 .. min (v1.Length - 1) (v2.Length - 1) do
                    let diff = float (v1.[i] - v2.[i])
                    let weight = if i % 4 = 0 then 1.0 else if i % 4 = 3 then -1.0 else 1.0
                    sum <- sum + weight * diff * diff
                abs sum |> sqrt
            
            | Wavelet ->
                // Wavelet-based distance using simple frequency approximation
                let mutable sum = 0.0
                for i in 0 .. min (v1.Length - 1) (v2.Length - 1) do
                    let freq1 = sin(float i * float v1.[i])
                    let freq2 = sin(float i * float v2.[i])
                    let diff = freq1 - freq2
                    sum <- sum + diff * diff
                sqrt sum

        /// Transform vector between geometric spaces
        member private _.TransformVector (vector: float32[]) (fromSpace: GeometricSpace) (toSpace: GeometricSpace) : float32[] =
            if fromSpace = toSpace then vector
            else
                match fromSpace, toSpace with
                | Euclidean, Hyperbolic(_) ->
                    // Map to Poincaré disk
                    let norm = vector |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                    if norm = 0.0 then vector
                    else vector |> Array.map (fun x -> float32 (tanh(float x / norm)))
                
                | Euclidean, Spherical(_) ->
                    // Normalize to unit sphere
                    let norm = vector |> Array.map (fun x -> float x * float x) |> Array.sum |> sqrt
                    if norm = 0.0 then vector
                    else vector |> Array.map (fun x -> float32 (float x / norm))
                
                | Hyperbolic(_), Euclidean ->
                    // Map from Poincaré disk to Euclidean
                    vector |> Array.map (fun x ->
                        let t = float x
                        if abs t >= 1.0 then float32 (float (Math.Sign(t)) * 10.0)
                        else float32 (0.5 * log((1.0 + t) / (1.0 - t))))
                
                | Spherical(_), Euclidean ->
                    // Map from sphere to Euclidean (stereographic projection)
                    if vector.Length > 0 then
                        let lastCoord = float vector.[vector.Length - 1]
                        if abs (lastCoord - 1.0) < 1e-6 then vector.[0..vector.Length-2]
                        else
                            vector.[0..vector.Length-2] |> Array.map (fun x -> 
                                float32 (float x / (1.0 - lastCoord)))
                    else vector
                
                | _ ->
                    // Default: return original vector
                    vector

        /// Add vector to store
        member this.AddVectorAsync(vector: GeometricVector) : Task<bool> =
            task {
                try
                    logger.LogDebug("Adding vector {Id} to {Space} space", vector.Id, vector.Space)
                    
                    // Add to main store
                    vectors.[vector.Id] <- vector
                    
                    // Add to space-specific index
                    let spaceIndex = spaceIndices.[vector.Space]
                    spaceIndex.[vector.Id] <- vector
                    
                    // If space transformation is enabled, add to other spaces
                    if config.EnableSpaceTransformation then
                        for kvp in spaceIndices do
                            if kvp.Key <> vector.Space then
                                let transformedValues = this.TransformVector vector.Values vector.Space kvp.Key
                                let transformedVector = { vector with Values = transformedValues; Space = kvp.Key }
                                kvp.Value.[vector.Id] <- transformedVector
                    
                    logger.LogInformation("Successfully added vector {Id} with {Dimensions} dimensions", 
                        vector.Id, vector.Values.Length)
                    
                    return true
                with
                | ex ->
                    logger.LogError(ex, "Failed to add vector {Id}", vector.Id)
                    return false
            }

        /// Search for similar vectors
        member this.SearchAsync(queryVector: float32[], space: GeometricSpace, topK: int) : Task<SearchResult[]> =
            task {
                try
                    logger.LogDebug("Searching for {TopK} similar vectors in {Space} space", topK, space)
                    
                    let results = ResizeArray<SearchResult>()
                    
                    // Search in the specified space
                    let spaceIndex = spaceIndices.[space]
                    for kvp in spaceIndex do
                        let vector = kvp.Value
                        let distance = this.ComputeDistance queryVector vector.Values space
                        let similarity = 1.0 / (1.0 + distance)
                        
                        results.Add({
                            Vector = vector
                            Distance = distance
                            Similarity = similarity
                            SpaceTransformed = vector.Space <> space
                        })
                    
                    // Sort by similarity (descending) and take top K
                    let sortedResults = 
                        results
                        |> Seq.sortByDescending (fun r -> r.Similarity)
                        |> Seq.take (min topK results.Count)
                        |> Array.ofSeq
                    
                    logger.LogInformation("Found {ResultCount} similar vectors in {Space} space", 
                        sortedResults.Length, space)
                    
                    return sortedResults
                with
                | ex ->
                    logger.LogError(ex, "Failed to search vectors in {Space} space", space)
                    return [||]
            }

        /// Get vector by ID
        member _.GetVectorAsync(id: string) : Task<GeometricVector option> =
            task {
                match vectors.TryGetValue(id) with
                | true, vector -> return Some vector
                | false, _ -> return None
            }

        /// Remove vector
        member _.RemoveVectorAsync(id: string) : Task<bool> =
            task {
                try
                    match vectors.TryRemove(id) with
                    | true, vector ->
                        // Remove from all space indices
                        for kvp in spaceIndices do
                            kvp.Value.TryRemove(id) |> ignore
                        
                        logger.LogInformation("Removed vector {Id}", id)
                        return true
                    | false, _ ->
                        logger.LogWarning("Vector {Id} not found for removal", id)
                        return false
                with
                | ex ->
                    logger.LogError(ex, "Failed to remove vector {Id}", id)
                    return false
            }

        /// Get statistics
        member _.GetStatistics() : Map<string, obj> =
            let totalVectors = vectors.Count
            let spaceDistribution = 
                spaceIndices 
                |> Seq.map (fun kvp -> (kvp.Key.ToString(), kvp.Value.Count :> obj))
                |> Map.ofSeq
            
            Map.ofList [
                ("total_vectors", totalVectors :> obj)
                ("space_distribution", spaceDistribution :> obj)
                ("default_space", config.DefaultSpace.ToString() :> obj)
                ("space_transformation_enabled", config.EnableSpaceTransformation :> obj)
                ("cuda_enabled", config.UseCuda :> obj)
                ("dimension_size", config.DimensionSize :> obj)
            ]

        /// Clear all vectors
        member _.ClearAsync() : Task =
            task {
                vectors.Clear()
                for kvp in spaceIndices do
                    kvp.Value.Clear()
                logger.LogInformation("Cleared all vectors from store")
            }

    /// Create geometric vector
    let createGeometricVector (id: string) (values: float32[]) (space: GeometricSpace) (metadata: Map<string, obj>) : GeometricVector =
        {
            Id = id
            Values = values
            Space = space
            Metadata = metadata
            Timestamp = DateTime.UtcNow
        }

    /// Default configuration for non-Euclidean vector store
    let defaultConfig = {
        DefaultSpace = Euclidean
        EnableSpaceTransformation = true
        MaxVectors = 100000
        DimensionSize = 768
        UseCuda = true
        PersistToDisk = false
        StoragePath = None
    }

    /// Create TARS non-Euclidean vector store
    let createVectorStore (config: VectorStoreConfig option) (logger: ILogger) : TarsNonEuclideanVectorStore =
        let finalConfig = config |> Option.defaultValue defaultConfig
        TarsNonEuclideanVectorStore(finalConfig, logger)
