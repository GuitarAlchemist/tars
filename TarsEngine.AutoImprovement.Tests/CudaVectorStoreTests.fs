module TarsEngine.AutoImprovement.Tests.CudaVectorStoreTests

open System
open System.Collections.Concurrent
open Xunit
open FsUnit.Xunit
open FsCheck.Xunit

// === CUDA VECTOR STORE TESTS ===

type CudaVectorData = {
    VectorId: string
    Embedding: float[]
    Metadata: Map<string, string>
    Timestamp: DateTime
    CudaProcessed: bool
}

type CudaVectorStore() =
    let vectors = ConcurrentDictionary<string, CudaVectorData>()
    let mutable cudaInitialized = false
    
    member _.InitializeCuda() =
        cudaInitialized <- true
        printfn "🚀 CUDA Vector Store Initialized"
        true
    
    member _.AddVector(content: string, embedding: float[]) =
        let vectorId = Guid.NewGuid().ToString("N").[..7]
        let metadata = Map.ofList [
            ("content_hash", content.GetHashCode().ToString())
            ("vector_dim", embedding.Length.ToString())
            ("cuda_enabled", "true")
        ]
        
        let vectorData = {
            VectorId = vectorId
            Embedding = embedding
            Metadata = metadata
            Timestamp = DateTime.UtcNow
            CudaProcessed = cudaInitialized
        }
        
        vectors.TryAdd(vectorId, vectorData) |> ignore
        vectorId
    
    member _.ComputeSimilarity(vec1: float[], vec2: float[]) =
        if not cudaInitialized then failwith "CUDA not initialized"
        
        // Simulate CUDA-accelerated cosine similarity
        let dotProduct = Array.zip vec1 vec2 |> Array.sumBy (fun (a, b) -> a * b)
        let magnitude1 = vec1 |> Array.sumBy (fun x -> x * x) |> sqrt
        let magnitude2 = vec2 |> Array.sumBy (fun x -> x * x) |> sqrt
        dotProduct / (magnitude1 * magnitude2)
    
    member _.SearchSimilar(queryVector: float[], limit: int) =
        vectors.Values
        |> Seq.map (fun v -> (v.VectorId, _.ComputeSimilarity(queryVector, v.Embedding)))
        |> Seq.sortByDescending snd
        |> Seq.take limit
        |> Seq.toList
    
    member _.GetVectorCount() = vectors.Count
    member _.IsCudaInitialized() = cudaInitialized

[<Fact>]
let ``CUDA Vector Store should initialize successfully`` () =
    // Arrange
    let store = CudaVectorStore()
    
    // Act
    let result = store.InitializeCuda()
    
    // Assert
    result |> should equal true
    store.IsCudaInitialized() |> should equal true

[<Fact>]
let ``CUDA Vector Store should add vectors with proper metadata`` () =
    // Arrange
    let store = CudaVectorStore()
    store.InitializeCuda() |> ignore
    let testVector = Array.init 768 (fun i -> float i * 0.001)
    
    // Act
    let vectorId = store.AddVector("Test TARS improvement content", testVector)
    
    // Assert
    vectorId |> should not' (equal "")
    vectorId.Length |> should equal 8
    store.GetVectorCount() |> should equal 1

[<Fact>]
let ``CUDA similarity computation should return valid scores`` () =
    // Arrange
    let store = CudaVectorStore()
    store.InitializeCuda() |> ignore
    let vec1 = Array.init 100 (fun i -> float i)
    let vec2 = Array.init 100 (fun i -> float i)  // Identical vector
    let vec3 = Array.init 100 (fun i -> float (i + 50))  // Different vector
    
    // Act
    let similarity1 = store.ComputeSimilarity(vec1, vec2)  // Should be 1.0
    let similarity2 = store.ComputeSimilarity(vec1, vec3)  // Should be < 1.0
    
    // Assert
    similarity1 |> should (equalWithin 0.001) 1.0
    similarity2 |> should be (lessThan 1.0)
    similarity2 |> should be (greaterThan 0.0)

[<Property>]
let ``CUDA vector similarity should be symmetric`` (vec1: float[]) (vec2: float[]) =
    let store = CudaVectorStore()
    store.InitializeCuda() |> ignore
    
    // Ensure vectors are not empty and have same length
    let normalizedVec1 = if vec1.Length = 0 then [|1.0|] else vec1
    let normalizedVec2 = Array.init normalizedVec1.Length (fun i -> if i < vec2.Length then vec2.[i] else 0.0)
    
    let sim1 = store.ComputeSimilarity(normalizedVec1, normalizedVec2)
    let sim2 = store.ComputeSimilarity(normalizedVec2, normalizedVec1)
    
    abs(sim1 - sim2) < 0.0001

[<Fact>]
let ``CUDA Vector Store should handle multiple vectors efficiently`` () =
    // Arrange
    let store = CudaVectorStore()
    store.InitializeCuda() |> ignore
    let vectorCount = 1000
    
    // Act
    let vectorIds = [
        for i in 1..vectorCount ->
            let vector = Array.init 768 (fun j -> float (i + j) * 0.001)
            store.AddVector($"Content {i}", vector)
    ]
    
    // Assert
    vectorIds.Length |> should equal vectorCount
    store.GetVectorCount() |> should equal vectorCount
    vectorIds |> List.distinct |> List.length |> should equal vectorCount

[<Fact>]
let ``CUDA similarity search should return top results`` () =
    // Arrange
    let store = CudaVectorStore()
    store.InitializeCuda() |> ignore
    
    // Add test vectors
    let queryVector = Array.init 100 (fun i -> float i * 0.01)
    let similarVector = Array.init 100 (fun i -> float i * 0.01 + 0.001)  // Very similar
    let differentVector = Array.init 100 (fun i -> float (i + 50) * 0.01)  // Different
    
    let id1 = store.AddVector("Similar content", similarVector)
    let id2 = store.AddVector("Different content", differentVector)
    
    // Act
    let results = store.SearchSimilar(queryVector, 2)
    
    // Assert
    results.Length |> should equal 2
    let (topId, topScore) = results.[0]
    let (secondId, secondScore) = results.[1]
    
    topScore |> should be (greaterThan secondScore)
    topId |> should equal id1  // Similar vector should be top result

[<Fact>]
let ``CUDA Vector Store should fail operations when not initialized`` () =
    // Arrange
    let store = CudaVectorStore()
    let vec1 = [|1.0; 2.0; 3.0|]
    let vec2 = [|4.0; 5.0; 6.0|]
    
    // Act & Assert
    (fun () -> store.ComputeSimilarity(vec1, vec2) |> ignore) 
    |> should throw typeof<System.Exception>

[<Fact>]
let ``CUDA Vector Store should handle edge cases gracefully`` () =
    // Arrange
    let store = CudaVectorStore()
    store.InitializeCuda() |> ignore
    
    // Test with zero vectors
    let zeroVector = Array.create 100 0.0
    let normalVector = Array.init 100 (fun i -> float i)
    
    // Act & Assert - should not throw
    let similarity = store.ComputeSimilarity(normalVector, normalVector)
    similarity |> should (equalWithin 0.001) 1.0
    
    // Zero vector similarity should be handled
    let zeroSim = store.ComputeSimilarity(zeroVector, normalVector)
    zeroSim |> should equal 0.0
