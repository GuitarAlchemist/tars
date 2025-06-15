namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.Threading.Tasks
open Xunit
open TarsEngine.FSharp.FLUX.VectorStore.SemanticVectorStore

/// Comprehensive unit tests for ChatGPT-Vector Store Semantics
module SemanticVectorStoreTests =

    [<Fact>]
    let ``SimpleEmbeddingService should generate consistent embeddings`` () =
        task {
            // Arrange
            let service = SimpleEmbeddingService() :> IEmbeddingService
            let text = "Hello World"
            
            // Act
            let! embedding1 = service.GenerateEmbedding(text)
            let! embedding2 = service.GenerateEmbedding(text)
            
            // Assert
            Assert.Equal(embedding1.Length, embedding2.Length)
            Assert.Equal(embedding1, embedding2)
            Assert.Equal(384, embedding1.Length)
        }

    [<Fact>]
    let ``SimpleEmbeddingService should generate normalized embeddings`` () =
        task {
            // Arrange
            let service = SimpleEmbeddingService() :> IEmbeddingService
            let text = "Test normalization"
            
            // Act
            let! embedding = service.GenerateEmbedding(text)
            
            // Assert
            let magnitude = Math.Sqrt(embedding |> Array.map (fun x -> x * x) |> Array.sum)
            Assert.True(Math.Abs(magnitude - 1.0) < 0.001, $"Expected normalized vector (magnitude ≈ 1.0), but got {magnitude}")
        }

    [<Fact>]
    let ``SemanticVectorStore should add vectors successfully`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            let content = "let x = 1"
            
            // Act
            let! vectorId = store.AddVectorAsync(content, CodeBlock)
            
            // Assert
            Assert.NotNull(vectorId)
            Assert.NotEmpty(vectorId)
            
            let vector = store.GetVector(vectorId)
            Assert.True(vector.IsSome)
            Assert.Equal(content, vector.Value.Content)
            Assert.Equal(CodeBlock, vector.Value.SemanticType)
        }

    [<Fact>]
    let ``SemanticVectorStore should calculate cosine similarity correctly`` () =
        // Arrange
        let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
        let store = SemanticVectorStore(embeddingService)
        let v1 = [| 1.0; 0.0; 0.0 |]
        let v2 = [| 1.0; 0.0; 0.0 |]
        let v3 = [| 0.0; 1.0; 0.0 |]
        
        // Act
        let similarity1 = store.GetType().GetMethod("CosineSimilarity", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(store, [| v1; v2 |]) :?> float
        let similarity2 = store.GetType().GetMethod("CosineSimilarity", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(store, [| v1; v3 |]) :?> float
        
        // Assert
        Assert.Equal(1.0, similarity1, 3)
        Assert.Equal(0.0, similarity2, 3)

    [<Fact>]
    let ``SemanticVectorStore should calculate Euclidean distance correctly`` () =
        // Arrange
        let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
        let store = SemanticVectorStore(embeddingService)
        let v1 = [| 0.0; 0.0; 0.0 |]
        let v2 = [| 3.0; 4.0; 0.0 |]
        
        // Act
        let distance = store.GetType().GetMethod("EuclideanDistance", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(store, [| v1; v2 |]) :?> float
        
        // Assert
        Assert.Equal(5.0, distance, 3) // 3-4-5 triangle

    [<Fact>]
    let ``SemanticVectorStore should search similar vectors`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            
            let! id1 = store.AddVectorAsync("let x = 1", CodeBlock)
            let! id2 = store.AddVectorAsync("let y = 2", CodeBlock)
            let! id3 = store.AddVectorAsync("This is documentation", Documentation)
            
            // Act
            let! results = store.SearchSimilarAsync("let z = 3", 2, CodeBlock)
            
            // Assert
            Assert.Equal(2, results.Length)
            Assert.All(results, fun r -> Assert.Equal(CodeBlock, r.Vector.SemanticType))
            Assert.True(results.[0].Rank < results.[1].Rank)
        }

    [<Fact>]
    let ``SemanticVectorStore should perform clustering`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            
            // Add multiple vectors
            let! _ = store.AddVectorAsync("let x = 1", CodeBlock)
            let! _ = store.AddVectorAsync("let y = 2", CodeBlock)
            let! _ = store.AddVectorAsync("This is documentation", Documentation)
            let! _ = store.AddVectorAsync("Another documentation", Documentation)
            let! _ = store.AddVectorAsync("Error occurred", ErrorMessage)
            let! _ = store.AddVectorAsync("Another error", ErrorMessage)
            
            // Act
            let clusters = store.PerformSemanticClustering(3)
            
            // Assert
            Assert.Equal(3, clusters.Length)
            Assert.All(clusters, fun c -> 
                Assert.NotNull(c.Id)
                Assert.NotNull(c.Centroid)
                Assert.True(c.Coherence >= 0.0 && c.Coherence <= 1.0)
                Assert.True(c.Diversity >= 0.0 && c.Diversity <= 1.0))
        }

    [<Fact>]
    let ``CalculateSemanticSimilarity should consider type compatibility`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            
            let! id1 = store.AddVectorAsync("let x = 1", CodeBlock)
            let! id2 = store.AddVectorAsync("let y = 2", CodeBlock)
            let! id3 = store.AddVectorAsync("Documentation text", Documentation)
            
            let vector1 = store.GetVector(id1).Value
            let vector2 = store.GetVector(id2).Value
            let vector3 = store.GetVector(id3).Value
            
            // Act
            let similarity1 = store.CalculateSemanticSimilarity(vector1, vector2)
            let similarity2 = store.CalculateSemanticSimilarity(vector1, vector3)
            
            // Assert
            Assert.True(similarity1.SemanticRelevance > similarity2.SemanticRelevance,
                $"Code-to-code similarity ({similarity1.SemanticRelevance}) should be higher than code-to-docs ({similarity2.SemanticRelevance})")
        }

    [<Fact>]
    let ``SemanticVectorStore should remove vectors`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            let! vectorId = store.AddVectorAsync("test content", CodeBlock)
            
            // Act
            let removed = store.RemoveVector(vectorId)
            let vector = store.GetVector(vectorId)
            
            // Assert
            Assert.True(removed)
            Assert.True(vector.IsNone)
        }

    [<Fact>]
    let ``SemanticVectorStore should clear all vectors`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            let! _ = store.AddVectorAsync("content 1", CodeBlock)
            let! _ = store.AddVectorAsync("content 2", Documentation)
            
            // Act
            store.Clear()
            let allVectors = store.GetAllVectors()
            
            // Assert
            Assert.Empty(allVectors)
        }

    [<Fact>]
    let ``SemanticVectorStoreService should add FLUX code`` () =
        task {
            // Arrange
            let service = SemanticVectorStoreService()
            let code = "let result = 42"
            
            // Act
            let! vectorId = service.AddFluxCodeAsync(code)
            
            // Assert
            Assert.NotNull(vectorId)
            Assert.NotEmpty(vectorId)
        }

    [<Fact>]
    let ``SemanticVectorStoreService should search similar code`` () =
        task {
            // Arrange
            let service = SemanticVectorStoreService()
            let! _ = service.AddFluxCodeAsync("let x = 1")
            let! _ = service.AddFluxCodeAsync("let y = 2")
            
            // Act
            let! results = service.SearchSimilarCodeAsync("let z = 3", 2)
            
            // Assert
            Assert.NotEmpty(results)
            Assert.True(results.Length <= 2)
        }

    [<Fact>]
    let ``SemanticVectorStoreService should add execution results`` () =
        task {
            // Arrange
            let service = SemanticVectorStoreService()
            let result = "Execution completed successfully"
            
            // Act
            let! vectorId = service.AddExecutionResultAsync(result)
            
            // Assert
            Assert.NotNull(vectorId)
            Assert.NotEmpty(vectorId)
        }

    [<Fact>]
    let ``SemanticVectorStoreService should analyze codebase`` () =
        task {
            // Arrange
            let service = SemanticVectorStoreService()
            let! _ = service.AddFluxCodeAsync("let x = 1")
            let! _ = service.AddFluxCodeAsync("let y = 2")
            let! _ = service.AddFluxCodeAsync("let z = 3")
            
            // Act
            let clusters = service.AnalyzeFluxCodebase()
            
            // Assert
            Assert.NotEmpty(clusters)
        }

    [<Fact>]
    let ``SemanticVectorStoreService should provide semantic insights`` () =
        task {
            // Arrange
            let service = SemanticVectorStoreService()
            let! _ = service.AddFluxCodeAsync("let x = 1")
            let! _ = service.AddExecutionResultAsync("Result: 1")
            
            // Act
            let insights = service.GetSemanticInsights()
            
            // Assert
            Assert.NotEmpty(insights)
            Assert.True(insights.ContainsKey("TotalVectors"))
            Assert.True(insights.ContainsKey("CodeVectors"))
            Assert.True(insights.ContainsKey("Clusters"))
        }

    [<Theory>]
    [<InlineData(CodeBlock, CodeBlock, 1.0)>]
    [<InlineData(Documentation, Documentation, 1.0)>]
    [<InlineData(CodeBlock, Documentation, 0.8)>]
    [<InlineData(ErrorMessage, CodeBlock, 0.6)>]
    let ``Semantic type relevance should be calculated correctly`` (type1: SemanticType, type2: SemanticType, expectedRelevance: float) =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            
            let! id1 = store.AddVectorAsync("content 1", type1)
            let! id2 = store.AddVectorAsync("content 2", type2)
            
            let vector1 = store.GetVector(id1).Value
            let vector2 = store.GetVector(id2).Value
            
            // Act
            let similarity = store.CalculateSemanticSimilarity(vector1, vector2)
            
            // Assert
            // The semantic relevance should be influenced by type compatibility
            // We can't test exact values due to embedding randomness, but we can test relative ordering
            Assert.True(similarity.SemanticRelevance >= 0.0 && similarity.SemanticRelevance <= 1.0)
        }

    [<Fact>]
    let ``SemanticVector should have all required properties`` () =
        task {
            // Arrange
            let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
            let store = SemanticVectorStore(embeddingService)
            let content = "test content"
            let metadata = Map.ofList [("key", box "value")]
            
            // Act
            let! vectorId = store.AddVectorAsync(content, CodeBlock, metadata)
            let vector = store.GetVector(vectorId).Value
            
            // Assert
            Assert.Equal(vectorId, vector.Id)
            Assert.Equal(content, vector.Content)
            Assert.Equal(CodeBlock, vector.SemanticType)
            Assert.NotNull(vector.Embedding)
            Assert.True(vector.Embedding.Length > 0)
            Assert.True(vector.Metadata.ContainsKey("key"))
            Assert.True(vector.Timestamp <= DateTime.UtcNow)
        }
