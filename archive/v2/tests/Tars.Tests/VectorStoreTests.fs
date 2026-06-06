namespace Tars.Tests

open System
open System.IO
open Microsoft.Data.Sqlite
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Cortex

type VectorStoreTests(output: ITestOutputHelper) =

    // Helper to create a simple test vector
    let createVector (values: float32 list) = values |> Array.ofList

    // SQLite cleanup helper - clears connection pools and forces GC before deleting
    let deleteIfExists path =
        SqliteConnection.ClearAllPools()
        GC.Collect()
        GC.WaitForPendingFinalizers()
        try
            if File.Exists(path) then File.Delete(path)
        with
        | :? IOException -> () // Ignore if file is still locked, temp dir will clean up

    // Similarity Module Tests

    [<Fact>]
    member _.``Similarity: Cosine similarity of identical vectors is 1``() =
        output.WriteLine("Starting test: Cosine similarity of identical vectors is 1")
        // Arrange
        let v1 = createVector [1.0f; 2.0f; 3.0f]
        let v2 = createVector [1.0f; 2.0f; 3.0f]

        // Act
        let similarity = Similarity.cosineSimilarity v1 v2

        // Assert
        Assert.Equal(1.0f, similarity, 5)
        output.WriteLine($"Similarity: {similarity}")

    [<Fact>]
    member _.``Similarity: Cosine similarity of orthogonal vectors is 0``() =
        output.WriteLine("Starting test: Cosine similarity of orthogonal vectors is 0")
        // Arrange
        let v1 = createVector [1.0f; 0.0f; 0.0f]
        let v2 = createVector [0.0f; 1.0f; 0.0f]

        // Act
        let similarity = Similarity.cosineSimilarity v1 v2

        // Assert
        Assert.Equal(0.0f, similarity, 5)
        output.WriteLine($"Similarity: {similarity}")

    [<Fact>]
    member _.``Similarity: Cosine similarity of opposite vectors is -1``() =
        output.WriteLine("Starting test: Cosine similarity of opposite vectors is -1")
        // Arrange
        let v1 = createVector [1.0f; 2.0f; 3.0f]
        let v2 = createVector [-1.0f; -2.0f; -3.0f]

        // Act
        let similarity = Similarity.cosineSimilarity v1 v2

        // Assert
        Assert.Equal(-1.0f, similarity, 5)
        output.WriteLine($"Similarity: {similarity}")

    [<Fact>]
    member _.``Similarity: Returns 0 for empty vectors``() =
        output.WriteLine("Starting test: Returns 0 for empty vectors")
        // Arrange
        let v1 = createVector []
        let v2 = createVector []

        // Act
        let similarity = Similarity.cosineSimilarity v1 v2

        // Assert
        Assert.Equal(0.0f, similarity)
        output.WriteLine($"Similarity: {similarity}")

    [<Fact>]
    member _.``Similarity: Returns 0 for different length vectors``() =
        output.WriteLine("Starting test: Returns 0 for different length vectors")
        // Arrange
        let v1 = createVector [1.0f; 2.0f]
        let v2 = createVector [1.0f; 2.0f; 3.0f]

        // Act
        let similarity = Similarity.cosineSimilarity v1 v2

        // Assert
        Assert.Equal(0.0f, similarity)
        output.WriteLine($"Similarity: {similarity}")

    [<Fact>]
    member _.``Similarity: Distance conversion works correctly``() =
        output.WriteLine("Starting test: Distance conversion works correctly")
        // Arrange & Act
        let distanceForIdentical = Similarity.similarityToDistance 1.0f
        let distanceForOrthogonal = Similarity.similarityToDistance 0.0f
        let distanceForOpposite = Similarity.similarityToDistance -1.0f

        // Assert
        Assert.Equal(0.0f, distanceForIdentical, 5)
        Assert.Equal(1.0f, distanceForOrthogonal, 5)
        Assert.Equal(2.0f, distanceForOpposite, 5)
        output.WriteLine($"Distances: identical={distanceForIdentical}, orthogonal={distanceForOrthogonal}, opposite={distanceForOpposite}")

    // InMemoryVectorStore Tests

    [<Fact>]
    member _.``InMemoryVectorStore: Can save and retrieve by search``() =
        task {
            output.WriteLine("Starting test: Can save and retrieve by search")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_save_retrieve"
            let vector = createVector [1.0f; 0.0f; 0.0f]
            let payload = Map [ "text", "Hello World" ]

            // Act
            do! vectorStore.SaveAsync(collection, "doc1", vector, payload)
            let! results = vectorStore.SearchAsync(collection, vector, 10)

            // Assert
            Assert.Single(results) |> ignore
            let (id, distance, meta) = results.Head
            Assert.Equal("doc1", id)
            Assert.Equal(0.0f, distance, 5)  // Distance should be 0 for identical vector
            Assert.Equal("Hello World", meta["text"])
            output.WriteLine($"Found document: {id} with distance {distance}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Search returns results ordered by similarity``() =
        task {
            output.WriteLine("Starting test: Search returns results ordered by similarity")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_ordering"
            
            // Create vectors with varying similarity to query
            let queryVector = createVector [1.0f; 0.0f; 0.0f]
            let identicalVector = createVector [1.0f; 0.0f; 0.0f]
            let similarVector = createVector [0.9f; 0.1f; 0.0f]
            let differentVector = createVector [0.0f; 1.0f; 0.0f]

            // Act
            do! vectorStore.SaveAsync(collection, "different", differentVector, Map.empty)
            do! vectorStore.SaveAsync(collection, "identical", identicalVector, Map.empty)
            do! vectorStore.SaveAsync(collection, "similar", similarVector, Map.empty)
            
            let! results = vectorStore.SearchAsync(collection, queryVector, 10)

            // Assert
            Assert.Equal(3, results.Length)
            let ids = results |> List.map (fun (id, _, _) -> id)
            Assert.Equal("identical", ids[0])  // Most similar (distance 0)
            Assert.Equal("similar", ids[1])    // Second most similar
            Assert.Equal("different", ids[2])  // Least similar (orthogonal)
            let orderStr = String.Join(", ", ids)
            output.WriteLine($"Order: {orderStr}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Search respects limit parameter``() =
        task {
            output.WriteLine("Starting test: Search respects limit parameter")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_limit"
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Add 5 documents
            for i in 1..5 do
                do! vectorStore.SaveAsync(collection, $"doc{i}", vector, Map.empty)

            // Act
            let! results = vectorStore.SearchAsync(collection, vector, 3)

            // Assert
            Assert.Equal(3, results.Length)
            output.WriteLine($"Returned {results.Length} results (limit was 3)")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Search on empty collection returns empty list``() =
        task {
            output.WriteLine("Starting test: Search on empty collection returns empty list")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_empty"
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Act
            let! results = vectorStore.SearchAsync(collection, vector, 10)

            // Assert
            Assert.Empty(results)
            output.WriteLine("Correctly returned empty list for empty collection")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Can update existing document``() =
        task {
            output.WriteLine("Starting test: Can update existing document")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_update"
            let vector1 = createVector [1.0f; 0.0f; 0.0f]
            let vector2 = createVector [0.0f; 1.0f; 0.0f]
            let payload1 = Map [ "version", "1" ]
            let payload2 = Map [ "version", "2" ]

            // Act
            do! vectorStore.SaveAsync(collection, "doc1", vector1, payload1)
            do! vectorStore.SaveAsync(collection, "doc1", vector2, payload2)
            
            let! entry = store.GetByIdAsync(collection, "doc1")

            // Assert
            Assert.True(entry.IsSome)
            let e = entry.Value
            Assert.Equal("2", e.Payload["version"])
            Assert.Equal(2, e.Version)  // Version should be incremented
            output.WriteLine($"Document updated to version {e.Version}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: GetCount returns correct count``() =
        task {
            output.WriteLine("Starting test: GetCount returns correct count")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_count"
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Act
            let! countBefore = store.GetCountAsync(collection)
            do! vectorStore.SaveAsync(collection, "doc1", vector, Map.empty)
            do! vectorStore.SaveAsync(collection, "doc2", vector, Map.empty)
            let! countAfter = store.GetCountAsync(collection)

            // Assert
            Assert.Equal(0, countBefore)
            Assert.Equal(2, countAfter)
            output.WriteLine($"Count: before={countBefore}, after={countAfter}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Delete removes document``() =
        task {
            output.WriteLine("Starting test: Delete removes document")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_delete"
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Act
            do! vectorStore.SaveAsync(collection, "doc1", vector, Map.empty)
            let! countBefore = store.GetCountAsync(collection)
            let! deleted = store.DeleteAsync(collection, "doc1")
            let! countAfter = store.GetCountAsync(collection)

            // Assert
            Assert.True(deleted)
            Assert.Equal(1, countBefore)
            Assert.Equal(0, countAfter)
            output.WriteLine($"Deleted: {deleted}, count: before={countBefore}, after={countAfter}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Checksum validation works``() =
        task {
            output.WriteLine("Starting test: Checksum validation works")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_checksum"
            let vector = createVector [1.0f; 2.0f; 3.0f]
            let payload = Map [ "key", "value" ]

            // Act
            do! vectorStore.SaveAsync(collection, "doc1", vector, payload)
            let! isValid = store.ValidateChecksumAsync(collection, "doc1")
            let! isValidMissing = store.ValidateChecksumAsync(collection, "nonexistent")

            // Assert
            Assert.True(isValid)
            Assert.False(isValidMissing)
            output.WriteLine($"Checksum valid: {isValid}, missing doc valid: {isValidMissing}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: GetCollections returns all collection names``() =
        task {
            output.WriteLine("Starting test: GetCollections returns all collection names")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection1 = "collection_a_getcoll"
            let collection2 = "collection_b_getcoll"
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Act
            do! vectorStore.SaveAsync(collection1, "doc1", vector, Map.empty)
            do! vectorStore.SaveAsync(collection2, "doc1", vector, Map.empty)
            let! collections = store.GetCollectionsAsync()

            // Assert
            Assert.Contains(collection1, collections)
            Assert.Contains(collection2, collections)
            let collStr = String.Join(", ", collections)
            output.WriteLine($"Collections: {collStr}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: DeleteCollection removes entire collection``() =
        task {
            output.WriteLine("Starting test: DeleteCollection removes entire collection")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let collection = "test_collection_delcoll"
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Act
            do! vectorStore.SaveAsync(collection, "doc1", vector, Map.empty)
            do! vectorStore.SaveAsync(collection, "doc2", vector, Map.empty)
            let! deleted = store.DeleteCollectionAsync(collection)
            let! collections = store.GetCollectionsAsync()

            // Assert
            Assert.True(deleted)
            Assert.DoesNotContain(collection, collections)
            output.WriteLine($"Collection deleted: {deleted}")
        }

    [<Fact>]
    member _.``InMemoryVectorStore: File persistence works``() =
        task {
            output.WriteLine("Starting test: File persistence works")
            // Arrange
            let store1 = InMemoryVectorStore()
            let vectorStore1 = store1 :> IVectorStore
            let collection = "persist_test"
            let vector = createVector [1.0f; 2.0f; 3.0f]
            let payload = Map [ "text", "Persisted data" ]
            let tempFile = Path.Combine(Path.GetTempPath(), $"vectorstore_test_{Guid.NewGuid()}.json")

            try
                // Act - Save and persist
                do! vectorStore1.SaveAsync(collection, "doc1", vector, payload)
                do! store1.PersistToFileAsync(tempFile)

                // Load into new store
                let store2 = InMemoryVectorStore()
                let! loaded = store2.LoadFromFileAsync(tempFile)
                let! entry = store2.GetByIdAsync(collection, "doc1")

                // Assert
                Assert.True(loaded)
                Assert.True(entry.IsSome)
                Assert.Equal("Persisted data", entry.Value.Payload["text"])
                // Compare vectors element by element
                Assert.Equal(vector.Length, entry.Value.Vector.Length)
                for i in 0 .. vector.Length - 1 do
                    Assert.Equal(vector[i], entry.Value.Vector[i])
                output.WriteLine($"Successfully persisted and loaded data from {tempFile}")
            finally
                // Cleanup
                if File.Exists(tempFile) then
                    File.Delete(tempFile)
        }

    [<Fact>]
    member _.``InMemoryVectorStore: Clear removes all data``() =
        task {
            output.WriteLine("Starting test: Clear removes all data")
            // Arrange
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore
            let vector = createVector [1.0f; 0.0f; 0.0f]

            // Act
            do! vectorStore.SaveAsync("collection1", "doc1", vector, Map.empty)
            do! vectorStore.SaveAsync("collection2", "doc1", vector, Map.empty)
            let! collectionsBefore = store.GetCollectionsAsync()
            do! store.ClearAsync()
            let! collectionsAfter = store.GetCollectionsAsync()

            // Assert
            Assert.Equal(2, collectionsBefore.Length)
            Assert.Empty(collectionsAfter)
            output.WriteLine($"Collections: before={collectionsBefore.Length}, after={collectionsAfter.Length}")
        }

    // ============================================================
    // SqliteVectorStore Tests
    // ============================================================

    [<Fact>]
    member _.``SqliteVectorStore: Can save and retrieve by search``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore save and retrieve")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_test_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let collection = "test_collection"
                let vector = createVector [1.0f; 0.0f; 0.0f]
                let payload = Map [ "text", "Hello SQLite" ]

                do! vectorStore.SaveAsync(collection, "doc1", vector, payload)
                let! results = vectorStore.SearchAsync(collection, vector, 10)

                Assert.Single(results) |> ignore
                let (id, distance, meta) = results.Head
                Assert.Equal("doc1", id)
                Assert.True(distance < 0.01f, "Distance should be near 0 for identical vector")
                Assert.Equal("Hello SQLite", meta["text"])
                output.WriteLine($"Found: {id} with distance {distance}")
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: Search returns results ordered by similarity``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore ordering")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_order_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let collection = "test_ordering"

                let queryVector = createVector [1.0f; 0.0f; 0.0f]
                let identicalVector = createVector [1.0f; 0.0f; 0.0f]
                let similarVector = createVector [0.9f; 0.1f; 0.0f]
                let differentVector = createVector [0.0f; 1.0f; 0.0f]

                do! vectorStore.SaveAsync(collection, "different", differentVector, Map.empty)
                do! vectorStore.SaveAsync(collection, "identical", identicalVector, Map.empty)
                do! vectorStore.SaveAsync(collection, "similar", similarVector, Map.empty)

                let! results = vectorStore.SearchAsync(collection, queryVector, 10)

                Assert.Equal(3, results.Length)
                let ids = results |> List.map (fun (id, _, _) -> id)
                Assert.Equal("identical", ids[0])
                Assert.Equal("similar", ids[1])
                Assert.Equal("different", ids[2])
                let orderStr = String.Join(", ", ids)
                output.WriteLine $"Order: %s{orderStr}"
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: Search respects limit parameter``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore limit")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_limit_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let collection = "test_limit"
                let vector = createVector [1.0f; 0.0f; 0.0f]

                for i in 1..5 do
                    do! vectorStore.SaveAsync(collection, $"doc{i}", vector, Map.empty)

                let! results = vectorStore.SearchAsync(collection, vector, 3)

                Assert.Equal(3, results.Length)
                output.WriteLine($"Returned {results.Length} results (limit was 3)")
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: Search on empty collection returns empty list``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore empty collection")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_empty_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let vector = createVector [1.0f; 0.0f; 0.0f]

                let! results = vectorStore.SearchAsync("nonexistent", vector, 10)

                Assert.Empty(results)
                output.WriteLine("Correctly returned empty list")
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: Can update existing document``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore update")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_update_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let collection = "test_update"
                let vector1 = createVector [1.0f; 0.0f; 0.0f]
                let vector2 = createVector [0.0f; 1.0f; 0.0f]
                let payload1 = Map [ "version", "1" ]
                let payload2 = Map [ "version", "2" ]

                do! vectorStore.SaveAsync(collection, "doc1", vector1, payload1)
                do! vectorStore.SaveAsync(collection, "doc1", vector2, payload2)

                // Search with new vector should find it
                let! results = vectorStore.SearchAsync(collection, vector2, 10)

                Assert.Single(results) |> ignore
                let (_, _, meta) = results.Head
                Assert.Equal("2", meta["version"])
                output.WriteLine("Document updated successfully")
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: GetCount returns correct count``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore count")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_count_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let collection = "test_count"
                let vector = createVector [1.0f; 0.0f; 0.0f]

                let! countBefore = store.GetCountAsync(collection)
                do! vectorStore.SaveAsync(collection, "doc1", vector, Map.empty)
                do! vectorStore.SaveAsync(collection, "doc2", vector, Map.empty)
                let! countAfter = store.GetCountAsync(collection)

                Assert.Equal(0, countBefore)
                Assert.Equal(2, countAfter)
                output.WriteLine($"Count: before={countBefore}, after={countAfter}")
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: GetCollections returns all collection names``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore collections")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_coll_{Guid.NewGuid()}.db")
            try
                let store = SqliteVectorStore(dbPath)
                let vectorStore = store :> IVectorStore
                let vector = createVector [1.0f; 0.0f; 0.0f]

                do! vectorStore.SaveAsync("collection_a", "doc1", vector, Map.empty)
                do! vectorStore.SaveAsync("collection_b", "doc1", vector, Map.empty)
                let! collections = store.GetCollectionsAsync()

                Assert.Contains("collection_a", collections)
                Assert.Contains("collection_b", collections)
                let collStr = String.Join(", ", collections)
                output.WriteLine $"Collections: %s{collStr}"
            finally
                deleteIfExists dbPath
        }

    [<Fact>]
    member _.``SqliteVectorStore: Persists data across instances``() =
        task {
            output.WriteLine("Starting test: SqliteVectorStore persistence")
            let dbPath = Path.Combine(Path.GetTempPath(), $"sqlitestore_persist_{Guid.NewGuid()}.db")
            try
                let collection = "persist_test"
                let vector = createVector [1.0f; 2.0f; 3.0f]
                let payload = Map [ "key", "persisted_value" ]

                // Save with first instance
                let store1 = SqliteVectorStore(dbPath)
                do! (store1 :> IVectorStore).SaveAsync(collection, "doc1", vector, payload)

                // Load with second instance
                let store2 = SqliteVectorStore(dbPath)
                let! results = (store2 :> IVectorStore).SearchAsync(collection, vector, 10)

                Assert.Single(results) |> ignore
                let (id, _, meta) = results.Head
                Assert.Equal("doc1", id)
                Assert.Equal("persisted_value", meta["key"])
                output.WriteLine("Data persisted successfully across instances")
            finally
                deleteIfExists dbPath
        }
