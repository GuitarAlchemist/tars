namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Core
open Tars.Cortex

module PersistenceTests =

    [<Fact>]
    let ``SqliteVectorStore can save and retrieve vectors`` () =
        task {
            // Arrange
            let dbPath = Path.Combine(Path.GetTempPath(), $"tars_test_{Guid.NewGuid()}.db")
            let store = SqliteVectorStore(dbPath) :> IVectorStore
            let collection = "test_collection"
            let id = "vec1"
            let vector = [| 0.1f; 0.2f; 0.3f |]
            let payload = Map [ "key", "value" ]

            try
                // Act
                do! store.SaveAsync(collection, id, vector, payload)
                let! results = store.SearchAsync(collection, vector, 1)

                // Assert
                Assert.NotEmpty(results)
                let (resId, score, resPayload) = results.Head
                Assert.Equal(id, resId)
                Assert.Equal("value", resPayload["key"])
                // Cosine similarity of identical vectors is 1.0, distance is 0.0 (or 1-sim depending on impl)
                // Tars.Cortex.Similarity.similarityToDistance usually does 1 - sim.
                // Let's check the implementation or just assert it's close to 0.
                Assert.True(score < 0.001f, $"Score {score} should be near 0 for identical vectors")

            finally
                // Cleanup
                Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools()

                if File.Exists(dbPath) then
                    File.Delete(dbPath)
        }
