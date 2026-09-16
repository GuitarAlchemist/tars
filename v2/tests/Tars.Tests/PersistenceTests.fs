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
                let hit = results.Head
                Assert.Equal(id, hit.Id)
                Assert.Equal("value", hit.Payload["key"])
                Assert.True(hit.Distance < 0.001f, $"Distance {hit.Distance} should be near 0 for identical vectors")

            finally
                // Cleanup
                Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools()

                if File.Exists(dbPath) then
                    File.Delete(dbPath)
        }
