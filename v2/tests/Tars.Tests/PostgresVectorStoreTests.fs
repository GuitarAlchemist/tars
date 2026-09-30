module Tars.Tests.PostgresVectorStoreTests

open System
open Xunit
open Tars.Core
open Tars.Cortex

// IMPORTANT: These tests require a running Postgres container with pgvector.
// docker run --name tars_postgres -e POSTGRES_PASSWORD=tars_password -e POSTGRES_DB=tars_memory -p 5432:5432 -d pgvector/pgvector:pg16

[<Collection("Postgres Integration")>]
type PostgresVectorStoreTests() =
    let connectionString =
        "Host=localhost;Port=5432;Database=tars_memory;Username=postgres;Password=tars_password"

    let testDim = 768

    [<Fact>]
    member this.``Save and Retrieve vector``() =
        task {
            if not (TestHelpers.requirePostgres ()) then () else

            let store = PostgresVectorStore(connectionString, testDim) :> IVectorStore

            let vector1 = [| 0.1f; 0.2f; 0.3f; 0.4f; 0.5f; 0.6f; 0.7f; 0.8f |]
            let vector3 = [| 0.11f; 0.21f; 0.31f; 0.41f; 0.51f; 0.61f; 0.71f; 0.81f |]

            let pad (v: float32[]) =
                let padded = Array.zeroCreate<float32> testDim
                Array.blit v 0 padded 0 v.Length
                padded

            let v1 = pad vector1
            let v3 = pad vector3

            let collection = "test_collection_" + Guid.NewGuid().ToString("N")
            let id = "vec1"
            let payload = Map [ "content", "hello world" ]

            do! store.SaveAsync(collection, id, v1, payload)

            let! count = (store :?> PostgresVectorStore).GetCountAsync(collection)
            Assert.Equal(1, count)

            let! results = store.SearchAsync(collection, v1, 1)
            Assert.NotEmpty(results)
            let hit = results.Head
            Assert.Equal(id, hit.Id)
            Assert.True(hit.Distance < 0.001f, $"Expected distance < 0.001, got {hit.Distance}")
            Assert.Equal("hello world", hit.Payload["content"])

            do! store.SaveAsync(collection, "vec3", v3, Map [ "content", "similar" ])
            let! searchResults = store.SearchAsync(collection, v1, 2)
            Assert.Equal(2, searchResults.Length)
            Assert.Equal("vec1", searchResults.Item(0).Id)
            Assert.Equal("vec3", searchResults.Item(1).Id)
        }

    [<Fact>]
    member this.``the schema creates what is missing and never drops anything``() =
        // Runs everywhere, Postgres or not. Every `tars evolve` and every UI start
        // constructs a store, and construction applied a schema that began with
        // `DROP TABLE IF EXISTS vectors` - wiping every persisted embedding (#284).
        let sql = PostgresVectorStore.SchemaSql 768

        Assert.DoesNotContain("DROP", sql.ToUpperInvariant())
        Assert.Contains("CREATE TABLE IF NOT EXISTS vectors", sql)
        Assert.Contains("vector(768)", sql)

    [<Fact>]
    member this.``a store constructed later keeps what an earlier one saved``() =
        task {
            if not (TestHelpers.requirePostgres ()) then () else

            let first = PostgresVectorStore(connectionString, testDim) :> IVectorStore
            let collection = "persist_" + Guid.NewGuid().ToString("N")
            do! first.SaveAsync(collection, "kept", Array.create testDim 0.1f, Map [ "content", "still here" ])

            // What a restart does.
            let second = PostgresVectorStore(connectionString, testDim)

            let! count = second.GetCountAsync(collection)
            Assert.Equal(1, count)
        }

