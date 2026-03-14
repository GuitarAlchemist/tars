namespace Tars.Tests

open Xunit
open Tars.Core
open Tars.Cortex

type AnnVectorStoreTests() =

    let v1 = [| 1.0f; 0.0f; 0.0f |]
    let v2 = [| 0.9f; 0.1f; 0.0f |]
    let v3 = [| 0.0f; 1.0f; 0.0f |]

    [<Fact>]
    member _.``AnnVectorStore returns nearest from same bucket``() =
        let store = AnnVectorStore(8) :> IVectorStore
        store.SaveAsync("c", "a", v1, Map.empty) |> Async.AwaitTask |> Async.RunSynchronously
        store.SaveAsync("c", "b", v2, Map.empty) |> Async.AwaitTask |> Async.RunSynchronously
        store.SaveAsync("c", "c", v3, Map.empty) |> Async.AwaitTask |> Async.RunSynchronously

        let results =
            store.SearchAsync("c", v1, 2)
            |> Async.AwaitTask
            |> Async.RunSynchronously

        Assert.True(results.Length >= 1)
        let (id, _, _) = results.[0]
        Assert.Equal("a", id)

    [<Fact>]
    member _.``AnnVectorStore falls back when bucket empty``() =
        let store = AnnVectorStore(2) :> IVectorStore
        store.SaveAsync("c", "a", v1, Map.empty) |> Async.AwaitTask |> Async.RunSynchronously

        // Query a very different vector to likely hit an empty bucket
        let q = [| 0.0f; 0.0f; 1.0f |]

        let results =
            store.SearchAsync("c", q, 1)
            |> Async.AwaitTask
            |> Async.RunSynchronously

        Assert.Equal(1, results.Length)
        let (id, _, _) = results.[0]
        // Should still return something due to fallback scan
        Assert.Equal("a", id)
