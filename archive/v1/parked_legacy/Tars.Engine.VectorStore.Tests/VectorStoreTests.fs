namespace Tars.Engine.VectorStore.Tests

open System
open Xunit
open Tars.Engine.VectorStore

module private TestHelpers =
    let createConfig rawDimension =
        { RawDimension = rawDimension
          EnableFFT = true
          EnableDual = true
          EnableProjective = true
          EnableHyperbolic = true
          EnableWavelet = true
          EnableMinkowski = true
          EnablePauli = true
          SpaceWeights = Map.ofList [
              ("raw", 1.0)
              ("fft", 0.8)
              ("phase", 0.5)
              ("dual", 0.7)
              ("projective", 0.6)
              ("hyperbolic", 0.6)
              ("wavelet", 0.5)
              ("minkowski", 0.4)
              ("pauli", 0.3)
          ]
          PersistToDisk = false
          StoragePath = None }

    let run (workflow: Async<'T>) = Async.RunSynchronously workflow

    let createSampleDocuments (generator: IEmbeddingGenerator) =
        [ "doc1", "hello world", ["greeting"]
          "doc2", "advanced vector mathematics", ["science"; "vector"]
          "doc3", "functional programming with F#", ["development"; "fsharp"] ]
        |> List.map (fun (id, text, tags) ->
            let embedding = generator.GenerateEmbedding text |> run
            VectorStoreUtils.createDocument id text embedding tags (Some "unit-tests"))

module EmbeddingGeneratorTests =

    [<Fact>]
    let ``Basic generator produces deterministic embeddings`` () =
        let config = TestHelpers.createConfig 16
        let generator = EmbeddingGeneratorFactory.createBasic config

        let embedding1 = generator.GenerateEmbedding "persistent vector" |> TestHelpers.run
        let embedding2 = generator.GenerateEmbedding "persistent vector" |> TestHelpers.run
        let embeddingDifferent = generator.GenerateEmbedding "different payload" |> TestHelpers.run

        Assert.Equal<float[]>(embedding1.Raw, embedding2.Raw)
        Assert.NotEqual<float[]>(embedding1.Raw, embeddingDifferent.Raw)
        Assert.All(embedding1.Raw, (fun value -> Assert.InRange(value, -1.0, 1.0)))
        Assert.Equal("persistent vector".Length.ToString(), embedding1.Metadata.["text_length"])
        Assert.Equal(64, embedding1.Metadata.["text_hash"].Length)

    [<Fact>]
    let ``Enhanced generator aggregates token embeddings`` () =
        let config = TestHelpers.createConfig 12
        let generator = EmbeddingGeneratorFactory.createEnhanced config (Some "endpoint")

        let embedding1 = generator.GenerateEmbedding "token level test" |> TestHelpers.run
        let embedding2 = generator.GenerateEmbedding "token level test" |> TestHelpers.run

        Assert.Equal<float[]>(embedding1.Raw, embedding2.Raw)
        Assert.Equal("endpoint", embedding1.Metadata.["model_endpoint"])
        Assert.All(embedding1.Raw, (fun value -> Assert.InRange(value, -1.0, 1.0)))

module VectorStoreIntegrationTests =

    [<Fact>]
    let ``Vector store supports add, search, and statistics`` () =
        let config = TestHelpers.createConfig 18
        let generator = EmbeddingGeneratorFactory.createBasic config
        let store = VectorStoreFactory.createInMemory config

        let documents = TestHelpers.createSampleDocuments generator
        store.AddDocuments documents |> TestHelpers.run

        // Basic search
        let queryEmbedding = generator.GenerateEmbedding "hello" |> TestHelpers.run
        let query = VectorStoreUtils.createQuery "hello" queryEmbedding 5 -1.0 Map.empty
        let results = store.Search query |> TestHelpers.run
        Assert.NotEmpty(results)
        Assert.Equal("doc1", results.Head.Document.Id)

        // Metadata filter
        let textLength = results.Head.Document.Embedding.Metadata.["text_length"]
        let filteredQuery = { query with Filters = Map.ofList [("text_length", textLength)] }
        let filteredResults = store.Search filteredQuery |> TestHelpers.run
        Assert.True(filteredResults |> List.exists (fun r -> r.Document.Id = "doc1"))

        // Tag filter (filter key does not matter because tags are value matched)
        let tagQuery = { query with Filters = Map.ofList [("tag", "science")] }
        let tagResults = store.Search tagQuery |> TestHelpers.run
        Assert.True(tagResults |> List.exists (fun r -> r.Document.Id = "doc2"))

        // Statistics
        let stats = VectorStoreUtils.getStatistics store |> TestHelpers.run
        Assert.Equal(documents.Length, stats.DocumentCount)
        Assert.Equal(float config.RawDimension, stats.AverageEmbeddingSize)
        Assert.True(stats.SpaceUsageStats.ContainsKey("raw"))
        Assert.True(stats.IndexSize > 0L)

        // Snapshot
        let snapshot = store.GetAllDocuments() |> TestHelpers.run
        Assert.Equal(documents.Length, snapshot.Length)
        Assert.True((DateTime.UtcNow - stats.LastUpdated) < TimeSpan.FromMinutes 1.0)

    [<Fact>]
    let ``Batch add reports progress`` () =
        let config = TestHelpers.createConfig 10
        let generator = EmbeddingGeneratorFactory.createBasic config
        let store = VectorStoreFactory.createInMemory config
        let documents = TestHelpers.createSampleDocuments generator

        let progressUpdates = ResizeArray<int * int>()
        VectorStoreUtils.batchAddDocuments store documents (fun processed total -> progressUpdates.Add(processed, total))
        |> TestHelpers.run

        Assert.NotEmpty(progressUpdates)
        let lastProcessed, lastTotal = progressUpdates |> Seq.last
        Assert.Equal(documents.Length, lastProcessed)
        Assert.Equal(documents.Length, lastTotal)

        let storedCount = store.GetDocumentCount() |> TestHelpers.run
        Assert.Equal(documents.Length, storedCount)

    [<Fact>]
    let ``Search with expansion preserves results`` () =
        let config = TestHelpers.createConfig 14
        let generator = EmbeddingGeneratorFactory.createBasic config
        let store = VectorStoreFactory.createInMemory config
        let documents = TestHelpers.createSampleDocuments generator
        store.AddDocuments documents |> TestHelpers.run

        let baseQuery =
            { VectorStoreUtils.createQuery "unrelated" (generator.GenerateEmbedding "unrelated" |> TestHelpers.run) 3 -1.0 Map.empty with
                MaxResults = 3 }

        let baseline = store.Search baseQuery |> TestHelpers.run
        let expanded = VectorStoreUtils.searchWithExpansion store baseQuery ["hello"; "vector"] |> TestHelpers.run

        Assert.True(expanded.Length >= baseline.Length)

module InMemoryStoreDeterminismTests =

    [<Fact>]
    let ``GetAllDocuments returns deterministic order by insertion`` () =
        let config = TestHelpers.createConfig 8
        let generator = EmbeddingGeneratorFactory.createBasic config
        let store = VectorStoreFactory.createInMemory config
        let docs = TestHelpers.createSampleDocuments generator

        docs |> List.iter (fun doc -> store.AddDocument doc |> TestHelpers.run)
        let fetched = store.GetAllDocuments() |> TestHelpers.run

        let originalIds = docs |> List.map (fun doc -> doc.Id) |> Set.ofList
        let fetchedIds = fetched |> List.map (fun doc -> doc.Id) |> Set.ofList
        Assert.Equal<Set<string>>(originalIds, fetchedIds)
