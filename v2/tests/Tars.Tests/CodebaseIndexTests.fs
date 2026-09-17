module Tars.Tests.CodebaseIndexTests

open System
open System.IO
open System.Threading.Tasks
open Xunit
open Tars.Llm
open Tars.Cortex
open Tars.Tools.Standard

// Issue #241: the four codebase MCP tools were registered but gated on a mutable index
// nothing ever filled, so every call answered "not initialized".

/// Embeds anything as a non-zero vector, so a search that claims to be semantic is
/// not quietly falling back because embedding failed.
type private EmbeddingLlm() =
    let refuse () =
        Task.FromException<LlmResponse>(NotSupportedException "not used")

    interface ILlmService with
        member _.CompleteAsync(_) = refuse ()
        member _.CompleteStreamAsync(_, _) = refuse ()
        member _.EmbedAsync(_) = Task.FromResult [| 1.0f; 0.0f |]

        member _.RouteAsync(_) =
            Task.FromException<Routing.RoutedBackend>(NotSupportedException "not used")

let private sandbox () =
    let dir =
        Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), $"rag-{Guid.NewGuid():N}"))

    File.WriteAllText(
        Path.Combine(dir.FullName, "Widget.fs"),
        "module Widget\n\nlet assembleWidget parts = List.length parts\n"
    )

    File.WriteAllText(Path.Combine(dir.FullName, "Unrelated.fs"), "module Unrelated\n\nlet gadget () = 42\n")

    dir.FullName

[<Fact>]
let ``a quick-ingested index answers by keyword instead of an empty semantic result`` () =
    // IngestQuickAsync stores chunks without embeddings. Searching the empty vector
    // collection used to return [], which reads as "no such code" rather than "no index".
    let root = sandbox ()

    try
        let index = CodebaseRAG.CodebaseIndex(InMemoryVectorStore(), EmbeddingLlm())

        index.IngestQuickAsync(root)
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> ignore

        let results =
            index.SearchAsync("assembleWidget", 5)
            |> Async.AwaitTask
            |> Async.RunSynchronously

        Assert.NotEmpty results
        Assert.All(results, fun r -> Assert.Contains("Widget.fs", r.Chunk.FilePath))
    finally
        Directory.Delete(root, true)

[<Fact>]
let ``the shared index is quick-ingested once and can be replaced`` () =
    let root = sandbox ()
    CodebaseRAG.SharedIndex.clear ()

    try
        Assert.True(CodebaseRAG.SharedIndex.tryGet().IsNone)

        let first =
            CodebaseRAG.SharedIndex.getOrQuickIngest root
            |> Async.AwaitTask
            |> Async.RunSynchronously

        let second =
            CodebaseRAG.SharedIndex.getOrQuickIngest root
            |> Async.AwaitTask
            |> Async.RunSynchronously

        Assert.Same(first, second)
        Assert.True(first.IsIngested)
        Assert.True(first.GetStats().TotalChunks > 0)

        // A caller with embeddings wins over the lazily built one.
        let better = CodebaseRAG.CodebaseIndex(InMemoryVectorStore(), EmbeddingLlm())
        CodebaseRAG.SharedIndex.set better

        let after =
            CodebaseRAG.SharedIndex.getOrQuickIngest root
            |> Async.AwaitTask
            |> Async.RunSynchronously

        Assert.Same(better, after)
    finally
        CodebaseRAG.SharedIndex.clear ()
        Directory.Delete(root, true)

[<Fact>]
let ``search_codebase returns results instead of reporting an uninitialized index`` () =
    if not (TestHelpers.requireTools ()) then
        ()
    else
        let root = sandbox ()
        let previousRoot = Environment.GetEnvironmentVariable "TARS_CODEBASE_ROOT"
        Environment.SetEnvironmentVariable("TARS_CODEBASE_ROOT", root)
        CodebaseRAG.SharedIndex.clear ()

        try
            let answer =
                SemanticCodeTools.searchCodebase """{"query": "assembleWidget"}"""
                |> Async.AwaitTask
                |> Async.RunSynchronously

            Assert.DoesNotContain("not initialized", answer)
            Assert.Contains("Widget.fs", answer)
        finally
            CodebaseRAG.SharedIndex.clear ()
            Environment.SetEnvironmentVariable("TARS_CODEBASE_ROOT", previousRoot)
            Directory.Delete(root, true)

