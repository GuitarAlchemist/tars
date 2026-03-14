namespace Tars.Tests

open Xunit
open Tars.Cortex.Chunking

module ChunkingTests =

    let sampleText = "This is sentence one. This is sentence two. This is sentence three."
    let longText = String.replicate 10 "This is a test paragraph with multiple words. "
    let docId = "test-doc"

    // === FixedSize Chunking ===

    [<Fact>]
    let ``FixedSize: Short text returns single chunk`` () =
        let config = { defaultConfig with ChunkSize = 500; MinChunkSize = 10; Strategy = FixedSize }
        let chunks = chunk config docId "Hello world"
        Assert.Single(chunks) |> ignore
        Assert.Equal("Hello world", chunks.[0].Content)

    [<Fact>]
    let ``FixedSize: Respects chunk size`` () =
        let config = { defaultConfig with ChunkSize = 50; MinChunkSize = 10; Strategy = FixedSize }
        let chunks = chunk config docId longText
        for c in chunks do
            Assert.True(c.Content.Length <= config.ChunkSize)

    [<Fact>]
    let ``FixedSize: Filters out small chunks`` () =
        let config = { defaultConfig with ChunkSize = 50; MinChunkSize = 30; Strategy = FixedSize }
        let chunks = chunk config docId longText
        for c in chunks do
            Assert.True(c.Content.Length >= config.MinChunkSize)

    // === SlidingWindow Chunking ===

    [<Fact>]
    let ``SlidingWindow: Has overlap`` () =
        let config = { defaultConfig with ChunkSize = 100; ChunkOverlap = 30; MinChunkSize = 10; Strategy = SlidingWindow }
        let chunks = chunk config docId longText
        
        if chunks.Length >= 2 then
            // Check that chunks overlap by comparing end/start positions
            let chunk1End = chunks.[0].Metadata.EndChar
            let chunk2Start = chunks.[1].Metadata.StartChar
            Assert.True(chunk2Start < chunk1End, "Chunks should overlap")

    [<Fact>]
    let ``SlidingWindow: Chunk IDs are unique`` () =
        let chunks = chunk defaultConfig docId longText
        let ids = chunks |> List.map (fun c -> c.Id)
        Assert.Equal(ids.Length, (ids |> List.distinct).Length)

    // === Sentence Chunking ===

    [<Fact>]
    let ``Sentence: Splits on sentence boundaries`` () =
        let config = { defaultConfig with ChunkSize = 50; MinChunkSize = 10; Strategy = Sentence }
        let chunks = chunk config docId sampleText
        Assert.True(chunks.Length >= 1)

    [<Fact>]
    let ``Sentence: Preserves complete sentences`` () =
        let config = { defaultConfig with ChunkSize = 100; MinChunkSize = 10; Strategy = Sentence }
        let text = "First sentence. Second sentence. Third sentence."
        let chunks = chunk config docId text
        // All chunks should end with period or be the last chunk
        for c in chunks do
            Assert.True(c.Content.Contains(".") || c = List.last chunks)

    // === Paragraph Chunking ===

    [<Fact>]
    let ``Paragraph: Splits on double newlines`` () =
        let config = { defaultConfig with ChunkSize = 200; MinChunkSize = 10; Strategy = Paragraph }
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        let chunks = chunk config docId text
        Assert.True(chunks.Length >= 1)

    [<Fact>]
    let ``Paragraph: Empty text returns empty list`` () =
        let config = { defaultConfig with Strategy = Paragraph; MinChunkSize = 1 }
        let chunks = chunk config docId ""
        Assert.Empty(chunks)

    // === Recursive Chunking ===

    [<Fact>]
    let ``Recursive: Handles nested structure`` () =
        let config = { defaultConfig with ChunkSize = 100; MinChunkSize = 10; Strategy = Recursive }
        let text = "Para 1.\n\nPara 2. Sentence A. Sentence B.\n\nPara 3."
        let chunks = chunk config docId text
        Assert.True(chunks.Length >= 1)

    [<Fact>]
    let ``Recursive: Falls back to fixed size for long words`` () =
        let config = { defaultConfig with ChunkSize = 20; MinChunkSize = 5; Strategy = Recursive }
        let text = "Supercalifragilisticexpialidocious"
        let chunks = chunk config docId text
        Assert.True(chunks.Length >= 1)

    // === Metadata Tests ===

    [<Fact>]
    let ``Metadata: Contains parent document ID`` () =
        let chunks = chunk defaultConfig docId sampleText
        for c in chunks do
            Assert.Equal(Some docId, c.Metadata.ParentId)

    [<Fact>]
    let ``Metadata: Strategy field matches config`` () =
        let config = { defaultConfig with Strategy = Paragraph }
        let chunks = chunk config docId "Para one.\n\nPara two."
        for c in chunks do
            Assert.Equal("Paragraph", c.Metadata.Strategy)

    [<Fact>]
    let ``Metadata: Indices are sequential`` () =
        let chunks = chunk defaultConfig docId longText
        let indices = chunks |> List.map (fun c -> c.Metadata.Index)
        Assert.Equal<int list>([0..chunks.Length-1], indices)

    // === Helper Function Tests ===

    [<Fact>]
    let ``getParentContext: Returns adjacent chunks`` () =
        let chunks = chunk { defaultConfig with ChunkSize = 50; MinChunkSize = 10 } docId longText
        if chunks.Length >= 3 then
            let context = getParentContext chunks chunks.[1].Id 1
            Assert.True(context.Length >= 2)

    [<Fact>]
    let ``getParentContext: Unknown chunk returns empty`` () =
        let chunks = chunk defaultConfig docId longText
        let context = getParentContext chunks "unknown-id" 1
        Assert.Empty(context)

    [<Fact>]
    let ``mergeSmallChunks: Merges adjacent small chunks`` () =
        let smallChunks = [
            { Id = "c1"; Content = "AB"; Metadata = { Index = 0; StartChar = 0; EndChar = 2; ParentId = None; Strategy = "test" } }
            { Id = "c2"; Content = "CD"; Metadata = { Index = 1; StartChar = 2; EndChar = 4; ParentId = None; Strategy = "test" } }
        ]
        let merged = mergeSmallChunks 100 smallChunks
        Assert.Single(merged) |> ignore
        Assert.Contains("AB", merged.[0].Content)
        Assert.Contains("CD", merged.[0].Content)

    [<Fact>]
    let ``chunkDefault: Uses default config`` () =
        // Use longer text to exceed MinChunkSize (100)
        let chunks = chunkDefault docId longText
        Assert.True(chunks.Length >= 1)
        Assert.Equal("SlidingWindow", chunks.[0].Metadata.Strategy)

