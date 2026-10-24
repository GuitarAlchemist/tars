namespace Tars.Tests

open Xunit
open Tars.Cortex.Chunking

module ChunkingTests =

    let sampleText =
        "This is sentence one. This is sentence two. This is sentence three."

    let longText = String.replicate 10 "This is a test paragraph with multiple words. "
    let docId = "test-doc"

    // === FixedSize Chunking ===

    [<Fact>]
    let ``FixedSize: Short text returns single chunk`` () =
        let config =
            { defaultConfig with
                ChunkSize = 500
                MinChunkSize = 10
                Strategy = FixedSize }

        let chunks = chunk config docId "Hello world"
        Assert.Single(chunks) |> ignore
        Assert.Equal("Hello world", chunks.[0].Content)

    [<Fact>]
    let ``FixedSize: Respects chunk size`` () =
        let config =
            { defaultConfig with
                ChunkSize = 50
                MinChunkSize = 10
                Strategy = FixedSize }

        let chunks = chunk config docId longText

        for c in chunks do
            Assert.True(c.Content.Length <= config.ChunkSize)

    [<Fact>]
    // Was "Filters out small chunks": it asserted the minimum was respected, which it
    // still is — but the small chunk is now folded into its neighbour rather than
    // deleted, so the name described a behaviour that was losing text.
    let ``FixedSize: No chunk comes back under the minimum`` () =
        let config =
            { defaultConfig with
                ChunkSize = 50
                MinChunkSize = 30
                Strategy = FixedSize }

        let chunks = chunk config docId longText

        for c in chunks do
            Assert.True(c.Content.Length >= config.MinChunkSize)

    // === SlidingWindow Chunking ===

    [<Fact>]
    let ``SlidingWindow: Has overlap`` () =
        let config =
            { defaultConfig with
                ChunkSize = 100
                ChunkOverlap = 30
                MinChunkSize = 10
                Strategy = SlidingWindow }

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
        let config =
            { defaultConfig with
                ChunkSize = 50
                MinChunkSize = 10
                Strategy = Sentence }

        let chunks = chunk config docId sampleText
        Assert.True(chunks.Length >= 1)

    [<Fact>]
    let ``Sentence: Preserves complete sentences`` () =
        let config =
            { defaultConfig with
                ChunkSize = 100
                MinChunkSize = 10
                Strategy = Sentence }

        let text = "First sentence. Second sentence. Third sentence."
        let chunks = chunk config docId text
        // All chunks should end with period or be the last chunk
        for c in chunks do
            Assert.True(c.Content.Contains(".") || c = List.last chunks)

    // === Paragraph Chunking ===

    [<Fact>]
    let ``Paragraph: Splits on double newlines`` () =
        let config =
            { defaultConfig with
                ChunkSize = 200
                MinChunkSize = 10
                Strategy = Paragraph }

        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        let chunks = chunk config docId text
        Assert.True(chunks.Length >= 1)

    [<Fact>]
    let ``Paragraph: Empty text returns empty list`` () =
        let config =
            { defaultConfig with
                Strategy = Paragraph
                MinChunkSize = 1 }

        let chunks = chunk config docId ""
        Assert.Empty(chunks)

    // === Recursive Chunking ===

    [<Fact>]
    let ``Recursive: Handles nested structure`` () =
        let config =
            { defaultConfig with
                ChunkSize = 100
                MinChunkSize = 10
                Strategy = Recursive }

        let text = "Para 1.\n\nPara 2. Sentence A. Sentence B.\n\nPara 3."
        let chunks = chunk config docId text
        Assert.True(chunks.Length >= 1)

    [<Fact>]
    let ``Recursive: Falls back to fixed size for long words`` () =
        let config =
            { defaultConfig with
                ChunkSize = 20
                MinChunkSize = 5
                Strategy = Recursive }

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
        let config =
            { defaultConfig with
                Strategy = Paragraph }

        let chunks = chunk config docId "Para one.\n\nPara two."

        for c in chunks do
            Assert.Equal("Paragraph", c.Metadata.Strategy)

    [<Fact>]
    let ``Metadata: Indices are sequential`` () =
        let chunks = chunk defaultConfig docId longText
        let indices = chunks |> List.map (fun c -> c.Metadata.Index)
        Assert.Equal<int list>([ 0 .. chunks.Length - 1 ], indices)

    // === Helper Function Tests ===

    [<Fact>]
    let ``getParentContext: Returns adjacent chunks`` () =
        let chunks =
            chunk
                { defaultConfig with
                    ChunkSize = 50
                    MinChunkSize = 10 }
                docId
                longText

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
        let smallChunks =
            [ { Id = "c1"
                Content = "AB"
                Metadata =
                  { Index = 0
                    StartChar = 0
                    EndChar = 2
                    ParentId = None
                    Strategy = "test" } }
              { Id = "c2"
                Content = "CD"
                Metadata =
                  { Index = 1
                    StartChar = 2
                    EndChar = 4
                    ParentId = None
                    Strategy = "test" } } ]

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


    // === Nothing is silently deleted ===
    //
    // Every strategy used to drop a piece shorter than MinChunkSize. That deleted
    // the tail of any document whose length did not divide evenly, and the whole of
    // any document under the minimum — which is 100 characters by default, so most
    // short documents chunked into nothing at all. The tests below hold the fix:
    // short text still produces a chunk, and every character of the source survives.

    /// Which characters of the source some chunk still covers, in order.
    ///
    /// Only meaningful for the strategies that cut by exact offsets; the accumulating
    /// ones keep approximate positions, so they are checked by content instead.
    let private coveredText (text: string) (chunks: Chunk list) =
        let covered = Array.zeroCreate<bool> text.Length

        for c in chunks do
            for i in c.Metadata.StartChar .. (min c.Metadata.EndChar text.Length) - 1 do
                covered.[i] <- true

        text.ToCharArray()
        |> Array.mapi (fun i ch -> i, ch)
        |> Array.filter (fun (i, _) -> covered.[i])
        |> Array.map (snd >> string)
        |> String.concat ""

    [<Fact>]
    let ``FixedSize: A document shorter than the minimum still produces a chunk`` () =
        let config =
            { defaultConfig with
                Strategy = FixedSize }

        let short = "Too short to matter."
        Assert.True(short.Length < config.MinChunkSize)

        let chunks = chunk config docId short

        Assert.Single(chunks) |> ignore
        Assert.Equal(short, chunks.[0].Content)

    [<Fact>]
    let ``SlidingWindow: A document shorter than the minimum still produces a chunk`` () =
        let chunks =
            chunk
                { defaultConfig with
                    Strategy = SlidingWindow }
                docId
                "Too short."

        Assert.Single(chunks) |> ignore
        Assert.Equal("Too short.", chunks.[0].Content)

    [<Fact>]
    let ``Sentence: A document shorter than the minimum still produces a chunk`` () =
        let chunks =
            chunk
                { defaultConfig with
                    Strategy = Sentence }
                docId
                "One sentence."

        Assert.Single(chunks) |> ignore
        Assert.Equal("One sentence.", chunks.[0].Content)

    [<Fact>]
    let ``Paragraph: A document shorter than the minimum still produces a chunk`` () =
        let chunks =
            chunk
                { defaultConfig with
                    Strategy = Paragraph }
                docId
                "One paragraph."

        Assert.Single(chunks) |> ignore
        Assert.Equal("One paragraph.", chunks.[0].Content)

    [<Fact>]
    let ``Recursive: A document shorter than the minimum still produces a chunk`` () =
        let chunks =
            chunk
                { defaultConfig with
                    Strategy = Recursive }
                docId
                "Short."

        Assert.Single(chunks) |> ignore
        Assert.Equal("Short.", chunks.[0].Content)

    [<Fact>]
    let ``FixedSize: The tail of an uneven document is not lost`` () =
        // 460 characters at a chunk size of 50 leaves a 10-character tail, which is
        // under the 30-character minimum. It used to disappear.
        let config =
            { defaultConfig with
                ChunkSize = 50
                MinChunkSize = 30
                Strategy = FixedSize }

        let chunks = chunk config docId longText

        Assert.Equal(longText.Length, chunks |> List.map (fun c -> c.Metadata.EndChar) |> List.max)
        Assert.Equal(longText, coveredText longText chunks)

    [<Fact>]
    let ``SlidingWindow: The tail of an uneven document is not lost`` () =
        let config =
            { defaultConfig with
                ChunkSize = 100
                ChunkOverlap = 30
                MinChunkSize = 60
                Strategy = SlidingWindow }

        let chunks = chunk config docId longText

        Assert.Equal(longText.Length, chunks |> List.map (fun c -> c.Metadata.EndChar) |> List.max)
        Assert.Equal(longText, coveredText longText chunks)

    [<Fact>]
    let ``Folding a short tail never produces a chunk under the minimum`` () =
        // Why the tail is folded into its neighbour rather than emitted on its own:
        // callers asked for a minimum, and they still get one.
        let config =
            { defaultConfig with
                ChunkSize = 50
                MinChunkSize = 30
                Strategy = FixedSize }

        let chunks = chunk config docId longText

        for c in chunks do
            Assert.True(
                c.Content.Length >= config.MinChunkSize,
                $"a chunk of {c.Content.Length} characters is under the minimum"
            )

    [<Fact>]
    let ``A minimum larger than the chunk size is read as no minimum`` () =
        // Taken literally it would reject every chunk and return nothing, which is
        // never what a caller meant by writing it.
        let config =
            { defaultConfig with
                ChunkSize = 20
                MinChunkSize = 500
                Strategy = FixedSize }

        let chunks = chunk config docId longText

        Assert.NotEmpty(chunks)
        Assert.Equal(longText, coveredText longText chunks)

    [<Fact>]
    let ``Empty text still produces nothing`` () =
        for strategy in [ FixedSize; SlidingWindow; Sentence; Paragraph; Recursive ] do
            Assert.Empty(
                chunk
                    { defaultConfig with
                        Strategy = strategy }
                    docId
                    ""
            )
