namespace Tars.Cortex

open System
open System.Text.RegularExpressions
open System.Threading.Tasks
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax

/// <summary>
/// Document chunking strategies for RAG pipelines.
/// Provides multiple approaches to split documents into retrievable chunks.
/// </summary>
module Chunking =

    /// <summary>Chunk metadata</summary>
    type ChunkMetadata =
        { Index: int
          StartChar: int
          EndChar: int
          ParentId: string option
          Strategy: string }

    /// <summary>A document chunk with content and metadata</summary>
    type Chunk =
        { Id: string
          Content: string
          Metadata: ChunkMetadata }

    /// <summary>Chunking configuration</summary>
    type ChunkingConfig =
        { ChunkSize: int
          ChunkOverlap: int
          MinChunkSize: int
          Strategy: ChunkingStrategy }

    and ChunkingStrategy =
        | FixedSize
        | SlidingWindow
        | Sentence
        | Paragraph
        | Semantic
        | Recursive
        | Agentic
        | Ast

    /// <summary>Default chunking configuration</summary>
    let defaultConfig =
        { ChunkSize = 512
          ChunkOverlap = 50
          MinChunkSize = 100
          Strategy = SlidingWindow }

    /// <summary>Split text into fixed-size chunks</summary>
    let fixedSizeChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let chunks = ResizeArray<Chunk>()
        let chunkSize = max 1 config.ChunkSize
        let mutable pos = 0
        let mutable idx = 0

        while pos < text.Length do
            let endPos = min (pos + chunkSize) text.Length
            let content = text.Substring(pos, endPos - pos)

            if content.Length >= config.MinChunkSize then
                chunks.Add(
                    { Id = $"{docId}_chunk_{idx}"
                      Content = content
                      Metadata =
                        { Index = idx
                          StartChar = pos
                          EndChar = endPos
                          ParentId = Some docId
                          Strategy = "FixedSize" } }
                )

                idx <- idx + 1

            pos <- pos + chunkSize

        chunks |> Seq.toList

    /// <summary>Split text using sliding window with overlap</summary>
    let slidingWindowChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let chunks = ResizeArray<Chunk>()
        let chunkSize = max 1 config.ChunkSize
        let overlap = min config.ChunkOverlap (chunkSize - 1)
        let step = max 1 (chunkSize - overlap)
        let mutable pos = 0
        let mutable idx = 0

        while pos < text.Length do
            let endPos = min (pos + chunkSize) text.Length
            let content = text.Substring(pos, endPos - pos)

            if content.Length >= config.MinChunkSize then
                chunks.Add(
                    { Id = $"{docId}_chunk_{idx}"
                      Content = content
                      Metadata =
                        { Index = idx
                          StartChar = pos
                          EndChar = endPos
                          ParentId = Some docId
                          Strategy = "SlidingWindow" } }
                )

                idx <- idx + 1

            pos <- pos + step

            if pos >= text.Length then
                pos <- text.Length

        chunks |> Seq.toList

    /// <summary>Split text by sentences</summary>
    let sentenceChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let sentencePattern = @"(?<=[.!?])\s+"

        let sentences =
            Regex.Split(text, sentencePattern)
            |> Array.filter (fun s -> s.Trim().Length > 0)

        let chunks = ResizeArray<Chunk>()
        let mutable currentChunk = ""
        let mutable startPos = 0
        let mutable idx = 0
        let mutable charPos = 0

        for sentence in sentences do
            if
                currentChunk.Length + sentence.Length > config.ChunkSize
                && currentChunk.Length >= config.MinChunkSize
            then
                chunks.Add(
                    { Id = $"{docId}_chunk_{idx}"
                      Content = currentChunk.Trim()
                      Metadata =
                        { Index = idx
                          StartChar = startPos
                          EndChar = charPos
                          ParentId = Some docId
                          Strategy = "Sentence" } }
                )

                idx <- idx + 1
                startPos <- charPos
                currentChunk <- sentence
            else
                currentChunk <- currentChunk + " " + sentence

            charPos <- charPos + sentence.Length + 1

        if currentChunk.Trim().Length >= config.MinChunkSize then
            chunks.Add(
                { Id = $"{docId}_chunk_{idx}"
                  Content = currentChunk.Trim()
                  Metadata =
                    { Index = idx
                      StartChar = startPos
                      EndChar = text.Length
                      ParentId = Some docId
                      Strategy = "Sentence" } }
            )

        chunks |> Seq.toList

    /// <summary>Split text by paragraphs</summary>
    let paragraphChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let paragraphs =
            text.Split([| "\n\n"; "\r\n\r\n" |], StringSplitOptions.RemoveEmptyEntries)

        let chunks = ResizeArray<Chunk>()
        let mutable currentChunk = ""
        let mutable startPos = 0
        let mutable idx = 0
        let mutable charPos = 0

        for para in paragraphs do
            let trimmed = para.Trim()

            if
                currentChunk.Length + trimmed.Length > config.ChunkSize
                && currentChunk.Length >= config.MinChunkSize
            then
                chunks.Add(
                    { Id = $"{docId}_chunk_{idx}"
                      Content = currentChunk.Trim()
                      Metadata =
                        { Index = idx
                          StartChar = startPos
                          EndChar = charPos
                          ParentId = Some docId
                          Strategy = "Paragraph" } }
                )

                idx <- idx + 1
                startPos <- charPos
                currentChunk <- trimmed
            else
                currentChunk <-
                    if currentChunk.Length = 0 then
                        trimmed
                    else
                        currentChunk + "\n\n" + trimmed

            charPos <- charPos + para.Length + 2

        if currentChunk.Trim().Length >= config.MinChunkSize then
            chunks.Add(
                { Id = $"{docId}_chunk_{idx}"
                  Content = currentChunk.Trim()
                  Metadata =
                    { Index = idx
                      StartChar = startPos
                      EndChar = text.Length
                      ParentId = Some docId
                      Strategy = "Paragraph" } }
            )

        chunks |> Seq.toList

    /// <summary>Recursive chunking - tries larger separators first, then smaller</summary>
    let recursiveChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let separators = [| "\n\n"; "\n"; ". "; " " |]

        let rec splitRecursive (content: string) (sepIdx: int) : string list =
            if content.Length <= config.ChunkSize then
                [ content ]
            elif sepIdx >= separators.Length then
                let mutable results = []
                let mutable pos = 0

                while pos < content.Length do
                    let endPos = min (pos + config.ChunkSize) content.Length
                    results <- results @ [ content.Substring(pos, endPos - pos) ]
                    pos <- endPos

                results
            else
                let sep = separators.[sepIdx]
                let parts = content.Split([| sep |], StringSplitOptions.RemoveEmptyEntries)

                if parts.Length <= 1 then
                    splitRecursive content (sepIdx + 1)
                else
                    parts |> Array.toList |> List.collect (fun p -> splitRecursive p (sepIdx + 1))

        let parts = splitRecursive text 0

        parts
        |> List.mapi (fun idx content ->
            let startPos = parts |> List.take idx |> List.sumBy (fun c -> c.Length)
            let endPos = startPos + content.Length

            { Id = $"{docId}_chunk_{idx}"
              Content = content.Trim()
              Metadata =
                { Index = idx
                  StartChar = startPos
                  EndChar = endPos
                  ParentId = Some docId
                  Strategy = "Recursive" } })
        |> List.filter (fun c -> c.Content.Length >= config.MinChunkSize)

    // F# Compiler Service Checker
    let private checker = lazy (FSharpChecker.Create())

    /// <summary>
    /// AST-based chunking - parses F# code and chunks by top-level declarations.
    /// </summary>
    let astChunkAsync (config: ChunkingConfig) (docId: string) (text: string) : Task<Chunk list> =
        let sourceText = SourceText.ofString text
        let fileName = "source.fs"

        let options =
            { FSharpParsingOptions.Default with
                SourceFiles = [| fileName |] }

        let addChunk (chunks: ResizeArray<Chunk>) (range: Range) (strategy: string) =
            try
                let extracted = sourceText.GetSubTextFromRange(range).ToString()

                if extracted.Length >= config.MinChunkSize then
                    chunks.Add(
                        { Id = $"{docId}_chunk_{chunks.Count}"
                          Content = extracted.Trim()
                          Metadata =
                            { Index = chunks.Count
                              StartChar = 0
                              EndChar = 0
                              ParentId = Some docId
                              Strategy = strategy } }
                    )
            with _ ->
                ()

        let rec walkDecls (chunks: ResizeArray<Chunk>) (decls: SynModuleDecl list) =
            for decl in decls do
                match decl with
                | SynModuleDecl.NestedModule(_, _, subDecls, _, _, _) -> walkDecls chunks subDecls
                | SynModuleDecl.Let(_, _, range) -> addChunk chunks range "Ast_Let"
                | SynModuleDecl.Types(_, range) -> addChunk chunks range "Ast_Type"
                | SynModuleDecl.Exception(_, range) -> addChunk chunks range "Ast_Exception"
                | _ -> ()

        task {
            try
                let! parseRes = checker.Value.ParseFile(fileName, sourceText, options) |> Async.StartAsTask
                let chunks = ResizeArray<Chunk>()

                match parseRes.ParseTree with
                | ParsedInput.ImplFile(implFile) ->
                    for content in implFile.Contents do
                        match content with
                        | SynModuleOrNamespace(decls = decls) -> walkDecls chunks decls
                | _ -> ()

                if chunks.Count = 0 then
                    return recursiveChunk config docId text
                else
                    return chunks |> Seq.toList

            with ex ->
                return recursiveChunk config docId text
        }

    /// <summary>
    /// Semantic chunking - splits based on embedding similarity between sentences.
    /// </summary>
    let semanticChunkAsync
        (config: ChunkingConfig)
        (docId: string)
        (text: string)
        (embedder: string -> Task<float32[]>)
        : Task<Chunk list> =
        task {
            let sentencePattern = @"(?<=[.!?])\s+"

            let sentences =
                Regex.Split(text, sentencePattern)
                |> Array.filter (fun s -> s.Trim().Length > 0)

            if sentences.Length = 0 then
                return []
            else
                let! embeddings = sentences |> Seq.map (fun s -> embedder s) |> Task.WhenAll
                let chunks = ResizeArray<Chunk>()
                let mutable currentChunkSentences = ResizeArray<string>()
                let mutable currentChunkStart = 0
                let mutable currentChunkIdx = 0
                let mutable lastEmbedding = embeddings.[0]
                let similarityThreshold = 0.7f

                for i in 0 .. sentences.Length - 1 do
                    let sentence = sentences.[i]
                    let embedding = embeddings.[i]

                    let similarity =
                        if currentChunkSentences.Count > 0 then
                            Similarity.cosineSimilarity lastEmbedding embedding
                        else
                            1.0f

                    let currentLength = currentChunkSentences |> Seq.sumBy (fun s -> s.Length)

                    let shouldSplit =
                        (similarity < similarityThreshold && currentLength >= config.MinChunkSize)
                        || (currentLength + sentence.Length > config.ChunkSize)

                    if shouldSplit && currentChunkSentences.Count > 0 then
                        let content = String.Join(" ", currentChunkSentences)

                        chunks.Add(
                            { Id = $"{docId}_chunk_{currentChunkIdx}"
                              Content = content
                              Metadata =
                                { Index = currentChunkIdx
                                  StartChar = currentChunkStart
                                  EndChar = currentChunkStart + content.Length
                                  ParentId = Some docId
                                  Strategy = "Semantic" } }
                        )

                        currentChunkIdx <- currentChunkIdx + 1
                        currentChunkStart <- currentChunkStart + content.Length + 1
                        currentChunkSentences.Clear()
                        currentChunkSentences.Add(sentence)
                    else
                        currentChunkSentences.Add(sentence)

                    lastEmbedding <- embedding

                if currentChunkSentences.Count > 0 then
                    let content = String.Join(" ", currentChunkSentences)

                    chunks.Add(
                        { Id = $"{docId}_chunk_{currentChunkIdx}"
                          Content = content
                          Metadata =
                            { Index = currentChunkIdx
                              StartChar = currentChunkStart
                              EndChar = text.Length
                              ParentId = Some docId
                              Strategy = "Semantic" } }
                    )

                return chunks |> Seq.toList
        }

    /// <summary>
    /// Agentic chunking - uses an LLM to identify logical sections.
    /// </summary>
    let agenticChunkAsync
        (config: ChunkingConfig)
        (docId: string)
        (text: string)
        (completer: string -> Task<string>)
        : Task<Chunk list> =
        task {
            let prompt =
                sprintf
                    """Analyze the following text and split it into logical sections. 
Return the sections in the following format:
### SECTION: [Title]
[Content]
### END SECTION

Text to split:
%s
"""
                    text

            let! response = completer prompt
            let chunks = ResizeArray<Chunk>()
            let pattern = @"### SECTION: (.*?)\r?\n(.*?)### END SECTION"
            let matches = Regex.Matches(response, pattern, RegexOptions.Singleline)
            let mutable idx = 0
            let mutable startPos = 0

            for m in matches do
                let title = m.Groups.[1].Value.Trim()
                let content = m.Groups.[2].Value.Trim()

                chunks.Add(
                    { Id = $"{docId}_chunk_{idx}"
                      Content = content
                      Metadata =
                        { Index = idx
                          StartChar = startPos
                          EndChar = startPos + content.Length
                          ParentId = Some docId
                          Strategy = "Agentic" } }
                )

                idx <- idx + 1
                startPos <- startPos + content.Length

            return chunks |> Seq.toList
        }

    /// <summary>Main chunking function - dispatches to appropriate strategy</summary>
    let chunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        match config.Strategy with
        | FixedSize -> fixedSizeChunk config docId text
        | SlidingWindow -> slidingWindowChunk config docId text
        | Sentence -> sentenceChunk config docId text
        | Paragraph -> paragraphChunk config docId text
        | Semantic -> sentenceChunk config docId text
        | Recursive -> recursiveChunk config docId text
        | Agentic -> sentenceChunk config docId text
        | Ast -> recursiveChunk config docId text

    /// <summary>Main chunking function (Async) - dispatches to appropriate strategy</summary>
    let chunkAsync
        (config: ChunkingConfig)
        (docId: string)
        (text: string)
        (embedder: (string -> Task<float32[]>) option)
        (completer: (string -> Task<string>) option)
        : Task<Chunk list> =
        task {
            match config.Strategy with
            | FixedSize -> return fixedSizeChunk config docId text
            | SlidingWindow -> return slidingWindowChunk config docId text
            | Sentence -> return sentenceChunk config docId text
            | Paragraph -> return paragraphChunk config docId text
            | Semantic ->
                match embedder with
                | Some e -> return! semanticChunkAsync config docId text e
                | None -> return sentenceChunk config docId text
            | Recursive -> return recursiveChunk config docId text
            | Agentic ->
                match completer with
                | Some c -> return! agenticChunkAsync config docId text c
                | None -> return sentenceChunk config docId text
            | Ast -> return! astChunkAsync config docId text
        }

    /// <summary>Chunk with default config</summary>
    let chunkDefault (docId: string) (text: string) : Chunk list = chunk defaultConfig docId text

    /// <summary>Get parent document context for a chunk</summary>
    let getParentContext (chunks: Chunk list) (chunkId: string) (windowSize: int) : Chunk list =
        match chunks |> List.tryFindIndex (fun c -> c.Id = chunkId) with
        | Some idx ->
            let startIdx = max 0 (idx - windowSize)
            let endIdx = min (chunks.Length - 1) (idx + windowSize)
            chunks.[startIdx..endIdx]
        | None -> []

    /// <summary>Merge adjacent chunks if they're small</summary>
    let mergeSmallChunks (minSize: int) (chunks: Chunk list) : Chunk list =
        let rec merge acc remaining =
            match remaining with
            | [] -> List.rev acc
            | [ x ] -> List.rev (x :: acc)
            | x :: y :: rest when x.Content.Length + y.Content.Length < minSize * 2 ->
                let merged =
                    { Id = x.Id
                      Content = x.Content + "\n" + y.Content
                      Metadata =
                        { x.Metadata with
                            EndChar = y.Metadata.EndChar } }

                merge acc (merged :: rest)
            | x :: rest -> merge (x :: acc) rest

        merge [] chunks
