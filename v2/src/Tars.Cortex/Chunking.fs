namespace Tars.Cortex

open System
open System.Text.RegularExpressions

/// <summary>
/// Document chunking strategies for RAG pipelines.
/// Provides multiple approaches to split documents into retrievable chunks.
/// </summary>
module Chunking =
    
    /// <summary>Chunk metadata</summary>
    type ChunkMetadata = {
        Index: int
        StartChar: int
        EndChar: int
        ParentId: string option
        Strategy: string
    }
    
    /// <summary>A document chunk with content and metadata</summary>
    type Chunk = {
        Id: string
        Content: string
        Metadata: ChunkMetadata
    }
    
    /// <summary>Chunking configuration</summary>
    type ChunkingConfig = {
        ChunkSize: int
        ChunkOverlap: int
        MinChunkSize: int
        Strategy: ChunkingStrategy
    }
    and ChunkingStrategy =
        | FixedSize
        | SlidingWindow
        | Sentence
        | Paragraph
        | Semantic
        | Recursive
    
    /// <summary>Default chunking configuration</summary>
    let defaultConfig = {
        ChunkSize = 512
        ChunkOverlap = 50
        MinChunkSize = 100
        Strategy = SlidingWindow
    }
    
    /// <summary>Split text into fixed-size chunks</summary>
    let fixedSizeChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let chunks = ResizeArray<Chunk>()
        // Clamp to avoid zero/negative sizes that can cause infinite loops
        let chunkSize = max 1 config.ChunkSize
        let mutable pos = 0
        let mutable idx = 0
        
        while pos < text.Length do
            let endPos = min (pos + chunkSize) text.Length
            let content = text.Substring(pos, endPos - pos)
            
            if content.Length >= config.MinChunkSize then
                chunks.Add({
                    Id = $"{docId}_chunk_{idx}"
                    Content = content
                    Metadata = {
                        Index = idx
                        StartChar = pos
                        EndChar = endPos
                        ParentId = Some docId
                        Strategy = "FixedSize"
                    }
                })
                idx <- idx + 1
            
            pos <- pos + chunkSize
        
        chunks |> Seq.toList
    
    /// <summary>Split text using sliding window with overlap</summary>
    let slidingWindowChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let chunks = ResizeArray<Chunk>()
        // Prevent overlap >= chunk size from creating a zero step (hang)
        let chunkSize = max 1 config.ChunkSize
        let overlap = min config.ChunkOverlap (chunkSize - 1)
        let step = max 1 (chunkSize - overlap)
        let mutable pos = 0
        let mutable idx = 0
        
        while pos < text.Length do
            let endPos = min (pos + chunkSize) text.Length
            let content = text.Substring(pos, endPos - pos)
            
            if content.Length >= config.MinChunkSize then
                chunks.Add({
                    Id = $"{docId}_chunk_{idx}"
                    Content = content
                    Metadata = {
                        Index = idx
                        StartChar = pos
                        EndChar = endPos
                        ParentId = Some docId
                        Strategy = "SlidingWindow"
                    }
                })
                idx <- idx + 1
            
            pos <- pos + step
            if pos >= text.Length then pos <- text.Length
        
        chunks |> Seq.toList
    
    /// <summary>Split text by sentences</summary>
    let sentenceChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        // Split on sentence boundaries
        let sentencePattern = @"(?<=[.!?])\s+"
        let sentences = Regex.Split(text, sentencePattern) |> Array.filter (fun s -> s.Trim().Length > 0)
        
        let chunks = ResizeArray<Chunk>()
        let mutable currentChunk = ""
        let mutable startPos = 0
        let mutable idx = 0
        let mutable charPos = 0
        
        for sentence in sentences do
            if currentChunk.Length + sentence.Length > config.ChunkSize && currentChunk.Length >= config.MinChunkSize then
                chunks.Add({
                    Id = $"{docId}_chunk_{idx}"
                    Content = currentChunk.Trim()
                    Metadata = {
                        Index = idx
                        StartChar = startPos
                        EndChar = charPos
                        ParentId = Some docId
                        Strategy = "Sentence"
                    }
                })
                idx <- idx + 1
                startPos <- charPos
                currentChunk <- sentence
            else
                currentChunk <- currentChunk + " " + sentence
            
            charPos <- charPos + sentence.Length + 1
        
        // Add remaining content
        if currentChunk.Trim().Length >= config.MinChunkSize then
            chunks.Add({
                Id = $"{docId}_chunk_{idx}"
                Content = currentChunk.Trim()
                Metadata = {
                    Index = idx
                    StartChar = startPos
                    EndChar = text.Length
                    ParentId = Some docId
                    Strategy = "Sentence"
                }
            })

        chunks |> Seq.toList

    /// <summary>Split text by paragraphs</summary>
    let paragraphChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let paragraphs = text.Split([|"\n\n"; "\r\n\r\n"|], StringSplitOptions.RemoveEmptyEntries)

        let chunks = ResizeArray<Chunk>()
        let mutable currentChunk = ""
        let mutable startPos = 0
        let mutable idx = 0
        let mutable charPos = 0

        for para in paragraphs do
            let trimmed = para.Trim()
            if currentChunk.Length + trimmed.Length > config.ChunkSize && currentChunk.Length >= config.MinChunkSize then
                chunks.Add({
                    Id = $"{docId}_chunk_{idx}"
                    Content = currentChunk.Trim()
                    Metadata = {
                        Index = idx
                        StartChar = startPos
                        EndChar = charPos
                        ParentId = Some docId
                        Strategy = "Paragraph"
                    }
                })
                idx <- idx + 1
                startPos <- charPos
                currentChunk <- trimmed
            else
                currentChunk <- if currentChunk.Length = 0 then trimmed else currentChunk + "\n\n" + trimmed

            charPos <- charPos + para.Length + 2

        if currentChunk.Trim().Length >= config.MinChunkSize then
            chunks.Add({
                Id = $"{docId}_chunk_{idx}"
                Content = currentChunk.Trim()
                Metadata = {
                    Index = idx
                    StartChar = startPos
                    EndChar = text.Length
                    ParentId = Some docId
                    Strategy = "Paragraph"
                }
            })

        chunks |> Seq.toList

    /// <summary>Recursive chunking - tries larger separators first, then smaller</summary>
    let recursiveChunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        let separators = [|"\n\n"; "\n"; ". "; " "|]

        let rec splitRecursive (content: string) (sepIdx: int) : string list =
            if content.Length <= config.ChunkSize then
                [content]
            elif sepIdx >= separators.Length then
                // Fall back to fixed size
                let mutable results = []
                let mutable pos = 0
                while pos < content.Length do
                    let endPos = min (pos + config.ChunkSize) content.Length
                    results <- results @ [content.Substring(pos, endPos - pos)]
                    pos <- endPos
                results
            else
                let sep = separators.[sepIdx]
                let parts = content.Split([|sep|], StringSplitOptions.RemoveEmptyEntries)
                if parts.Length <= 1 then
                    splitRecursive content (sepIdx + 1)
                else
                    parts
                    |> Array.toList
                    |> List.collect (fun p -> splitRecursive p (sepIdx + 1))

        let parts = splitRecursive text 0
        parts
        |> List.mapi (fun idx content ->
            // Track approximate offsets by accumulating lengths up to this chunk
            let startPos =
                parts
                |> List.take idx
                |> List.sumBy (fun c -> c.Length)

            let endPos = startPos + content.Length
            {
                Id = $"{docId}_chunk_{idx}"
                Content = content.Trim()
                Metadata = {
                    Index = idx
                    StartChar = startPos
                    EndChar = endPos
                    ParentId = Some docId
                    Strategy = "Recursive"
                }
            })
        |> List.filter (fun c -> c.Content.Length >= config.MinChunkSize)

    /// <summary>Main chunking function - dispatches to appropriate strategy</summary>
    let chunk (config: ChunkingConfig) (docId: string) (text: string) : Chunk list =
        match config.Strategy with
        | FixedSize -> fixedSizeChunk config docId text
        | SlidingWindow -> slidingWindowChunk config docId text
        | Sentence -> sentenceChunk config docId text
        | Paragraph -> paragraphChunk config docId text
        | Semantic -> sentenceChunk config docId text  // Fallback - true semantic needs embeddings
        | Recursive -> recursiveChunk config docId text

    /// <summary>Chunk with default config</summary>
    let chunkDefault (docId: string) (text: string) : Chunk list =
        chunk defaultConfig docId text

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
            | [x] -> List.rev (x :: acc)
            | x :: y :: rest when x.Content.Length + y.Content.Length < minSize * 2 ->
                let merged = {
                    Id = x.Id
                    Content = x.Content + "\n" + y.Content
                    Metadata = { x.Metadata with EndChar = y.Metadata.EndChar }
                }
                merge acc (merged :: rest)
            | x :: rest -> merge (x :: acc) rest

        merge [] chunks
