/// <summary>
/// Codebase RAG - Retrieval Augmented Generation over TARS Source Code
/// =====================================================================
/// Enables TARS to search and understand its own codebase for:
/// - Evolution tasks: Find relevant patterns and examples
/// - Refactoring: Discover similar code structures
/// - Self-improvement: Learn from existing implementations
/// </summary>
namespace Tars.Cortex

open System
open System.IO
open System.Threading.Tasks
open System.Collections.Concurrent
open Tars.Core
open Tars.Llm

/// Codebase RAG for source code retrieval
module CodebaseRAG =

    // =========================================================================
    // Types
    // =========================================================================

    /// A code chunk with metadata
    type CodeChunk = {
        Id: string
        FilePath: string
        ModuleName: string
        Content: string
        StartLine: int
        EndLine: int
        ChunkType: ChunkType
        Embedding: float32[] option
    }
    
    and ChunkType =
        | Function
        | Module
        | Type
        | Comment
        | File

    /// Search result with relevance score
    type SearchResult = {
        Chunk: CodeChunk
        Score: float
        Highlights: string list
    }

    /// Ingestion statistics
    type IngestionStats = {
        FilesProcessed: int
        ChunksCreated: int
        TotalLines: int
        Duration: TimeSpan
    }

    /// Configuration for the codebase RAG
    type CodebaseRAGConfig = {
        ChunkSize: int
        ChunkOverlap: int
        ExcludePatterns: string list
        IncludeExtensions: string list
        MaxFileSize: int64
    }

    let defaultConfig = {
        ChunkSize = 1500  // ~30 lines
        ChunkOverlap = 200
        ExcludePatterns = ["obj"; "bin"; ".git"; "packages"; "node_modules"]
        IncludeExtensions = [".fs"; ".fsx"; ".fsi"; ".md"; ".json"]
        MaxFileSize = 100_000L  // 100KB max
    }

    let private collectionName = "tars_codebase"

    // =========================================================================
    // Chunking
    // =========================================================================

    /// Extract module name from F# content
    let private extractModuleName (content: string) =
        let lines = content.Split([|'\n'|])
        lines
        |> Array.tryFind (fun l -> 
            l.TrimStart().StartsWith("module ") || 
            l.TrimStart().StartsWith("namespace "))
        |> Option.map (fun l -> l.Trim())
        |> Option.defaultValue "Unknown"

    /// Chunk a file into semantic pieces
    let private chunkFile (config: CodebaseRAGConfig) (filePath: string) : CodeChunk list =
        try
            let content = File.ReadAllText(filePath)
            let lines = content.Split([|'\n'|])
            let moduleName = extractModuleName content
            
            if int64 content.Length > config.MaxFileSize then
                // For large files, create a summary chunk only
                [{
                    Id = Guid.NewGuid().ToString("N").[..8]
                    FilePath = filePath
                    ModuleName = moduleName
                    Content = $"// File: %s{filePath} (%d{lines.Length} lines, truncated due to size)\n%s{content.Substring(0, min 2000 content.Length)}"
                    StartLine = 1
                    EndLine = lines.Length
                    ChunkType = File
                    Embedding = None
                }]
            else
                // Smart chunking based on F# constructs
                let mutable chunks: CodeChunk list = []
                let mutable currentChunk = ResizeArray<string>()
                let mutable chunkStart = 1
                let mutable inFunction = false
                let mutable inType = false
                
                for i, line in lines |> Array.indexed do
                    let trimmed = line.TrimStart()
                    
                    // Detect function/type boundaries for smarter chunking
                    let isNewConstruct = 
                        trimmed.StartsWith("let ") ||
                        trimmed.StartsWith("let rec ") ||
                        trimmed.StartsWith("type ") ||
                        trimmed.StartsWith("module ") ||
                        trimmed.StartsWith("/// <summary>")
                    
                    // If we hit a new construct and have accumulated enough, create a chunk
                    if isNewConstruct && currentChunk.Count > 10 then
                        let chunkContent = String.Join("\n", currentChunk)
                        if chunkContent.Trim().Length > 50 then
                            chunks <- chunks @ [{
                                Id = Guid.NewGuid().ToString("N").[..8]
                                FilePath = filePath
                                ModuleName = moduleName
                                Content = chunkContent
                                StartLine = chunkStart
                                EndLine = i
                                ChunkType = if inType then Type elif inFunction then Function else Module
                                Embedding = None
                            }]
                        currentChunk.Clear()
                        chunkStart <- i + 1
                    
                    currentChunk.Add(line)
                    
                    // Track construct type
                    if trimmed.StartsWith("type ") then inType <- true; inFunction <- false
                    elif trimmed.StartsWith("let ") || trimmed.StartsWith("let rec ") then inFunction <- true; inType <- false
                    
                    // Force chunk if too large
                    if currentChunk.Count >= config.ChunkSize / 50 then // ~50 chars per line
                        let chunkContent = String.Join("\n", currentChunk)
                        if chunkContent.Trim().Length > 50 then
                            chunks <- chunks @ [{
                                Id = Guid.NewGuid().ToString("N").[..8]
                                FilePath = filePath
                                ModuleName = moduleName
                                Content = chunkContent
                                StartLine = chunkStart
                                EndLine = i + 1
                                ChunkType = Module
                                Embedding = None
                            }]
                        currentChunk.Clear()
                        chunkStart <- i + 2
                
                // Final chunk
                if currentChunk.Count > 0 then
                    let chunkContent = String.Join("\n", currentChunk)
                    if chunkContent.Trim().Length > 50 then
                        chunks <- chunks @ [{
                            Id = Guid.NewGuid().ToString("N").[..8]
                            FilePath = filePath
                            ModuleName = moduleName
                            Content = chunkContent
                            StartLine = chunkStart
                            EndLine = lines.Length
                            ChunkType = Module
                            Embedding = None
                        }]
                
                chunks
        with ex ->
            printfn $"Warning: Failed to chunk %s{filePath}: %s{ex.Message}"
            []

    // =========================================================================
    // CodebaseIndex - In-Memory Index with Vector Store Backing
    // =========================================================================

    /// In-memory codebase index
    type CodebaseIndex(vectorStore: IVectorStore, llmService: ILlmService) =
        let chunks = ConcurrentDictionary<string, CodeChunk>()
        let mutable ingested = false
        
        /// Check if the index has been populated
        member _.IsIngested = ingested
        
        /// Ingest the codebase from a directory
        member this.IngestAsync(rootPath: string, ?config: CodebaseRAGConfig) =
            task {
                let cfg = config |> Option.defaultValue defaultConfig
                let startTime = DateTime.UtcNow
                let mutable filesProcessed = 0
                let mutable totalLines = 0
                
                if Directory.Exists(rootPath) then
                    let files = 
                        cfg.IncludeExtensions
                        |> List.collect (fun ext ->
                            Directory.GetFiles(rootPath, "*" + ext, SearchOption.AllDirectories)
                            |> Array.toList)
                        |> List.filter (fun f ->
                            not (cfg.ExcludePatterns |> List.exists (fun p -> f.Contains(p))))

                    printfn $"📁 Indexing %d{files.Length} files from %s{rootPath}..."

                    let allChunks = files |> List.collect (chunkFile cfg)
                    printfn $"🧩 Total chunks to embed: %d{allChunks.Length}"

                    // Process in batches of 10 to avoid overwhelming Ollama
                    let batches = allChunks |> List.chunkBySize 10
                    let mutable completed = 0
                    
                    for batch in batches do
                        let! batchResults = 
                            batch 
                            |> List.map (fun chunk -> 
                                task {
                                    try
                                        let! embedding = llmService.EmbedAsync(chunk.Content)
                                        if embedding.Length > 0 then
                                            return Some (chunk, embedding)
                                        else return None
                                    with _ -> return None
                                })
                            |> Task.WhenAll
                        
                        for result in batchResults do
                            match result with
                            | Some (chunk, embedding) ->
                                let enrichedChunk = { chunk with Embedding = Some embedding }
                                chunks.[chunk.Id] <- enrichedChunk
                                
                                let metadata = Map.ofList [
                                    "file", chunk.FilePath
                                    "lines", $"%d{chunk.StartLine}-%d{chunk.EndLine}"
                                ]
                                do! vectorStore.SaveAsync(collectionName, chunk.Id, embedding, metadata)
                                totalLines <- totalLines + (chunk.EndLine - chunk.StartLine + 1)
                            | None -> ()
                            
                        completed <- completed + batch.Length
                        if completed % 50 = 0 || completed = allChunks.Length then
                            let elapsed = DateTime.UtcNow - startTime
                            let speed = float completed / elapsed.TotalSeconds
                            let remaining = allChunks.Length - completed
                            let etaSeconds = if speed > 0.0 then float remaining / speed else 0.0
                            let eta = TimeSpan.FromSeconds(etaSeconds)

                            printfn $"   [Progress] Embedded %d{completed}/%d{allChunks.Length} chunks... (%.1f{speed} chunks/s, ETA: %02d{int eta.TotalHours}:%02d{eta.Minutes}:%02d{eta.Seconds})"

                    filesProcessed <- files.Length
                    ingested <- true
                    printfn $"✓ Ingested %d{chunks.Count} chunks from %d{filesProcessed} files"


                return {
                    FilesProcessed = filesProcessed
                    ChunksCreated = chunks.Count
                    TotalLines = totalLines
                    Duration = DateTime.UtcNow - startTime
                }
            }
        
        /// Quick ingest without embeddings (faster)
        member this.IngestQuickAsync(rootPath: string, ?config: CodebaseRAGConfig) =
            task {
                let cfg = config |> Option.defaultValue defaultConfig
                let startTime = DateTime.UtcNow
                let mutable filesProcessed = 0
                let mutable totalLines = 0
                
                if Directory.Exists(rootPath) then
                    let files = 
                        cfg.IncludeExtensions
                        |> List.collect (fun ext ->
                            Directory.GetFiles(rootPath, "*" + ext, SearchOption.AllDirectories)
                            |> Array.toList)
                        |> List.filter (fun f ->
                            not (cfg.ExcludePatterns |> List.exists (fun p -> f.Contains(p))))
                    
                    for file in files do
                        let fileChunks = chunkFile cfg file
                        filesProcessed <- filesProcessed + 1
                        
                        for chunk in fileChunks do
                            chunks.[chunk.Id] <- chunk
                            totalLines <- totalLines + (chunk.EndLine - chunk.StartLine + 1)
                    
                    ingested <- true
                
                return {
                    FilesProcessed = filesProcessed
                    ChunksCreated = chunks.Count
                    TotalLines = totalLines
                    Duration = DateTime.UtcNow - startTime
                }
            }
        
        /// Search with embedding (semantic search)
        member this.SearchAsync(query: string, topK: int) =
            task {
                if not ingested then
                    return []
                else
                    try
                        let! queryEmbedding = llmService.EmbedAsync(query)
                        if queryEmbedding.Length > 0 then
                            let! results = vectorStore.SearchAsync(collectionName, queryEmbedding, topK)
                            
                            return 
                                results
                                |> List.choose (fun (id, score, _meta) ->
                                    match chunks.TryGetValue(id) with
                                    | true, chunk -> 
                                        Some {
                                            Chunk = chunk
                                            Score = float score
                                            Highlights = []
                                        }
                                    | false, _ -> None)
                        else
                            // Fall back to keyword search
                            return this.SearchKeyword(query, topK)
                    with _ ->
                        // Fall back to keyword search on error
                        return this.SearchKeyword(query, topK)
            }

        
        /// Keyword-based search (fallback)
        member this.SearchKeyword(query: string, topK: int) : SearchResult list =
            let queryLower = query.ToLowerInvariant()
            let queryTerms = queryLower.Split([|' '; '.'; ':'; '('; ')'; '"'|], StringSplitOptions.RemoveEmptyEntries)
            
            chunks.Values
            |> Seq.map (fun chunk ->
                let contentLower = chunk.Content.ToLowerInvariant()
                let matchCount = 
                    queryTerms 
                    |> Array.sumBy (fun term -> 
                        if contentLower.Contains(term) then 1 else 0)
                let score = float matchCount / float (max 1 queryTerms.Length)
                {
                    Chunk = chunk
                    Score = score
                    Highlights = queryTerms |> Array.filter contentLower.Contains |> Array.toList
                })
            |> Seq.filter (fun r -> r.Score > 0.0)
            |> Seq.sortByDescending (fun r -> r.Score)
            |> Seq.truncate topK
            |> Seq.toList
        
        /// Get chunk by ID
        member _.GetChunk(id: string) =
            match chunks.TryGetValue(id) with
            | true, chunk -> Some chunk
            | false, _ -> None
        
        /// Get all chunks for a file
        member _.GetChunksForFile(filePath: string) =
            chunks.Values
            |> Seq.filter (fun c -> c.FilePath = filePath)
            |> Seq.sortBy (fun c -> c.StartLine)
            |> Seq.toList
        
        /// Get statistics
        member _.GetStats() =
            let totalChunks = chunks.Count
            let withEmbeddings = chunks.Values |> Seq.filter (fun c -> c.Embedding.IsSome) |> Seq.length
            let files = chunks.Values |> Seq.map (fun c -> c.FilePath) |> Seq.distinct |> Seq.length
            {| TotalChunks = totalChunks; WithEmbeddings = withEmbeddings; Files = files |}

    // =========================================================================
    // Context Building for Prompts
    // =========================================================================

    /// Build context from search results for prompts
    let buildContext (results: SearchResult list) (maxTokens: int) =
        let mutable context = ""
        let mutable tokenEstimate = 0
        
        for result in results do
            let entry = $"\n[File: %s{result.Chunk.FilePath}, Lines: %d{result.Chunk.StartLine}-%d{result.Chunk.EndLine}]\n```fsharp\n%s{result.Chunk.Content}\n```\n"

            let entryTokens = entry.Length / 4 // Rough token estimate
            if tokenEstimate + entryTokens < maxTokens then
                context <- context + entry
                tokenEstimate <- tokenEstimate + entryTokens
        
        context

    /// Build a summary of what code is relevant to a task
    let buildTaskContext (index: CodebaseIndex) (taskDescription: string) =
        task {
            let! results = index.SearchAsync(taskDescription, 5)
            
            if results.IsEmpty then
                return None
            else
                let summary = 
                    results
                    |> List.map (fun r -> 
                        sprintf "- %s (lines %d-%d): %s" 
                            (Path.GetFileName r.Chunk.FilePath)
                            r.Chunk.StartLine
                            r.Chunk.EndLine
                            (if r.Chunk.Content.Length > 100 
                             then r.Chunk.Content.Substring(0, 100).Replace("\n", " ") + "..."
                             else r.Chunk.Content.Replace("\n", " ")))
                    |> String.concat "\n"
                
                return Some $"\n[RELEVANT CODEBASE CONTEXT]\n%s{summary}\n"
        }
