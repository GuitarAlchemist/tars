namespace TarsEngine.FSharp.Cli.Services

open System
open System.IO
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Services.ChromaDB

/// Codebase vector store service for TARS
module CodebaseVectorStore =
    
    type CodeFile = {
        Path: string
        Content: string
        Language: string
        LastModified: DateTime
        Size: int64
    }
    
    type CodeChunk = {
        Id: string
        FilePath: string
        Content: string
        StartLine: int
        EndLine: int
        ChunkType: string // "function", "class", "module", etc.
        Language: string
    }
    
    type CodebaseIndex = {
        TotalFiles: int
        TotalChunks: int
        Languages: string list
        LastIndexed: DateTime
    }
    
    let private config = ChromaDB.defaultConfig
    
    /// Extract code chunks from a file
    let extractCodeChunks (codeFile: CodeFile) =
        let lines = codeFile.Content.Split('\n')
        let chunks = ResizeArray<CodeChunk>()
        
        // Simple chunking strategy - split by functions/classes
        let mutable currentChunk = []
        let mutable startLine = 1
        let mutable chunkId = 0
        
        for i, line in lines |> Array.indexed do
            currentChunk <- line :: currentChunk
            
            // Simple heuristic for chunk boundaries
            if line.Trim().StartsWith("let ") || 
               line.Trim().StartsWith("type ") || 
               line.Trim().StartsWith("module ") ||
               line.Trim().StartsWith("namespace ") ||
               (currentChunk.Length > 50) then
                
                if currentChunk.Length > 1 then
                    let chunk = {
                        Id = $"{codeFile.Path}_{chunkId}"
                        FilePath = codeFile.Path
                        Content = String.Join("\n", currentChunk |> List.rev)
                        StartLine = startLine
                        EndLine = i + 1
                        ChunkType = "code_block"
                        Language = codeFile.Language
                    }
                    chunks.Add(chunk)
                    chunkId <- chunkId + 1
                
                currentChunk <- []
                startLine <- i + 2
        
        // Add remaining chunk
        if currentChunk.Length > 0 then
            let chunk = {
                Id = $"{codeFile.Path}_{chunkId}"
                FilePath = codeFile.Path
                Content = String.Join("\n", currentChunk |> List.rev)
                StartLine = startLine
                EndLine = lines.Length
                ChunkType = "code_block"
                Language = codeFile.Language
            }
            chunks.Add(chunk)
        
        chunks |> Seq.toList
    
    /// Index a single file
    let indexFileAsync (filePath: string) =
        task {
            try
                if File.Exists(filePath) then
                    let content = File.ReadAllText(filePath)
                    let fileInfo = FileInfo(filePath)
                    let language = Path.GetExtension(filePath).TrimStart('.')
                    
                    let codeFile = {
                        Path = filePath
                        Content = content
                        Language = language
                        LastModified = fileInfo.LastWriteTime
                        Size = fileInfo.Length
                    }
                    
                    let chunks = extractCodeChunks codeFile
                    
                    // Add chunks to vector store
                    for chunk in chunks do
                        let document = {
                            Id = chunk.Id
                            Content = chunk.Content
                            Metadata = Map.ofList [
                                ("file_path", chunk.FilePath :> obj)
                                ("language", chunk.Language :> obj)
                                ("chunk_type", chunk.ChunkType :> obj)
                                ("start_line", chunk.StartLine :> obj)
                                ("end_line", chunk.EndLine :> obj)
                            ]
                            Embedding = None
                        }
                        
                        let! result = ChromaDB.addDocumentAsync config document
                        match result with
                        | Ok _ -> ()
                        | Error err -> printfn "Error indexing chunk %s: %s" chunk.Id err
                    
                    return Ok chunks.Length
                else
                    return Error $"File not found: {filePath}"
            with
            | ex -> return Error ex.Message
        }
    
    /// Index entire codebase directory
    let indexCodebaseAsync (rootPath: string) =
        task {
            try
                let codeExtensions = [".fs"; ".fsx"; ".cs"; ".py"; ".js"; ".ts"; ".md"]
                let files = Directory.GetFiles(rootPath, "*", SearchOption.AllDirectories)
                            |> Array.filter (fun f -> codeExtensions |> List.exists (fun ext -> f.EndsWith(ext)))
                
                let mutable totalChunks = 0
                let languages = ResizeArray<string>()
                
                for file in files do
                    let! result = indexFileAsync file
                    match result with
                    | Ok chunkCount -> 
                        totalChunks <- totalChunks + chunkCount
                        let lang = Path.GetExtension(file).TrimStart('.')
                        if not (languages.Contains(lang)) then
                            languages.Add(lang)
                    | Error err -> printfn "Error indexing %s: %s" file err
                
                let index = {
                    TotalFiles = files.Length
                    TotalChunks = totalChunks
                    Languages = languages |> Seq.toList
                    LastIndexed = DateTime.UtcNow
                }
                
                return Ok index
            with
            | ex -> return Error ex.Message
        }
    
    /// Search codebase for similar code
    let searchCodeAsync (query: string) (limit: int) =
        task {
            let! result = ChromaDB.searchSimilarAsync config query limit
            return result
        }
    
    /// Get codebase statistics
    let getStatsAsync () =
        task {
            let! result = ChromaDB.getStatsAsync config
            return result
        }
