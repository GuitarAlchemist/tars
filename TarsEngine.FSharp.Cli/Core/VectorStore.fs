namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Collections.Generic
open System.Threading.Tasks
open System.Diagnostics
open Microsoft.Extensions.Logging
open Spectre.Console

/// Represents a document in the vector store
type Document = {
    Id: string
    Path: string
    Content: string
    Size: int64
    LastModified: DateTime
    FileType: string
    Embedding: float[] option
}

/// Ingestion metrics for tracking performance
type IngestionMetrics = {
    DirectoriesScanned: int
    FilesProcessed: int
    TotalSizeBytes: int64
    IngestionTimeMs: int64
    FilesPerSecond: float
    BytesPerSecond: float
    EmbeddingsGenerated: int
}

/// In-memory vector store for TARS codebase and content
type CodebaseVectorStore(logger: ILogger<CodebaseVectorStore>) =
    
    let documents = Dictionary<string, Document>()
    let mutable lastIngestionMetrics: IngestionMetrics option = None
    
    // File extensions to include in ingestion
    let includedExtensions = Set.ofList [
        // Code files
        ".fs"; ".fsx"; ".cs"; ".csx"; ".py"; ".js"; ".ts"; ".jsx"; ".tsx"
        ".java"; ".kt"; ".scala"; ".go"; ".rs"; ".cpp"; ".c"; ".h"; ".hpp"
        ".php"; ".rb"; ".swift"; ".dart"; ".lua"; ".r"; ".m"; ".mm"

        // Configuration and data files
        ".json"; ".yaml"; ".yml"; ".xml"; ".toml"; ".ini"; ".cfg"; ".conf"
        ".properties"; ".env"; ".editorconfig"; ".gitignore"; ".gitattributes"

        // Documentation and text files
        ".md"; ".txt"; ".rst"; ".adoc"; ".tex"; ".rtf"

        // Web files
        ".html"; ".htm"; ".css"; ".scss"; ".sass"; ".less"; ".vue"

        // Database and query files
        ".sql"; ".sqlite"; ".db"

        // Scripts and automation
        ".sh"; ".bash"; ".zsh"; ".ps1"; ".cmd"; ".bat"; ".makefile"

        // Project and build files
        ".proj"; ".csproj"; ".fsproj"; ".vbproj"; ".sln"; ".targets"; ".props"
        ".gradle"; ".maven"; ".pom"; ".build"; ".cmake"; ".dockerfile"

        // TARS specific files
        ".tars"; ".trsx"

        // Other important files
        ".lock"; ".log"; ".config"
    ]
    
    // Directories to exclude from ingestion
    let excludedDirectories = Set.ofList [
        "bin"; "obj"; "node_modules"; ".git"; ".vs"; ".vscode"
        "packages"; "target"; "dist"; "build"; "out"
    ]
    
    member private this.ShouldIncludeFile(filePath: string) =
        let extension = Path.GetExtension(filePath).ToLower()
        let fileName = Path.GetFileName(filePath).ToLower()
        let fileNameWithoutExt = Path.GetFileNameWithoutExtension(filePath).ToLower()

        // Include files with known extensions
        if includedExtensions.Contains(extension) then true
        // Include important files without extensions
        elif fileName = "readme" || fileName = "license" || fileName = "changelog" ||
             fileName = "makefile" || fileName = "dockerfile" || fileName = "jenkinsfile" ||
             fileName = "vagrantfile" || fileName = "gemfile" || fileName = "rakefile" ||
             fileName = "procfile" || fileName = "requirements" || fileName = "pipfile" ||
             fileName = "cargo" || fileName = "go.mod" || fileName = "go.sum" then true
        // Include files that start with important prefixes
        elif fileNameWithoutExt.StartsWith("readme") || fileNameWithoutExt.StartsWith("license") ||
             fileNameWithoutExt.StartsWith("changelog") || fileNameWithoutExt.StartsWith("contributing") ||
             fileNameWithoutExt.StartsWith("install") || fileNameWithoutExt.StartsWith("setup") then true
        else false
    
    member private this.ShouldIncludeDirectory(dirPath: string) =
        let dirName = Path.GetFileName(dirPath).ToLower()
        let userProfileTarsDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")

        // Exclude user profile .tars directory specifically, but allow repository .tars
        if dirPath.Equals(userProfileTarsDir, StringComparison.OrdinalIgnoreCase) then
            false
        else
            not (excludedDirectories.Contains(dirName))

    member private this.FindRepositoryRoot(startPath: string) =
        let rec findRoot (currentPath: string) =
            if String.IsNullOrEmpty(currentPath) || currentPath = Path.GetPathRoot(currentPath) then
                startPath // Fallback to start path if no repo root found
            else
                // Look for common repository indicators
                let gitDir = Path.Combine(currentPath, ".git")
                let slnFiles = Directory.GetFiles(currentPath, "*.sln")
                let readmeFiles = Directory.GetFiles(currentPath, "README*")

                if Directory.Exists(gitDir) || slnFiles.Length > 0 || readmeFiles.Length > 0 then
                    currentPath
                else
                    findRoot (Directory.GetParent(currentPath).FullName)

        let repoRoot = findRoot startPath
        logger.LogInformation($"Repository root detected: {repoRoot}")
        repoRoot
    
    member private this.GenerateSimpleEmbedding(content: string) =
        // Simple hash-based embedding for demonstration
        // In production, this would use a real embedding model
        let hash = content.GetHashCode()
        let embedding = Array.create 384 0.0 // Standard embedding dimension
        
        // Generate pseudo-embedding based on content characteristics
        let words = content.Split([|' '; '\n'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
        let wordCount = words.Length
        let charCount = content.Length
        
        // Fill embedding with content-based features
        embedding.[0] <- float wordCount / 1000.0
        embedding.[1] <- float charCount / 10000.0
        embedding.[2] <- float (hash % 1000) / 1000.0
        
        // Add more features based on content type
        if content.Contains("function") || content.Contains("def") then
            embedding.[3] <- 1.0 // Code indicator
        if content.Contains("class") || content.Contains("type") then
            embedding.[4] <- 1.0 // Type definition indicator
        if content.Contains("TODO") || content.Contains("FIXME") then
            embedding.[5] <- 1.0 // Task indicator
            
        embedding
    
    member private this.ProcessFile(filePath: string) =
        try
            let fileInfo = FileInfo(filePath)
            let content = File.ReadAllText(filePath)
            let embedding = this.GenerateSimpleEmbedding(content)
            
            let document = {
                Id = Guid.NewGuid().ToString()
                Path = filePath
                Content = content
                Size = fileInfo.Length
                LastModified = fileInfo.LastWriteTime
                FileType = Path.GetExtension(filePath)
                Embedding = Some embedding
            }
            
            documents.[filePath] <- document
            Some document
        with
        | ex ->
            logger.LogWarning($"Failed to process file {filePath}: {ex.Message}")
            None
    
    member private this.ScanDirectory(rootPath: string) =
        let rec scanRecursive (currentPath: string) (metrics: IngestionMetrics) =
            if this.ShouldIncludeDirectory(currentPath) then
                let mutable updatedMetrics = { metrics with DirectoriesScanned = metrics.DirectoriesScanned + 1 }
                
                // Process files in current directory
                try
                    let files = Directory.GetFiles(currentPath)
                    for file in files do
                        if this.ShouldIncludeFile(file) then
                            match this.ProcessFile(file) with
                            | Some doc ->
                                updatedMetrics <- {
                                    updatedMetrics with
                                        FilesProcessed = updatedMetrics.FilesProcessed + 1
                                        TotalSizeBytes = updatedMetrics.TotalSizeBytes + doc.Size
                                        EmbeddingsGenerated = updatedMetrics.EmbeddingsGenerated + 1
                                }
                            | None -> ()
                with
                | ex -> logger.LogWarning($"Failed to scan directory {currentPath}: {ex.Message}")
                
                // Process subdirectories
                try
                    let subdirs = Directory.GetDirectories(currentPath)
                    for subdir in subdirs do
                        updatedMetrics <- scanRecursive subdir updatedMetrics
                with
                | ex -> logger.LogWarning($"Failed to scan subdirectories of {currentPath}: {ex.Message}")
                
                updatedMetrics
            else
                metrics
        
        let initialMetrics = {
            DirectoriesScanned = 0
            FilesProcessed = 0
            TotalSizeBytes = 0L
            IngestionTimeMs = 0L
            FilesPerSecond = 0.0
            BytesPerSecond = 0.0
            EmbeddingsGenerated = 0
        }
        
        scanRecursive rootPath initialMetrics
    
    member this.IngestCodebase() =
        task {
            let stopwatch = Stopwatch.StartNew()

            AnsiConsole.MarkupLine("[bold cyan]🔄 Ingesting entire TARS repository into vector store...[/]")

            // Clear existing documents
            documents.Clear()

            // Get paths to ingest - only repository root and subdirectories
            let currentDir = Directory.GetCurrentDirectory()
            let repoRoot = this.FindRepositoryRoot(currentDir)

            let pathsToIngest = [
                (repoRoot, "TARS Repository")
            ]

            // Validate paths exist
            let validPaths =
                pathsToIngest
                |> List.filter (fun (path, _) ->
                    if Directory.Exists(path) then
                        true
                    else
                        AnsiConsole.MarkupLine($"[yellow]⚠️ Path not found: {path}[/]")
                        false)

            AnsiConsole.MarkupLine($"[dim]Note: Including repository .tars directory, excluding user profile .tars directory[/]")
            
            let mutable totalMetrics = {
                DirectoriesScanned = 0
                FilesProcessed = 0
                TotalSizeBytes = 0L
                IngestionTimeMs = 0L
                FilesPerSecond = 0.0
                BytesPerSecond = 0.0
                EmbeddingsGenerated = 0
            }
            
            // Process each valid path
            for (path, description) in validPaths do
                AnsiConsole.MarkupLine($"[yellow]📁 Scanning {description}: {path}[/]")
                let metrics = this.ScanDirectory(path)
                
                totalMetrics <- {
                    DirectoriesScanned = totalMetrics.DirectoriesScanned + metrics.DirectoriesScanned
                    FilesProcessed = totalMetrics.FilesProcessed + metrics.FilesProcessed
                    TotalSizeBytes = totalMetrics.TotalSizeBytes + metrics.TotalSizeBytes
                    IngestionTimeMs = totalMetrics.IngestionTimeMs
                    FilesPerSecond = totalMetrics.FilesPerSecond
                    BytesPerSecond = totalMetrics.BytesPerSecond
                    EmbeddingsGenerated = totalMetrics.EmbeddingsGenerated + metrics.EmbeddingsGenerated
                }
            
            stopwatch.Stop()
            
            // Calculate final metrics
            let finalMetrics = {
                totalMetrics with
                    IngestionTimeMs = stopwatch.ElapsedMilliseconds
                    FilesPerSecond = if stopwatch.ElapsedMilliseconds > 0L then float totalMetrics.FilesProcessed / (float stopwatch.ElapsedMilliseconds / 1000.0) else 0.0
                    BytesPerSecond = if stopwatch.ElapsedMilliseconds > 0L then float totalMetrics.TotalSizeBytes / (float stopwatch.ElapsedMilliseconds / 1000.0) else 0.0
            }
            
            lastIngestionMetrics <- Some finalMetrics
            
            // Display metrics
            this.ShowIngestionMetrics(finalMetrics)
            
            return finalMetrics
        }
    
    member private this.ShowIngestionMetrics(metrics: IngestionMetrics) =
        let metricsTable = Table()
        metricsTable.Border <- TableBorder.Rounded
        metricsTable.BorderStyle <- Style.Parse("green")
        
        metricsTable.AddColumn(TableColumn("[bold cyan]Metric[/]")) |> ignore
        metricsTable.AddColumn(TableColumn("[bold yellow]Value[/]").RightAligned()) |> ignore
        
        let sizeInMB = float metrics.TotalSizeBytes / (1024.0 * 1024.0)
        let timeInSeconds = float metrics.IngestionTimeMs / 1000.0
        
        let metricsData = [
            ("Directories Scanned", $"{metrics.DirectoriesScanned:N0}")
            ("Files Processed", $"{metrics.FilesProcessed:N0}")
            ("Total Size", $"{sizeInMB:F2} MB")
            ("Embeddings Generated", $"{metrics.EmbeddingsGenerated:N0}")
            ("Ingestion Time", $"{timeInSeconds:F2} seconds")
            ("Files/Second", $"{metrics.FilesPerSecond:F1}")
            ("MB/Second", $"{metrics.BytesPerSecond / (1024.0 * 1024.0):F2}")
        ]
        
        for (metric, value) in metricsData do
            metricsTable.AddRow(
                $"[cyan]{metric}[/]",
                $"[yellow]{value}[/]"
            ) |> ignore
        
        let metricsPanel = Panel(metricsTable)
        metricsPanel.Header <- PanelHeader("[bold green]📊 Ingestion Metrics[/]")
        metricsPanel.Border <- BoxBorder.Double
        AnsiConsole.Write(metricsPanel)
    
    member this.GetDocumentCount() = documents.Count
    
    member this.GetTotalSize() = 
        documents.Values 
        |> Seq.sumBy (fun doc -> doc.Size)
    
    member this.GetLastIngestionMetrics() = lastIngestionMetrics
    
    member this.SearchDocuments(query: string, maxResults: int) =
        // Simple text-based search for demonstration
        // In production, this would use vector similarity search
        documents.Values
        |> Seq.filter (fun doc ->
            doc.Content.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            doc.Path.Contains(query, StringComparison.OrdinalIgnoreCase))
        |> Seq.truncate maxResults
        |> Seq.toList

    member this.HybridSearch(query: string, maxResults: int) =
        // Hybrid search combining text search and semantic similarity
        let textResults = this.SearchDocuments(query, maxResults * 2)
        let queryEmbedding = this.GenerateSimpleEmbedding(query)

        // Score documents by combining text relevance and semantic similarity
        let scoredResults =
            textResults
            |> List.map (fun doc ->
                let textScore =
                    let contentMatches = doc.Content.Split([|' '; '\n'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
                                        |> Array.filter (fun word -> word.Contains(query, StringComparison.OrdinalIgnoreCase))
                                        |> Array.length
                    let pathScore = if doc.Path.Contains(query, StringComparison.OrdinalIgnoreCase) then 10 else 0
                    float (contentMatches + pathScore)

                let semanticScore =
                    match doc.Embedding with
                    | Some embedding -> this.CosineSimilarity(queryEmbedding, embedding)
                    | None -> 0.0

                let hybridScore = (textScore * 0.7) + (semanticScore * 0.3)
                (doc, hybridScore))
            |> List.sortByDescending snd
            |> List.truncate maxResults  // Use truncate instead of take to avoid bounds errors
            |> List.map fst

        scoredResults

    member private this.CosineSimilarity(a: float[], b: float[]) =
        if a.Length <> b.Length then 0.0
        else
            let dotProduct = Array.zip a b |> Array.sumBy (fun (x, y) -> x * y)
            let magnitudeA = sqrt (Array.sumBy (fun x -> x * x) a)
            let magnitudeB = sqrt (Array.sumBy (fun x -> x * x) b)
            if magnitudeA = 0.0 || magnitudeB = 0.0 then 0.0
            else dotProduct / (magnitudeA * magnitudeB)

    member this.SearchByFileType(fileType: string, query: string option, maxResults: int) =
        let typeFiltered = this.GetDocumentsByType(fileType)
        match query with
        | Some q ->
            typeFiltered
            |> List.filter (fun doc -> doc.Content.Contains(q, StringComparison.OrdinalIgnoreCase))
            |> List.truncate maxResults
        | None ->
            typeFiltered |> List.truncate maxResults

    member this.SearchByPath(pathPattern: string, maxResults: int) =
        documents.Values
        |> Seq.filter (fun doc -> doc.Path.Contains(pathPattern, StringComparison.OrdinalIgnoreCase))
        |> Seq.truncate maxResults
        |> Seq.toList
    
    member this.GetDocumentsByType(fileType: string) =
        documents.Values
        |> Seq.filter (fun doc -> doc.FileType.Equals(fileType, StringComparison.OrdinalIgnoreCase))
        |> Seq.toList
    
    member this.GetAllDocuments() =
        documents.Values |> Seq.toList
