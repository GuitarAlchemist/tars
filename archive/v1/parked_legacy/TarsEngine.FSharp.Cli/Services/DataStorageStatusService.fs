namespace TarsEngine.FSharp.Cli.Services

open System
open System.IO
open System.Text
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core

/// Comprehensive data storage status and metrics service
type DataStorageStatusService(
    logger: ILogger<DataStorageStatusService>,
    vectorStore: CodebaseVectorStore,
    learningMemoryService: LearningMemoryService option,
    chatSessionService: ChatSessionService option) =
    
    /// Get comprehensive storage status across all TARS components
    member this.GetComprehensiveStorageStatus() =
        logger.LogInformation("📊 STORAGE STATUS: Gathering comprehensive data storage metrics")
        
        // Vector Store Metrics
        let vectorStoreMetrics = this.GetVectorStoreMetrics()
        
        // Learning Memory Metrics
        let learningMemoryMetrics = 
            match learningMemoryService with
            | Some service -> Some (service.GetMemoryStats())
            | None -> None
        
        // Session Storage Metrics
        let sessionMetrics = this.GetSessionStorageMetrics()
        
        // File System Metrics
        let fileSystemMetrics = this.GetFileSystemMetrics()
        
        // Total aggregated metrics
        let totalMetrics = this.CalculateTotalMetrics(vectorStoreMetrics, learningMemoryMetrics, sessionMetrics, fileSystemMetrics)
        
        {|
            Timestamp = DateTime.UtcNow
            VectorStore = vectorStoreMetrics
            LearningMemory = learningMemoryMetrics
            SessionStorage = sessionMetrics
            FileSystem = fileSystemMetrics
            TotalAggregated = totalMetrics
            IndexingCapabilities = this.GetIndexingCapabilities()
            StorageLocations = this.GetStorageLocations()
        |}
    
    /// Get vector store specific metrics
    member private this.GetVectorStoreMetrics() =
        try
            // Get vector store statistics
            let documentCount = vectorStore.GetDocumentCount()
            let indexSize = vectorStore.GetIndexSizeBytes()
            let vectorDimensions = vectorStore.GetVectorDimensions()
            
            {|
                Type = "CUDA Vector Store (Non-Euclidean)"
                DocumentCount = documentCount
                IndexSizeBytes = indexSize
                IndexSizeMB = float indexSize / (1024.0 * 1024.0)
                VectorDimensions = vectorDimensions
                EstimatedTokens = documentCount * 500 // Rough estimate
                IndexingType = "Semantic Vector Embeddings"
                SearchCapabilities = ["Hybrid Search"; "Semantic Similarity"; "Keyword Matching"]
                CudaAccelerated = true
                Status = "Active"
            |}
        with
        | ex ->
            logger.LogWarning(ex, "Failed to get vector store metrics")
            {|
                Type = "CUDA Vector Store (Non-Euclidean)"
                DocumentCount = 0
                IndexSizeBytes = 0L
                IndexSizeMB = 0.0
                VectorDimensions = 0
                EstimatedTokens = 0
                IndexingType = "Semantic Vector Embeddings"
                SearchCapabilities = ["Hybrid Search"; "Semantic Similarity"; "Keyword Matching"]
                CudaAccelerated = true
                Status = "Error"
            |}
    
    /// Get session storage metrics
    member private this.GetSessionStorageMetrics() =
        match chatSessionService with
        | Some service ->
            let activeSessions = service.GetActiveSessions()
            let totalSessions = activeSessions.Length
            
            let totalMessages = 
                activeSessions 
                |> List.sumBy (fun s -> s.Messages.Length)
            
            let totalMemoryItems = 
                activeSessions 
                |> List.sumBy (fun s -> s.Memory.Facts.Count + s.Memory.UserPreferences.Count + s.Memory.ContextVariables.Count)
            
            let estimatedContentSize = 
                activeSessions 
                |> List.sumBy (fun s -> 
                    s.Messages |> List.sumBy (fun m -> m.Content.Length))
            
            {|
                Type = "Session Memory (In-Memory)"
                ActiveSessions = totalSessions
                TotalMessages = totalMessages
                TotalMemoryItems = totalMemoryItems
                ContentSizeBytes = estimatedContentSize
                ContentSizeMB = float estimatedContentSize / (1024.0 * 1024.0)
                EstimatedTokens = estimatedContentSize / 4
                IndexingType = "Hash-based Session Lookup"
                Persistence = "Temporary (Session-based)"
                Status = "Active"
            |}
        | None ->
            {|
                Type = "Session Memory (In-Memory)"
                ActiveSessions = 0
                TotalMessages = 0
                TotalMemoryItems = 0
                ContentSizeBytes = 0
                ContentSizeMB = 0.0
                EstimatedTokens = 0
                IndexingType = "Hash-based Session Lookup"
                Persistence = "Temporary (Session-based)"
                Status = "Not Available"
            |}
    
    /// Get file system storage metrics
    member private this.GetFileSystemMetrics() =
        try
            let currentDir = Directory.GetCurrentDirectory()
            let tarsFiles = Directory.GetFiles(currentDir, "*.tars", SearchOption.AllDirectories)
            let mdFiles = Directory.GetFiles(currentDir, "TARS_Session_*.md", SearchOption.TopDirectoryOnly)
            let configFiles = Directory.GetFiles(currentDir, "*.yaml", SearchOption.AllDirectories)
            
            let totalFileSize = 
                (tarsFiles |> Array.append mdFiles |> Array.append configFiles)
                |> Array.sumBy (fun file -> 
                    try (FileInfo(file)).Length
                    with _ -> 0L)
            
            {|
                Type = "File System Storage"
                TarsMetascripts = tarsFiles.Length
                SessionTranscripts = mdFiles.Length
                ConfigurationFiles = configFiles.Length
                TotalFiles = tarsFiles.Length + mdFiles.Length + configFiles.Length
                TotalSizeBytes = totalFileSize
                TotalSizeMB = float totalFileSize / (1024.0 * 1024.0)
                IndexingType = "File System Directory Structure"
                Persistence = "Permanent (Disk-based)"
                Status = "Active"
            |}
        with
        | ex ->
            logger.LogWarning(ex, "Failed to get file system metrics")
            {|
                Type = "File System Storage"
                TarsMetascripts = 0
                SessionTranscripts = 0
                ConfigurationFiles = 0
                TotalFiles = 0
                TotalSizeBytes = 0L
                TotalSizeMB = 0.0
                IndexingType = "File System Directory Structure"
                Persistence = "Permanent (Disk-based)"
                Status = "Error"
            |}
    
    /// Calculate total aggregated metrics
    member private this.CalculateTotalMetrics(vectorStore, learningMemory, sessionStorage, fileSystem) =
        let totalSizeBytes = 
            vectorStore.IndexSizeBytes + 
            int64 sessionStorage.ContentSizeBytes + 
            fileSystem.TotalSizeBytes +
            (match learningMemory with 
             | Some lm -> int64 lm.StorageMetrics.TotalContentSizeBytes 
             | None -> 0L)
        
        let totalTokens = 
            vectorStore.EstimatedTokens + 
            sessionStorage.EstimatedTokens + 
            (match learningMemory with 
             | Some lm -> lm.StorageMetrics.EstimatedTokens 
             | None -> 0)
        
        let totalEntries = 
            vectorStore.DocumentCount + 
            sessionStorage.TotalMessages + 
            fileSystem.TotalFiles +
            (match learningMemory with 
             | Some lm -> lm.TotalKnowledge 
             | None -> 0)
        
        {|
            TotalSizeBytes = totalSizeBytes
            TotalSizeMB = float totalSizeBytes / (1024.0 * 1024.0)
            TotalEstimatedTokens = totalTokens
            TotalDataEntries = totalEntries
            StorageComponents = 4
            ActiveComponents = [
                if vectorStore.Status = "Active" then "Vector Store"
                if sessionStorage.Status = "Active" then "Session Memory"
                if fileSystem.Status = "Active" then "File System"
                match learningMemory with 
                | Some _ -> "Learning Memory" 
                | None -> ()
            ]
        |}
    
    /// Get indexing capabilities across all components
    member private this.GetIndexingCapabilities() =
        {|
            VectorStore = {|
                Type = "Non-Euclidean CUDA Vector Store"
                Capabilities = ["Semantic Search"; "Hybrid Search"; "Vector Similarity"; "CUDA Acceleration"]
                Dimensions = "High-dimensional embeddings"
                SearchComplexity = "O(log n) with CUDA optimization"
            |}
            LearningMemory = {|
                Type = "RDF Triple Store + In-Memory Cache"
                Capabilities = ["Tag-based Search"; "Confidence Filtering"; "Temporal Queries"; "Source Tracking"]
                Indexing = "Hash-based + RDF SPARQL"
                SearchComplexity = "O(1) cache lookup, O(log n) RDF queries"
            |}
            SessionMemory = {|
                Type = "In-Memory Hash Tables"
                Capabilities = ["Session Lookup"; "Message History"; "Memory Context"]
                Indexing = "Hash-based session IDs"
                SearchComplexity = "O(1) session lookup"
            |}
            FileSystem = {|
                Type = "Directory Structure + File Metadata"
                Capabilities = ["File Pattern Matching"; "Directory Traversal"; "Metadata Queries"]
                Indexing = "File system directory tree"
                SearchComplexity = "O(n) directory scan"
            |}
        |}
    
    /// Get storage locations and accessibility
    member private this.GetStorageLocations() =
        {|
            VectorStore = {|
                Location = "GPU Memory (CUDA)"
                Accessibility = "TarsKnowledgeService, EnhancedKnowledgeService"
                Persistence = "Memory-based (rebuilt on startup)"
                BackupStrategy = "Periodic index snapshots"
            |}
            LearningMemory = {|
                Location = "RDF Triple Store + RAM Cache"
                Accessibility = "LearningMemoryService, EnhancedKnowledgeService"
                Persistence = "Persistent (RDF database)"
                BackupStrategy = "RDF database backups"
            |}
            SessionMemory = {|
                Location = "Application Memory (RAM)"
                Accessibility = "ChatSessionService, Interactive Sessions"
                Persistence = "Session-scoped (lost on session end)"
                BackupStrategy = "Transcript saving to disk"
            |}
            FileSystem = {|
                Location = "Local Disk Storage"
                Accessibility = "All TARS components, File operations"
                Persistence = "Permanent (disk-based)"
                BackupStrategy = "File system backups"
            |}
        |}
    
    /// Generate detailed storage report
    member this.GenerateStorageReport() =
        let status = this.GetComprehensiveStorageStatus()
        let sb = StringBuilder()

        sb.AppendLine("# 📊 TARS Data Storage Status Report") |> ignore
        sb.AppendLine() |> ignore
        let timestampStr = status.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
        sb.AppendLine($"**Generated:** {timestampStr} UTC") |> ignore
        sb.AppendLine() |> ignore

        // Total Summary
        sb.AppendLine("## 🎯 Total Storage Summary") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine($"- **Total Data Size:** {status.TotalAggregated.TotalSizeMB:F2} MB") |> ignore
        sb.AppendLine($"- **Estimated Tokens:** {status.TotalAggregated.TotalEstimatedTokens:N0}") |> ignore
        sb.AppendLine($"- **Total Entries:** {status.TotalAggregated.TotalDataEntries:N0}") |> ignore
        let activeComponents = String.Join(", ", status.TotalAggregated.ActiveComponents)
        sb.AppendLine($"- **Active Components:** {activeComponents}") |> ignore
        sb.AppendLine() |> ignore

        sb.ToString()
