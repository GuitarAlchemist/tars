namespace TarsEngine.FSharp.Cli.Services

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Service for performing file system operations in metascripts.
/// </summary>
type FileOperationsService(logger: ILogger<FileOperationsService>) =
    
    /// <summary>
    /// Scan directory for files matching patterns.
    /// </summary>
    member this.ScanDirectory(rootPath: string, patterns: string[]) =
        try
            logger.LogInformation("📁 TARS: Scanning directory: {Path}", rootPath)
            
            if not (Directory.Exists(rootPath)) then
                logger.LogWarning("Directory does not exist: {Path}", rootPath)
                Map.empty
            else
                let results = Dictionary<string, string[]>()
                
                for pattern in patterns do
                    try
                        let files = Directory.GetFiles(rootPath, pattern, SearchOption.AllDirectories)
                        results.[pattern] <- files
                        logger.LogInformation("Found {Count} files matching pattern '{Pattern}'", files.Length, pattern)
                    with
                    | ex -> 
                        logger.LogWarning(ex, "Error scanning for pattern '{Pattern}': {Error}", pattern, ex.Message)
                        results.[pattern] <- [||]
                
                logger.LogInformation("✅ TARS: Directory scan completed")
                results |> Seq.map (|KeyValue|) |> Map.ofSeq
        with
        | ex ->
            logger.LogError(ex, "❌ TARS: Directory scan failed: {Error}", ex.Message)
            Map.empty
    
    /// <summary>
    /// Count lines of code in files.
    /// </summary>
    member this.CountLinesOfCode(filePaths: string[]) =
        try
            logger.LogInformation("📊 TARS: Counting lines of code in {Count} files", filePaths.Length)
            
            let mutable totalLines = 0
            let mutable processedFiles = 0
            
            for filePath in filePaths do
                try
                    if File.Exists(filePath) then
                        let lines = File.ReadAllLines(filePath)
                        totalLines <- totalLines + lines.Length
                        processedFiles <- processedFiles + 1
                with
                | ex -> 
                    logger.LogWarning(ex, "Error reading file '{File}': {Error}", filePath, ex.Message)
            
            logger.LogInformation("✅ TARS: Counted {TotalLines} lines in {ProcessedFiles} files", totalLines, processedFiles)
            (totalLines, processedFiles)
        with
        | ex ->
            logger.LogError(ex, "❌ TARS: Line counting failed: {Error}", ex.Message)
            (0, 0)
    
    /// <summary>
    /// Create directory if it doesn't exist.
    /// </summary>
    member this.EnsureDirectoryExists(directoryPath: string) =
        try
            if not (Directory.Exists(directoryPath)) then
                Directory.CreateDirectory(directoryPath) |> ignore
                logger.LogInformation("📁 TARS: Created directory: {Path}", directoryPath)
            true
        with
        | ex ->
            logger.LogError(ex, "❌ TARS: Failed to create directory '{Path}': {Error}", directoryPath, ex.Message)
            false
    
    /// <summary>
    /// Write content to file.
    /// </summary>
    member this.WriteFileAsync(filePath: string, content: string) =
        task {
            try
                // Ensure directory exists
                let directory = Path.GetDirectoryName(filePath)
                if not (String.IsNullOrEmpty(directory)) then
                    this.EnsureDirectoryExists(directory) |> ignore
                
                // Write file
                do! File.WriteAllTextAsync(filePath, content, Encoding.UTF8)
                logger.LogInformation("📄 TARS: File written successfully: {Path}", filePath)
                return Ok ()
            with
            | ex ->
                let error = sprintf "Failed to write file '%s': %s" filePath ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Read file content.
    /// </summary>
    member this.ReadFileAsync(filePath: string) =
        task {
            try
                if not (File.Exists(filePath)) then
                    return Error (sprintf "File does not exist: %s" filePath)
                else
                    let! content = File.ReadAllTextAsync(filePath, Encoding.UTF8)
                    logger.LogInformation("📄 TARS: File read successfully: {Path}", filePath)
                    return Ok content
            with
            | ex ->
                let error = sprintf "Failed to read file '%s': %s" filePath ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Get file information.
    /// </summary>
    member this.GetFileInfo(filePath: string) =
        try
            if File.Exists(filePath) then
                let fileInfo = FileInfo(filePath)
                Some {|
                    Path = filePath
                    Size = fileInfo.Length
                    CreatedAt = fileInfo.CreationTime
                    ModifiedAt = fileInfo.LastWriteTime
                    Extension = fileInfo.Extension
                |}
            else
                None
        with
        | ex ->
            logger.LogWarning(ex, "Error getting file info for '{Path}': {Error}", filePath, ex.Message)
            None
    
    /// <summary>
    /// Copy file to destination.
    /// </summary>
    member this.CopyFileAsync(sourcePath: string, destinationPath: string, overwrite: bool) =
        task {
            try
                if not (File.Exists(sourcePath)) then
                    return Error (sprintf "Source file does not exist: %s" sourcePath)
                else
                    // Ensure destination directory exists
                    let directory = Path.GetDirectoryName(destinationPath)
                    if not (String.IsNullOrEmpty(directory)) then
                        this.EnsureDirectoryExists(directory) |> ignore
                    
                    File.Copy(sourcePath, destinationPath, overwrite)
                    logger.LogInformation("📄 TARS: File copied: {Source} -> {Destination}", sourcePath, destinationPath)
                    return Ok ()
            with
            | ex ->
                let error = sprintf "Failed to copy file '%s' to '%s': %s" sourcePath destinationPath ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Delete file if it exists.
    /// </summary>
    member this.DeleteFile(filePath: string) =
        try
            if File.Exists(filePath) then
                File.Delete(filePath)
                logger.LogInformation("🗑️ TARS: File deleted: {Path}", filePath)
            true
        with
        | ex ->
            logger.LogError(ex, "❌ TARS: Failed to delete file '{Path}': {Error}", filePath, ex.Message)
            false
