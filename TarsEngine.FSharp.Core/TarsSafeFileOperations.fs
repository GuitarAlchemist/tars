namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Safe file operations with timeout support for TARS auto-improvement
/// Fixes the critical hanging I/O issue encountered in evolution experiments
module TarsSafeFileOperations =

    /// File operation result with comprehensive error information
    type FileOperationResult<'T> = {
        Success: bool
        Result: 'T option
        ErrorMessage: string option
        TimedOut: bool
        ExecutionTimeMs: int64
    }

    /// File backup information
    type FileBackup = {
        OriginalPath: string
        BackupPath: string
        BackupTime: DateTime
        FileSize: int64
        Checksum: string
    }

    /// Safe file operations service with timeout and error handling
    type TarsSafeFileOperationsService(logger: ILogger<TarsSafeFileOperationsService>) =

        let defaultTimeoutMs = 30000 // 30 seconds default timeout

        /// Calculate simple checksum for file integrity
        member private this.CalculateChecksum(content: string) : string =
            use sha256 = System.Security.Cryptography.SHA256.Create()
            let bytes = System.Text.Encoding.UTF8.GetBytes(content)
            let hash = sha256.ComputeHash(bytes)
            Convert.ToBase64String(hash)

        /// Read file with timeout and error handling
        member this.ReadFileWithTimeout(filePath: string, ?timeoutMs: int) : Async<FileOperationResult<string>> = async {
            let timeout = defaultArg timeoutMs defaultTimeoutMs
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            try
                use cts = new CancellationTokenSource(timeout)
                
                logger.LogDebug($"📖 Reading file: {Path.GetFileName(filePath)}")
                
                let! content = File.ReadAllTextAsync(filePath, cts.Token) |> Async.AwaitTask
                stopwatch.Stop()
                
                logger.LogDebug($"✅ File read successfully: {content.Length} characters in {stopwatch.ElapsedMilliseconds}ms")
                
                return {
                    Success = true
                    Result = Some content
                    ErrorMessage = None
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
            with
            | :? OperationCanceledException ->
                stopwatch.Stop()
                logger.LogWarning($"⏰ File read timeout: {filePath} (>{timeout}ms)")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some $"File read timeout after {timeout}ms"
                    TimedOut = true
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
            | ex ->
                stopwatch.Stop()
                logger.LogError(ex, $"❌ File read failed: {filePath}")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some ex.Message
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
        }

        /// Write file with timeout and backup
        member this.WriteFileWithBackup(filePath: string, content: string, ?timeoutMs: int) : Async<FileOperationResult<FileBackup option>> = async {
            let timeout = defaultArg timeoutMs defaultTimeoutMs
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            try
                use cts = new CancellationTokenSource(timeout)
                
                logger.LogDebug($"📝 Writing file: {Path.GetFileName(filePath)}")
                
                // Create backup if file exists
                let backup = 
                    if File.Exists(filePath) then
                        let backupPath = $"{filePath}.backup.{DateTime.UtcNow:yyyyMMdd_HHmmss}"
                        let originalContent = File.ReadAllText(filePath)
                        File.Copy(filePath, backupPath)
                        
                        Some {
                            OriginalPath = filePath
                            BackupPath = backupPath
                            BackupTime = DateTime.UtcNow
                            FileSize = FileInfo(filePath).Length
                            Checksum = this.CalculateChecksum(originalContent)
                        }
                    else
                        None

                // Write new content
                do! File.WriteAllTextAsync(filePath, content, cts.Token) |> Async.AwaitTask
                stopwatch.Stop()
                
                logger.LogInformation($"✅ File written successfully: {content.Length} characters in {stopwatch.ElapsedMilliseconds}ms")
                if backup.IsSome then
                    logger.LogDebug($"💾 Backup created: {backup.Value.BackupPath}")
                
                return {
                    Success = true
                    Result = Some backup
                    ErrorMessage = None
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
            with
            | :? OperationCanceledException ->
                stopwatch.Stop()
                logger.LogWarning($"⏰ File write timeout: {filePath} (>{timeout}ms)")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some $"File write timeout after {timeout}ms"
                    TimedOut = true
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
            | ex ->
                stopwatch.Stop()
                logger.LogError(ex, $"❌ File write failed: {filePath}")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some ex.Message
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
        }

        /// Read multiple files with timeout
        member this.ReadMultipleFiles(filePaths: string array, ?timeoutMs: int) : Async<FileOperationResult<Map<string, string>>> = async {
            let timeout = defaultArg timeoutMs (defaultTimeoutMs * 2) // Longer timeout for multiple files
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            try
                use cts = new CancellationTokenSource(timeout)
                
                logger.LogDebug($"📖 Reading {filePaths.Length} files...")
                
                let results = ResizeArray<string * string>()
                
                for filePath in filePaths do
                    if cts.Token.IsCancellationRequested then
                        raise (OperationCanceledException())
                    
                    try
                        let! content = File.ReadAllTextAsync(filePath, cts.Token) |> Async.AwaitTask
                        results.Add((filePath, content))
                    with
                    | ex ->
                        logger.LogWarning(ex, $"Failed to read file: {filePath}")
                        // Continue with other files
                
                stopwatch.Stop()
                
                let resultMap = results |> Seq.toArray |> Map.ofArray
                
                logger.LogInformation($"✅ Read {resultMap.Count}/{filePaths.Length} files in {stopwatch.ElapsedMilliseconds}ms")
                
                return {
                    Success = true
                    Result = Some resultMap
                    ErrorMessage = None
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
            with
            | :? OperationCanceledException ->
                stopwatch.Stop()
                logger.LogWarning($"⏰ Multiple file read timeout (>{timeout}ms)")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some $"Multiple file read timeout after {timeout}ms"
                    TimedOut = true
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
            | ex ->
                stopwatch.Stop()
                logger.LogError(ex, "❌ Multiple file read failed")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some ex.Message
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
        }

        /// Restore file from backup
        member this.RestoreFromBackup(backup: FileBackup) : Async<FileOperationResult<unit>> = async {
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            try
                logger.LogInformation($"🔄 Restoring file from backup: {backup.OriginalPath}")
                
                if File.Exists(backup.BackupPath) then
                    File.Copy(backup.BackupPath, backup.OriginalPath, true)
                    stopwatch.Stop()
                    
                    logger.LogInformation($"✅ File restored successfully in {stopwatch.ElapsedMilliseconds}ms")
                    
                    return {
                        Success = true
                        Result = Some ()
                        ErrorMessage = None
                        TimedOut = false
                        ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                    }
                else
                    stopwatch.Stop()
                    let error = $"Backup file not found: {backup.BackupPath}"
                    logger.LogError(error)
                    
                    return {
                        Success = false
                        Result = None
                        ErrorMessage = Some error
                        TimedOut = false
                        ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                    }
            with
            | ex ->
                stopwatch.Stop()
                logger.LogError(ex, $"❌ File restore failed: {backup.OriginalPath}")
                return {
                    Success = false
                    Result = None
                    ErrorMessage = Some ex.Message
                    TimedOut = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                }
        }

        /// Clean up old backup files
        member this.CleanupBackups(directory: string, olderThanDays: int) : Async<int> = async {
            try
                logger.LogDebug($"🧹 Cleaning up backups older than {olderThanDays} days in: {directory}")
                
                let cutoffDate = DateTime.UtcNow.AddDays(-float olderThanDays)
                let backupFiles = Directory.GetFiles(directory, "*.backup.*", SearchOption.AllDirectories)
                
                let mutable deletedCount = 0
                
                for backupFile in backupFiles do
                    try
                        let fileInfo = FileInfo(backupFile)
                        if fileInfo.CreationTimeUtc < cutoffDate then
                            File.Delete(backupFile)
                            deletedCount <- deletedCount + 1
                    with
                    | ex ->
                        logger.LogWarning(ex, $"Failed to delete backup file: {backupFile}")
                
                logger.LogInformation($"🧹 Cleaned up {deletedCount} old backup files")
                return deletedCount
            with
            | ex ->
                logger.LogError(ex, "Backup cleanup failed")
                return 0
        }

        /// Verify file integrity using checksum
        member this.VerifyFileIntegrity(filePath: string, expectedChecksum: string) : Async<bool> = async {
            try
                let! readResult = this.ReadFileWithTimeout(filePath)
                
                match readResult.Result with
                | Some content ->
                    let actualChecksum = this.CalculateChecksum(content)
                    let isValid = actualChecksum = expectedChecksum
                    
                    if isValid then
                        logger.LogDebug($"✅ File integrity verified: {Path.GetFileName(filePath)}")
                    else
                        logger.LogWarning($"❌ File integrity check failed: {Path.GetFileName(filePath)}")
                    
                    return isValid
                | None ->
                    logger.LogError($"❌ Cannot verify integrity - file read failed: {filePath}")
                    return false
            with
            | ex ->
                logger.LogError(ex, $"File integrity verification failed: {filePath}")
                return false
        }

        /// Safe directory operations
        member this.EnsureDirectoryExists(directoryPath: string) : bool =
            try
                if not (Directory.Exists(directoryPath)) then
                    Directory.CreateDirectory(directoryPath) |> ignore
                    logger.LogDebug($"📁 Created directory: {directoryPath}")
                true
            with
            | ex ->
                logger.LogError(ex, $"Failed to create directory: {directoryPath}")
                false

    /// Static helper functions for common file operations
    module SafeFileHelpers =
        
        /// Quick safe file read
        let safeReadFile (filePath: string) (logger: ILogger<_>) = async {
            let service = TarsSafeFileOperationsService(logger)
            let! result = service.ReadFileWithTimeout(filePath)
            return result.Result
        }

        /// Quick safe file write
        let safeWriteFile (filePath: string) (content: string) (logger: ILogger<_>) = async {
            let service = TarsSafeFileOperationsService(logger)
            let! result = service.WriteFileWithBackup(filePath, content)
            return result.Success
        }

        /// Check if file operations are working
        let testFileOperations (testDir: string) (logger: ILogger<_>) = async {
            try
                let service = TarsSafeFileOperationsService(logger)
                let testFile = Path.Combine(testDir, "test_file_ops.txt")
                let testContent = "TARS file operations test"

                // Test write
                let! writeResult = service.WriteFileWithBackup(testFile, testContent)
                if not writeResult.Success then
                    return false
                else
                    // Test read
                    let! readResult = service.ReadFileWithTimeout(testFile)
                    if not readResult.Success then
                        return false
                    else
                        // Cleanup
                        try File.Delete(testFile) with | _ -> ()
                        return true
            with
            | ex ->
                logger.LogError(ex, "File operations test failed")
                return false
        }
