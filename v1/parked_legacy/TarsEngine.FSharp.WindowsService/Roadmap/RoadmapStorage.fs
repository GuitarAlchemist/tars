namespace TarsEngine.FSharp.WindowsService.Roadmap

open System
open System.IO
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open YamlDotNet.Serialization
open Newtonsoft.Json

/// <summary>
/// Roadmap storage configuration
/// </summary>
type RoadmapStorageConfig = {
    RoadmapDirectory: string
    BackupDirectory: string
    MaxBackups: int
    AutoBackupInterval: TimeSpan
    EnableVersioning: bool
    EnableCompression: bool
}

/// <summary>
/// Roadmap storage events
/// </summary>
type RoadmapStorageEvent =
    | RoadmapCreated of TarsRoadmap
    | RoadmapUpdated of TarsRoadmap * TarsRoadmap // old, new
    | RoadmapDeleted of string
    | AchievementUpdated of Achievement * AchievementUpdate
    | BackupCreated of string
    | StorageError of string * Exception

/// <summary>
/// Roadmap storage system for persistent roadmap management
/// </summary>
type RoadmapStorage(config: RoadmapStorageConfig, logger: ILogger<RoadmapStorage>) =
    
    let roadmaps = ConcurrentDictionary<string, TarsRoadmap>()
    let achievementUpdates = ConcurrentQueue<AchievementUpdate>()
    let storageEvents = ConcurrentQueue<RoadmapStorageEvent>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable backupTask: Task option = None
    
    let yamlSerializer = SerializerBuilder().Build()
    let yamlDeserializer = DeserializerBuilder().Build()
    
    let maxUpdateHistory = 10000
    
    /// Start the roadmap storage system
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting roadmap storage system...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Ensure directories exist
            this.EnsureDirectoriesExist()
            
            // Load existing roadmaps
            do! this.LoadExistingRoadmapsAsync()
            
            // Start backup task if enabled
            if config.AutoBackupInterval > TimeSpan.Zero then
                let backupLoop = this.BackupLoopAsync(cancellationTokenSource.Value.Token)
                backupTask <- Some backupLoop
            
            logger.LogInformation($"Roadmap storage started. Loaded {roadmaps.Count} roadmaps.")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start roadmap storage system")
            isRunning <- false
            raise
    }
    
    /// Stop the roadmap storage system
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping roadmap storage system...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for backup task to complete
            match backupTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Backup task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for backup task to complete")
            | None -> ()
            
            // Save all roadmaps before shutdown
            do! this.SaveAllRoadmapsAsync()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            backupTask <- None
            
            logger.LogInformation("Roadmap storage stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping roadmap storage system")
    }
    
    /// Save roadmap to storage
    member this.SaveRoadmapAsync(roadmap: TarsRoadmap) = task {
        try
            logger.LogDebug($"Saving roadmap: {roadmap.Title}")
            
            // Validate roadmap
            let validation = RoadmapHelpers.validateRoadmap roadmap
            if not validation.IsValid then
                let errors = String.Join("; ", validation.Errors)
                logger.LogWarning($"Invalid roadmap: {errors}")
                return Error $"Invalid roadmap: {errors}"
            
            // Update roadmap in memory
            let oldRoadmap = roadmaps.TryGetValue(roadmap.Id)
            roadmaps.[roadmap.Id] <- roadmap
            
            // Save to file
            let fileName = $"{roadmap.Id}.roadmap.yaml"
            let filePath = Path.Combine(config.RoadmapDirectory, fileName)
            
            let yamlContent = yamlSerializer.Serialize(roadmap)
            do! File.WriteAllTextAsync(filePath, yamlContent)
            
            // Create backup if versioning is enabled
            if config.EnableVersioning then
                do! this.CreateVersionBackupAsync(roadmap)
            
            // Emit event
            let event = 
                match oldRoadmap with
                | true, old -> RoadmapUpdated (old, roadmap)
                | false, _ -> RoadmapCreated roadmap
            
            storageEvents.Enqueue(event)
            
            logger.LogInformation($"Roadmap saved successfully: {roadmap.Title}")
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Error saving roadmap: {roadmap.Title}")
            storageEvents.Enqueue(StorageError ($"Save failed for {roadmap.Title}", ex))
            return Error ex.Message
    }
    
    /// Load roadmap from storage
    member this.LoadRoadmapAsync(roadmapId: string) = task {
        try
            logger.LogDebug($"Loading roadmap: {roadmapId}")
            
            // Check memory cache first
            match roadmaps.TryGetValue(roadmapId) with
            | true, roadmap -> 
                return Some roadmap
            | false, _ ->
                // Load from file
                let fileName = $"{roadmapId}.roadmap.yaml"
                let filePath = Path.Combine(config.RoadmapDirectory, fileName)
                
                if File.Exists(filePath) then
                    let! content = File.ReadAllTextAsync(filePath)
                    let roadmap = yamlDeserializer.Deserialize<TarsRoadmap>(content)
                    
                    // Cache in memory
                    roadmaps.[roadmapId] <- roadmap
                    
                    logger.LogDebug($"Roadmap loaded from file: {roadmap.Title}")
                    return Some roadmap
                else
                    logger.LogDebug($"Roadmap file not found: {roadmapId}")
                    return None
                    
        with
        | ex ->
            logger.LogError(ex, $"Error loading roadmap: {roadmapId}")
            storageEvents.Enqueue(StorageError ($"Load failed for {roadmapId}", ex))
            return None
    }
    
    /// Delete roadmap from storage
    member this.DeleteRoadmapAsync(roadmapId: string) = task {
        try
            logger.LogDebug($"Deleting roadmap: {roadmapId}")
            
            // Remove from memory
            roadmaps.TryRemove(roadmapId) |> ignore
            
            // Delete file
            let fileName = $"{roadmapId}.roadmap.yaml"
            let filePath = Path.Combine(config.RoadmapDirectory, fileName)
            
            if File.Exists(filePath) then
                File.Delete(filePath)
            
            // Emit event
            storageEvents.Enqueue(RoadmapDeleted roadmapId)
            
            logger.LogInformation($"Roadmap deleted: {roadmapId}")
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Error deleting roadmap: {roadmapId}")
            storageEvents.Enqueue(StorageError ($"Delete failed for {roadmapId}", ex))
            return Error ex.Message
    }
    
    /// Get all roadmaps
    member this.GetAllRoadmapsAsync() = task {
        return roadmaps.Values |> List.ofSeq
    }
    
    /// Search roadmaps
    member this.SearchRoadmapsAsync(query: string) = task {
        try
            let queryLower = query.ToLower()
            
            let matchingRoadmaps = 
                roadmaps.Values
                |> Seq.filter (fun roadmap ->
                    roadmap.Title.ToLower().Contains(queryLower) ||
                    roadmap.Description.ToLower().Contains(queryLower) ||
                    roadmap.Phases |> List.exists (fun phase ->
                        phase.Title.ToLower().Contains(queryLower) ||
                        phase.Milestones |> List.exists (fun milestone ->
                            milestone.Title.ToLower().Contains(queryLower) ||
                            milestone.Achievements |> List.exists (fun achievement ->
                                achievement.Title.ToLower().Contains(queryLower) ||
                                achievement.Description.ToLower().Contains(queryLower)))))
                |> List.ofSeq
            
            return matchingRoadmaps
            
        with
        | ex ->
            logger.LogError(ex, $"Error searching roadmaps: {query}")
            return []
    }
    
    /// Update achievement
    member this.UpdateAchievementAsync(roadmapId: string, achievementId: string, update: AchievementUpdate) = task {
        try
            logger.LogDebug($"Updating achievement: {achievementId} in roadmap: {roadmapId}")
            
            match! this.LoadRoadmapAsync(roadmapId) with
            | Some roadmap ->
                // Find and update achievement
                let updatedRoadmap = this.UpdateAchievementInRoadmap(roadmap, achievementId, update)
                
                match updatedRoadmap with
                | Some updated ->
                    // Save updated roadmap
                    let! saveResult = this.SaveRoadmapAsync(updated)
                    
                    match saveResult with
                    | Ok () ->
                        // Record update
                        achievementUpdates.Enqueue(update)
                        
                        // Keep update history manageable
                        while achievementUpdates.Count > maxUpdateHistory do
                            achievementUpdates.TryDequeue() |> ignore
                        
                        // Find updated achievement for event
                        let updatedAchievement = this.FindAchievementInRoadmap(updated, achievementId)
                        match updatedAchievement with
                        | Some achievement ->
                            storageEvents.Enqueue(AchievementUpdated (achievement, update))
                        | None -> ()
                        
                        logger.LogInformation($"Achievement updated: {achievementId}")
                        return Ok ()
                    
                    | Error error ->
                        return Error error
                
                | None ->
                    let error = $"Achievement not found: {achievementId}"
                    logger.LogWarning(error)
                    return Error error
            
            | None ->
                let error = $"Roadmap not found: {roadmapId}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Error updating achievement: {achievementId}")
            return Error ex.Message
    }
    
    /// Get achievement updates
    member this.GetAchievementUpdatesAsync(achievementId: string option, limit: int option) = task {
        let updates = 
            achievementUpdates
            |> Seq.filter (fun update ->
                match achievementId with
                | Some id -> update.AchievementId = id
                | None -> true)
            |> Seq.sortByDescending (fun update -> update.UpdatedAt)
            |> (fun seq ->
                match limit with
                | Some l -> seq |> Seq.take l
                | None -> seq)
            |> List.ofSeq
        
        return updates
    }
    
    /// Create backup
    member this.CreateBackupAsync() = task {
        try
            logger.LogDebug("Creating roadmap backup...")
            
            let timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")
            let backupDir = Path.Combine(config.BackupDirectory, $"backup-{timestamp}")
            Directory.CreateDirectory(backupDir) |> ignore
            
            // Copy all roadmap files
            let roadmapFiles = Directory.GetFiles(config.RoadmapDirectory, "*.roadmap.yaml")
            for file in roadmapFiles do
                let fileName = Path.GetFileName(file)
                let destPath = Path.Combine(backupDir, fileName)
                File.Copy(file, destPath)
            
            // Create backup manifest
            let manifest = {|
                CreatedAt = DateTime.UtcNow
                RoadmapCount = roadmapFiles.Length
                TotalSize = roadmapFiles |> Array.sumBy (fun f -> FileInfo(f).Length)
                Version = "1.0"
            |}
            
            let manifestPath = Path.Combine(backupDir, "manifest.json")
            let manifestJson = JsonConvert.SerializeObject(manifest, Formatting.Indented)
            do! File.WriteAllTextAsync(manifestPath, manifestJson)
            
            // Cleanup old backups
            this.CleanupOldBackups()
            
            storageEvents.Enqueue(BackupCreated backupDir)
            
            logger.LogInformation($"Backup created: {backupDir}")
            return Ok backupDir
            
        with
        | ex ->
            logger.LogError(ex, "Error creating backup")
            storageEvents.Enqueue(StorageError ("Backup creation failed", ex))
            return Error ex.Message
    }
    
    /// Get storage events
    member this.GetStorageEvents(limit: int option) =
        let events = 
            storageEvents
            |> Seq.take (limit |> Option.defaultValue storageEvents.Count)
            |> List.ofSeq
        
        events
    
    /// Ensure directories exist
    member private this.EnsureDirectoriesExist() =
        let directories = [
            config.RoadmapDirectory
            config.BackupDirectory
            Path.Combine(config.RoadmapDirectory, "versions")
        ]
        
        for directory in directories do
            if not (Directory.Exists(directory)) then
                Directory.CreateDirectory(directory) |> ignore
                logger.LogDebug($"Created directory: {directory}")
    
    /// Load existing roadmaps from disk
    member private this.LoadExistingRoadmapsAsync() = task {
        try
            let roadmapFiles = Directory.GetFiles(config.RoadmapDirectory, "*.roadmap.yaml")
            
            for filePath in roadmapFiles do
                try
                    let! content = File.ReadAllTextAsync(filePath)
                    let roadmap = yamlDeserializer.Deserialize<TarsRoadmap>(content)
                    roadmaps.[roadmap.Id] <- roadmap
                    logger.LogDebug($"Loaded roadmap: {roadmap.Title}")
                with
                | ex ->
                    logger.LogWarning(ex, $"Failed to load roadmap from {filePath}")
            
            logger.LogInformation($"Loaded {roadmaps.Count} roadmaps from storage")
            
        with
        | ex ->
            logger.LogError(ex, "Error loading existing roadmaps")
    }
    
    /// Save all roadmaps to disk
    member private this.SaveAllRoadmapsAsync() = task {
        try
            logger.LogDebug("Saving all roadmaps...")
            
            for kvp in roadmaps do
                let roadmap = kvp.Value
                let fileName = $"{roadmap.Id}.roadmap.yaml"
                let filePath = Path.Combine(config.RoadmapDirectory, fileName)
                
                let yamlContent = yamlSerializer.Serialize(roadmap)
                do! File.WriteAllTextAsync(filePath, yamlContent)
            
            logger.LogInformation($"Saved {roadmaps.Count} roadmaps to storage")
            
        with
        | ex ->
            logger.LogError(ex, "Error saving all roadmaps")
    }
    
    /// Create version backup
    member private this.CreateVersionBackupAsync(roadmap: TarsRoadmap) = task {
        try
            let timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")
            let versionDir = Path.Combine(config.RoadmapDirectory, "versions", roadmap.Id)
            Directory.CreateDirectory(versionDir) |> ignore
            
            let versionFile = Path.Combine(versionDir, $"{roadmap.Id}-{timestamp}.yaml")
            let yamlContent = yamlSerializer.Serialize(roadmap)
            do! File.WriteAllTextAsync(versionFile, yamlContent)
            
            logger.LogDebug($"Created version backup: {versionFile}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to create version backup for roadmap: {roadmap.Id}")
    }
    
    /// Backup loop for automatic backups
    member private this.BackupLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting backup loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    do! Task.Delay(config.AutoBackupInterval, cancellationToken)
                    
                    if not cancellationToken.IsCancellationRequested then
                        let! _ = this.CreateBackupAsync()
                        ()
                    
                with
                | :? OperationCanceledException ->
                    () // Exit the loop
                | ex ->
                    logger.LogWarning(ex, "Error in backup loop")
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Backup loop cancelled")
        | ex ->
            logger.LogError(ex, "Backup loop failed")
    }
    
    /// Cleanup old backups
    member private this.CleanupOldBackups() =
        try
            let backupDirs = Directory.GetDirectories(config.BackupDirectory, "backup-*")
            let sortedDirs = backupDirs |> Array.sortByDescending id
            
            if sortedDirs.Length > config.MaxBackups then
                let dirsToDelete = sortedDirs |> Array.skip config.MaxBackups
                
                for dir in dirsToDelete do
                    Directory.Delete(dir, true)
                    logger.LogDebug($"Deleted old backup: {dir}")
        with
        | ex ->
            logger.LogWarning(ex, "Error cleaning up old backups")
    
    /// Update achievement in roadmap
    member private this.UpdateAchievementInRoadmap(roadmap: TarsRoadmap, achievementId: string, update: AchievementUpdate) =
        // This would recursively search and update the achievement in the roadmap structure
        // Implementation would traverse phases -> milestones -> achievements
        // For brevity, returning None here - full implementation would update the nested structure
        None
    
    /// Find achievement in roadmap
    member private this.FindAchievementInRoadmap(roadmap: TarsRoadmap, achievementId: string) =
        roadmap.Phases
        |> List.collect (fun phase -> phase.Milestones)
        |> List.collect (fun milestone -> milestone.Achievements)
        |> List.tryFind (fun achievement -> achievement.Id = achievementId)
