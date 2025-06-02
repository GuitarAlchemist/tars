namespace TarsEngine.FSharp.Consciousness

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging

/// <summary>
/// Hybrid Consciousness Storage System
/// Combines volatile in-memory state with persistent storage for optimal performance and durability
/// </summary>
module HybridConsciousnessStorage =
    
    /// Storage tier for different types of consciousness data
    type StorageTier =
        | Volatile      // In-memory only, ultra-fast
        | Cached        // In-memory with async persistence
        | Persistent    // Immediate disk write
        | LongTerm      // Compressed, archived storage
    
    /// Consciousness data classification
    type ConsciousnessDataType =
        | WorkingMemory of importance: float
        | CurrentThoughts
        | EmotionalState
        | AttentionFocus
        | PersonalityTraits
        | SelfAwareness
        | ConsciousnessLevel
        | AgentContributions
        | ConversationContext
        | LongTermMemory of significance: float
    
    /// Storage configuration for each data type
    type StorageConfig = {
        DataType: ConsciousnessDataType
        Tier: StorageTier
        MaxVolatileItems: int option
        PersistenceInterval: TimeSpan option
        CompressionThreshold: int option
        RetentionPeriod: TimeSpan option
    }
    
    /// Consciousness state entry
    type ConsciousnessEntry = {
        Id: string
        DataType: ConsciousnessDataType
        Content: obj
        Timestamp: DateTime
        Importance: float
        Tier: StorageTier
        LastAccessed: DateTime
        AccessCount: int
    }
    
    /// <summary>
    /// Hybrid Consciousness Storage Manager
    /// Manages multi-tiered storage for optimal performance and persistence
    /// </summary>
    type HybridConsciousnessStorageManager(storageConfigs: StorageConfig list, basePath: string, logger: ILogger<HybridConsciousnessStorageManager>) =
        
        // Volatile storage (in-memory)
        let volatileStorage = ConcurrentDictionary<string, ConsciousnessEntry>()
        
        // Cached storage (in-memory + async persistence)
        let cachedStorage = ConcurrentDictionary<string, ConsciousnessEntry>()
        
        // Storage configurations by data type
        let configMap = storageConfigs |> List.map (fun c -> c.DataType, c) |> Map.ofList
        
        // Persistence queue for async writes
        let persistenceQueue = ConcurrentQueue<ConsciousnessEntry>()
        
        // Background persistence task
        let mutable persistenceTask: Task option = None
        
        /// <summary>
        /// Get storage configuration for data type
        /// </summary>
        member private this.GetStorageConfig(dataType: ConsciousnessDataType) =
            configMap.TryFind(dataType) |> Option.defaultValue {
                DataType = dataType
                Tier = Cached
                MaxVolatileItems = Some 100
                PersistenceInterval = Some (TimeSpan.FromSeconds(5.0))
                CompressionThreshold = Some 1000
                RetentionPeriod = Some (TimeSpan.FromDays(30.0))
            }
        
        /// <summary>
        /// Store consciousness data
        /// </summary>
        member this.Store(id: string, dataType: ConsciousnessDataType, content: obj, importance: float) =
            task {
                let config = this.GetStorageConfig(dataType)
                let entry = {
                    Id = id
                    DataType = dataType
                    Content = content
                    Timestamp = DateTime.UtcNow
                    Importance = importance
                    Tier = config.Tier
                    LastAccessed = DateTime.UtcNow
                    AccessCount = 1
                }
                
                match config.Tier with
                | Volatile ->
                    // Store in volatile memory only
                    volatileStorage.[id] <- entry
                    this.ManageVolatileCapacity(dataType, config)
                    logger.LogDebug("Stored {DataType} in volatile storage: {Id}", dataType, id)
                
                | Cached ->
                    // Store in cache and queue for persistence
                    cachedStorage.[id] <- entry
                    persistenceQueue.Enqueue(entry)
                    logger.LogDebug("Stored {DataType} in cached storage: {Id}", dataType, id)
                
                | Persistent ->
                    // Immediate disk write
                    do! this.WriteToDisk(entry)
                    cachedStorage.[id] <- entry
                    logger.LogDebug("Stored {DataType} in persistent storage: {Id}", dataType, id)
                
                | LongTerm ->
                    // Compressed long-term storage
                    do! this.WriteToLongTermStorage(entry)
                    logger.LogDebug("Stored {DataType} in long-term storage: {Id}", dataType, id)
            }
        
        /// <summary>
        /// Retrieve consciousness data
        /// </summary>
        member this.Retrieve(id: string) =
            task {
                // Try volatile storage first (fastest)
                match volatileStorage.TryGetValue(id) with
                | (true, entry) ->
                    let updatedEntry = { entry with LastAccessed = DateTime.UtcNow; AccessCount = entry.AccessCount + 1 }
                    volatileStorage.[id] <- updatedEntry
                    return Some updatedEntry
                | (false, _) ->
                    // Try cached storage
                    match cachedStorage.TryGetValue(id) with
                    | (true, entry) ->
                        let updatedEntry = { entry with LastAccessed = DateTime.UtcNow; AccessCount = entry.AccessCount + 1 }
                        cachedStorage.[id] <- updatedEntry
                        return Some updatedEntry
                    | (false, _) ->
                        // Try loading from disk
                        let! diskEntry = this.LoadFromDisk(id)
                        match diskEntry with
                        | Some entry ->
                            // Promote to cache for faster future access
                            let updatedEntry = { entry with LastAccessed = DateTime.UtcNow; AccessCount = entry.AccessCount + 1 }
                            cachedStorage.[id] <- updatedEntry
                            return Some updatedEntry
                        | None ->
                            return None
            }
        
        /// <summary>
        /// Query consciousness data by type and criteria
        /// </summary>
        member this.Query(dataType: ConsciousnessDataType, predicate: ConsciousnessEntry -> bool, maxResults: int) =
            task {
                let results = ResizeArray<ConsciousnessEntry>()
                
                // Search volatile storage
                for entry in volatileStorage.Values do
                    if entry.DataType = dataType && predicate(entry) && results.Count < maxResults then
                        results.Add(entry)
                
                // Search cached storage
                for entry in cachedStorage.Values do
                    if entry.DataType = dataType && predicate(entry) && results.Count < maxResults then
                        results.Add(entry)
                
                // If still need more results, search disk
                if results.Count < maxResults then
                    let! diskResults = this.QueryDisk(dataType, predicate, maxResults - results.Count)
                    results.AddRange(diskResults)
                
                return results |> Seq.toList
            }
        
        /// <summary>
        /// Manage volatile storage capacity
        /// </summary>
        member private this.ManageVolatileCapacity(dataType: ConsciousnessDataType, config: StorageConfig) =
            match config.MaxVolatileItems with
            | Some maxItems ->
                let currentItems = 
                    volatileStorage.Values 
                    |> Seq.filter (fun e -> e.DataType = dataType)
                    |> Seq.length
                
                if currentItems > maxItems then
                    // Remove least important and least recently accessed items
                    let itemsToRemove = 
                        volatileStorage.Values
                        |> Seq.filter (fun e -> e.DataType = dataType)
                        |> Seq.sortBy (fun e -> e.Importance * float e.AccessCount + (DateTime.UtcNow - e.LastAccessed).TotalHours)
                        |> Seq.take (currentItems - maxItems)
                    
                    for item in itemsToRemove do
                        volatileStorage.TryRemove(item.Id) |> ignore
                        // Move to cached storage if important
                        if item.Importance > 0.7 then
                            cachedStorage.[item.Id] <- { item with Tier = Cached }
                            persistenceQueue.Enqueue(item)
            | None -> ()
        
        /// <summary>
        /// Write entry to disk
        /// </summary>
        member private this.WriteToDisk(entry: ConsciousnessEntry) =
            task {
                try
                    let fileName = $"{entry.DataType}_{entry.Id}.json"
                    let filePath = Path.Combine(basePath, "consciousness", fileName)
                    Directory.CreateDirectory(Path.GetDirectoryName(filePath)) |> ignore
                    
                    let json = JsonSerializer.Serialize(entry, JsonSerializerOptions(WriteIndented = true))
                    do! File.WriteAllTextAsync(filePath, json)
                with
                | ex -> logger.LogError(ex, "Failed to write consciousness entry to disk: {Id}", entry.Id)
            }
        
        /// <summary>
        /// Load entry from disk
        /// </summary>
        member private this.LoadFromDisk(id: string) =
            task {
                try
                    let files = Directory.GetFiles(Path.Combine(basePath, "consciousness"), $"*_{id}.json")
                    if files.Length > 0 then
                        let! json = File.ReadAllTextAsync(files.[0])
                        let entry = JsonSerializer.Deserialize<ConsciousnessEntry>(json)
                        return Some entry
                    else
                        return None
                with
                | ex -> 
                    logger.LogError(ex, "Failed to load consciousness entry from disk: {Id}", id)
                    return None
            }
        
        /// <summary>
        /// Query disk storage
        /// </summary>
        member private this.QueryDisk(dataType: ConsciousnessDataType, predicate: ConsciousnessEntry -> bool, maxResults: int) =
            task {
                try
                    let results = ResizeArray<ConsciousnessEntry>()
                    let consciousnessDir = Path.Combine(basePath, "consciousness")
                    
                    if Directory.Exists(consciousnessDir) then
                        let files = Directory.GetFiles(consciousnessDir, $"{dataType}_*.json")
                        
                        for file in files do
                            if results.Count < maxResults then
                                let! json = File.ReadAllTextAsync(file)
                                let entry = JsonSerializer.Deserialize<ConsciousnessEntry>(json)
                                if predicate(entry) then
                                    results.Add(entry)
                    
                    return results |> Seq.toList
                with
                | ex ->
                    logger.LogError(ex, "Failed to query disk storage for {DataType}", dataType)
                    return []
            }
        
        /// <summary>
        /// Write to long-term compressed storage
        /// </summary>
        member private this.WriteToLongTermStorage(entry: ConsciousnessEntry) =
            task {
                // Implementation for compressed long-term storage
                // Could use compression, database, or cloud storage
                logger.LogInformation("Writing to long-term storage: {Id}", entry.Id)
            }
        
        /// <summary>
        /// Start background persistence
        /// </summary>
        member this.StartBackgroundPersistence() =
            persistenceTask <- Some (Task.Run(fun () ->
                task {
                    while true do
                        try
                            let mutable entry = Unchecked.defaultof<ConsciousnessEntry>
                            while persistenceQueue.TryDequeue(&entry) do
                                do! this.WriteToDisk(entry)
                            
                            do! Task.Delay(1000) // Check every second
                        with
                        | ex -> logger.LogError(ex, "Error in background persistence")
                } |> Async.AwaitTask |> Async.RunSynchronously
            ))
        
        /// <summary>
        /// Get storage statistics
        /// </summary>
        member this.GetStorageStatistics() =
            Map.ofList [
                ("volatileEntries", volatileStorage.Count :> obj)
                ("cachedEntries", cachedStorage.Count :> obj)
                ("pendingPersistence", persistenceQueue.Count :> obj)
                ("memoryUsage", GC.GetTotalMemory(false) :> obj)
            ]
