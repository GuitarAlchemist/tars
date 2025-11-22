namespace TarsEngine.FSharp.Cli.Core

open System
open System.Collections.Concurrent
open System.IO
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem

/// Unified Caching System - Multi-level caching using unified architecture
module UnifiedCache =
    
    /// Cache levels
    type CacheLevel =
        | Memory
        | Disk
        | Distributed
    
    /// Cache entry with metadata
    type CacheEntry<'T> = {
        Key: string
        Value: 'T
        CreatedAt: DateTime
        LastAccessed: DateTime
        AccessCount: int64
        ExpiresAt: DateTime option
        Size: int64
        Tags: string list
        CorrelationId: string
    }
    
    /// Cache invalidation strategy
    type InvalidationStrategy =
        | TimeToLive of TimeSpan
        | LeastRecentlyUsed of maxEntries: int
        | LeastFrequentlyUsed of maxEntries: int
        | TagBased of tags: string list
        | Manual
    
    /// Cache statistics
    type CacheStatistics = {
        TotalEntries: int64
        MemoryEntries: int64
        DiskEntries: int64
        DistributedEntries: int64
        HitCount: int64
        MissCount: int64
        HitRatio: float
        TotalSize: int64
        MemorySize: int64
        DiskSize: int64
        LastCleanup: DateTime
        OperationsPerSecond: float
    }
    
    /// Cache configuration
    type CacheConfiguration = {
        MaxMemoryEntries: int
        MaxDiskEntries: int
        MaxMemorySize: int64
        MaxDiskSize: int64
        DefaultTtl: TimeSpan
        CleanupInterval: TimeSpan
        CompressionEnabled: bool
        EncryptionEnabled: bool
        DiskCachePath: string
        DistributedCacheEndpoint: string option
    }
    
    /// Cache context
    type CacheContext = {
        ConfigManager: UnifiedConfigurationManager
        ProofGenerator: UnifiedProofGenerator
        Logger: ITarsLogger
        Configuration: CacheConfiguration
        CorrelationId: string
    }
    
    /// Create cache context
    let createCacheContext (logger: ITarsLogger) (configManager: UnifiedConfigurationManager) (proofGenerator: UnifiedProofGenerator) =
        let config = {
            MaxMemoryEntries = ConfigurationExtensions.getInt configManager "tars.cache.maxMemoryEntries" 10000
            MaxDiskEntries = ConfigurationExtensions.getInt configManager "tars.cache.maxDiskEntries" 100000
            MaxMemorySize = ConfigurationExtensions.getInt configManager "tars.cache.maxMemorySize" (100 * 1024 * 1024) |> int64 // 100MB
            MaxDiskSize = ConfigurationExtensions.getInt configManager "tars.cache.maxDiskSize" (1024 * 1024 * 1024) |> int64 // 1GB
            DefaultTtl = TimeSpan.FromMinutes(ConfigurationExtensions.getFloat configManager "tars.cache.defaultTtlMinutes" 60.0)
            CleanupInterval = TimeSpan.FromMinutes(ConfigurationExtensions.getFloat configManager "tars.cache.cleanupIntervalMinutes" 15.0)
            CompressionEnabled = ConfigurationExtensions.getBool configManager "tars.cache.compressionEnabled" true
            EncryptionEnabled = ConfigurationExtensions.getBool configManager "tars.cache.encryptionEnabled" false
            DiskCachePath = ConfigurationExtensions.getString configManager "tars.cache.diskPath" "./cache"
            DistributedCacheEndpoint = 
                let endpoint = ConfigurationExtensions.getString configManager "tars.cache.distributedEndpoint" ""
                if String.IsNullOrEmpty(endpoint) then None else Some endpoint
        }
        
        {
            ConfigManager = configManager
            ProofGenerator = proofGenerator
            Logger = logger
            Configuration = config
            CorrelationId = generateCorrelationId()
        }
    
    /// Calculate cache entry size
    let calculateEntrySize<'T> (entry: CacheEntry<'T>) : int64 =
        try
            let json = JsonSerializer.Serialize(entry.Value)
            int64 (json.Length * 2) // Approximate UTF-16 size
        with
        | _ -> 1024L // Default size if serialization fails
    
    /// Check if entry is expired
    let isExpired (entry: CacheEntry<'T>) : bool =
        match entry.ExpiresAt with
        | Some expiry -> DateTime.UtcNow > expiry
        | None -> false
    
    /// Update entry access statistics
    let updateAccessStats (entry: CacheEntry<'T>) : CacheEntry<'T> =
        { entry with 
            LastAccessed = DateTime.UtcNow
            AccessCount = entry.AccessCount + 1L }
    
    /// Unified Cache implementation
    type UnifiedCacheManager(logger: ITarsLogger, configManager: UnifiedConfigurationManager, proofGenerator: UnifiedProofGenerator) =
        
        let context = createCacheContext logger configManager proofGenerator
        let memoryCache = ConcurrentDictionary<string, CacheEntry<obj>>()
        let mutable statistics = {
            TotalEntries = 0L
            MemoryEntries = 0L
            DiskEntries = 0L
            DistributedEntries = 0L
            HitCount = 0L
            MissCount = 0L
            HitRatio = 0.0
            TotalSize = 0L
            MemorySize = 0L
            DiskSize = 0L
            LastCleanup = DateTime.UtcNow
            OperationsPerSecond = 0.0
        }
        
        let mutable lastOperationTime = DateTime.UtcNow
        let mutable operationCount = 0L
        
        // Ensure disk cache directory exists
        do
            if not (Directory.Exists(context.Configuration.DiskCachePath)) then
                Directory.CreateDirectory(context.Configuration.DiskCachePath) |> ignore
        
        /// Update operation statistics
        let updateOperationStats() =
            operationCount <- operationCount + 1L
            let now = DateTime.UtcNow
            let timeDiff = (now - lastOperationTime).TotalSeconds
            if timeDiff > 0.0 then
                statistics <- { statistics with OperationsPerSecond = float operationCount / timeDiff }
            lastOperationTime <- now
        
        /// Generate cache operation proof
        let generateCacheProof (operation: string) (key: string) =
            task {
                let! proofResult =
                    ProofExtensions.generateExecutionProof
                        context.ProofGenerator
                        $"CacheOperation_{operation}_{key}"
                        context.CorrelationId
                
                return match proofResult with
                       | Success (proof, _) -> Some proof.ProofId
                       | Failure _ -> None
            }
        
        /// Get cache entry from memory
        member private this.GetFromMemory(key: string) : CacheEntry<obj> option =
            match memoryCache.TryGetValue(key) with
            | true, entry when not (isExpired entry) ->
                let updatedEntry = updateAccessStats entry
                memoryCache.TryUpdate(key, updatedEntry, entry) |> ignore
                Some updatedEntry
            | true, entry ->
                // Entry is expired, remove it
                memoryCache.TryRemove(key) |> ignore
                None
            | false, _ -> None
        
        /// Store cache entry in memory
        member private this.StoreInMemory(key: string, value: obj, ttl: TimeSpan option) : TarsResult<unit, TarsError> =
            try
                let entry = {
                    Key = key
                    Value = value
                    CreatedAt = DateTime.UtcNow
                    LastAccessed = DateTime.UtcNow
                    AccessCount = 0L
                    ExpiresAt = ttl |> Option.map (fun t -> DateTime.UtcNow.Add(t))
                    Size = calculateEntrySize { Key = key; Value = value; CreatedAt = DateTime.UtcNow; LastAccessed = DateTime.UtcNow; AccessCount = 0L; ExpiresAt = None; Tags = []; CorrelationId = context.CorrelationId }
                    Tags = []
                    CorrelationId = context.CorrelationId
                }
                
                // Check memory limits
                if memoryCache.Count >= context.Configuration.MaxMemoryEntries then
                    this.CleanupMemoryCache()
                
                memoryCache.AddOrUpdate(key, entry, fun _ _ -> entry) |> ignore
                
                // Update statistics
                statistics <-
                    { statistics with
                        MemoryEntries = int64 memoryCache.Count
                        TotalEntries = statistics.TotalEntries + 1L
                        MemorySize = statistics.MemorySize + entry.Size
                        TotalSize = statistics.TotalSize + entry.Size }
                
                Success ((), Map [("key", box key); ("level", box "Memory")])
            
            with
            | ex ->
                let error = ExecutionError ($"Failed to store in memory cache: {ex.Message}", Some ex)
                Failure (error, context.CorrelationId)
        
        /// Cleanup memory cache using LRU strategy
        member private this.CleanupMemoryCache() : unit =
            try
                let entries = memoryCache.ToArray()
                let sortedEntries = 
                    entries 
                    |> Array.sortBy (fun kvp -> kvp.Value.LastAccessed)
                    |> Array.take (entries.Length / 4) // Remove 25% of entries
                
                for kvp in sortedEntries do
                    memoryCache.TryRemove(kvp.Key) |> ignore
                
                context.Logger.LogInformation(context.CorrelationId, $"Cleaned up {sortedEntries.Length} entries from memory cache")
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "CacheCleanupError" "Memory cache cleanup failed" (Some ex), ex)
        
        /// Get cache entry from disk
        member private this.GetFromDisk(key: string) : Task<CacheEntry<obj> option> =
            task {
                try
                    let filePath = Path.Combine(context.Configuration.DiskCachePath, $"{key}.cache")
                    if File.Exists(filePath) then
                        let! json = File.ReadAllTextAsync(filePath)
                        let entry = JsonSerializer.Deserialize<CacheEntry<obj>>(json)
                        
                        if not (isExpired entry) then
                            return Some (updateAccessStats entry)
                        else
                            // Entry is expired, delete file
                            File.Delete(filePath)
                            return None
                    else
                        return None
                
                with
                | ex ->
                    context.Logger.LogError(context.CorrelationId, TarsError.create "DiskCacheError" "Disk cache read failed" (Some ex), ex)
                    return None
            }
        
        /// Store cache entry on disk
        member private this.StoreToDisk(key: string, value: obj, ttl: TimeSpan option) : Task<TarsResult<unit, TarsError>> =
            task {
                try
                    let entry = {
                        Key = key
                        Value = value
                        CreatedAt = DateTime.UtcNow
                        LastAccessed = DateTime.UtcNow
                        AccessCount = 0L
                        ExpiresAt = ttl |> Option.map (fun t -> DateTime.UtcNow.Add(t))
                        Size = calculateEntrySize { Key = key; Value = value; CreatedAt = DateTime.UtcNow; LastAccessed = DateTime.UtcNow; AccessCount = 0L; ExpiresAt = None; Tags = []; CorrelationId = context.CorrelationId }
                        Tags = []
                        CorrelationId = context.CorrelationId
                    }
                    
                    let json = JsonSerializer.Serialize(entry)
                    let filePath = Path.Combine(context.Configuration.DiskCachePath, $"{key}.cache")
                    
                    do! File.WriteAllTextAsync(filePath, json)
                    
                    // Update statistics
                    statistics <-
                        { statistics with
                            DiskEntries = statistics.DiskEntries + 1L
                            TotalEntries = statistics.TotalEntries + 1L
                            DiskSize = statistics.DiskSize + entry.Size
                            TotalSize = statistics.TotalSize + entry.Size }
                    
                    return Success ((), Map [("key", box key); ("level", box "Disk")])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to store to disk cache: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get cached value
        member this.GetAsync<'T>(key: string) : Task<TarsResult<'T option, TarsError>> =
            task {
                try
                    updateOperationStats()
                    context.Logger.LogInformation(context.CorrelationId, $"Cache GET: {key}")
                    
                    // Try memory cache first
                    match this.GetFromMemory(key) with
                    | Some entry ->
                        statistics <- { statistics with HitCount = statistics.HitCount + 1L }
                        let! _ = generateCacheProof "GET_MEMORY" key
                        return Success (Some (entry.Value :?> 'T), Map [("source", box "Memory")])
                    
                    | None ->
                        // Try disk cache
                        let! diskEntry = this.GetFromDisk(key)
                        match diskEntry with
                        | Some entry ->
                            // Promote to memory cache
                            let! _ = this.StoreInMemory(key, entry.Value, entry.ExpiresAt |> Option.map (fun exp -> exp - DateTime.UtcNow))
                            statistics <- { statistics with HitCount = statistics.HitCount + 1L }
                            let! _ = generateCacheProof "GET_DISK" key
                            return Success (Some (entry.Value :?> 'T), Map [("source", box "Disk")])
                        
                        | None ->
                            statistics <- { statistics with MissCount = statistics.MissCount + 1L }
                            let! _ = generateCacheProof "GET_MISS" key
                            return Success (None, Map [("source", box "Miss")])
                
                with
                | ex ->
                    let error = ExecutionError ($"Cache get failed: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Set cached value
        member this.SetAsync<'T>(key: string, value: 'T, ?ttl: TimeSpan) : Task<TarsResult<unit, TarsError>> =
            task {
                try
                    updateOperationStats()
                    let effectiveTtl = ttl |> Option.defaultValue context.Configuration.DefaultTtl
                    context.Logger.LogInformation(context.CorrelationId, $"Cache SET: {key}")
                    
                    // Store in memory cache
                    let memoryResult = this.StoreInMemory(key, box value, Some effectiveTtl)
                    
                    // Also store in disk cache for persistence
                    let! diskResult = this.StoreToDisk(key, box value, Some effectiveTtl)
                    
                    let! _ = generateCacheProof "SET" key
                    
                    match memoryResult, diskResult with
                    | Success _, Success _ ->
                        return Success ((), Map [("key", box key); ("ttl", box effectiveTtl.TotalMinutes)])
                    | Success _, Failure (error, _) ->
                        context.Logger.LogWarning(context.CorrelationId, $"Disk cache failed but memory cache succeeded: {TarsError.toString error}")
                        return Success ((), Map [("key", box key); ("warning", box "Disk cache failed")])
                    | Failure (error, _), _ ->
                        return Failure (error, context.CorrelationId)
                
                with
                | ex ->
                    let error = ExecutionError ($"Cache set failed: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Remove cached value
        member this.RemoveAsync(key: string) : Task<TarsResult<bool, TarsError>> =
            task {
                try
                    updateOperationStats()
                    context.Logger.LogInformation(context.CorrelationId, $"Cache REMOVE: {key}")
                    
                    let memoryRemoved = memoryCache.TryRemove(key) |> fst
                    
                    let diskPath = Path.Combine(context.Configuration.DiskCachePath, $"{key}.cache")
                    let diskRemoved = 
                        if File.Exists(diskPath) then
                            File.Delete(diskPath)
                            true
                        else
                            false
                    
                    let! _ = generateCacheProof "REMOVE" key
                    
                    return Success (memoryRemoved || diskRemoved, Map [
                        ("key", box key)
                        ("memoryRemoved", box memoryRemoved)
                        ("diskRemoved", box diskRemoved)
                    ])
                
                with
                | ex ->
                    let error = ExecutionError ($"Cache remove failed: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Clear all cache
        member this.ClearAsync() : Task<TarsResult<int64, TarsError>> =
            task {
                try
                    updateOperationStats()
                    context.Logger.LogInformation(context.CorrelationId, "Cache CLEAR ALL")
                    
                    let memoryCount = int64 memoryCache.Count
                    memoryCache.Clear()
                    
                    let diskFiles = Directory.GetFiles(context.Configuration.DiskCachePath, "*.cache")
                    for file in diskFiles do
                        File.Delete(file)
                    
                    let totalCleared = memoryCount + int64 diskFiles.Length
                    
                    // Reset statistics
                    statistics <-
                        { statistics with
                            TotalEntries = 0L
                            MemoryEntries = 0L
                            DiskEntries = 0L
                            TotalSize = 0L
                            MemorySize = 0L
                            DiskSize = 0L
                            LastCleanup = DateTime.UtcNow }
                    
                    let! _ = generateCacheProof "CLEAR_ALL" "all"
                    
                    return Success (totalCleared, Map [("entriesCleared", box totalCleared)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Cache clear failed: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get cache statistics
        member this.GetStatistics() : CacheStatistics =
            let totalOps = statistics.HitCount + statistics.MissCount
            let hitRatio = if totalOps > 0L then float statistics.HitCount / float totalOps else 0.0
            
            { statistics with 
                HitRatio = hitRatio
                MemoryEntries = int64 memoryCache.Count }
        
        /// Cleanup expired entries
        member this.CleanupAsync() : Task<TarsResult<int64, TarsError>> =
            task {
                try
                    context.Logger.LogInformation(context.CorrelationId, "Cache cleanup started")
                    
                    let mutable cleanedCount = 0L
                    
                    // Cleanup memory cache
                    let expiredMemoryKeys = 
                        memoryCache.ToArray()
                        |> Array.filter (fun kvp -> isExpired kvp.Value)
                        |> Array.map (fun kvp -> kvp.Key)
                    
                    for key in expiredMemoryKeys do
                        if memoryCache.TryRemove(key) |> fst then
                            cleanedCount <- cleanedCount + 1L
                    
                    // Cleanup disk cache
                    let diskFiles = Directory.GetFiles(context.Configuration.DiskCachePath, "*.cache")
                    for file in diskFiles do
                        try
                            let json = File.ReadAllText(file)
                            let entry = JsonSerializer.Deserialize<CacheEntry<obj>>(json)
                            if isExpired entry then
                                File.Delete(file)
                                cleanedCount <- cleanedCount + 1L
                        with
                        | _ -> () // Skip corrupted files
                    
                    statistics <- { statistics with LastCleanup = DateTime.UtcNow }
                    
                    let! _ = generateCacheProof "CLEANUP" "expired"
                    
                    return Success (cleanedCount, Map [("cleanedEntries", box cleanedCount)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Cache cleanup failed: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Dispose resources
        interface IDisposable with
            member this.Dispose() =
                memoryCache.Clear()
