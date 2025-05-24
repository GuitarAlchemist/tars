namespace TarsEngine.DSL

open System
open System.Collections.Generic
open System.Security.Cryptography

/// <summary>
/// Module for caching parsed chunks to improve performance.
/// </summary>
module ChunkCache =
    /// <summary>
    /// Cache of parsed chunks.
    /// </summary>
    let private cache = Dictionary<string, ChunkParseResult>()
    
    /// <summary>
    /// Maximum number of entries in the cache.
    /// </summary>
    let private maxCacheSize = 1000
    
    /// <summary>
    /// Last access times for cache entries.
    /// </summary>
    let private lastAccessTimes = Dictionary<string, DateTime>()
    
    /// <summary>
    /// Calculate a hash for a chunk.
    /// </summary>
    /// <param name="chunk">The chunk to hash.</param>
    /// <returns>A hash string for the chunk.</returns>
    let private calculateHash (chunk: CodeChunk) =
        use sha256 = SHA256.Create()
        let hashBytes = sha256.ComputeHash(Text.Encoding.UTF8.GetBytes(chunk.Content))
        BitConverter.ToString(hashBytes).Replace("-", "")
    
    /// <summary>
    /// Get a cached parse result for a chunk.
    /// </summary>
    /// <param name="chunk">The chunk to get the cached result for.</param>
    /// <returns>The cached parse result, or None if not found.</returns>
    let getCachedResult (chunk: CodeChunk) =
        let hash = calculateHash chunk
        
        match cache.TryGetValue(hash) with
        | true, result ->
            // Update last access time
            lastAccessTimes.[hash] <- DateTime.Now
            Some result
        | false, _ -> None
    
    /// <summary>
    /// Cache a parse result for a chunk.
    /// </summary>
    /// <param name="chunk">The chunk to cache the result for.</param>
    /// <param name="result">The parse result to cache.</param>
    let cacheResult (chunk: CodeChunk) (result: ChunkParseResult) =
        let hash = calculateHash chunk
        
        // Check if we need to evict an entry
        if cache.Count >= maxCacheSize then
            // Find the least recently used entry
            let oldestHash = 
                lastAccessTimes
                |> Seq.minBy (fun kvp -> kvp.Value)
                |> fun kvp -> kvp.Key
            
            // Remove the oldest entry
            cache.Remove(oldestHash) |> ignore
            lastAccessTimes.Remove(oldestHash) |> ignore
        
        // Add the new entry
        cache.[hash] <- result
        lastAccessTimes.[hash] <- DateTime.Now
    
    /// <summary>
    /// Clear the cache.
    /// </summary>
    let clearCache() =
        cache.Clear()
        lastAccessTimes.Clear()
    
    /// <summary>
    /// Get the number of entries in the cache.
    /// </summary>
    /// <returns>The number of entries in the cache.</returns>
    let getCacheSize() =
        cache.Count
