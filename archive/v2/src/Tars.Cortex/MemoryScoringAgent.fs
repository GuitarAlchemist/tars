namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core

/// <summary>
/// Agent responsible for the lifecycle of long-term memories.
/// Defines retention policies, pruning schedules, and consolidation logic.
/// </summary>
module MemoryScoringAgent =

    /// <summary>
    /// Runs a decay cycle on a specific memory collection.
    /// Removes low-value or expired memories and enforces capacity limits.
    /// </summary>
    /// <param name="store">The vector store instance.</param>
    /// <param name="collectionName">Target collection.</param>
    /// <param name="capacity">Maximum number of items to retain.</param>
    /// <returns>Number of items pruned.</returns>
    let runDecayCycle (store: IVectorStore) (collectionName: string) (capacity: int) =
        task {
            // Policy 1: Remove items explicitly marked as 'ephemeral'
            // This allows short-term working notes to be cleaned up automatically.
            let decayPolicy (metadata: Map<string, string>) =
                match metadata.TryFind "lifecycle" with
                | Some "ephemeral" -> true
                | Some "transient" -> 
                    // prune if expired
                    match metadata.TryFind "expires_at" with
                    | Some dateStr ->
                        match System.DateTimeOffset.TryParse dateStr with
                        | true, expiresAt -> System.DateTimeOffset.UtcNow > expiresAt
                        | _ -> false // invalid date, keep safe
                    | None -> false
                | _ -> 
                    // Check generic expiration even if not explicitly transient?
                    match metadata.TryFind "expires_at" with
                    | Some dateStr ->
                        match System.DateTimeOffset.TryParse dateStr with
                        | true, expiresAt -> System.DateTimeOffset.UtcNow > expiresAt
                        | _ -> false
                    | None -> false

            // The PruneAsync method in IVectorStore implementations (like InMemory/Sqlite)
            // typically enforcing the capacity limit using LRU (Least Recently Used) logic
            // AFTER applying the filter.
            let! prunedCount = store.PruneAsync(collectionName, capacity, decayPolicy)
            
            return prunedCount
        }

    /// <summary>
    /// Promotes a memory to a more permanent state (e.g., higher importance).
    /// </summary>
    let promote (store: IVectorStore) (collectionName: string) (id: string) =
        task {
            // Update metadata to reflect promotion
            let updates = Map [ "lifecycle", "permanent"; "importance", "high" ]
            do! store.UpdateMetadataAsync(collectionName, id, updates)
        }
