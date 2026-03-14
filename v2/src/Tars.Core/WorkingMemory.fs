/// Working Memory Capacitor for Evolution Engine
/// Provides importance-based memory management with time decay
namespace Tars.Core

open System
open System.Collections.Generic

/// Importance scoring for memory items
type ImportanceScore =
    { BaseImportance: float
      Recency: float
      Relevance: float
      SuccessWeight: float }

    member this.Total =
        this.BaseImportance + this.Recency + this.Relevance + this.SuccessWeight

/// A memory item with importance tracking
type MemoryEntry<'T> =
    { Content: 'T
      CreatedAt: DateTime
      LastAccessedAt: DateTime
      AccessCount: int
      Importance: ImportanceScore
      Tags: string list }

    /// Calculate decayed importance based on age
    member this.DecayedImportance(currentTime: DateTime) =
        let age = currentTime - this.CreatedAt
        let hoursSinceCreation = age.TotalHours

        // Exponential decay: half-life of 24 hours
        let decayFactor = Math.Pow(0.5, hoursSinceCreation / 24.0)

        // Boost for recent access
        let accessAge = (currentTime - this.LastAccessedAt).TotalHours

        let accessBoost =
            if accessAge < 1.0 then 0.3
            elif accessAge < 6.0 then 0.1
            else 0.0

        (this.Importance.Total * decayFactor) + accessBoost

/// Working Memory with capacity limits and importance-based pruning
type WorkingMemory<'T>(capacity: int) =

    let items = List<MemoryEntry<'T>>()
    let lockObj = obj ()

    /// Current number of items in memory
    member _.Count = lock lockObj (fun () -> items.Count)

    /// Current capacity
    member _.Capacity = capacity

    /// Add an item to working memory
    member this.Add(content: 'T, importance: ImportanceScore, ?tags: string list) =
        let entry =
            { Content = content
              CreatedAt = DateTime.UtcNow
              LastAccessedAt = DateTime.UtcNow
              AccessCount = 1
              Importance = importance
              Tags = tags |> Option.defaultValue [] }

        lock lockObj (fun () ->
            items.Add(entry)

            // Auto-prune if over capacity
            if items.Count > capacity then
                this.PruneInternal(capacity))

    /// Add a simple item with default importance
    member this.AddSimple(content: 'T, baseImportance: float) =
        let importance =
            { BaseImportance = baseImportance
              Recency = 0.0
              Relevance = 0.0
              SuccessWeight = 0.0 }

        this.Add(content, importance)

    /// Get all items sorted by current importance (descending)
    member _.GetAll() =
        let now = DateTime.UtcNow
        lock lockObj (fun () -> items |> Seq.toList |> List.sortByDescending (fun e -> e.DecayedImportance(now)))

    /// Get top N most important items
    member this.GetTop(n: int) = this.GetAll() |> List.truncate n

    /// Find items by tag
    member _.FindByTag(tag: string) =
        lock lockObj (fun () -> items |> Seq.filter (fun e -> e.Tags |> List.contains tag) |> Seq.toList)

    /// Update last accessed time for matching items
    member _.Touch(predicate: 'T -> bool) =
        lock lockObj (fun () ->
            for i in 0 .. items.Count - 1 do
                if predicate items.[i].Content then
                    items.[i] <-
                        { items.[i] with
                            LastAccessedAt = DateTime.UtcNow
                            AccessCount = items.[i].AccessCount + 1 })

    /// Prune to target size, keeping highest importance items
    member private this.PruneInternal(targetSize: int) =
        if items.Count <= targetSize then
            ()
        else
            let now = DateTime.UtcNow

            let sorted =
                items
                |> Seq.toList
                |> List.sortByDescending (fun e -> e.DecayedImportance(now))
                |> List.truncate targetSize

            items.Clear()

            for item in sorted do
                items.Add(item)

    /// Explicit prune operation
    member this.Prune(?targetSize: int) =
        let target = targetSize |> Option.defaultValue capacity
        lock lockObj (fun () -> this.PruneInternal(target))

    /// Clear all memory
    member _.Clear() = lock lockObj (fun () -> items.Clear())

    /// Get memory statistics
    member _.Statistics() =
        lock lockObj (fun () ->
            if items.Count = 0 then
                "{empty}"
            else
                let now = DateTime.UtcNow
                let avgImportance = items |> Seq.averageBy (fun e -> e.DecayedImportance(now))

                let oldestAge =
                    items |> Seq.map (fun e -> (now - e.CreatedAt).TotalHours) |> Seq.max

                $"[Memory] Count: %d{items.Count}/%d{capacity} | Avg Importance: %.2f{avgImportance} | Oldest: %.1f{oldestAge}h"
        )
