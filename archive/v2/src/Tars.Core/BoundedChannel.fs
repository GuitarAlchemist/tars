namespace Tars.Core

// BoundedChannel - Backpressure-aware message channels
// Phase 6.1 of the TARS v2 Roadmap

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks

/// Behavior when channel is full
type FullBehavior =
    | RejectNew
    | DropOldest
    | Block

/// Result of write attempt
type WriteResult =
    | Written
    | Rejected of reason: string
    | Dropped of droppedItem: obj

/// Statistics for the bounded channel
type ChannelStats =
    { TotalWrites: int64
      TotalReads: int64
      TotalRejections: int64
      TotalDropped: int64
      CurrentCount: int
      HighWaterMark: int }

/// A bounded channel with backpressure support
type BoundedChannel<'T>(capacity: int, fullBehavior: FullBehavior) =
    let queue = ConcurrentQueue<'T>()
    let mutable count = 0
    let lockObj = obj ()
    let notEmpty = new SemaphoreSlim(0, capacity)
    let notFull = new SemaphoreSlim(capacity, capacity)

    // Stats
    let mutable totalWrites = 0L
    let mutable totalReads = 0L
    let mutable totalRejections = 0L
    let mutable totalDropped = 0L
    let mutable highWaterMark = 0

    /// Capacity of the channel
    member _.Capacity = capacity

    /// Current item count
    member _.Count = count

    /// Whether channel is full
    member _.IsFull = count >= capacity

    /// Whether channel is empty
    member _.IsEmpty = count = 0

    /// Get channel statistics
    member _.Stats =
        { TotalWrites = totalWrites
          TotalReads = totalReads
          TotalRejections = totalRejections
          TotalDropped = totalDropped
          CurrentCount = count
          HighWaterMark = highWaterMark }

    /// Internal read helper (assumes notEmpty permit is already acquired/handled)
    member private _.doRead() : 'T option =
        lock lockObj (fun () ->
            match queue.TryDequeue() with
            | true, item ->
                count <- count - 1
                Interlocked.Increment(&totalReads) |> ignore

                try
                    notFull.Release() |> ignore
                with _ ->
                    ()

                Some item
            | false, _ -> None)

    /// Try to write an item (non-blocking)
    member _.TryWrite(item: 'T) : WriteResult =
        // 1. Try to acquire space permit
        if notFull.Wait(0) then
            lock lockObj (fun () ->
                queue.Enqueue(item)
                count <- count + 1
                highWaterMark <- max highWaterMark count
                Interlocked.Increment(&totalWrites) |> ignore
                notEmpty.Release() |> ignore
                Written)
        else
            match fullBehavior with
            | Block -> Rejected "Channel full (blocking mode)"
            | RejectNew ->
                Interlocked.Increment(&totalRejections) |> ignore
                Rejected "Channel full"
            | DropOldest ->
                lock lockObj (fun () ->
                    // Attempt to make space by dropping
                    match queue.TryDequeue() with
                    | true, dropped ->
                        // We dequeued (logic: +1 empty slot) and enqueue (logic: -1 empty slot).
                        // Net change to notFull is 0. So we don't touch semaphore.
                        Interlocked.Increment(&totalDropped) |> ignore
                        queue.Enqueue(item)
                        Interlocked.Increment(&totalWrites) |> ignore
                        Dropped(box dropped)
                    | false, _ ->
                        // Queue is empty but we couldn't acquire permit?
                        // This implies pure reservation race (WriteAsync holds permit but hasn't written).
                        Interlocked.Increment(&totalRejections) |> ignore
                        Rejected "Channel reserved")

    /// Write an item with blocking
    member this.WriteAsync(item: 'T) : Task<WriteResult> =
        task {
            match fullBehavior with
            | Block ->
                do! notFull.WaitAsync()

                lock lockObj (fun () ->
                    queue.Enqueue(item)
                    count <- count + 1
                    highWaterMark <- max highWaterMark count
                    Interlocked.Increment(&totalWrites) |> ignore
                    notEmpty.Release() |> ignore)

                return Written
            | _ -> return this.TryWrite(item)
        }

    /// Try to read an item (non-blocking)
    member this.TryRead() : 'T option =
        // Must acquire permit first
        if notEmpty.Wait(0) then
            match this.doRead () with
            | Some item -> Some item
            | None ->
                // Invariant broken - permit acquired but no item
                try
                    notEmpty.Release() |> ignore
                with _ ->
                    ()

                None
        else
            None

    /// Read an item with blocking
    member this.ReadAsync() : Task<'T> =
        task {
            do! notEmpty.WaitAsync()

            match this.doRead () with
            | Some item -> return item
            | None ->
                // Should hopefully not happen if invariants hold
                try
                    notEmpty.Release() |> ignore
                with _ ->
                    ()

                return raise (InvalidOperationException("Channel empty after permit acquired"))
        }

    /// Read an item with timeout
    member this.ReadAsync(timeout: TimeSpan) : Task<'T option> =
        task {
            let! acquired = notEmpty.WaitAsync(timeout)

            if acquired then
                match this.doRead () with
                | Some item -> return Some item
                | None ->
                    try
                        notEmpty.Release() |> ignore
                    with _ ->
                        ()

                    return None
            else
                return None
        }

    /// Drain all items from the channel
    member _.DrainAll() : 'T list =
        lock lockObj (fun () ->
            let items = queue |> Seq.toList
            let drainedCount = items.Length

            if drainedCount > 0 then
                queue.Clear()
                count <- 0
                Interlocked.Add(&totalReads, int64 drainedCount) |> ignore
                // Release permits for space (notFull)
                try
                    notFull.Release(drainedCount) |> ignore
                with _ ->
                    ()
                // Consume permits for items (notEmpty)
                // We trust drainedCount matches available permits
                for _ in 1..drainedCount do
                    notEmpty.Wait(0) |> ignore

            items)

    interface IDisposable with
        member _.Dispose() =
            notEmpty.Dispose()
            notFull.Dispose()

module BoundedChannel =

    /// Create a bounded channel with reject behavior
    let createRejecting<'T> capacity =
        new BoundedChannel<'T>(capacity, RejectNew)

    /// Create a bounded channel with drop-oldest behavior
    let createDropping<'T> capacity =
        new BoundedChannel<'T>(capacity, DropOldest)

    /// Create a bounded channel with blocking behavior
    let createBlocking<'T> capacity = new BoundedChannel<'T>(capacity, Block)
