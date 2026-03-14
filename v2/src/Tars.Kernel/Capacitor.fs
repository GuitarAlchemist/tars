namespace Tars.Kernel

open System
open System.Threading.Tasks

/// <summary>
/// A capacitor that buffers items and releases them in batches.
/// Useful for smoothing out bursts of activity or efficient batch processing.
/// </summary>
type BufferAgent<'T>(capacity: int, timeWindow: TimeSpan, onFlush: 'T list -> Task) =

    let agent =
        MailboxProcessor.Start(fun inbox ->
            let rec loop (buffer: 'T list) (lastFlush: DateTime) =
                async {
                    // Calculate timeout for next flush
                    let elapsed = DateTime.UtcNow - lastFlush
                    let remaining = timeWindow - elapsed

                    let timeout =
                        if remaining.TotalMilliseconds > 0.0 then
                            int remaining.TotalMilliseconds
                        else
                            0

                    // Wait for message or timeout
                    let! msg = inbox.TryReceive(timeout)

                    match msg with
                    | Some item ->
                        let newBuffer = item :: buffer

                        if newBuffer.Length >= capacity then
                            // Flush due to capacity
                            try
                                do! onFlush (List.rev newBuffer) |> Async.AwaitTask
                            with _ ->
                                () // Suppress errors in flush to keep agent alive

                            return! loop [] DateTime.UtcNow
                        else
                            return! loop newBuffer lastFlush
                    | None ->
                        // Timeout occurred
                        if not buffer.IsEmpty then
                            try
                                do! onFlush (List.rev buffer) |> Async.AwaitTask
                            with _ ->
                                ()

                            return! loop [] DateTime.UtcNow
                        else
                            // If empty and timed out, just reset timer (effectively waiting for next item)
                            // Actually, if empty, we can wait indefinitely?
                            // No, because we want to respect the time window relative to the *first* item.
                            // But if buffer is empty, there is no first item.
                            // So if buffer is empty, we should wait indefinitely for the first item.
                            let! firstItem = inbox.Receive()
                            return! loop [ firstItem ] DateTime.UtcNow
                }
            // Initial state: empty buffer, so wait for first item
            async {
                let! firstItem = inbox.Receive()
                return! loop [ firstItem ] DateTime.UtcNow
            })

    /// <summary>
    /// Adds an item to the buffer. Thread-safe.
    /// </summary>
    member this.Accumulate(item: 'T) = agent.Post(item)
