/// Gates - Transistor-like flow control primitives
/// Phase 6.7.3 of the TARS v2 Roadmap
namespace Tars.Core

open System
open System.Threading
open System.Threading.Tasks

/// Result of attempting to pass through a gate
type GateResult =
    | Passed
    | Blocked of reason: string
    | TimedOut

/// A condition-based gate that controls message/task flow
type Gate =
    { Name: string
      Condition: unit -> Task<bool>
      OnPass: unit -> Task<unit>
      OnBlock: unit -> Task<unit> }

module Gate =

    /// Create a gate with a simple condition
    let create name condition =
        { Name = name
          Condition = condition
          OnPass = fun () -> task { return () }
          OnBlock = fun () -> task { return () } }

    /// Try to pass through the gate
    let tryPassAsync (gate: Gate) : Task<GateResult> =
        task {
            let! canPass = gate.Condition()

            if canPass then
                do! gate.OnPass()
                return Passed
            else
                do! gate.OnBlock()
                return Blocked $"Gate '{gate.Name}' condition not met"
        }

    /// Wait for gate to open with timeout
    let waitForOpenAsync (gate: Gate) (timeout: TimeSpan) (pollInterval: TimeSpan) : Task<GateResult> =
        let deadline = DateTime.UtcNow + timeout

        let rec loop () =
            task {
                if DateTime.UtcNow >= deadline then
                    return TimedOut
                else
                    let! canPass = gate.Condition()

                    if canPass then
                        do! gate.OnPass()
                        return Passed
                    else
                        do! Task.Delay(pollInterval)
                        return! loop ()
            }

        loop ()

/// A mutex gate that limits concurrent access
type MutexGate(maxConcurrent: int) =
    let semaphore = new SemaphoreSlim(maxConcurrent, maxConcurrent)
    let mutable totalAcquires = 0
    let mutable totalReleases = 0

    /// Current number of available slots
    member _.Available = semaphore.CurrentCount

    /// Maximum concurrent allowed
    member _.MaxConcurrent = maxConcurrent

    /// Total successful acquires
    member _.TotalAcquires = totalAcquires

    /// Total releases
    member _.TotalReleases = totalReleases

    /// Try to acquire access immediately
    member _.TryAcquire() =
        let acquired = semaphore.Wait(0)

        if acquired then
            Interlocked.Increment(&totalAcquires) |> ignore

        acquired

    /// Try to acquire access with timeout
    member _.TryAcquireAsync(timeout: TimeSpan) : Task<bool> =
        task {
            let! acquired = semaphore.WaitAsync(timeout)

            if acquired then
                Interlocked.Increment(&totalAcquires) |> ignore

            return acquired
        }

    /// Acquire access (blocking)
    member _.AcquireAsync() : Task<unit> =
        task {
            do! semaphore.WaitAsync()
            Interlocked.Increment(&totalAcquires) |> ignore
        }

    /// Release access
    member _.Release() =
        semaphore.Release() |> ignore
        Interlocked.Increment(&totalReleases) |> ignore

    /// Execute action with mutex protection
    member this.WithLockAsync(action: unit -> Task<'T>) : Task<'T> =
        task {
            do! this.AcquireAsync()

            try
                return! action ()
            finally
                this.Release()
        }

    interface IDisposable with
        member _.Dispose() = semaphore.Dispose()

/// A join gate that waits for multiple signals
type JoinGate(requiredSignals: int) =
    let mutable signals = 0
    let lockObj = obj ()
    let completionSource = TaskCompletionSource<unit>()

    /// Current signal count
    member _.SignalCount = signals

    /// Required signals to open
    member _.RequiredSignals = requiredSignals

    /// Whether gate is open
    member _.IsOpen = signals >= requiredSignals

    /// Send a signal
    member _.Signal() =
        lock lockObj (fun () ->
            signals <- signals + 1

            if signals >= requiredSignals then
                completionSource.TrySetResult() |> ignore)

    /// Wait for gate to open
    member _.WaitAsync() : Task<unit> = task { do! completionSource.Task }

    /// Wait with timeout
    member _.WaitAsync(timeout: TimeSpan) : Task<bool> =
        task {
            let timeoutTask = Task.Delay(timeout)
            let! completed = Task.WhenAny(completionSource.Task, timeoutTask)
            return completed = completionSource.Task
        }

    /// Reset the gate
    member _.Reset() = lock lockObj (fun () -> signals <- 0)
