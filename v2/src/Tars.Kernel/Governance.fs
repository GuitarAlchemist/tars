namespace Tars.Kernel

open System
open System.Collections.Concurrent
open Tars.Core

/// Tracks the health of agents. If an agent fails repeatedly, it is "tripped" to prevent cascading failures.
/// Implements the "Jidoka" (Stop the Line) philosophy.
type CircuitBreaker(failureThreshold: int, resetTimeout: TimeSpan) =
    let failureCounts = ConcurrentDictionary<string, int>()
    let lastFailureTime = ConcurrentDictionary<string, DateTime>()

    /// Checks if an agent is healthy enough to receive messages
    member this.CanRequest(agentId: string) =
        match lastFailureTime.TryGetValue(agentId) with
        | true, lastFail ->
            if DateTime.UtcNow - lastFail > resetTimeout then
                // Reset if timeout passed (Half-Open logic simplified)
                failureCounts.TryRemove(agentId) |> ignore
                lastFailureTime.TryRemove(agentId) |> ignore
                true
            else
                // Check if failures exceed threshold
                match failureCounts.TryGetValue(agentId) with
                | true, count -> count < failureThreshold
                | false, _ -> true
        | false, _ -> true

    member this.RecordSuccess(agentId: string) =
        failureCounts.TryRemove(agentId) |> ignore
        lastFailureTime.TryRemove(agentId) |> ignore

    member this.RecordFailure(agentId: string) =
        failureCounts.AddOrUpdate(agentId, 1, (fun _ c -> c + 1)) |> ignore

        lastFailureTime.AddOrUpdate(agentId, DateTime.UtcNow, (fun _ _ -> DateTime.UtcNow))
        |> ignore
