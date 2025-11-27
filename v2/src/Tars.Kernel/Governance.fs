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



/// Manages the "Cognitive Budget" (Tokens/Cost/Time) for tasks.
/// Implements the "Resource Controller" pattern.
type BudgetGovernor(maxTokens: int<token>, maxCalls: int, maxTime: TimeSpan) =
    let usage = ConcurrentDictionary<Guid, int<token> * int * DateTime>() // Tokens, Calls, StartTime

    new(maxTokensArg: int) = BudgetGovernor(maxTokensArg * 1<token>, 100, TimeSpan.FromMinutes(5.0))

    member this.RecordUsage(correlationId: Guid, tokens: int<token>) =
        usage.AddOrUpdate(
            correlationId,
            (tokens, 1, DateTime.UtcNow),
            (fun _ (t, c, start) -> (t + tokens, c + 1, start))
        )
        |> ignore

    member this.GetRemainingBudget(correlationId: Guid) =
        match usage.TryGetValue(correlationId) with
        | true, (t, c, start) ->
            let remainingTokens = max 0<token> (maxTokens - t)
            let remainingCalls = max 0 (maxCalls - c)
            let elapsed = DateTime.UtcNow - start

            let remainingTime =
                if elapsed > maxTime then
                    TimeSpan.Zero
                else
                    maxTime - elapsed

            (remainingTokens, remainingCalls, remainingTime)
        | false, _ -> (maxTokens, maxCalls, maxTime)

    member this.CanSpend(correlationId: Guid, estimatedTokens: int<token>) =
        match usage.TryGetValue(correlationId) with
        | true, (t, c, start) ->
            let tokensOk = (t + estimatedTokens) <= maxTokens
            let callsOk = (c + 1) <= maxCalls
            let timeOk = (DateTime.UtcNow - start) <= maxTime
            tokensOk && callsOk && timeOk
        | false, _ ->
            // First usage, initialize start time implicitly on next record
            true
