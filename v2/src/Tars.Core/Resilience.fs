namespace Tars.Core

// <summary>
// Functional resilience patterns for Tars operations.
// Includes Retry with Exponential Backoff and Circuit Breaker logic.
// </summary>

open System
open System.Threading.Tasks

module Resilience =

    /// <summary>Configuration for retry behavior.</summary>
    type RetryPolicy =
        { MaxRetries: int
          BaseDelayMs: int
          MaxDelayMs: int
          Jitter: bool }

    /// <summary>Default retry policy (3 retries, start 100ms).</summary>
    let defaultRetry =
        { MaxRetries = 3
          BaseDelayMs = 100
          MaxDelayMs = 2000
          Jitter = true }

    /// <summary>Executes a task with retry logic.</summary>
    let retryAsync (policy: RetryPolicy) (operation: unit -> Task<'T>) : Task<'T> =
        let rec loop attempt =
            task {
                try
                    return! operation ()
                with ex ->
                    if attempt >= policy.MaxRetries then
                        return raise ex
                    else
                        let delay =
                            let baseDelay = float policy.BaseDelayMs * (2.0 ** float attempt)
                            let cappedDelay = min baseDelay (float policy.MaxDelayMs)

                            let jitter =
                                if policy.Jitter then
                                    Random.Shared.NextDouble() * 0.2 + 0.9
                                else
                                    1.0

                            int (cappedDelay * jitter)

                        do! Task.Delay delay
                        return! loop (attempt + 1)
            }

        loop 0

    /// <summary>State of a Circuit Breaker.</summary>
    type CircuitState =
        | Closed
        | Open of nextTry: DateTime
        | HalfOpen

    /// <summary>Simple functional Circuit Breaker (in-memory state).</summary>
    type CircuitBreaker(failureThreshold: int, openDuration: TimeSpan) =
        let mutable failures = 0
        let mutable state = Closed
        let lockObj = obj ()

        member this.State = state

        member this.ExecuteAsync(operation: unit -> Task<'T>) : Task<'T> =
            let shouldRun =
                lock lockObj (fun () ->
                    match state with
                    | Closed -> true
                    | Open nextTry ->
                        if DateTime.UtcNow >= nextTry then
                            state <- HalfOpen
                            true
                        else
                            false
                    | HalfOpen -> true // Tentative execution
                )

            if not shouldRun then
                Task.FromException<'T>(InvalidOperationException("CircuitBreaker is OPEN"))
            else
                task {
                    try
                        let! result = operation ()

                        lock lockObj (fun () ->
                            if state = HalfOpen then
                                state <- Closed

                            failures <- 0)

                        return result
                    with ex ->
                        lock lockObj (fun () ->
                            failures <- failures + 1

                            if failures >= failureThreshold then
                                state <- Open(DateTime.UtcNow.Add(openDuration))
                            else if state = HalfOpen then
                                state <- Open(DateTime.UtcNow.Add(openDuration)) // Immediate reopen on fail in half-open
                        )

                        return! Task.FromException<'T>(ex)
                }
