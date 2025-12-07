/// <summary>
/// Tools for defining and executing resilient operations.
/// </summary>
module Tars.Tools.ResilienceTools

open System.Threading.Tasks
open Tars.Core
open Tars.Core.Resilience

open Tars.Tools

/// <summary>
/// Wraps an operation with retry logic using exponential backoff.
/// Use this when calling external services that might fail transiently.
/// </summary>
[<TarsToolAttribute("retry_with_backoff",
                    "Retries an operation with exponential backoff. Input: { \"max_retries\": 3, \"base_delay_ms\": 100 }")>]
let retry_with_backoff (max_retries: int) (base_delay_ms: int) (operation: unit -> Task<'T>) : Task<'T> =
    let policy =
        { defaultRetry with
            MaxRetries = max_retries
            BaseDelayMs = base_delay_ms }

    retryAsync policy operation
