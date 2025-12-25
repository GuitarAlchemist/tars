namespace Tars.Core

open System
open System.Collections.Concurrent

/// Failure taxonomy for tool execution
type ToolFailureCategory =
    | NoFailure
    | Timeout
    | DependencyFailure
    | ValidationFailure
    | LogicError
    | CircuitBreakerOpen
    | Unauthorized
    | Unknown

/// Record of a single tool execution
type ToolExecutionRecord =
    { ToolName: string
      Timestamp: DateTime
      DurationMs: float
      Input: string
      Output: string option // Success output or Error message
      ErrorCategory: ToolFailureCategory
      IsSuccess: bool
      CorrelationId: string option }

/// Global append-only ledger for tool execution auditing
module ToolLedger =

    // Last N records (ring buffer style)
    let private MAX_RECORDS = 500
    let private records = ConcurrentQueue<ToolExecutionRecord>()

    /// Record a tool execution
    let record
        (toolName: string)
        (input: string)
        (output: string)
        (durationMs: float)
        (isSuccess: bool)
        (category: ToolFailureCategory)
        (correlationId: string option)
        =
        // Truncate input/output for safety/memory
        let truncatedInput =
            if input.Length > 2048 then
                input.Substring(0, 2048) + "... [truncated]"
            else
                input

        let truncatedOutput =
            if output.Length > 2048 then
                output.Substring(0, 2048) + "... [truncated]"
            else
                output

        let item =
            { ToolName = toolName
              Timestamp = DateTime.UtcNow
              DurationMs = durationMs
              Input = truncatedInput
              Output = Some truncatedOutput
              ErrorCategory = category
              IsSuccess = isSuccess
              CorrelationId = correlationId }

        records.Enqueue(item)

        // Trim if too large
        let mutable dummy = Unchecked.defaultof<ToolExecutionRecord>

        while records.Count > MAX_RECORDS && records.TryDequeue(&dummy) do
            ()

        // Also push to standard metrics if available
        Metrics.record
            toolName
            (if isSuccess then "success" else "failure")
            durationMs
            None
            (Map.ofList [ ("category", category.ToString()) ])

    /// Get all records
    let getRecords () = records.ToArray() |> Array.toList

    /// Get last N failures
    let getFailures count =
        records.ToArray()
        |> Array.filter (fun r -> not r.IsSuccess)
        |> Array.sortByDescending (fun r -> r.Timestamp)
        |> Array.truncate count
        |> Array.toList

    /// Get health stats per tool
    let getStats () =
        records.ToArray()
        |> Array.groupBy (fun r -> r.ToolName)
        |> Array.map (fun (name, rs) ->
            let total = rs.Length
            let successes = rs |> Array.filter (fun r -> r.IsSuccess) |> Array.length
            let errors = total - successes
            let avgDuration = rs |> Array.averageBy (fun r -> r.DurationMs)

            let lastError =
                rs
                |> Array.filter (fun r -> not r.IsSuccess)
                |> Array.sortByDescending (fun r -> r.Timestamp)
                |> Array.tryHead

            (name, total, errors, avgDuration, lastError))
        |> Array.sortBy (fun (name, _, _, _, _) -> name)
        |> Array.toList
