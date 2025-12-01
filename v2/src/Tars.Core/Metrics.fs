namespace Tars.Core

open System
open System.Collections.Concurrent
open System.IO

/// <summary>
/// Lightweight in-memory metrics recorder for agent events.
/// Designed to be cheap to call and flushable to CSV for time-series tracking.
/// </summary>
module Metrics =

    type MetricEvent =
        { Timestamp: DateTime
          AgentId: AgentId option
          Kind: string
          Status: string
          DurationMs: float
          Details: Map<string, string> }

    let private events = ConcurrentQueue<MetricEvent>()
    let mutable private autoFlushTarget: string option = None

    let private sanitize (value: string) =
        if String.IsNullOrWhiteSpace value then
            ""
        else
            "\"" + value.Replace("\"", "\"\"") + "\""

    /// <summary>Record a metric event.</summary>
    let record (kind: string) (status: string) (durationMs: float) (agentId: AgentId option) (details: Map<string, string>) =
        events.Enqueue(
            { Timestamp = DateTime.UtcNow
              AgentId = agentId
              Kind = kind
              Status = status
              DurationMs = durationMs
              Details = details }
        )

    /// <summary>Record a metric event with optional parameters.</summary>
    let recordSimple (kind: string) (status: string) (agentId: AgentId option) (durationMs: float option) (details: Map<string, string> option) =
        let duration = defaultArg durationMs 0.0
        let detailMap = defaultArg details Map.empty
        record kind status duration agentId detailMap

    /// <summary>Return current metrics snapshot.</summary>
    let snapshot () = events.ToArray() |> Array.toList

    /// <summary>Clear recorded metrics.</summary>
    let clear () =
        let mutable tmp = Unchecked.defaultof<MetricEvent>
        while events.TryDequeue(&tmp) do
            ()

    /// <summary>Flush metrics to CSV for offline tracking.</summary>
    let dumpCsv (path: string) =
        let header = "timestamp,agent_id,kind,status,duration_ms,details"

        let lines =
            events
            |> Seq.map (fun e ->
                let agentId = e.AgentId |> Option.map (fun (AgentId id) -> id.ToString()) |> Option.defaultValue ""
                let detailText =
                    if e.Details.IsEmpty then
                        ""
                    else
                        e.Details
                        |> Seq.map (fun kv -> $"{kv.Key}={kv.Value}")
                        |> String.concat ";"

                $"{e.Timestamp:o},{sanitize agentId},{sanitize e.Kind},{sanitize e.Status},{e.DurationMs},{sanitize detailText}")
            |> Seq.toList

        File.WriteAllLines(path, header :: lines)

    /// <summary>Enable automatic flush on process exit (or call manually).</summary>
    let enableAutoFlush (path: string) =
        autoFlushTarget <- Some path
        AppDomain.CurrentDomain.ProcessExit.Add(fun _ -> dumpCsv path)

    do
        match Environment.GetEnvironmentVariable "TARS_METRICS_PATH" with
        | null
        | "" -> ()
        | path -> enableAutoFlush path
