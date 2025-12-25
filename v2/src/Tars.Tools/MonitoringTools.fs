/// <summary>
/// Tools for system monitoring and observability.
/// </summary>
module Tars.Tools.MonitoringTools

open System.Threading.Tasks
open Tars.Core
open Tars.Tools
open System.Text.Json

/// <summary>
/// Performs a basic health check of the system.
/// Returns true if the system is responding, false otherwise.
/// </summary>
[<TarsToolAttribute("health_check", "Checks if the system is healthy. No input required.")>]
let health_check () : Task<bool> =
    task {
        // In a real scenario, this would check DB connection, LLM connectivity, etc.
        // For now, it just confirms the agent is running.
        return true
    }

/// <summary>
/// Retrieves a snapshot of current system metrics.
/// Returns a list of metric events captured recently.
/// </summary>
[<TarsToolAttribute("get_metrics", "Retrieves system metrics/telemetry. No input required.")>]
let get_metrics () : Task<Metrics.MetricEvent list> = task { return Metrics.snapshot () }

/// <summary>
/// Lists the most recent tool execution failures.
/// </summary>
[<TarsToolAttribute("list_tool_errors", "Lists recent tool execution failures. Input JSON: { \"count\": 10 }")>]
let list_tool_errors (args: string) =
    task {
        let count = 
            try
                let doc = JsonDocument.Parse(args).RootElement
                match doc.TryGetProperty("count") with
                | (true, p) -> p.GetInt32()
                | _ -> 10
            with _ -> 10
        
        let failures = Tars.Core.ToolLedger.getFailures count
        if failures.IsEmpty then
            return "No tool failures recorded."
        else
            let lines = 
                failures 
                |> List.map (fun (f: Tars.Core.ToolExecutionRecord) -> 
                    let err = f.Output |> Option.defaultValue "Unknown error"
                    sprintf "- [%s] **%s**: %s\n  Category: %A, Latency: %.1fms" (f.Timestamp.ToString("T")) f.ToolName err f.ErrorCategory f.DurationMs)
                |> String.concat "\n"
            return sprintf "**Last %d Tool Failures:**\n%s" failures.Length lines
    }

/// <summary>
/// Provides a comprehensive status report of all registered tools.
/// </summary>
[<TarsToolAttribute("report_status", "Provides a comprehensive status report of all registered tools. No input required.")>]
let report_status () =
    task {
        let stats = Tars.Core.ToolLedger.getStats()
        if stats.IsEmpty then
            return "No tool activity recorded yet."
        else
            let lines = 
                stats 
                |> List.map (fun (name, total, errors, avgDur, lastErr) -> 
                    let healthIcon = if errors = 0 then "✅" else "⚠️"
                    let status = if errors = 0 then "Healthy" else sprintf "Degraded (%d errors)" errors
                    let lastErrPart = 
                        match lastErr with
                        | Some (e: Tars.Core.ToolExecutionRecord) -> 
                            let msg = e.Output |> Option.defaultValue "Unknown"
                            sprintf "\n  ! Last Error: %s" msg
                        | None -> ""
                    sprintf "%s **%s**: %s\n  Calls: %d, Avg: %.1fms%s" healthIcon name status total avgDur lastErrPart)
                |> String.concat "\n\n"
            return sprintf "### TARS Tool Health Report\n\n%s" lines
    }
