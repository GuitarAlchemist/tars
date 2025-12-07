/// <summary>
/// Tools for system monitoring and observability.
/// </summary>
module Tars.Tools.MonitoringTools

open System.Threading.Tasks
open Tars.Core

open Tars.Tools

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
