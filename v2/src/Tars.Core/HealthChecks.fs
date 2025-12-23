/// Health Check System for Production Monitoring
/// Provides readiness, liveness, and dependency health probes
namespace Tars.Core

open System
open System.Threading.Tasks
open System.Diagnostics

/// Health check result
type HealthStatus =
    | Healthy
    | Degraded of reason: string
    | Unhealthy of reason: string

/// Health check entry
type HealthCheckResult =
    { Name: string
      Status: HealthStatus
      Duration: TimeSpan
      Tags: string list
      Data: Map<string, obj> }

/// Overall health report
type HealthReport =
    { Status: HealthStatus
      TotalDuration: TimeSpan
      Checks: HealthCheckResult list
      Timestamp: DateTime }

/// Health check interface
type IHealthCheck =
    abstract member Name: string
    abstract member Tags: string list
    abstract member CheckAsync: unit -> Task<HealthStatus>

/// Simple health check from a function
type FunctionalHealthCheck(name: string, tags: string list, check: unit -> Task<HealthStatus>) =
    interface IHealthCheck with
        member _.Name = name
        member _.Tags = tags
        member _.CheckAsync() = check ()

/// Health check registry
type HealthCheckRegistry() =

    let mutable checks: IHealthCheck list = []

    /// Register a health check
    member _.Register(check: IHealthCheck) = checks <- check :: checks

    /// Register a simple check
    member this.RegisterSimple(name: string, tags: string list, check: unit -> Task<HealthStatus>) =
        this.Register(FunctionalHealthCheck(name, tags, check))

    /// Run all health checks
    member _.RunAllAsync() =
        task {
            let sw = Stopwatch.StartNew()

            let! results =
                checks
                |> List.map (fun hc ->
                    task {
                        let checkSw = Stopwatch.StartNew()

                        try
                            let! status = hc.CheckAsync()
                            checkSw.Stop()

                            return
                                { Name = hc.Name
                                  Status = status
                                  Duration = checkSw.Elapsed
                                  Tags = hc.Tags
                                  Data = Map.empty }
                        with ex ->
                            checkSw.Stop()

                            return
                                { Name = hc.Name
                                  Status = Unhealthy $"Exception: {ex.Message}"
                                  Duration = checkSw.Elapsed
                                  Tags = hc.Tags
                                  Data = Map.empty }
                    })
                |> Task.WhenAll

            sw.Stop()

            let overallStatus =
                if
                    results
                    |> Array.exists (fun r ->
                        match r.Status with
                        | Unhealthy _ -> true
                        | _ -> false)
                then
                    Unhealthy "One or more checks failed"
                elif
                    results
                    |> Array.exists (fun r ->
                        match r.Status with
                        | Degraded _ -> true
                        | _ -> false)
                then
                    Degraded "One or more checks degraded"
                else
                    Healthy

            return
                { Status = overallStatus
                  TotalDuration = sw.Elapsed
                  Checks = results |> Array.toList
                  Timestamp = DateTime.UtcNow }
        }

    /// Run checks with specific tags only
    member _.RunByTagAsync(tag: string) =
        task {
            let sw = Stopwatch.StartNew()
            let tagged = checks |> List.filter (fun c -> c.Tags |> List.contains tag)

            let! results =
                tagged
                |> List.map (fun hc ->
                    task {
                        let checkSw = Stopwatch.StartNew()

                        try
                            let! status = hc.CheckAsync()
                            checkSw.Stop()

                            return
                                { Name = hc.Name
                                  Status = status
                                  Duration = checkSw.Elapsed
                                  Tags = hc.Tags
                                  Data = Map.empty }
                        with ex ->
                            checkSw.Stop()

                            return
                                { Name = hc.Name
                                  Status = Unhealthy $"Exception: {ex.Message}"
                                  Duration = checkSw.Elapsed
                                  Tags = hc.Tags
                                  Data = Map.empty }
                    })
                |> Task.WhenAll

            sw.Stop()

            let overallStatus =
                if
                    results
                    |> Array.exists (fun r ->
                        match r.Status with
                        | Unhealthy _ -> true
                        | _ -> false)
                then
                    Unhealthy "One or more checks failed"
                elif
                    results
                    |> Array.exists (fun r ->
                        match r.Status with
                        | Degraded _ -> true
                        | _ -> false)
                then
                    Degraded "One or more checks degraded"
                else
                    Healthy

            return
                { Status = overallStatus
                  TotalDuration = sw.Elapsed
                  Checks = results |> Array.toList
                  Timestamp = DateTime.UtcNow }
        }

/// Common health check implementations
module HealthChecks =

    /// Always healthy (for basic liveness)
    let alwaysHealthy name =
        FunctionalHealthCheck(name, [ "liveness" ], fun () -> Task.FromResult(Healthy))

    /// Memory check - warn if usage too high
    let memoryCheck thresholdMB =
        FunctionalHealthCheck(
            "memory",
            [ "readiness" ],
            fun () ->
                task {
                    let memoryMB = float (GC.GetTotalMemory(false)) / 1024.0 / 1024.0

                    if memoryMB > thresholdMB then
                        return Degraded $"Memory usage {memoryMB:F1}MB exceeds threshold {thresholdMB}MB"
                    else
                        return Healthy
                }
        )

    /// Uptime check
    let uptimeCheck (minUptime: TimeSpan) (startTime: DateTime) =
        FunctionalHealthCheck(
            "uptime",
            [ "readiness" ],
            fun () ->
                task {
                    let uptime = DateTime.UtcNow - startTime

                    if uptime < minUptime then
                        return Degraded $"Uptime {uptime.TotalSeconds:F0}s below minimum {minUptime.TotalSeconds:F0}s"
                    else
                        return Healthy
                }
        )

    /// Custom dependency check
    let dependencyCheck name (check: unit -> Task<bool>) =
        FunctionalHealthCheck(
            name,
            [ "readiness"; "dependency" ],
            fun () ->
                task {
                    try
                        let! isHealthy = check ()

                        if isHealthy then
                            return Healthy
                        else
                            return Unhealthy "Dependency check returned false"
                    with ex ->
                        return Unhealthy $"Dependency error: {ex.Message}"
                }
        )

    /// Format health report as string
    let formatReport (report: HealthReport) =
        let statusIcon status =
            match status with
            | Healthy -> "✓"
            | Degraded _ -> "⚠"
            | Unhealthy _ -> "✗"

        let statusStr status =
            match status with
            | Healthy -> "Healthy"
            | Degraded reason -> $"Degraded: {reason}"
            | Unhealthy reason -> $"Unhealthy: {reason}"

        let lines =
            [ sprintf "Health Report [%s]" (report.Timestamp.ToString("HH:mm:ss"))
              sprintf "Overall: %s %s" (statusIcon report.Status) (statusStr report.Status)
              sprintf "Duration: %.1fms" report.TotalDuration.TotalMilliseconds
              ""
              "Checks:" ]

        let checkLines =
            report.Checks
            |> List.map (fun c ->
                sprintf
                    "  %s %s: %s (%.1fms)"
                    (statusIcon c.Status)
                    c.Name
                    (statusStr c.Status)
                    c.Duration.TotalMilliseconds)

        String.concat "\n" (lines @ checkLines)

    /// Format as JSON
    let formatJson (report: HealthReport) =
        let statusStr status =
            match status with
            | Healthy -> "healthy"
            | Degraded _ -> "degraded"
            | Unhealthy _ -> "unhealthy"

        let checksJson =
            report.Checks
            |> List.map (fun c ->
                sprintf
                    """{"name":"%s","status":"%s","durationMs":%d}"""
                    c.Name
                    (statusStr c.Status)
                    (int c.Duration.TotalMilliseconds))
            |> String.concat ","

        sprintf
            """{"status":"%s","totalDurationMs":%d,"timestamp":"%s","checks":[%s]}"""
            (statusStr report.Status)
            (int report.TotalDuration.TotalMilliseconds)
            (report.Timestamp.ToString("o"))
            checksJson
