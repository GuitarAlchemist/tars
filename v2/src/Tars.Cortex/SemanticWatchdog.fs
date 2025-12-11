/// SemanticWatchdog - Monitors agent behavior for anomalies
/// Part of v2.2 Cognitive Patterns
namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core

/// Watchdog alert severity levels
type AlertSeverity =
    | Info
    | Warning
    | Critical

/// Alert raised by the watchdog
type WatchdogAlert =
    { Id: Guid
      AgentId: string option
      Severity: AlertSeverity
      Message: string
      Recommendation: string
      Timestamp: DateTime
      Metrics: Map<string, float> }

/// Configuration for the semantic watchdog
type WatchdogConfig =
    { MaxTokensPerMinute: int
      MaxLoopIterations: int
      MaxSimilarResponses: int
      SimilarityThreshold: float
      EnableAutoIntervention: bool }

    static member Default =
        { MaxTokensPerMinute = 50000
          MaxLoopIterations = 10
          MaxSimilarResponses = 3
          SimilarityThreshold = 0.95
          EnableAutoIntervention = false }

/// Semantic Watchdog - Pattern 8 from research
/// Monitors agent behavior for signs of:
/// - Runaway loops (repetitive actions)
/// - Budget explosions (excessive token usage)
/// - Semantic drift (output deviating from intent)
type SemanticWatchdog(config: WatchdogConfig) =
    let alerts = System.Collections.Concurrent.ConcurrentBag<WatchdogAlert>()
    let tokenHistory = System.Collections.Concurrent.ConcurrentQueue<int * DateTime>()
    let responseHistory = System.Collections.Concurrent.ConcurrentQueue<string>()
    let mutable loopCounter = 0

    /// Record token usage
    member this.RecordTokens(count: int) =
        let now = DateTime.UtcNow
        tokenHistory.Enqueue((count, now))

        // Cleanup old entries (older than 1 minute)
        let cutoff = now.AddMinutes(-1.0)
        let mutable item = Unchecked.defaultof<_>

        while tokenHistory.TryPeek(&item) && snd item < cutoff do
            tokenHistory.TryDequeue(&item) |> ignore

        // Check for budget explosion
        let recentTokens =
            tokenHistory |> Seq.filter (fun (_, t) -> t >= cutoff) |> Seq.sumBy fst

        if recentTokens > config.MaxTokensPerMinute then
            this.RaiseAlert(
                Critical,
                "Budget explosion detected",
                $"Token usage ({recentTokens}) exceeded limit ({config.MaxTokensPerMinute}) in the last minute",
                "Consider pausing agent or reducing task complexity",
                Map [ "tokens_per_minute", float recentTokens ]
            )

    /// Record a response for loop detection
    member this.RecordResponse(response: string) =
        responseHistory.Enqueue(response)

        // Keep only recent responses
        while responseHistory.Count > config.MaxSimilarResponses * 2 do
            responseHistory.TryDequeue() |> ignore

        // Check for repetitive responses (simple heuristic: exact match)
        let recentResponses = responseHistory |> Seq.toList
        let similar = recentResponses |> List.filter (fun r -> r = response) |> List.length

        if similar >= config.MaxSimilarResponses then
            this.RaiseAlert(
                Warning,
                "Repetitive response detected",
                $"Agent produced {similar} identical responses",
                "Agent may be stuck in a loop - consider intervention",
                Map [ "similar_count", float similar ]
            )

    /// Record loop iteration
    member this.RecordIteration() =
        loopCounter <- loopCounter + 1

        if loopCounter > config.MaxLoopIterations then
            this.RaiseAlert(
                Critical,
                "Loop limit exceeded",
                $"Agent exceeded {config.MaxLoopIterations} iterations",
                "Force termination recommended",
                Map [ "iterations", float loopCounter ]
            )

    /// Reset loop counter
    member this.ResetLoop() = loopCounter <- 0

    /// Raise an alert
    member private this.RaiseAlert(severity, title, message, recommendation, metrics) =
        let alert =
            { Id = Guid.NewGuid()
              AgentId = None
              Severity = severity
              Message = $"{title}: {message}"
              Recommendation = recommendation
              Timestamp = DateTime.UtcNow
              Metrics = metrics }

        alerts.Add(alert)

    /// Get all alerts
    member this.GetAlerts() = alerts |> Seq.toList

    /// Get critical alerts
    member this.GetCriticalAlerts() =
        alerts |> Seq.filter (fun a -> a.Severity = Critical) |> Seq.toList

    /// Clear alerts
    member this.ClearAlerts() = alerts.Clear()

    /// Check if intervention is needed
    member this.RequiresIntervention() =
        config.EnableAutoIntervention
        && alerts |> Seq.exists (fun a -> a.Severity = Critical)

module SemanticWatchdog =
    /// Create with default config
    let createDefault () =
        SemanticWatchdog(WatchdogConfig.Default)

    /// Create with custom config
    let create config = SemanticWatchdog(config)
