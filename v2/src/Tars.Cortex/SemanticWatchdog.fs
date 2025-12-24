namespace Tars.Cortex

// SemanticWatchdog - Monitors agent behavior for anomalies
// Part of v2.2 Cognitive Patterns

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

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
          SimilarityThreshold = 0.95 // High threshold for semantic repetition
          EnableAutoIntervention = false }

/// Semantic Watchdog - Pattern 8 from research
/// Monitors agent behavior for signs of:
/// - Runaway loops (repetitive actions)
/// - Budget explosions (excessive token usage)
/// - Semantic drift (output deviating from intent)
type SemanticWatchdog(config: WatchdogConfig) =
    let alerts = System.Collections.Concurrent.ConcurrentBag<WatchdogAlert>()
    let tokenHistory = System.Collections.Concurrent.ConcurrentQueue<int * DateTime>()
    // Store both text and its embedding
    let responseHistory =
        System.Collections.Concurrent.ConcurrentQueue<string * float32[] option>()

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

    /// Record a response for loop detection (Legacy synchronous version)
    member this.RecordResponse(response: string) =
        responseHistory.Enqueue(response, None)

        // Keep only recent responses
        while responseHistory.Count > config.MaxSimilarResponses * 2 do
            responseHistory.TryDequeue() |> ignore

        // Check for repetitive responses (simple heuristic: exact match)
        let recentResponses = responseHistory |> Seq.toList

        let similar =
            recentResponses |> List.filter (fun (r, _) -> r = response) |> List.length

        if similar >= config.MaxSimilarResponses then
            this.RaiseAlert(
                Warning,
                "Repetitive response detected",
                $"Agent produced {similar} identical responses",
                "Agent may be stuck in a loop - consider intervention",
                Map [ "similar_count", float similar ]
            )

    /// Record and analyze response using embeddings for neural-symbolic loop detection
    member this.RecordResponseAsync(response: string, llm: ILlmService) =
        async {
            // 1. Get embedding for the new response
            let! embedding = llm.EmbedAsync response |> Async.AwaitTask

            responseHistory.Enqueue(response, Some embedding)

            // 2. Keep only recent responses
            while responseHistory.Count > config.MaxSimilarResponses * 2 do
                responseHistory.TryDequeue() |> ignore

            let recent = responseHistory |> Seq.toList

            // 3. Perform semantic similarity check against previous responses
            let semanticallySimilar =
                recent
                |> List.choose (fun (text, emb) ->
                    match emb with
                    | Some e ->
                        try
                            let sim = MetricSpace.cosineSimilarity e embedding

                            if float sim >= config.SimilarityThreshold then
                                Some text
                            else
                                None
                        with _ ->
                            None
                    | None -> if text = response then Some text else None)

            let similarCount = semanticallySimilar.Length

            if similarCount >= config.MaxSimilarResponses then
                let severity =
                    if similarCount > config.MaxSimilarResponses * 2 then
                        Critical
                    else
                        Warning

                this.RaiseAlert(
                    severity,
                    "Semantic loop detected",
                    $"Agent produced {similarCount} semantically similar responses (Similarity > {config.SimilarityThreshold})",
                    "Agent is repeating itself even if phrasing differs. Suggest resetting context or changing strategy.",
                    Map
                        [ "semantic_similar_count", float similarCount
                          "similarity_threshold", config.SimilarityThreshold ]
                )
        }

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
