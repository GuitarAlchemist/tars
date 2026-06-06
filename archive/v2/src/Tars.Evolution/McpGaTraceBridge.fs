namespace Tars.Evolution

/// MCP tools for ingesting GA trace artifacts into the TARS promotion pipeline.
/// Enables cross-repo pattern discovery: GA orchestrator events → TARS promotion staircase.
module McpGaTraceBridge =

    open System
    open System.Text.Json
    open System.Text.Json.Serialization
    open Tars.Core

    let private jsonOptions =
        let o = JsonSerializerOptions(WriteIndented = true)
        o.Converters.Add(JsonFSharpConverter())
        o

    // =========================================================================
    // Tool: ingest_ga_traces — Import GA traces into TARS promotion pipeline
    // =========================================================================

    type IngestInput = {
        Count: int option
        SinceIso: string option
        MinOccurrences: int option
    }

    type IngestResponse = {
        TracesRead: int
        ArtifactsIngested: int
        PipelineResults: int
        Promotions: int
        Rejections: int
        Details: string list
    }

    let private ingestGaTraces (input: string) : Result<string, string> =
        try
            let req =
                try JsonSerializer.Deserialize<IngestInput>(input, jsonOptions)
                with _ -> { Count = None; SinceIso = None; MinOccurrences = None }

            let count = req.Count |> Option.defaultValue 100
            let minOcc = req.MinOccurrences |> Option.defaultValue 3
            let since =
                req.SinceIso
                |> Option.bind (fun s ->
                    match DateTimeOffset.TryParse(s) with
                    | true, dto -> Some dto.UtcDateTime
                    | false, _ -> None)

            let artifacts = GaTraceBridge.ingest count since

            if artifacts.IsEmpty then
                Result.Ok (JsonSerializer.Serialize(
                    { TracesRead = 0; ArtifactsIngested = 0; PipelineResults = 0
                      Promotions = 0; Rejections = 0
                      Details = ["No GA trace files found in ~/.ga/traces/"] }, jsonOptions))
            else
                let results = PromotionPipeline.run minOcc artifacts
                let promotions = results |> List.filter (fun r ->
                    match r.Decision with GovernanceDecision.Approve _ -> true | _ -> false)
                let rejections = results |> List.filter (fun r ->
                    match r.Decision with GovernanceDecision.Reject _ -> true | _ -> false)

                let details =
                    results |> List.map (fun r ->
                        let decision = match r.Decision with
                                       | GovernanceDecision.Approve reason -> $"APPROVED: {reason}"
                                       | GovernanceDecision.Reject reason -> $"REJECTED: {reason}"
                                       | GovernanceDecision.Defer reason -> $"DEFERRED: {reason}"
                        $"{r.Candidate.Record.PatternName} → {PromotionLevel.label r.Candidate.ProposedLevel}: {decision}")

                let response = {
                    TracesRead = artifacts.Length
                    ArtifactsIngested = artifacts.Length
                    PipelineResults = results.Length
                    Promotions = promotions.Length
                    Rejections = rejections.Length
                    Details = details
                }
                Result.Ok (JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to ingest GA traces: {ex.Message}"

    // =========================================================================
    // Tool: ga_trace_stats — View GA trace bridge statistics
    // =========================================================================

    type StatsResponse = {
        FileCount: int
        Oldest: string option
        Newest: string option
        EventTypes: Map<string, int>
    }

    let private gaTraceStats (_input: string) : Result<string, string> =
        try
            let s = GaTraceBridge.stats ()
            let response = {
                FileCount = s.FileCount
                Oldest = s.Oldest
                Newest = s.Newest
                EventTypes = s.EventTypes
            }
            Result.Ok (JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to get GA trace stats: {ex.Message}"

    // =========================================================================
    // Tool registration
    // =========================================================================

    // =========================================================================
    // Tool: promotion_index — View ranked promotion index
    // =========================================================================

    let private promotionIndexTool (_input: string) : Result<string, string> =
        try
            let index = PromotionIndex.refresh ()
            Result.Ok (JsonSerializer.Serialize(index, jsonOptions))
        with ex ->
            Result.Error $"Failed to build promotion index: {ex.Message}"

    // =========================================================================
    // Tool: export_insights — Export meta-cognitive insights for GA consumption
    // =========================================================================

    let private exportInsightsTool (_input: string) : Result<string, string> =
        try
            let path = InsightExporter.export ()
            let snapshot = InsightExporter.loadLatest ()
            match snapshot with
            | Some s ->
                let summary =
                    {| ExportedTo = path
                       PatternScoreCount = s.PatternScores.Length
                       GapCount = s.Gaps.Length
                       PromotedCount = s.PromotedPatterns.Length
                       OutcomeSummary = s.OutcomeSummary
                       Recommendations = s.RecommendedActions |}
                Result.Ok (JsonSerializer.Serialize(summary, jsonOptions))
            | None ->
                Result.Ok $"{{\"ExportedTo\": \"{path}\", \"Note\": \"Export succeeded but re-read failed\"}}"
        with ex ->
            Result.Error $"Failed to export insights: {ex.Message}"

    // =========================================================================
    // Tool registration
    // =========================================================================

    /// Create GA trace bridge MCP tools
    let createTools () : Tool list =
        [ { Name = "ingest_ga_traces"
            Description = "Import GA orchestrator trace artifacts from ~/.ga/traces/ into the TARS promotion pipeline. Discovers cross-repo patterns (routing decisions, agent responses, skill executions) and promotes recurring ones up the staircase. Input: {\"Count\": 100, \"SinceIso\": \"2025-01-01T00:00:00Z\", \"MinOccurrences\": 3} (all optional)."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return ingestGaTraces input } }
          { Name = "ga_trace_stats"
            Description = "View GA trace bridge statistics: file count, date range, and event type distribution from ~/.ga/traces/. No input required."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return gaTraceStats input } }
          { Name = "promotion_index"
            Description = "View the ranked promotion index: all discovered patterns sorted by promotion level, score, and weight. Includes cross-repo GA patterns. The index drives pattern selection in the WoT agent. No input required."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return promotionIndexTool input } }
          { Name = "export_insights"
            Description = "Export TARS meta-cognitive insights to ~/.tars/insights/ for cross-repo consumption. Includes pattern scores, capability gaps, promoted patterns, outcome summary, and recommendations. Guitar Alchemist reads these to prioritize skill development. No input required."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return exportInsightsTool input } } ]
