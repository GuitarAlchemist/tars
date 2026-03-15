namespace Tars.Evolution

/// Exports TARS meta-cognitive insights to ~/.tars/insights/ for cross-repo consumption.
/// Guitar Alchemist reads these to prioritize skill development and detect gaps.
module InsightExporter =

    open System
    open System.IO
    open System.Text.Json
    open Tars.Cortex
    open Tars.Cortex.WoTTypes

    let private jsonOptions =
        JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    let private insightsDir () =
        let dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "insights")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        dir

    // =========================================================================
    // Insight types — lightweight DTOs for cross-repo consumption
    // =========================================================================

    type PatternScoreInsight =
        { PatternKind: string
          Score: float
          GoalKeywords: string list }

    type GapInsight =
        { Domain: string
          FailureRate: float
          SampleSize: int
          Remedy: string
          DetectedAt: string }

    type PromotedPatternInsight =
        { PatternName: string
          Level: string
          LevelRank: int
          Score: float
          Weight: float
          Contexts: string list }

    type InsightSnapshot =
        { Timestamp: string
          TarsVersion: string
          PatternScores: PatternScoreInsight list
          Gaps: GapInsight list
          PromotedPatterns: PromotedPatternInsight list
          OutcomeSummary: Map<string, int>
          RecommendedActions: string list }

    // =========================================================================
    // Export functions
    // =========================================================================

    /// Build an insight snapshot from current TARS state.
    let buildSnapshot () : InsightSnapshot =
        // Pattern scores from selector
        let selector = PatternSelector.HistoryAwareSelector()
        let sampleGoals =
            [ "explain music theory"; "search for patterns"; "compare approaches"
              "explore alternatives"; "refactor code"; "route to agent" ]

        let patternScores =
            sampleGoals
            |> List.collect (fun goal ->
                let scores = (selector :> IPatternSelector).Score(goal)
                scores
                |> Map.toList
                |> List.map (fun (kind, score) ->
                    { PatternKind = sprintf "%A" kind
                      Score = score
                      GoalKeywords = goal.Split(' ') |> Array.toList }))
            |> List.groupBy (fun p -> p.PatternKind)
            |> List.map (fun (kind, entries) ->
                { PatternKind = kind
                  Score = entries |> List.averageBy (fun e -> e.Score)
                  GoalKeywords = entries |> List.collect (fun e -> e.GoalKeywords) |> List.distinct })

        // Gaps from pattern outcomes
        let outcomes = PatternOutcomeStore.loadAll ()
        let gapsByGoal =
            outcomes
            |> List.groupBy (fun o -> o.Goal)
            |> List.choose (fun (goal, entries) ->
                let fails = entries |> List.filter (fun e -> not e.Success) |> List.length
                let total = entries.Length
                let rate = float fails / float total
                if rate > 0.3 && total >= 2 then
                    Some { Domain = goal
                           FailureRate = rate
                           SampleSize = total
                           Remedy = if rate > 0.7 then "switch_pattern" else "improve_prompt"
                           DetectedAt = DateTime.UtcNow.ToString("o") }
                else None)

        // Promoted patterns from index
        let promoted =
            match PromotionIndex.load () with
            | Some index ->
                index.Entries
                |> List.map (fun e ->
                    { PatternName = e.PatternName
                      Level = sprintf "%A" e.Level
                      LevelRank = e.LevelRank
                      Score = e.Score
                      Weight = e.Weight
                      Contexts = e.Contexts })
            | None -> []

        // Outcome summary
        let outcomeSummary =
            outcomes
            |> List.countBy (fun o -> if o.Success then "success" else "failure")
            |> Map.ofList

        // Recommendations
        let recommendations =
            [ if gapsByGoal |> List.exists (fun g -> g.FailureRate > 0.5) then
                "High failure rate detected — consider using reasoning model for complex tasks"
              if promoted |> List.exists (fun p -> p.LevelRank >= 4) then
                "Grammar-rule patterns available — enable promotion boost in pattern selector"
              if outcomes.Length < 10 then
                "Limited outcome data — run more evolution cycles for reliable gap detection" ]

        { Timestamp = DateTime.UtcNow.ToString("o")
          TarsVersion = "2.0"
          PatternScores = patternScores
          Gaps = gapsByGoal
          PromotedPatterns = promoted
          OutcomeSummary = outcomeSummary
          RecommendedActions = recommendations }

    /// Export the current insight snapshot to disk.
    let export () : string =
        let snapshot = buildSnapshot ()
        let path = Path.Combine(insightsDir (), "latest.json")
        let json = JsonSerializer.Serialize(snapshot, jsonOptions)
        File.WriteAllText(path, json)

        // Also write a timestamped copy for history
        let histPath = Path.Combine(insightsDir (), $"snapshot_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json")
        File.WriteAllText(histPath, json)

        path

    /// Load the latest insight snapshot from disk (for cross-repo consumption).
    let loadLatest () : InsightSnapshot option =
        try
            let path = Path.Combine(insightsDir (), "latest.json")
            if File.Exists path then
                let json = File.ReadAllText path
                Some (JsonSerializer.Deserialize<InsightSnapshot>(json, jsonOptions))
            else None
        with _ -> None
