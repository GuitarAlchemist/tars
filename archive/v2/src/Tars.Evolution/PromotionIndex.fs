namespace Tars.Evolution

/// Bridges the promotion pipeline output to the agent execution layer.
/// Reads promoted patterns from recurrence/lineage stores and exposes
/// a ranked index that pattern selectors can query at runtime.
module PromotionIndex =

    open System
    open System.IO
    open System.Text.Json
    open System.Text.Json.Serialization

    // ── Index entry: what the agent layer needs to know ─────────────

    type IndexEntry = {
        PatternId: string
        PatternName: string
        Level: PromotionLevel
        LevelRank: int
        Score: float
        OccurrenceCount: int
        Contexts: string list
        RollbackExpansion: string option
        Weight: float
        LastPromoted: DateTime
    }

    type PromotionIndexData = {
        Entries: IndexEntry list
        GeneratedAt: DateTime
        PatternCount: int
    }

    let private jsonOptions =
        let o = JsonSerializerOptions(WriteIndented = true)
        o.Converters.Add(JsonFSharpConverter())
        o

    let private indexPath () =
        let dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tars", "promotion")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        Path.Combine(dir, "index.json")

    // ── Build index from live promotion stores ──────────────────────

    /// Build the index from current recurrence records and weights.
    let build () : PromotionIndexData =
        let records = PromotionPipeline.getRecurrenceRecords ()
        let weights = WeightedGrammar.load ()
        let lineage = PromotionPipeline.getLineageRecords ()

        // Find the most recent rollback expansion from lineage for each pattern
        let rollbackByPattern =
            lineage
            |> List.choose (fun l ->
                l.RollbackExpansion |> Option.map (fun rb -> l.PatternId, rb))
            |> List.groupBy fst
            |> List.map (fun (pid, pairs) -> pid, pairs |> List.last |> snd)
            |> Map.ofList

        let entries =
            records
            |> List.map (fun r ->
                let weight =
                    weights
                    |> List.tryFind (fun w -> w.PatternId = r.PatternId)
                    |> Option.map (fun w -> w.Weight)
                    |> Option.defaultValue 0.5

                let lastPromoted =
                    r.PromotionHistory
                    |> List.tryLast
                    |> Option.map snd
                    |> Option.defaultValue r.LastSeen

                { PatternId = r.PatternId
                  PatternName = r.PatternName
                  Level = r.CurrentLevel
                  LevelRank = PromotionLevel.rank r.CurrentLevel
                  Score = r.AverageScore
                  OccurrenceCount = r.OccurrenceCount
                  Contexts = r.Contexts
                  RollbackExpansion = rollbackByPattern |> Map.tryFind r.PatternId
                  Weight = weight
                  LastPromoted = lastPromoted })
            |> List.sortByDescending (fun e -> e.LevelRank, e.Score, e.Weight)

        { Entries = entries
          GeneratedAt = DateTime.UtcNow
          PatternCount = entries.Length }

    /// Persist the index to disk for fast loading by agent layer.
    let save (index: PromotionIndexData) : unit =
        try
            let json = JsonSerializer.Serialize(index, jsonOptions)
            File.WriteAllText(indexPath (), json)
        with _ -> () // Best-effort

    /// Load the persisted index from disk.
    let load () : PromotionIndexData option =
        try
            let path = indexPath ()
            if File.Exists path then
                let json = File.ReadAllText path
                Some (JsonSerializer.Deserialize<PromotionIndexData>(json, jsonOptions))
            else
                None
        with _ -> None

    /// Build and persist in one call.
    let refresh () : PromotionIndexData =
        let index = build ()
        save index
        index

    // ── Query helpers for pattern selection ──────────────────────────

    /// Get patterns at or above a minimum promotion level, ranked by score.
    let atOrAbove (minLevel: PromotionLevel) (index: PromotionIndexData) : IndexEntry list =
        let minRank = PromotionLevel.rank minLevel
        index.Entries
        |> List.filter (fun e -> e.LevelRank >= minRank)

    /// Find the best pattern matching a goal by substring search on contexts.
    let findForGoal (goal: string) (index: PromotionIndexData) : IndexEntry option =
        let goalLower = goal.ToLowerInvariant()
        index.Entries
        |> List.sortByDescending (fun e ->
            let contextMatch =
                e.Contexts
                |> List.sumBy (fun c ->
                    let cLower = c.ToLowerInvariant()
                    // Count how many words from goal appear in context
                    goalLower.Split([|' '; ','; '.'; '?'; '!'|], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.sumBy (fun w -> if cLower.Contains(w) then 1 else 0))
            // Composite score: level rank * 10 + context match * 2 + weight + score
            float (e.LevelRank * 10) + float (contextMatch * 2) + e.Weight + e.Score)
        |> List.tryHead

    /// Score all patterns for a goal, returning (entry, score) pairs.
    let scoreForGoal (goal: string) (index: PromotionIndexData) : (IndexEntry * float) list =
        let goalLower = goal.ToLowerInvariant()
        index.Entries
        |> List.map (fun e ->
            let contextMatch =
                e.Contexts
                |> List.sumBy (fun c ->
                    let cLower = c.ToLowerInvariant()
                    goalLower.Split([|' '; ','; '.'; '?'; '!'|], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.sumBy (fun w -> if cLower.Contains(w) then 1 else 0))
            let score =
                float (e.LevelRank * 10) + float (contextMatch * 2) + e.Weight + e.Score
            e, score)
        |> List.sortByDescending snd
