namespace Tars.Cortex

open System
open System.IO
open System.Text.Json
open Tars.Core
open ReasoningPattern
open Tars.Cortex.WoTTypes

/// <summary>
/// Persistent store for pattern selection outcomes.
/// Records which patterns were selected for which goals and whether they succeeded.
/// </summary>
module PatternOutcomeStore =

    /// A recorded outcome of a pattern selection and execution.
    type PatternOutcome =
        { PatternKind: PatternKind
          Goal: string
          Success: bool
          DurationMs: int64
          Timestamp: DateTime }

    /// JSON-friendly DTO for serialization (PatternKind is a DU, so we store it as a string).
    type PatternOutcomeDto =
        { PatternKind: string
          Goal: string
          Success: bool
          DurationMs: int64
          Timestamp: DateTime }

    let private toDto (o: PatternOutcome) : PatternOutcomeDto =
        { PatternKind = sprintf "%A" o.PatternKind
          Goal = o.Goal
          Success = o.Success
          DurationMs = o.DurationMs
          Timestamp = o.Timestamp }

    let private parseKind (s: string) : PatternKind =
        match s.ToLowerInvariant() with
        | s when s.Contains("chainofthought") || s.Contains("chain") -> ChainOfThought
        | s when s.Contains("react") -> ReAct
        | s when s.Contains("planandexecute") -> PlanAndExecute
        | s when s.Contains("graphofthoughts") || s.Contains("graph") -> GraphOfThoughts
        | s when s.Contains("treeofthoughts") || s.Contains("tree") -> TreeOfThoughts
        | s when s.Contains("workflowofthought") || s.Contains("workflow") -> WorkflowOfThought
        | other -> Custom other

    let private fromDto (d: PatternOutcomeDto) : PatternOutcome =
        { PatternKind = parseKind d.PatternKind
          Goal = d.Goal
          Success = d.Success
          DurationMs = d.DurationMs
          Timestamp = d.Timestamp }

    let private jsonOptions =
        JsonSerializerOptions(
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    let private outcomePath () =
        let dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        Path.Combine(dir, "pattern_outcomes.json")

    /// Load all recorded outcomes from disk.
    let loadAll () : PatternOutcome list =
        try
            let path = outcomePath ()
            if File.Exists path then
                let json = File.ReadAllText path
                let dtos = JsonSerializer.Deserialize<PatternOutcomeDto list>(json, jsonOptions)
                dtos |> List.map fromDto
            else
                []
        with _ -> []

    /// Record a new outcome, appending it to the on-disk store.
    let record (outcome: PatternOutcome) : unit =
        try
            let existing = loadAll () |> List.map toDto
            let updated = existing @ [ toDto outcome ]
            let json = JsonSerializer.Serialize(updated, jsonOptions)
            File.WriteAllText(outcomePath (), json)
        with _ -> () // Best-effort — don't crash if disk write fails

/// <summary>
/// Selects the best reasoning pattern for a given goal.
/// Supports both heuristic selection and (future) learned selection.
/// </summary>
module PatternSelector =

    /// <summary>
    /// A strategy for selecting patterns.
    /// </summary>
    type SelectorStrategy =
        | RuleBased
        | HistoryBased
        | LlmBased

    /// <summary>
    /// Service for selecting and retrieving reasoning patterns.
    /// </summary>
    type PatternLibraryService() =

        let mutable patterns =
            [ Library.linearCoT; Library.criticRefinement; Library.parallelBrainstorming ]
            |> List.map (fun p -> p.Name, p)
            |> Map.ofList

        /// <summary>
        /// Register a new pattern or update an existing one.
        /// </summary>
        member _.Register(pattern: ReasoningPattern) =
            patterns <- patterns |> Map.add pattern.Name pattern

        /// <summary>
        /// Get a compiled list of all available patterns.
        /// </summary>
        member _.GetAll() = patterns |> Map.values |> Seq.toList

        /// <summary>
        /// Select the best pattern for the goal using simple heuristics.
        /// </summary>
        member _.Select(goal: string) : ReasoningPattern =
            let lowerGoal = goal.ToLowerInvariant()

            if
                lowerGoal.Contains("brainstorm")
                || lowerGoal.Contains("ideas")
                || lowerGoal.Contains("generate")
            then
                patterns.["Parallel Brainstorming"]
            else if
                lowerGoal.Contains("check")
                || lowerGoal.Contains("verify")
                || lowerGoal.Contains("critique")
                || lowerGoal.Contains("improve")
            then
                patterns.["Critic Refinement"]
            else
                // Default to CoT
                patterns.["Linear Chain of Thought"]

        /// <summary>
        /// Select a pattern by exact name.
        /// </summary>
        member _.GetByName(name: string) = patterns |> Map.tryFind name

    /// Lightweight promotion index entry — deserialized from ~/.tars/promotion/index.json
    /// without depending on Tars.Evolution (avoids circular reference).
    type PromotedEntry = {
        PatternName: string
        LevelRank: int
        Score: float
        Weight: float
        Contexts: string list
    }

    /// Load promoted patterns from the persisted index file on disk.
    let private loadPromotedEntries () : PromotedEntry list =
        try
            let path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".tars", "promotion", "index.json")
            if File.Exists path then
                use doc = JsonDocument.Parse(File.ReadAllText path)
                let entries = doc.RootElement.GetProperty("Entries")
                [ for e in entries.EnumerateArray() do
                    { PatternName =
                        (try e.GetProperty("PatternName").GetString() with _ -> "")
                      LevelRank =
                        (try e.GetProperty("LevelRank").GetInt32() with _ -> 0)
                      Score =
                        (try e.GetProperty("Score").GetDouble() with _ -> 0.0)
                      Weight =
                        (try e.GetProperty("Weight").GetDouble() with _ -> 0.5)
                      Contexts =
                        (try [ for c in e.GetProperty("Contexts").EnumerateArray() do c.GetString() ]
                         with _ -> []) } ]
            else []
        with _ -> []

    /// <summary>
    /// A pattern selector that uses golden trace history, recorded pattern
    /// outcomes, AND the promotion index (from cross-repo pattern discovery)
    /// to select the best reasoning pattern for a goal.
    /// </summary>
    type HistoryAwareSelector() =
        let goldenHistory = lazy (
            try
                GoldenTraceStore.listAll()
                |> List.choose (fun name ->
                    match GoldenTraceStore.load name with
                    | Result.Ok golden -> Some golden
                    | Result.Error _ -> None)
            with _ -> [])

        /// Lazy-load promoted patterns from disk.
        let promotedPatterns = lazy (loadPromotedEntries ())

        let parsePatternKind (s: string) =
            if isNull s then None else
            match s.ToLowerInvariant() with
            | s when s.Contains("chain") -> Some ChainOfThought
            | s when s.Contains("react") -> Some ReAct
            | s when s.Contains("graph") -> Some GraphOfThoughts
            | s when s.Contains("tree") -> Some TreeOfThoughts
            | s when s.Contains("workflow") -> Some WorkflowOfThought
            | _ -> None

        let goldenScores () =
            goldenHistory.Value
            |> List.choose (fun g -> parsePatternKind g.PatternKind)
            |> List.countBy id
            |> List.map (fun (kind, count) -> kind, float count)
            |> Map.ofList

        /// Compute a net score from recorded outcomes per PatternKind.
        /// Each success adds +1, each failure subtracts -0.5.
        let outcomeScores () =
            let outcomes = PatternOutcomeStore.loadAll ()
            outcomes
            |> List.groupBy (fun o -> o.PatternKind)
            |> List.map (fun (kind, entries) ->
                let score =
                    entries
                    |> List.sumBy (fun e -> if e.Success then 1.0 else -0.5)
                kind, score)
            |> Map.ofList

        let heuristicScore (goal: string) =
            let g = goal.ToLowerInvariant()
            [ ChainOfThought, (if g.Contains("explain") || g.Contains("step") then 0.8 else 0.4)
              ReAct, (if g.Contains("search") || g.Contains("find") || g.Contains("look") then 0.8 else 0.3)
              GraphOfThoughts, (if g.Contains("compare") || g.Contains("alternative") then 0.8 else 0.2)
              TreeOfThoughts, (if g.Contains("explore") || g.Contains("brainstorm") then 0.8 else 0.2)
              WorkflowOfThought, (if g.Contains("workflow") || g.Contains("pipeline") then 0.8 else 0.3) ]
            |> Map.ofList

        /// Score boost from promoted patterns (cross-repo discovery).
        /// Patterns at Builder+ level that contextually match the goal
        /// boost WorkflowOfThought (the most flexible pattern kind).
        /// Score a promoted entry against a goal by context word overlap.
        let scoreEntry (goalLower: string) (entry: PromotedEntry) : float =
            let words = goalLower.Split([|' '; ','; '.'; '?'; '!'|], StringSplitOptions.RemoveEmptyEntries)
            let contextMatch =
                entry.Contexts
                |> List.sumBy (fun c ->
                    let cLower = c.ToLowerInvariant()
                    words |> Array.sumBy (fun w -> if cLower.Contains(w) then 1 else 0))
            float (entry.LevelRank * 10) + float (contextMatch * 2) + entry.Weight + entry.Score

        let promotionBoost (goal: string) : Map<PatternKind, float> =
            let entries = promotedPatterns.Value
            if entries.IsEmpty then Map.empty
            else
                let goalLower = goal.ToLowerInvariant()
                let scored =
                    entries
                    |> List.map (fun e -> e, scoreEntry goalLower e)
                    |> List.sortByDescending snd

                match scored with
                | [] -> Map.empty
                | (topEntry, topScore) :: _ ->
                    // Only boost if context actually matches the goal well.
                    // Cap at 0.08 so promotion never overrides heuristic margins
                    // (heuristic range: 0.2-0.8, so 0.08 is a tiebreaker not an override).
                    let baseScore = float (topEntry.LevelRank * 10) + topEntry.Weight + topEntry.Score
                    let contextSignal = topScore - baseScore
                    let boost =
                        if contextSignal >= 4.0 then 0.08
                        elif contextSignal >= 2.0 then 0.04
                        else 0.0

                    let name = topEntry.PatternName.ToLowerInvariant()
                    let kindToBoost =
                        if name.Contains("routing") || name.Contains("pipeline") || name.Contains("orchestrat") then
                            WorkflowOfThought
                        elif name.Contains("skill") || name.Contains("fastpath") then
                            PlanAndExecute
                        elif name.Contains("hook") || name.Contains("fsm") then
                            WorkflowOfThought
                        elif name.Contains("confidence") || name.Contains("evidence") then
                            ChainOfThought
                        else
                            WorkflowOfThought

                    [ kindToBoost, boost
                      WorkflowOfThought, boost * 0.5 ]
                    |> List.distinctBy fst
                    |> Map.ofList

        let combineScores (goal: string) =
            let heuristic = heuristicScore goal
            let golden = goldenScores ()
            let maxGolden = golden |> Map.values |> Seq.append [1.0] |> Seq.max
            let outcomes = outcomeScores ()
            let maxOutcome = outcomes |> Map.values |> Seq.map abs |> Seq.append [1.0] |> Seq.max
            let promoted = promotionBoost goal

            heuristic
            |> Map.map (fun kind score ->
                let goldenBoost = golden |> Map.tryFind kind |> Option.defaultValue 0.0
                let outcomeBoost = outcomes |> Map.tryFind kind |> Option.defaultValue 0.0
                let promoBoost = promoted |> Map.tryFind kind |> Option.defaultValue 0.0
                score
                + 0.3 * (goldenBoost / maxGolden)
                + 0.2 * (outcomeBoost / maxOutcome)
                + promoBoost) // Direct addition — promoted patterns already normalized

        /// Record the outcome of a pattern execution so future selections can learn from it.
        member _.RecordOutcome(patternKind: PatternKind, goal: string, success: bool, durationMs: int64) =
            PatternOutcomeStore.record
                { PatternKind = patternKind
                  Goal = goal
                  Success = success
                  DurationMs = durationMs
                  Timestamp = DateTime.UtcNow }

        interface IPatternSelector with
            member _.Recommend(goal, _state) =
                combineScores goal |> Map.toList |> List.maxBy snd |> fst

            member _.Score(goal) =
                combineScores goal
