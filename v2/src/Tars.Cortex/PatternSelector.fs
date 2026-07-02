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

        /// Probe for the sibling ix repo once per selector instance. When present,
        /// pattern scoring delegates the Beta-Binomial + softmax bandit math to
        /// ix's `grammar.weights` skill; otherwise it runs the same math in F#.
        let ixConfig = lazy (
            match IxSkill.discover () with
            | Some c when IxSkill.isAvailable c -> Some c
            | _ -> None)

        /// In-memory cache for bandit scores to avoid repeated disk I/O and IX calls.
        let mutable cachedBanditScores: Map<PatternKind, float> option = None

        /// Stable string key for a PatternKind (used as the ix rule id).
        let kindKey (k: PatternKind) = sprintf "%A" k

        let keyToKind (s: string) : PatternKind option =
            match s.ToLowerInvariant() with
            | s when s.Contains("chain") -> Some ChainOfThought
            | s when s.Contains("react") -> Some ReAct
            | s when s.Contains("planandexecute") || s.Contains("plan") -> Some PlanAndExecute
            | s when s.Contains("graph") -> Some GraphOfThoughts
            | s when s.Contains("tree") -> Some TreeOfThoughts
            | s when s.Contains("workflow") -> Some WorkflowOfThought
            | _ -> None

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

        /// Pattern selection as a multi-armed bandit: each PatternKind is an arm
        /// with a Beta(α,β) success posterior (α = successes+1, β = failures+1,
        /// a Laplace prior). Returns a softmax distribution over the arms — higher
        /// for kinds that have succeeded more often, but never collapsing to zero
        /// so under-explored kinds keep a chance. Delegates to ix `grammar.weights`
        /// when available; otherwise computes the identical math in F#.
        let banditScores () : Map<PatternKind, float> =
            match cachedBanditScores with
            | Some scores -> scores
            | None ->
                let perKind =
                    PatternOutcomeStore.loadAll ()
                    |> List.groupBy (fun o -> o.PatternKind)
                    |> List.map (fun (kind, es) ->
                        let succ = es |> List.filter (fun e -> e.Success) |> List.length
                        let fail = es.Length - succ
                        kind, float succ + 1.0, float fail + 1.0)

                let scores =
                    if List.isEmpty perKind then Map.empty
                    else
                        // F# reference implementation: Beta mean α/(α+β), then softmax.
                        let fsharp () =
                            let means = perKind |> List.map (fun (k, a, b) -> k, a / (a + b))
                            let exps = means |> List.map (fun (k, m) -> k, exp m)
                            let z = exps |> List.sumBy snd
                            exps |> List.map (fun (k, e) -> k, e / z) |> Map.ofList

                        let viaIx () =
                            match ixConfig.Value with
                            | None -> None
                            | Some c ->
                                try
                                    let rules =
                                        perKind
                                        |> List.map (fun (k, a, b) -> {| id = kindKey k; alpha = a; beta = b |})
                                    let input = JsonSerializer.Serialize({| rules = rules; temperature = 1.0 |})
                                    match (IxSkill.runSkillJson c "grammar.weights" input).GetAwaiter().GetResult() with
                                    | Result.Error _ -> None
                                    | Result.Ok json ->
                                        use d = JsonDocument.Parse(json)
                                        let probs = d.RootElement.GetProperty("probabilities")
                                        let m =
                                            [ for p in probs.EnumerateArray() ->
                                                p.GetProperty("rule_id").GetString(),
                                                p.GetProperty("probability").GetDouble() ]
                                            |> List.choose (fun (id, pr) ->
                                                keyToKind id |> Option.map (fun k -> k, pr))
                                            |> Map.ofList
                                        if Map.isEmpty m then None else Some m
                                with _ -> None

                        viaIx () |> Option.defaultWith fsharp
                cachedBanditScores <- Some scores
                scores

        let heuristicScore (goal: string) =
            let g = goal.ToLowerInvariant()
            [ ChainOfThought, (if g.Contains("explain") || g.Contains("step") || g.Contains("summarize") || g.Contains("describe") then 0.8 else 0.4)
              ReAct, (if g.Contains("search") || g.Contains("find") || g.Contains("look") || g.Contains("scan") || g.Contains("debug") then 0.8 else 0.3)
              GraphOfThoughts, (if g.Contains("compare") || g.Contains("alternative") || g.Contains("tradeoff") then 0.8 else 0.2)
              TreeOfThoughts, (if g.Contains("explore") || g.Contains("brainstorm") || g.Contains("generate ideas") then 0.8 else 0.2)
              WorkflowOfThought, (if g.Contains("workflow") || g.Contains("pipeline") || g.Contains("refactor") || g.Contains("fix") then 0.8 else 0.3) ]
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
            let bandit = banditScores ()
            let promoted = promotionBoost goal

            heuristic
            |> Map.map (fun kind score ->
                let goldenBoost = golden |> Map.tryFind kind |> Option.defaultValue 0.0
                // Bandit boost is already a probability in [0,1]; no normalization.
                let banditBoost = bandit |> Map.tryFind kind |> Option.defaultValue 0.0
                let promoBoost = promoted |> Map.tryFind kind |> Option.defaultValue 0.0
                score
                + 0.3 * (goldenBoost / maxGolden)
                + 0.2 * banditBoost
                + promoBoost) // Direct addition — promoted patterns already normalized

        /// Record the outcome of a pattern execution so future selections can learn from it.
        member this.RecordOutcome(outcome: PatternOutcome) =
            // 1. Persist to disk
            PatternOutcomeStore.record outcome

            // 2. Invalidate/update cache
            cachedBanditScores <- None // Simple invalidation forces reload next time Score() is called

        interface IPatternSelector with
            member _.Recommend(goal, _state) =
                combineScores goal |> Map.toList |> List.maxBy snd |> fst

            member _.Score(goal) =
                combineScores goal

            member this.RecordOutcome(outcome) =
                this.RecordOutcome(outcome)
