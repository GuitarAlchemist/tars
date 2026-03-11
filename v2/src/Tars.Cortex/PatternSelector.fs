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

    /// <summary>
    /// A pattern selector that uses golden trace history and recorded pattern
    /// outcomes to boost patterns that have succeeded in the past, combined
    /// with goal-based heuristics.
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

        let combineScores (goal: string) =
            let heuristic = heuristicScore goal
            let golden = goldenScores ()
            let maxGolden = golden |> Map.values |> Seq.append [1.0] |> Seq.max
            let outcomes = outcomeScores ()
            let maxOutcome = outcomes |> Map.values |> Seq.map abs |> Seq.append [1.0] |> Seq.max
            heuristic
            |> Map.map (fun kind score ->
                let goldenBoost = golden |> Map.tryFind kind |> Option.defaultValue 0.0
                let outcomeBoost = outcomes |> Map.tryFind kind |> Option.defaultValue 0.0
                score
                + 0.3 * (goldenBoost / maxGolden)
                + 0.2 * (outcomeBoost / maxOutcome))

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
