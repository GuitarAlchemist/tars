namespace Tars.Cortex

open System
open Tars.Core
open ReasoningPattern
open Tars.Cortex.WoTTypes

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
    /// A pattern selector that uses golden trace history to boost patterns
    /// that have succeeded in the past, combined with goal-based heuristics.
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
            match s.ToLowerInvariant() with
            | s when s.Contains("chain") -> Some ChainOfThought
            | s when s.Contains("react") -> Some ReAct
            | s when s.Contains("graph") -> Some GraphOfThoughts
            | s when s.Contains("tree") -> Some TreeOfThoughts
            | s when s.Contains("workflow") -> Some WorkflowOfThought
            | _ -> None

        let historyScores () =
            goldenHistory.Value
            |> List.choose (fun g -> parsePatternKind g.PatternKind)
            |> List.countBy id
            |> List.map (fun (kind, count) -> kind, float count)
            |> Map.ofList

        let heuristicScore (goal: string) =
            let g = goal.ToLowerInvariant()
            [ ChainOfThought, (if g.Contains("explain") || g.Contains("step") then 0.8 else 0.4)
              ReAct, (if g.Contains("search") || g.Contains("find") || g.Contains("look") then 0.8 else 0.3)
              GraphOfThoughts, (if g.Contains("compare") || g.Contains("alternative") then 0.8 else 0.2)
              TreeOfThoughts, (if g.Contains("explore") || g.Contains("brainstorm") then 0.8 else 0.2)
              WorkflowOfThought, (if g.Contains("workflow") || g.Contains("pipeline") then 0.8 else 0.3) ]
            |> Map.ofList

        interface IPatternSelector with
            member _.Recommend(goal, _state) =
                let heuristic = heuristicScore goal
                let history = historyScores ()
                let maxHistory = history |> Map.values |> Seq.append [1.0] |> Seq.max
                let combined =
                    heuristic
                    |> Map.map (fun kind score ->
                        let histBoost = history |> Map.tryFind kind |> Option.defaultValue 0.0
                        score + 0.3 * (histBoost / maxHistory))
                combined |> Map.toList |> List.maxBy snd |> fst

            member _.Score(goal) =
                let heuristic = heuristicScore goal
                let history = historyScores ()
                let maxHistory = history |> Map.values |> Seq.append [1.0] |> Seq.max
                heuristic
                |> Map.map (fun kind score ->
                    let histBoost = history |> Map.tryFind kind |> Option.defaultValue 0.0
                    score + 0.3 * (histBoost / maxHistory))
