namespace Tars.Cortex

open System
open Tars.Core
open ReasoningPattern

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
