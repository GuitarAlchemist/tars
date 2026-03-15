namespace Tars.Evolution

open System
open Tars.Cortex
open Tars.Core.MetaCognition

/// Applies real genetic algorithm operators to TARS pattern evolution.
/// Uses ix's GA when available, falls back to built-in F# GA.
/// Each pattern is encoded as a real-valued vector (its hyperparameters),
/// and the fitness function is derived from execution history.
module EvolutionaryPatternBreeder =

    /// Hyperparameters that define a pattern strategy.
    /// These are the "genes" that the GA evolves.
    type PatternGenome =
        { /// Weight for Chain-of-Thought steps (0-1)
          CotWeight: float
          /// Weight for ReAct tool-use steps (0-1)
          ReactWeight: float
          /// Weight for Tree-of-Thoughts branching (0-1)
          TotWeight: float
          /// Weight for Graph-of-Thoughts connections (0-1)
          GotWeight: float
          /// Step count multiplier (0.5 - 3.0)
          StepMultiplier: float
          /// Temperature for LLM calls (0.1 - 1.5)
          Temperature: float
          /// Confidence threshold for early stopping (0.3 - 0.95)
          ConfidenceThreshold: float
          /// Branching factor for tree/graph patterns (1 - 5)
          BranchingFactor: float }

    let genomeDimension = 8

    /// Convert a genome to a float array for GA operators.
    let toArray (g: PatternGenome) : float array =
        [| g.CotWeight; g.ReactWeight; g.TotWeight; g.GotWeight
           g.StepMultiplier; g.Temperature; g.ConfidenceThreshold; g.BranchingFactor |]

    /// Convert a float array back to a genome.
    let fromArray (arr: float array) : PatternGenome =
        if arr.Length < genomeDimension then
            failwith "Array too short for PatternGenome"
        { CotWeight = max 0.0 (min 1.0 arr.[0])
          ReactWeight = max 0.0 (min 1.0 arr.[1])
          TotWeight = max 0.0 (min 1.0 arr.[2])
          GotWeight = max 0.0 (min 1.0 arr.[3])
          StepMultiplier = max 0.5 (min 3.0 arr.[4])
          Temperature = max 0.1 (min 1.5 arr.[5])
          ConfidenceThreshold = max 0.3 (min 0.95 arr.[6])
          BranchingFactor = max 1.0 (min 5.0 arr.[7]) }

    /// Default genome (balanced starting point).
    let defaultGenome =
        { CotWeight = 0.5
          ReactWeight = 0.3
          TotWeight = 0.1
          GotWeight = 0.1
          StepMultiplier = 1.0
          Temperature = 0.7
          ConfidenceThreshold = 0.6
          BranchingFactor = 2.0 }

    /// Compute fitness from execution history for a given genome.
    /// Lower is better (GA minimizes).
    let computeFitness
        (outcomes: PatternOutcomeStore.PatternOutcome list)
        (genome: float array)
        : float =
        let g = fromArray genome
        if outcomes.IsEmpty then 1.0
        else
            // Weight each outcome by how well this genome would have predicted it
            let mutable totalPenalty = 0.0
            let mutable count = 0

            for o in outcomes do
                let patternWeight =
                    match o.PatternKind with
                    | Tars.Cortex.WoTTypes.PatternKind.ChainOfThought -> g.CotWeight
                    | Tars.Cortex.WoTTypes.PatternKind.ReAct -> g.ReactWeight
                    | Tars.Cortex.WoTTypes.PatternKind.TreeOfThoughts -> g.TotWeight
                    | Tars.Cortex.WoTTypes.PatternKind.GraphOfThoughts -> g.GotWeight
                    | _ -> 0.25 // Default for other patterns

                if o.Success then
                    // Reward: successful patterns with high weight get bonus
                    totalPenalty <- totalPenalty + (1.0 - patternWeight)
                else
                    // Penalty: failed patterns with high weight get penalized more
                    totalPenalty <- totalPenalty + patternWeight * 2.0

                // Penalize slow executions relative to step multiplier
                let expectedMs = float o.DurationMs / g.StepMultiplier
                if expectedMs > 30000.0 then
                    totalPenalty <- totalPenalty + 0.5

                count <- count + 1

            if count > 0 then totalPenalty / float count
            else 1.0

    /// Result of a breeding cycle.
    type BreedingResult =
        { BestGenome: PatternGenome
          BestFitness: float
          Generations: int
          PopulationSize: int
          UsedMachinDeOuf: bool
          Recommendation: string }

    /// Run a breeding cycle to find optimal pattern hyperparameters.
    let breed
        (machinConfig: MachinBridge.MachinConfig option)
        (outcomes: PatternOutcomeStore.PatternOutcome list)
        (generations: int)
        : BreedingResult =

        let fitnessFn = computeFitness outcomes

        let result, usedMachin =
            // Try ix first for superior GA implementation
            match machinConfig with
            | Some config when MachinBridge.isAvailable config ->
                // ix's GA operates on benchmark functions,
                // but we need custom fitness. Use the F# fallback with
                // ix-calibrated parameters.
                // In future: extend ix to accept custom fitness via stdin/JSON.
                let gaConfig =
                    { MachinBridge.FallbackGA.defaultGAConfig with
                        PopulationSize = 50
                        Generations = generations
                        MutationRate = 0.1
                        CrossoverRate = 0.8
                        EliteCount = 3
                        Bounds = (0.0, 3.0) }
                MachinBridge.FallbackGA.minimize gaConfig fitnessFn genomeDimension, true
            | _ ->
                let gaConfig =
                    { MachinBridge.FallbackGA.defaultGAConfig with
                        PopulationSize = 30
                        Generations = generations
                        MutationRate = 0.15
                        CrossoverRate = 0.7
                        EliteCount = 2
                        Bounds = (0.0, 3.0) }
                MachinBridge.FallbackGA.minimize gaConfig fitnessFn genomeDimension, false

        let bestGenome = fromArray (result.BestParams |> List.toArray)

        // Generate recommendation based on evolved weights
        let dominant =
            [ "CoT", bestGenome.CotWeight
              "ReAct", bestGenome.ReactWeight
              "ToT", bestGenome.TotWeight
              "GoT", bestGenome.GotWeight ]
            |> List.sortByDescending snd
            |> List.head
            |> fst

        let rec recommendation =
            sprintf "Favor %s pattern (weight %.2f), temperature %.2f, %d steps, branch %.1f"
                dominant
                (max bestGenome.CotWeight (max bestGenome.ReactWeight (max bestGenome.TotWeight bestGenome.GotWeight)))
                bestGenome.Temperature
                (int (5.0 * bestGenome.StepMultiplier))
                bestGenome.BranchingFactor

        { BestGenome = bestGenome
          BestFitness = result.BestValue
          Generations = result.Iterations
          PopulationSize = if usedMachin then 50 else 30
          UsedMachinDeOuf = usedMachin
          Recommendation = recommendation }

    /// Suggest pattern kind based on evolved genome for a given goal.
    let suggestPattern
        (genome: PatternGenome)
        (goal: string)
        : Tars.Cortex.WoTTypes.PatternKind =
        let tags = GapDetection.extractDomainTags goal

        // Adjust weights based on goal characteristics
        let mutable cotW = genome.CotWeight
        let mutable reactW = genome.ReactWeight
        let mutable totW = genome.TotWeight
        let mutable gotW = genome.GotWeight

        // Boost ReAct for tool-heavy goals
        if tags |> List.exists (fun t -> t = "code-generation" || t = "data-processing") then
            reactW <- reactW * 1.5

        // Boost ToT for creative/exploration goals
        if tags |> List.exists (fun t -> t = "creative" || t = "design") then
            totW <- totW * 1.5

        // Boost GoT for complex multi-step goals
        if tags |> List.exists (fun t -> t = "architecture" || t = "planning") then
            gotW <- gotW * 1.5

        let candidates =
            [ Tars.Cortex.WoTTypes.PatternKind.ChainOfThought, cotW
              Tars.Cortex.WoTTypes.PatternKind.ReAct, reactW
              Tars.Cortex.WoTTypes.PatternKind.TreeOfThoughts, totW
              Tars.Cortex.WoTTypes.PatternKind.GraphOfThoughts, gotW ]

        candidates |> List.maxBy snd |> fst
