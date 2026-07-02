namespace Tars.Evolution

open System
open System.Text.Json
open System.Threading.Tasks
open Tars.Core

/// Bridge to ix's Rust-based ML algorithms.
///
/// The low-level subprocess handling lives in `Tars.Core.IxSkill` (the single
/// seam to ix); this module adds the Evolution-specific skill wrappers and a
/// built-in F# fallback so TARS never hard-depends on the Rust toolchain.
module MachinBridge =

    /// Result from an ix optimization call.
    type OptimizeResult =
        { BestValue: float
          BestParams: float list
          Iterations: int
          Converged: bool }

    /// Configuration for the ix bridge.
    type MachinConfig =
        { /// Path to the cargo executable (used when no prebuilt binary is found).
          SkillPath: string
          /// Timeout for subprocess calls
          Timeout: TimeSpan
          /// Working directory — the ix repo root (cargo + target/ live here).
          WorkingDir: string option }

    let defaultConfig =
        { SkillPath = "cargo"
          Timeout = TimeSpan.FromSeconds(30.0)
          WorkingDir = None }

    let private toIxConfig (c: MachinConfig) : IxSkill.Config =
        { CargoPath = c.SkillPath
          Timeout = c.Timeout
          RepoDir = c.WorkingDir }

    /// Check if ix is available (binary present, or cargo can resolve ix-skill).
    let isAvailable (config: MachinConfig) : bool =
        IxSkill.isAvailable (toIxConfig config)

    /// Run an ix skill with a JSON input document, returning raw JSON stdout.
    let runSkillJson
        (config: MachinConfig)
        (skill: string)
        (inputJson: string)
        : Task<Result<string, string>> =
        IxSkill.runSkillJson (toIxConfig config) skill inputJson

    /// Parse ix optimization output.
    let parseOptimizeOutput (output: string) : OptimizeResult =
        // ix output format:
        //   GeneticAlgorithm:
        //     Best value:   0.000123
        //     Best params:  [0.001, -0.002, 0.003]
        //     Iterations:   500
        //     Converged:    true
        let mutable bestValue = 0.0
        let mutable bestParams = []
        let mutable iterations = 0
        let mutable converged = false

        for line in output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries) do
            let trimmed = line.Trim()
            if trimmed.StartsWith("Best value:") then
                let v = trimmed.Replace("Best value:", "").Trim()
                bestValue <- try float v with _ -> 0.0
            elif trimmed.StartsWith("Best params:") then
                let v = trimmed.Replace("Best params:", "").Trim()
                let cleaned = v.Trim('[', ']')
                bestParams <-
                    cleaned.Split(',')
                    |> Array.map (fun s -> try float (s.Trim()) with _ -> 0.0)
                    |> Array.toList
            elif trimmed.StartsWith("Iterations:") then
                let v = trimmed.Replace("Iterations:", "").Trim()
                iterations <- try int v with _ -> 0
            elif trimmed.StartsWith("Converged:") then
                let v = trimmed.Replace("Converged:", "").Trim()
                converged <- v = "true"

        { BestValue = bestValue
          BestParams = bestParams
          Iterations = iterations
          Converged = converged }

    /// Minimize a benchmark function via ix's `optimize` skill.
    ///
    /// `func` is one of `sphere | rosenbrock | rastrigin`; `method` is one of
    /// `sgd | adam | pso | annealing` (ix's optimize surface — it does not yet
    /// accept a caller-supplied fitness closure, so custom-fitness work stays on
    /// the F# fallback in EvolutionaryPatternBreeder).
    let runOptimize
        (config: MachinConfig)
        (func: string)
        (method: string)
        (dim: int)
        (maxIter: int)
        : Task<Result<OptimizeResult, string>> =
        task {
            let input =
                JsonSerializer.Serialize(
                    {| ``function`` = func
                       dimensions = dim
                       method = method
                       max_iter = maxIter |})
            let! result = runSkillJson config "optimize" input
            match result with
            | Result.Error err -> return Result.Error err
            | Result.Ok json ->
                try
                    use doc = JsonDocument.Parse(json)
                    let root = doc.RootElement
                    let bestParams =
                        [ for e in root.GetProperty("best_params").EnumerateArray() -> e.GetDouble() ]
                    return
                        Result.Ok
                            { BestValue = root.GetProperty("best_value").GetDouble()
                              BestParams = bestParams
                              Iterations = root.GetProperty("iterations").GetInt32()
                              Converged = root.GetProperty("converged").GetBoolean() }
                with ex ->
                    return Result.Error (sprintf "ix optimize parse error: %s" ex.Message)
        }

    // =========================================================================
    // Built-in F# fallback implementations
    // =========================================================================

    /// Minimal genetic algorithm in pure F# for when ix is unavailable.
    module FallbackGA =

        type Individual =
            { Genes: float array
              Fitness: float }

        type GAConfig =
            { PopulationSize: int
              Generations: int
              MutationRate: float
              CrossoverRate: float
              EliteCount: int
              Bounds: float * float }

        let defaultGAConfig =
            { PopulationSize = 30
              Generations = 100
              MutationRate = 0.1
              CrossoverRate = 0.8
              EliteCount = 2
              Bounds = (0.0, 1.0) }

        let private rng = Random(42)

        let private randomIndividual (dim: int) (lo: float) (hi: float) =
            { Genes = Array.init dim (fun _ -> lo + rng.NextDouble() * (hi - lo))
              Fitness = Double.MaxValue }

        let private tournamentSelect (pop: Individual array) (k: int) =
            let candidates = Array.init k (fun _ -> pop.[rng.Next(pop.Length)])
            candidates |> Array.minBy (fun i -> i.Fitness)

        let private blxCrossover (a: Individual) (b: Individual) (alpha: float) =
            let genes =
                Array.init a.Genes.Length (fun i ->
                    let lo = min a.Genes.[i] b.Genes.[i] - alpha * abs (a.Genes.[i] - b.Genes.[i])
                    let hi = max a.Genes.[i] b.Genes.[i] + alpha * abs (a.Genes.[i] - b.Genes.[i])
                    lo + rng.NextDouble() * (hi - lo))
            { Genes = genes; Fitness = Double.MaxValue }

        let private mutate (ind: Individual) (rate: float) (lo: float) (hi: float) =
            let genes =
                ind.Genes |> Array.map (fun g ->
                    if rng.NextDouble() < 0.3 then
                        let noise = (rng.NextDouble() - 0.5) * 2.0 * rate
                        max lo (min hi (g + noise))
                    else g)
            { ind with Genes = genes }

        /// Run GA to minimize a fitness function.
        let minimize
            (config: GAConfig)
            (fitnessFn: float array -> float)
            (dim: int)
            : OptimizeResult =
            let lo, hi = config.Bounds

            // Initialize
            let mutable pop =
                Array.init config.PopulationSize (fun _ ->
                    let ind = randomIndividual dim lo hi
                    { ind with Fitness = fitnessFn ind.Genes })

            let mutable bestFitness = Double.MaxValue
            let mutable bestGenes = Array.empty

            for _ in 0 .. config.Generations - 1 do
                pop <- pop |> Array.sortBy (fun i -> i.Fitness)

                if pop.[0].Fitness < bestFitness then
                    bestFitness <- pop.[0].Fitness
                    bestGenes <- Array.copy pop.[0].Genes

                let elite = pop.[.. config.EliteCount - 1]
                let offspring = ResizeArray<Individual>(elite)

                while offspring.Count < config.PopulationSize do
                    let p1 = tournamentSelect pop 3
                    let p2 = tournamentSelect pop 3
                    let child =
                        if rng.NextDouble() < config.CrossoverRate then
                            blxCrossover p1 p2 0.5
                        else
                            { p1 with Fitness = Double.MaxValue }
                    let mutated = mutate child config.MutationRate lo hi
                    let evaluated = { mutated with Fitness = fitnessFn mutated.Genes }
                    offspring.Add(evaluated)

                pop <- offspring.ToArray() |> Array.take config.PopulationSize

            { BestValue = bestFitness
              BestParams = bestGenes |> Array.toList
              Iterations = config.Generations
              Converged = bestFitness < 1e-6 }
