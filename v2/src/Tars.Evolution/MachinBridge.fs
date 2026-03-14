namespace Tars.Evolution

open System
open System.Diagnostics
open System.Text
open System.Text.Json
open System.Threading.Tasks

/// Bridge to MachinDeOuf's Rust-based ML algorithms.
/// Calls `machin-skill` CLI for genetic algorithms, optimization, and clustering.
/// Falls back to built-in F# implementations when machin-skill is not available.
module MachinBridge =

    /// Result from a MachinDeOuf optimization call.
    type OptimizeResult =
        { BestValue: float
          BestParams: float list
          Iterations: int
          Converged: bool }

    /// Configuration for the MachinDeOuf bridge.
    type MachinConfig =
        { /// Path to machin-skill executable
          SkillPath: string
          /// Timeout for subprocess calls
          Timeout: TimeSpan
          /// Working directory for machin-skill
          WorkingDir: string option }

    let defaultConfig =
        { SkillPath = "cargo"
          Timeout = TimeSpan.FromSeconds(30.0)
          WorkingDir = None }

    /// Check if machin-skill is available.
    let isAvailable (config: MachinConfig) : bool =
        try
            let psi = ProcessStartInfo()
            psi.FileName <- config.SkillPath
            psi.Arguments <- "run -p machin-skill -- list"
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true
            match config.WorkingDir with
            | Some dir -> psi.WorkingDirectory <- dir
            | None -> ()

            use proc = Process.Start(psi)
            proc.WaitForExit(5000) |> ignore
            proc.ExitCode = 0
        with _ -> false

    /// Execute machin-skill CLI and parse output.
    let private executeSkill
        (config: MachinConfig)
        (args: string)
        : Task<Result<string, string>> =
        task {
            try
                let psi = ProcessStartInfo()
                psi.FileName <- config.SkillPath
                psi.Arguments <- sprintf "run -p machin-skill -- %s" args
                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.CreateNoWindow <- true
                match config.WorkingDir with
                | Some dir -> psi.WorkingDirectory <- dir
                | None -> ()

                use proc = new Process()
                proc.StartInfo <- psi

                let stdout = StringBuilder()
                let stderr = StringBuilder()

                proc.OutputDataReceived.Add(fun e ->
                    if not (isNull e.Data) then
                        stdout.AppendLine(e.Data) |> ignore)
                proc.ErrorDataReceived.Add(fun e ->
                    if not (isNull e.Data) then
                        stderr.AppendLine(e.Data) |> ignore)

                proc.Start() |> ignore
                proc.BeginOutputReadLine()
                proc.BeginErrorReadLine()

                let! completed =
                    Task.Run(fun () ->
                        proc.WaitForExit(int config.Timeout.TotalMilliseconds))

                if not completed then
                    try proc.Kill() with _ -> ()
                    return Error "machin-skill timed out"
                elif proc.ExitCode <> 0 then
                    return Error (stderr.ToString().Trim())
                else
                    return Ok (stdout.ToString().Trim())
            with ex ->
                return Error (sprintf "machin-skill error: %s" ex.Message)
        }

    /// Parse machin-skill optimization output.
    let parseOptimizeOutput (output: string) : OptimizeResult =
        // machin-skill output format:
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

    /// Run genetic algorithm optimization via machin-skill.
    let runGeneticAlgorithm
        (config: MachinConfig)
        (dim: int)
        (maxGenerations: int)
        (fitnessFunction: string)
        : Task<Result<OptimizeResult, string>> =
        task {
            let args =
                sprintf "optimize --algo genetic --function %s --dim %d --max-iter %d"
                    fitnessFunction dim maxGenerations
            let! result = executeSkill config args
            match result with
            | Ok output -> return Ok (parseOptimizeOutput output)
            | Error err -> return Error err
        }

    /// Run differential evolution via machin-skill.
    let runDifferentialEvolution
        (config: MachinConfig)
        (dim: int)
        (maxGenerations: int)
        (fitnessFunction: string)
        : Task<Result<OptimizeResult, string>> =
        task {
            let args =
                sprintf "optimize --algo differential --function %s --dim %d --max-iter %d"
                    fitnessFunction dim maxGenerations
            let! result = executeSkill config args
            match result with
            | Ok output -> return Ok (parseOptimizeOutput output)
            | Error err -> return Error err
        }

    /// Run PSO via machin-skill.
    let runPSO
        (config: MachinConfig)
        (dim: int)
        (maxIterations: int)
        (fitnessFunction: string)
        : Task<Result<OptimizeResult, string>> =
        task {
            let args =
                sprintf "optimize --algo pso --function %s --dim %d --max-iter %d"
                    fitnessFunction dim maxIterations
            let! result = executeSkill config args
            match result with
            | Ok output -> return Ok (parseOptimizeOutput output)
            | Error err -> return Error err
        }

    // =========================================================================
    // Built-in F# fallback implementations
    // =========================================================================

    /// Minimal genetic algorithm in pure F# for when machin-skill is unavailable.
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
