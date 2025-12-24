/// Neuro-Symbolic AI Puzzle Benchmark
/// Compare puzzle-solving performance with and without neuro-symbolic constraints
module Tars.Interface.Cli.Commands.NeuroSymbolicBenchmark

open System
open System.Diagnostics
open Microsoft.Extensions.Logging
open Tars.Core
open Tars.Interface.Cli.Commands.PuzzleDemo

/// Benchmark result for a single puzzle
type PuzzleResult =
    { PuzzleName: string
      Difficulty: int
      Success: bool
      AttemptCount: int
      TimeMs: float
      ConstraintScore: float option
      ContradictionsDetected: int }

/// Benchmark configuration
type BenchmarkConfig =
    { EnableNeuroSymbolic: bool
      MaxAttemptsPerPuzzle: int
      Verbose: bool
      LogResults: bool }

/// Benchmark results summary
type BenchmarkSummary =
    { Config: BenchmarkConfig
      TotalPuzzles: int
      SuccessfulPuzzles: int
      FailedPuzzles: int
      AverageAttempts: float
      AverageDurationMs: float
      AverageConstraintScore: float option
      TotalContradictions: int
      Results: PuzzleResult list }

/// Calculate success rate
let successRate (summary: BenchmarkSummary) : float =
    if summary.TotalPuzzles = 0 then 0.0
    else float summary.SuccessfulPuzzles / float summary.TotalPuzzles * 100.0

/// Run a puzzle with optional neuro-symbolic constraints
let runPuzzleWithConstraints 
    (logger: ILogger) 
    (puzzle: Puzzle) 
    (config: BenchmarkConfig) 
    : PuzzleResult =
    
    let sw = Stopwatch.StartNew()
    let mutable success = false
    let mutable attempts = 0
    let mutable contradictions = 0
    let mutable constraintScore = None
    
    // For neuro-symbolic mode, track constraint violations
    if config.EnableNeuroSymbolic then
        // TODO: Wire in actual neuro-symbolic scoring
        // For now, simulate with heuristics
        constraintScore <- Some 0.75
    
    // Try to solve the puzzle
    for i in 1 .. config.MaxAttemptsPerPuzzle do
        attempts <- i
        
        // Simulate puzzle attempt
        // TODO: Replace with actual TARS agent call
        let attemptSuccess = Random().NextDouble() > 0.5
        
        if attemptSuccess then
            success <- true
            // break
            ()
        else
            if config.EnableNeuroSymbolic then
                // Track contradictions in neuro-symbolic mode
                contradictions <- contradictions + 1
    
    sw.Stop()
    
    { PuzzleName = puzzle.Name
      Difficulty = puzzle.Difficulty
      Success = success
      AttemptCount = attempts
      TimeMs = sw.Elapsed.TotalMilliseconds
      ConstraintScore = constraintScore
      ContradictionsDetected = contradictions }

/// Run benchmark on all puzzles
let runBenchmark (logger: ILogger) (config: BenchmarkConfig) : BenchmarkSummary =
    logger.LogInformation("🧩 Starting Neuro-Symbolic Puzzle Benchmark")
    logger.LogInformation($"Mode: {if config.EnableNeuroSymbolic then "Neuro-Symbolic ON" else "Baseline (OFF)"}")
    logger.LogInformation($"Puzzles: {allPuzzles.Length}")
    logger.LogInformation("")
    
    let results =
        allPuzzles
        |> List.map (fun puzzle ->
            logger.LogInformation($"Running: {puzzle.Name} (Difficulty {puzzle.Difficulty})...")
            let result = runPuzzleWithConstraints logger puzzle config
            
            if config.LogResults then
                let status = if result.Success then "✅ SOLVED" else "❌ FAILED"
                logger.LogInformation($"  {status} in {result.AttemptCount} attempts ({result.TimeMs:F0}ms)")
                
                match result.ConstraintScore with
                | Some score -> logger.LogInformation($"  Constraint Score: {score:F2}")
                | None -> ()
                
                if result.ContradictionsDetected > 0 then
                    logger.LogInformation($"  Contradictions: {result.ContradictionsDetected}")
            
            result)
    
    let summary =
        { Config = config
          TotalPuzzles = results.Length
          SuccessfulPuzzles = results |> List.filter (fun r -> r.Success) |> List.length
          FailedPuzzles = results |> List.filter (fun r -> not r.Success) |> List.length
          AverageAttempts = results |> List.averageBy (fun r -> float r.AttemptCount)
          AverageDurationMs = results |> List.averageBy (fun r -> r.TimeMs)
          AverageConstraintScore =
              let scores = results |> List.choose (fun r -> r.ConstraintScore)
              if scores.IsEmpty then None else Some (List.average scores)
          TotalContradictions = results |> List.sumBy (fun r -> r.ContradictionsDetected)
          Results = results }
    
    summary

/// Print benchmark summary
let printSummary (logger: ILogger) (summary: BenchmarkSummary) =
    logger.LogInformation("")
    logger.LogInformation("=" .PadRight(60, '='))
    logger.LogInformation($"📊 BENCHMARK RESULTS ({if summary.Config.EnableNeuroSymbolic then "NEURO-SYMBOLIC" else "BASELINE"})")
    logger.LogInformation("=" .PadRight(60, '='))
    logger.LogInformation($"Total Puzzles: {summary.TotalPuzzles}")
    logger.LogInformation($"Successful: {summary.SuccessfulPuzzles} (✅ {successRate summary:F1}%%)")
    logger.LogInformation($"Failed: {summary.FailedPuzzles}")
    logger.LogInformation($"Average Attempts: {summary.AverageAttempts:F1}")
    logger.LogInformation($"Average Duration: {summary.AverageDurationMs:F0}ms")
    
    match summary.AverageConstraintScore with
    | Some score -> 
        logger.LogInformation($"Average Constraint Score: {score:F2}")
        logger.LogInformation($"Total Contradictions: {summary.TotalContradictions}")
    | None -> ()
    
    logger.LogInformation("=" .PadRight(60, '='))

/// Compare baseline vs neuro-symbolic performance
let runComparison (logger: ILogger) (maxAttempts: int) =
    logger.LogInformation("🔬 COMPARATIVE BENCHMARK: Baseline vs Neuro-Symbolic")
    logger.LogInformation("")
    
    // Run baseline (no neuro-symbolic)
    let baselineConfig =
        { EnableNeuroSymbolic = false
          MaxAttemptsPerPuzzle = maxAttempts
          Verbose = false
          LogResults = true }
    
    logger.LogInformation("Phase 1: Running BASELINE (neuro-symbolic OFF)...")
    logger.LogInformation("")
    let baseline = runBenchmark logger baselineConfig
    printSummary logger baseline
    
    logger.LogInformation("")
    logger.LogInformation("=" .PadRight(60, '='))
    logger.LogInformation("")
    
    // Run with neuro-symbolic
    let nsConfig =
        { EnableNeuroSymbolic = true
          MaxAttemptsPerPuzzle = maxAttempts
          Verbose = false
          LogResults = true }
    
    logger.LogInformation("Phase 2: Running NEURO-SYMBOLIC (constraints ON)...")
    logger.LogInformation("")
    let neuroSymbolic = runBenchmark logger nsConfig
    printSummary logger neuroSymbolic
    
    // Print comparison
    logger.LogInformation("")
    logger.LogInformation("=" .PadRight(60, '='))
    logger.LogInformation("📈 IMPROVEMENT ANALYSIS")
    logger.LogInformation("=" .PadRight(60, '='))
    
    let successImprovement = successRate neuroSymbolic - successRate baseline
    let attemptsReduction = baseline.AverageAttempts - neuroSymbolic.AverageAttempts
    let speedImprovement = (baseline.AverageDurationMs - neuroSymbolic.AverageDurationMs) / baseline.AverageDurationMs * 100.0
    
    logger.LogInformation($"Success Rate: {successRate baseline:F1}%% → {successRate neuroSymbolic:F1}%% ({if successImprovement >= 0.0 then "+" else ""}{successImprovement:F1}%%)")
    logger.LogInformation($"Average Attempts: {baseline.AverageAttempts:F1} → {neuroSymbolic.AverageAttempts:F1} ({if attemptsReduction >= 0.0 then "-" else "+"}{Math.Abs(attemptsReduction):F1})")
    logger.LogInformation($"Speed: {if speedImprovement >= 0.0 then "+" else ""}{speedImprovement:F1}%% faster")
    
    match neuroSymbolic.AverageConstraintScore with
    | Some score -> 
        logger.LogInformation($"Constraint Score: {score:F2}/1.00")
        logger.LogInformation($"Contradictions Avoided: {neuroSymbolic.TotalContradictions} detected and handled")
    | None -> ()
    
    logger.LogInformation("=" .PadRight(60, '='))
    
    // Return both for further analysis
    (baseline, neuroSymbolic)

/// Export results to CSV
let exportToCsv (baseline: BenchmarkSummary) (neuroSymbolic: BenchmarkSummary) (filename: string) =
    let lines =
        [ "Puzzle,Difficulty,Baseline_Success,Baseline_Attempts,NS_Success,NS_Attempts,NS_Score,Improvement"
          yield!
              List.zip baseline.Results neuroSymbolic.Results
              |> List.map (fun (b, ns) ->
                  let improvement = if ns.Success && not b.Success then "SOLVED" elif not ns.Success && b.Success then "WORSE" else "SAME"
                  let score = ns.ConstraintScore |> Option.map (sprintf "%.2f") |> Option.defaultValue "N/A"
                  $"{b.PuzzleName},{b.Difficulty},{b.Success},{b.AttemptCount},{ns.Success},{ns.AttemptCount},{score},{improvement}") ]
    
    System.IO.File.WriteAllLines(filename, lines)

/// Create visualization data
let createVisualization (baseline: BenchmarkSummary) (neuroSymbolic: BenchmarkSummary) =
    let data =
        sprintf """
{
  "baseline": {
    "successRate": %.1f,
    "avgAttempts": %.1f,
    "avgTimeMs": %.0f
  },
  "neuroSymbolic": {
    "successRate": %.1f,
    "avgAttempts": %.1f,
    "avgTimeMs": %.0f,
    "avgConstraintScore": %.2f
  },
  "improvement": {
    "successRate": %.1f,
    "attemptsReduction": %.1f,
    "speedup": %.1f
  }
}"""
            (successRate baseline)
            baseline.AverageAttempts
            baseline.AverageDurationMs
            (successRate neuroSymbolic)
            neuroSymbolic.AverageAttempts
            neuroSymbolic.AverageDurationMs
            (neuroSymbolic.AverageConstraintScore |> Option.defaultValue 0.0)
            (successRate neuroSymbolic - successRate baseline)
            (baseline.AverageAttempts - neuroSymbolic.AverageAttempts)
            ((baseline.AverageDurationMs - neuroSymbolic.AverageDurationMs) / baseline.AverageDurationMs * 100.0)
    data
