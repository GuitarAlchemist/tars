/// Benchmark Command - Test neuro-symbolic AI on puzzles
module Tars.Interface.Cli.Commands.Benchmark

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Core.Puzzles
open Tars.Interface.Cli.Commands.NeuroSymbolicBenchmark
open Tars.Interface.Cli.Commands.PuzzleDemo

/// Run the benchmark
let run (logger: ILogger) (args: string array) : Task<int> =
    task {
        try
            // Parse arguments
            let maxAttempts =
                let mutable d = 0

                if args.Length > 0 && Int32.TryParse(args.[0], &d) then
                    d
                else
                    5 // Default: 5 attempts per puzzle

            let mode = if args.Length > 1 then args.[1] else "compare"
            let filter = if args.Length > 2 then Some args.[2] else None

            let puzzles =
                match filter with
                | Some f ->
                    logger.Information($"Filtering puzzles by '{f}'")

                    Puzzles.all
                    |> List.filter (fun p -> p.Name.Contains(f, StringComparison.InvariantCultureIgnoreCase))
                | None -> Puzzles.all

            if puzzles.IsEmpty then
                logger.Error("No puzzles found matching filter.")
                return 1
            else
                match mode.ToLowerInvariant() with
                | "baseline" ->
                    // Run baseline only
                    let config =
                        { EnableNeuroSymbolic = false
                          MaxAttemptsPerPuzzle = maxAttempts
                          Verbose = true
                          LogResults = true }

                    let! summary = runBenchmark logger puzzles config
                    printSummary logger summary
                    return 0

                | "neurosymbolic"
                | "ns" ->
                    // Run neuro-symbolic only
                    let config =
                        { EnableNeuroSymbolic = true
                          MaxAttemptsPerPuzzle = maxAttempts
                          Verbose = true
                          LogResults = true }

                    let! summary = runBenchmark logger puzzles config
                    printSummary logger summary
                    return 0

                | "compare"
                | _ ->
                    // Run comparison (default)
                    let! (baseline, ns) = runComparison logger maxAttempts filter

                    // Export results
                    let timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
                    let csvFile = $"benchmark_results_{timestamp}.csv"
                    let jsonFile = $"benchmark_viz_{timestamp}.json"

                    exportToCsv baseline ns csvFile
                    logger.Information($"📄 Results exported to: {csvFile}")

                    let vizData = createVisualization baseline ns
                    System.IO.File.WriteAllText(jsonFile, vizData)
                    logger.Information($"📊 Visualization data: {jsonFile}")

                    return 0

        with ex ->
            logger.Error($"❌ Benchmark failed: {ex.Message}")
            return 1
    }

/// Show help
let help () =
    printfn
        "%s"
        """
📊 TARS Neuro-Symbolic Benchmark

Tests puzzle-solving performance with and without neuro-symbolic constraints.

USAGE:
    tars benchmark [max-attempts] [mode]

ARGUMENTS:
    max-attempts    Maximum attempts per puzzle (default: 5)
    mode            Run mode:
                    - compare (default): Run both baseline and neuro-symbolic
                    - baseline: Run baseline only (no constraints)
                    - neurosymbolic (or ns): Run neuro-symbolic only

EXAMPLES:
    tars benchmark                  # Compare baseline vs NS (5 attempts each)
    tars benchmark 10               # Compare with 10 attempts each
    tars benchmark 5 baseline       # Baseline only
    tars benchmark 5 ns             # Neuro-symbolic only

OUTPUT:
    - Console: Real-time progress and results
    - CSV: benchmark_results_TIMESTAMP.csv
    - JSON: benchmark_viz_TIMESTAMP.json (for visualization)

PUZZLES TESTED (9 total):
    1. River Crossing (Difficulty 2)
    2. Knights and Knaves (Difficulty 3)
    3. Tower of Hanoi (Difficulty 3)
    4. Logic Grid (Difficulty 4)
    5. Math Word Problem (Difficulty 2)
    6. Cryptarithmetic (Difficulty 4)
    7. Monty Hall (Difficulty 3)
    8. Cheryl's Birthday (Difficulty 5)
    9. Scheduling Problem (Difficulty 4)

METRICS TRACKED:
    - Success Rate (%)
    - Average Attempts
    - Average Duration (ms)
    - Constraint Score (0.0-1.0, neuro-symbolic mode only)
    - Contradictions Detected

EXPECTED IMPROVEMENTS WITH NEURO-SYMBOLIC:
    - Higher success rate (+10-30%)
    - Fewer attempts required (-20-40%)
    - Faster solving (constraints guide search)
    - Fewer contradictions (detected and avoided)
"""
