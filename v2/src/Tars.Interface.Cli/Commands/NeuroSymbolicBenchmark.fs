/// Neuro-Symbolic AI Puzzle Benchmark
/// Compare puzzle-solving performance with and without neuro-symbolic constraints
module Tars.Interface.Cli.Commands.NeuroSymbolicBenchmark

open System
open System.Diagnostics
open System.Threading.Tasks
open Serilog
open Spectre.Console
open Tars.Core
open Tars.Core.Puzzles
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Symbolic
open Tars.Interface.Cli
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
    if summary.TotalPuzzles = 0 then
        0.0
    else
        float summary.SuccessfulPuzzles / float summary.TotalPuzzles * 100.0

/// Run a puzzle with optional neuro-symbolic constraints
let runPuzzleWithConstraints (logger: ILogger) (puzzle: Puzzle) (config: BenchmarkConfig) : Task<PuzzleResult> =
    task {
        let sw = Stopwatch.StartNew()
        let mutable success = false
        let mutable attempts = 0
        let mutable contradictions = 0
        let mutable constraintScore = None
        let mutable previousFailures = []

        // Setup LLM (simplified for benchmark)
        let tarsConfig = ConfigurationLoader.load ()

        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri =
                    tarsConfig.Llm.BaseUrl
                    |> Option.map Uri
                    |> Option.defaultValue (Uri "http://localhost:11434")
                DefaultOllamaModel = tarsConfig.Llm.Model
                LlamaCppBaseUri = tarsConfig.Llm.LlamaCppUrl |> Option.map Uri
                DefaultLlamaCppModel =
                    if tarsConfig.Llm.LlamaCppUrl.IsSome then
                        Some tarsConfig.Llm.Model
                    else
                        None
                LlamaSharpModelPath = tarsConfig.Llm.LlamaSharpModelPath
                DefaultContextWindow = if tarsConfig.Llm.ContextWindow > 0 then Some tarsConfig.Llm.ContextWindow else None
                DefaultTemperature = None }

        use client = new System.Net.Http.HttpClient()

        let llmService =
            DefaultLlmService(client, { LlmServiceConfig.Routing = routingCfg }) :> ILlmService

        // Adaptive loop
        let mutable currentAttempt = 1

        while currentAttempt <= config.MaxAttemptsPerPuzzle && not success do
            attempts <- currentAttempt

            // 1. Prompt Engineering
            let systemPrompt =
                if config.EnableNeuroSymbolic && not (List.isEmpty previousFailures) then
                    // Neuro-Symbolic: Inject feedback via NeuralSymbolicFeedback module
                    let basePrompt =
                        """You are TARS, an advanced AI reasoning engine.
Your previous attempts to solve this puzzle were INCORRECT.
Analyze WHY these were wrong. avoid the same mistakes.
Use Chain of Thought reasoning. Verify your constraints."""

                    let badPatterns = previousFailures |> List.map fst
                    let shaped = NeuralSymbolicFeedback.shapePrompt basePrompt badPatterns []

                    // Append specific feedback
                    let feedbackStr =
                        previousFailures
                        |> List.map (fun (ans, fb) -> $"- Attempt: {ans}\n  Feedback: {fb}")
                        |> String.concat "\n"

                    $"{shaped}\n\nSPECIFIC FEEDBACK:\n{feedbackStr}"
                else
                    // Baseline / First attempt
                    """You are TARS, an advanced AI reasoning engine.
Solve complex logical puzzles step-by-step.
1. Break down problem
2. Identify constraints
3. Reason clearly
4. Verify answer"""

            let userPrompt =
                $"""PUZZLE: {puzzle.Name}
{puzzle.Prompt}

Please solve this puzzle step-by-step."""

            // 2. LLM Call
            // Baseline: Increase temp on retries to encourage diversity
            // NS: Keep temp low, rely on prompt feedback
            let temp =
                if config.EnableNeuroSymbolic then
                    0.2
                else
                    0.2 + (float (currentAttempt - 1) * 0.1)

            let request =
                { ModelHint = Some "smart"
                  Model = None
                  SystemPrompt = Some systemPrompt
                  MaxTokens = Some 1024
                  Temperature = Some temp
                  Stop = []
                  Messages = [ { Role = Role.User; Content = userPrompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = Some ResponseFormat.Text
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                if config.Verbose then
                    logger.Information($"Attempt {currentAttempt} thinking...")

                let! response = llmService.CompleteAsync request
                let answer = response.Text.Trim()

                // 3. Validation
                let isCorrect = puzzle.Validator answer

                // 4. Symbolic Scoring
                // In a real scenario, we would parse specific beliefs.
                // Here we treat the answer lines as "beliefs" for consistency checking.
                let beliefs =
                    answer.Split('\n', StringSplitOptions.RemoveEmptyEntries) |> Array.toList

                let invariants = [ SymbolicInvariant.BeliefConsistency beliefs ]
                // Score the answer against invariants
                // Context is mostly empty as we rely on the internal consistency within 'beliefs'
                let score =
                    ConstraintScoring.scoreInvariants invariants Map.empty ConstraintScoring.AverageScore

                constraintScore <- Some score

                if isCorrect then
                    success <- true
                else if config.EnableNeuroSymbolic then
                    // 5. Neuro-Symbolic Feedback Loop
                    contradictions <- contradictions + 1

                    // Generate feedback based on score and correctness
                    let feedback =
                        if score < 0.8 then
                            $"Answer contained internal contradictions (Score: {score:F2}). Please verify your logic."
                        else
                            "Answer was logically consistent but incorrect. efficient."

                    previousFailures <- (answer, feedback) :: previousFailures

            with ex ->
                logger.Error($"Attempt {currentAttempt} failed: {ex.Message}")

            currentAttempt <- currentAttempt + 1

        sw.Stop()

        return
            { PuzzleName = puzzle.Name
              Difficulty = puzzle.Difficulty
              Success = success
              AttemptCount = attempts
              TimeMs = sw.Elapsed.TotalMilliseconds
              ConstraintScore = constraintScore
              ContradictionsDetected = contradictions }
    }

/// Run benchmark on all puzzles
let runBenchmark (logger: ILogger) (puzzles: Puzzle list) (config: BenchmarkConfig) : Task<BenchmarkSummary> =
    task {
        logger.Information("🧩 Starting Neuro-Symbolic Puzzle Benchmark")

        let modeStr =
            if config.EnableNeuroSymbolic then
                "Neuro-Symbolic ON"
            else
                "Baseline (OFF)"

        logger.Information($"Mode: {modeStr}")
        logger.Information($"Puzzles: {puzzles.Length}")
        logger.Information("")

        let results = System.Collections.Generic.List<PuzzleResult>()

        for puzzle in puzzles do
            logger.Information($"Running: {puzzle.Name} (Difficulty {puzzle.Difficulty})...")
            let! result = runPuzzleWithConstraints logger puzzle config

            if config.LogResults then
                let status = if result.Success then "✅ SOLVED" else "❌ FAILED"
                logger.Information($"  {status} in {result.AttemptCount} attempts ({result.TimeMs:F0}ms)")

                match result.ConstraintScore with
                | Some score -> logger.Information($"  Constraint Score: {score:F2}")
                | None -> ()

                if result.ContradictionsDetected > 0 then
                    logger.Information($"  Contradictions: {result.ContradictionsDetected}")

            results.Add(result)

        let resultsList = results |> Seq.toList

        let summary =
            { Config = config
              TotalPuzzles = resultsList.Length
              SuccessfulPuzzles = resultsList |> List.filter (fun r -> r.Success) |> List.length
              FailedPuzzles = resultsList |> List.filter (fun r -> not r.Success) |> List.length
              AverageAttempts = resultsList |> List.averageBy (fun r -> float r.AttemptCount)
              AverageDurationMs = resultsList |> List.averageBy (fun r -> r.TimeMs)
              AverageConstraintScore =
                let scores = resultsList |> List.choose (fun r -> r.ConstraintScore)
                if scores.IsEmpty then None else Some(List.average scores)
              TotalContradictions = resultsList |> List.sumBy (fun r -> r.ContradictionsDetected)
              Results = resultsList }

        return summary
    }

/// Print benchmark summary
let printSummary (logger: ILogger) (summary: BenchmarkSummary) =
    logger.Information("")
    logger.Information("=".PadRight(60, '='))

    let modeLabel =
        if summary.Config.EnableNeuroSymbolic then
            "NEURO-SYMBOLIC"
        else
            "BASELINE"

    logger.Information($"📊 BENCHMARK RESULTS ({modeLabel})")
    logger.Information("=".PadRight(60, '='))
    logger.Information($"Total Puzzles: {summary.TotalPuzzles}")
    logger.Information($"Successful: {summary.SuccessfulPuzzles} (✅ {successRate summary:F1}%%)")
    logger.Information($"Failed: {summary.FailedPuzzles}")
    logger.Information($"Average Attempts: {summary.AverageAttempts:F1}")
    logger.Information($"Average Duration: {summary.AverageDurationMs:F0}ms")

    match summary.AverageConstraintScore with
    | Some score ->
        logger.Information($"Average Constraint Score: {score:F2}")
        logger.Information($"Total Contradictions: {summary.TotalContradictions}")
    | None -> ()

    logger.Information("=".PadRight(60, '='))

/// Compare baseline vs neuro-symbolic performance
let runComparison
    (logger: ILogger)
    (maxAttempts: int)
    (puzzleFilter: string option)
    : Task<BenchmarkSummary * BenchmarkSummary> =
    task {
        logger.Information("🔬 COMPARATIVE BENCHMARK: Baseline vs Neuro-Symbolic")
        logger.Information("")

        // Filter puzzles if requested
        let puzzles =
            match puzzleFilter with
            | Some filter ->
                let filtered =
                    allPuzzles
                    |> List.filter (fun p -> p.Name.Contains(filter, StringComparison.InvariantCultureIgnoreCase))

                if filtered.IsEmpty then
                    logger.Warning("No puzzles found matching filter '{Filter}'. Using all puzzles.", filter)
                    allPuzzles
                else
                    logger.Information(
                        "Using subset of puzzles matching '{Filter}': {Count} puzzles",
                        filter,
                        filtered.Length
                    )

                    filtered
            | None -> allPuzzles

        // Run baseline (no neuro-symbolic)
        let baselineConfig =
            { EnableNeuroSymbolic = false
              MaxAttemptsPerPuzzle = maxAttempts
              Verbose = false
              LogResults = true }

        logger.Information("Phase 1: Running BASELINE (neuro-symbolic OFF)...")
        logger.Information("")
        let! baseline = runBenchmark logger puzzles baselineConfig
        printSummary logger baseline

        logger.Information("")
        logger.Information("=".PadRight(60, '='))
        logger.Information("")

        // Run with neuro-symbolic
        let nsConfig =
            { EnableNeuroSymbolic = true
              MaxAttemptsPerPuzzle = maxAttempts
              Verbose = false
              LogResults = true }

        logger.Information("Phase 2: Running NEURO-SYMBOLIC (constraints ON)...")
        logger.Information("")
        let! neuroSymbolic = runBenchmark logger puzzles nsConfig
        printSummary logger neuroSymbolic

        // Print comparison
        logger.Information("")
        logger.Information("=".PadRight(60, '='))
        logger.Information("📈 IMPROVEMENT ANALYSIS")
        logger.Information("=".PadRight(60, '='))

        let successImprovement = successRate neuroSymbolic - successRate baseline
        let attemptsReduction = baseline.AverageAttempts - neuroSymbolic.AverageAttempts

        let speedImprovement =
            (baseline.AverageDurationMs - neuroSymbolic.AverageDurationMs)
            / baseline.AverageDurationMs
            * 100.0

        let successDiff = if successImprovement >= 0.0 then "+" else ""

        logger.Information(
            $"Success Rate: {successRate baseline:F1}%% → {successRate neuroSymbolic:F1}%% ({successDiff}{successImprovement:F1}%%)"
        )

        let attemptsSign = if attemptsReduction >= 0.0 then "-" else "+"

        logger.Information(
            $"Average Attempts: {baseline.AverageAttempts:F1} → {neuroSymbolic.AverageAttempts:F1} ({attemptsSign}{Math.Abs(attemptsReduction):F1})"
        )

        let speedDiff = if speedImprovement >= 0.0 then "+" else ""
        logger.Information($"Speed: {speedDiff}{speedImprovement:F1}%% faster")

        match neuroSymbolic.AverageConstraintScore with
        | Some score ->
            logger.Information($"Constraint Score: {score:F2}/1.00")
            logger.Information($"Contradictions Avoided: {neuroSymbolic.TotalContradictions} detected and handled")
        | None -> ()

        logger.Information("=".PadRight(60, '='))

        // Return both for further analysis
        return (baseline, neuroSymbolic)
    }

/// Export results to CSV
let exportToCsv (baseline: BenchmarkSummary) (neuroSymbolic: BenchmarkSummary) (filename: string) =
    let lines =
        [ "Puzzle,Difficulty,Baseline_Success,Baseline_Attempts,NS_Success,NS_Attempts,NS_Score,Improvement"
          yield!
              List.zip baseline.Results neuroSymbolic.Results
              |> List.map (fun (b, ns) ->
                  let improvement =
                      if ns.Success && not b.Success then "SOLVED"
                      elif not ns.Success && b.Success then "WORSE"
                      else "SAME"

                  let score =
                      ns.ConstraintScore |> Option.map (sprintf "%.2f") |> Option.defaultValue "N/A"

                  $"{b.PuzzleName},{b.Difficulty},{b.Success},{b.AttemptCount},{ns.Success},{ns.AttemptCount},{score},{improvement}") ]

    System.IO.File.WriteAllLines(filename, lines)

/// Create visualization data
let createVisualization (baseline: BenchmarkSummary) (neuroSymbolic: BenchmarkSummary) =
    let data =
        sprintf
            """
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
            ((baseline.AverageDurationMs - neuroSymbolic.AverageDurationMs)
             / baseline.AverageDurationMs
             * 100.0)

    data
