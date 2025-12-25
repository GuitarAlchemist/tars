module Tars.Interface.Cli.Commands.PuzzleDemo

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Core.Puzzles // Use shared puzzles
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Interface.Cli
open Spectre.Console

// ============================================================================
// COMMAND-LINE OPTIONS
// ============================================================================

/// Output format for puzzle results
type OutputFormat =
    | Text
    | Json

/// Command-line options for the puzzle demo
type PuzzleOptions =
    {
        /// Run all puzzles
        All: bool
        /// Maximum difficulty level (1-5)
        Difficulty: int option
        /// Specific puzzle name to run
        PuzzleName: string option
        /// Show verbose output
        Verbose: bool
        /// Output format (text or json)
        OutputFormat: OutputFormat
        /// Number of benchmark iterations (0 = no benchmark)
        BenchmarkRuns: int
        /// Path to export results (None = no export)
        ExportPath: string option
    }

/// Default options
let defaultOptions =
    { All = false
      Difficulty = None
      PuzzleName = None
      Verbose = false
      OutputFormat = Text
      BenchmarkRuns = 0
      ExportPath = None }

// ============================================================================
// BENCHMARK AND EXPORT TYPES
// ============================================================================

/// Benchmark statistics for a puzzle
type BenchmarkStats =
    { Runs: int
      MinLatencyMs: int64
      MaxLatencyMs: int64
      AvgLatencyMs: float
      StdDevMs: float }

/// JSON-serializable puzzle result
type PuzzleJsonResult =
    { PuzzleName: string
      Difficulty: int
      Description: string
      Success: bool
      LatencyMs: int64
      Answer: string
      BenchmarkStats: BenchmarkStats option }

/// JSON-serializable demo result
type DemoJsonResult =
    { Timestamp: DateTime
      TotalPuzzles: int
      SuccessfulPuzzles: int
      FailedPuzzles: int
      TotalSolveTimeMs: int64
      Results: PuzzleJsonResult list }

// ============================================================================
// PUZZLE DEMO - Complex reasoning puzzles to test TARS capabilities
// ============================================================================

/// All available puzzles
let allPuzzles = Puzzles.all

// ============================================================================
// PUZZLE RUNNER
// ============================================================================

/// Run a single puzzle and measure latency
let runPuzzle (logger: ILogger) (puzzle: Puzzle) (verbose: bool) =
    task {
        let sw = System.Diagnostics.Stopwatch.StartNew()
        // Log puzzle details
        logger.Information("🧩 Solving: {Name} (Difficulty: {Difficulty})", puzzle.Name, puzzle.Difficulty)

        if verbose then
            AnsiConsole.MarkupLine($"[grey]📝 Description: {Markup.Escape(puzzle.Description)}[/]")
            AnsiConsole.MarkupLine($"[grey]❓ Prompt: {Markup.Escape(puzzle.Prompt)}[/]")

        // 1. Setup LLM service
        let config = ConfigurationLoader.load ()

        if verbose then
            AnsiConsole.MarkupLine("[bold]🔧 Configuration Loaded[/]")
            AnsiConsole.MarkupLine($"[grey]  Provider: {config.Llm.Provider}[/]")
            AnsiConsole.MarkupLine($"[grey]  Model: {config.Llm.Model}[/]")

        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri =
                    config.Llm.BaseUrl
                    |> Option.map Uri
                    |> Option.defaultValue (Uri "http://localhost:11434")
                DefaultOllamaModel = config.Llm.Model
                LlamaCppBaseUri = config.Llm.LlamaCppUrl |> Option.map Uri
                DefaultLlamaCppModel =
                    if config.Llm.LlamaCppUrl.IsSome then
                        Some config.Llm.Model
                    else
                        None
                LlamaSharpModelPath = config.Llm.LlamaSharpModelPath
                DefaultContextWindow = if config.Llm.ContextWindow > 0 then Some config.Llm.ContextWindow else None }

        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        use client = new System.Net.Http.HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(5.0) // Increase timeout for slow local models
        let llmService = DefaultLlmService(client, serviceConfig) :> ILlmService

        // 2. Build the prompt
        // Use a system prompt that encourages reasoning
        let systemPrompt =
            """You are TARS, an advanced AI reasoning engine. 
Your goal is to solve complex logical puzzles.

STRATEGY:
1. Break down the problem into key components
2. Identify constraints and rules
3. Reason step-by-step (Chain of Thought)
4. Verify your reasoning against the constraints
5. Provide a final, clear answer

If the puzzle involves logic, use deductive reasoning. 
If it involves math, show your calculations.
If it involves planning, list the steps clearly.

Be precise and avoid guessing."""

        let userPrompt =
            $"""PUZZLE: {puzzle.Name}
DESCRIPTION: {puzzle.Description}

PROBLEM:
{puzzle.Prompt}

Please solve this puzzle step-by-step."""

        let request =
            { ModelHint = Some "reasoning"
              Model = None
              SystemPrompt = Some systemPrompt
              MaxTokens = Some 1024
              Temperature = Some 0.2
              Stop = []
              Messages = [ { Role = Role.User; Content = userPrompt } ]
              Tools = []
              ToolChoice = None
              ResponseFormat = Some ResponseFormat.Text
              Stream = false
              JsonMode = false
              Seed = None
              ContextWindow = None }

        // 3. Execute LLM call with verbose status or quiet direct call
        // Display backend info (always show this for transparency)
        let routed = chooseBackend routingCfg request
        AnsiConsole.MarkupLine($"[bold blue]🔌 Provider:[/] [yellow]{routed.Backend}[/]")
        AnsiConsole.MarkupLine($"[bold blue]📡 Endpoint:[/] [cyan]{routed.Endpoint}[/]")
        
        // PRE-FLIGHT: Warmup & Speed Test
        AnsiConsole.MarkupLine("[grey]🔥 Warming up model (loading into VRAM)...[/]")
        let warmupReq =
            { ModelHint = Some "fast"
              Model = None
              SystemPrompt = None
              MaxTokens = Some 5  // Just enough to trigger load
              Temperature = Some 0.0
              Stop = []
              Messages = [ { Role = Role.User; Content = "hi" } ]
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None
              ContextWindow = None }
        
        // 1. Warmup call (forces model load)
        let! _ = llmService.CompleteAsync warmupReq

        // 2. Actual speed test
        AnsiConsole.MarkupLine("[grey]⏱️  Running speed test...[/]")
        let speedTestReq =
            { warmupReq with 
                MaxTokens = Some 100 
                Messages = [ { Role = Role.User; Content = "Count from 1 to 20. Output: 1, 2, 3..." } ] 
            }

        let preflightSw = System.Diagnostics.Stopwatch.StartNew()
        let! preflightResp = llmService.CompleteAsync speedTestReq
        preflightSw.Stop()

        match preflightResp.Usage with
        | Some u ->
            let preflightTokPerSec = float u.CompletionTokens / preflightSw.Elapsed.TotalSeconds
            // Adjust thresholds for the high-end GPU expectations
            if preflightTokPerSec < 20.0 then
                AnsiConsole.MarkupLine($"[bold red]⚠️  SLOW: {preflightTokPerSec:F1} tok/s[/] (expected 100+)")
            elif preflightTokPerSec < 80.0 then
                AnsiConsole.MarkupLine($"[bold yellow]⚡ {preflightTokPerSec:F1} tok/s[/] (moderate)")
            else
                AnsiConsole.MarkupLine($"[bold green]🚀 FAST: {preflightTokPerSec:F1} tok/s[/]")
        | None ->
            AnsiConsole.MarkupLine($"[grey]⏱️  {preflightSw.ElapsedMilliseconds}ms (no token count)[/]")
        AnsiConsole.WriteLine()

        let! answer =
            if verbose then
                task {
                    AnsiConsole.MarkupLine($"[bold purple]🚀 EXECUTING LLM REQUEST[/]")

                    AnsiConsole.MarkupLine("[bold]📨 Request Details:[/]")
                    let hintStr = request.ModelHint |> Option.defaultValue "none"
                    AnsiConsole.MarkupLine($"[grey]  Hint: {hintStr}[/]")
                    let maxTokensStr = request.MaxTokens |> Option.map string |> Option.defaultValue "default"
                    AnsiConsole.MarkupLine($"[grey]  MaxTokens: {maxTokensStr}[/]")
                    AnsiConsole.WriteLine()

                    let sw = System.Diagnostics.Stopwatch.StartNew()
                    let! response =
                        AnsiConsole
                            .Status()
                            .Spinner(Spinner.Known.Dots)
                            .SpinnerStyle(Style(foreground = Color.Cyan1))
                            .StartAsync(
                                "🧠 TARS is thinking...",
                                fun ctx ->
                                    task {
                                        let! resp = llmService.CompleteAsync request
                                        ctx.Status <- $"✅ Complete!"
                                        do! Task.Delay(500)
                                        return resp
                                    }
                            )
                    sw.Stop()
                    
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold green]✅ LLM REQUEST COMPLETED[/]")

                    match response.Usage with
                    | Some u ->
                        let tokensPerSec = float u.CompletionTokens / sw.Elapsed.TotalSeconds
                        AnsiConsole.MarkupLine(
                            $"[grey]  Tokens: {u.TotalTokens} (prompt: {u.PromptTokens}, completion: {u.CompletionTokens})[/]"
                        )
                        AnsiConsole.MarkupLine($"[bold yellow]  ⚡ Speed: {tokensPerSec:F1} tok/s[/]")
                        AnsiConsole.MarkupLine($"[grey]  Time: {sw.Elapsed.TotalSeconds:F1}s[/]")
                    | None ->
                        AnsiConsole.MarkupLine($"[grey]  Time: {sw.Elapsed.TotalSeconds:F1}s[/]")

                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine($"[grey]  Output Length: {response.Text.Length} chars[/]")
                    AnsiConsole.MarkupLine("[bold]Reasoning:[/]")
                    AnsiConsole.WriteLine(response.Text)
                    return response.Text
                }
            else
                task {
                    // Non-verbose: still show timing and token rate
                    let sw = System.Diagnostics.Stopwatch.StartNew()
                    let! response = llmService.CompleteAsync request
                    sw.Stop()
                    
                    match response.Usage with
                    | Some u ->
                        let tokensPerSec = float u.CompletionTokens / sw.Elapsed.TotalSeconds
                        AnsiConsole.MarkupLine($"[bold yellow]⚡ {tokensPerSec:F1} tok/s[/] | {u.CompletionTokens} tokens in {sw.Elapsed.TotalSeconds:F1}s")
                    | None ->
                        AnsiConsole.MarkupLine($"[grey]Time: {sw.Elapsed.TotalSeconds:F1}s[/]")
                    
                    AnsiConsole.MarkupLine("[bold]Reasoning:[/]")
                    AnsiConsole.WriteLine(response.Text)
                    return response.Text
                }

        let answer = answer.Trim()

        if verbose then
            AnsiConsole.Write(new Rule("[bold blue]Full Answer[/]"))
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(answer)}[/]")
            AnsiConsole.Write(new Rule())

        let isCorrect = puzzle.Validator answer

        if isCorrect then
            logger.Information("✅ {Name}: SOLVED", puzzle.Name)

            if verbose then
                AnsiConsole.MarkupLine("[green]CORRECT![/]")
        else
            logger.Warning("❌ {Name}: FAILED", puzzle.Name)

            if verbose then
                AnsiConsole.MarkupLine("[red]INCORRECT[/]")
                AnsiConsole.MarkupLine($"[yellow]Expected answer to contain logic matching: {puzzle.ExpectedAnswer}[/]")

        sw.Stop()
        return (puzzle, isCorrect, answer, sw.ElapsedMilliseconds)
    }

/// Display logo
let private logo () =
    AnsiConsole.Write((new FigletText("TARS PUZZLES")).Color(Color.Blue))

/// List all available puzzles
let listPuzzles () =
    AnsiConsole.Write(new Rule("[bold]Available Puzzles[/]"))
    let table = new Table()
    table.AddColumn("Name") |> ignore
    table.AddColumn("Difficulty") |> ignore
    table.AddColumn("Description") |> ignore

    for p in allPuzzles do
        table.AddRow(p.Name, p.Difficulty.ToString(), p.Description) |> ignore

    AnsiConsole.Write(table)
    logo ()
    0

/// Calculate benchmark statistics
let private calculateBenchmarkStats (latencies: int64 list) =
    if latencies.IsEmpty then
        None
    else
        let min = List.min latencies
        let max = List.max latencies
        let avg = List.average (latencies |> List.map float)

        let variance =
            latencies
            |> List.map float
            |> List.map (fun x -> (x - avg) ** 2.0)
            |> List.average

        let stdDev = sqrt variance

        Some
            { Runs = latencies.Length
              MinLatencyMs = min
              MaxLatencyMs = max
              AvgLatencyMs = avg
              StdDevMs = stdDev }

/// Run a single puzzle multiple times for benchmarking
let private runPuzzleBenchmark (logger: ILogger) (puzzle: Puzzle) (verbose: bool) (runs: int) =
    task {
        let latencies = System.Collections.Generic.List<int64>()
        let mutable lastResult = (puzzle, false, "", 0L)

        for i in 1..runs do
            if verbose then
                logger.Information("  Run {Current}/{Total}...", i, runs)

            let! result = runPuzzle logger puzzle false
            let (_, _, _, latency) = result
            latencies.Add(latency)
            lastResult <- result

        let stats = calculateBenchmarkStats (latencies |> Seq.toList)
        return (lastResult, stats)
    }

/// Run a set of puzzles
let private runPuzzles (logger: ILogger) (puzzles: Puzzle list) (verbose: bool) =
    task {
        logo ()
        logger.Information("Found {Count} puzzles", puzzles.Length)

        let mutable correctCount = 0
        let results = System.Collections.Generic.List<string * bool * string>()

        for puzzle in puzzles do
            try
                let! (_, correct, _, _) = runPuzzle logger puzzle verbose

                if correct then
                    correctCount <- correctCount + 1

                results.Add((puzzle.Name, correct, if correct then "Pass" else "Fail"))
            with ex ->
                logger.Error(ex, "Error running puzzle {Name}", puzzle.Name)
                results.Add((puzzle.Name, false, "Error"))

        // Summary
        AnsiConsole.Write(new Rule("[bold]SUMMARY[/]"))
        let table = new Table()
        table.AddColumn("Puzzle") |> ignore
        table.AddColumn("Result") |> ignore

        for (name, correct, status) in results do
            let color = if correct then "green" else "red"
            table.AddRow(name, $"[{color}]{status}[/]") |> ignore

        AnsiConsole.Write(table)

        if puzzles.Length > 0 then
            let score = float correctCount / float puzzles.Length * 100.0

            logger.Information(
                "🏁 Finished. Score: {Correct}/{Total} ({Score:F1}%)",
                correctCount,
                puzzles.Length,
                score
            )

        return 0
    }

/// Run puzzles with options (benchmark, export, etc.)
let runPuzzlesWithOptions (logger: ILogger) (puzzles: Puzzle list) (options: PuzzleOptions) =
    task {
        if options.OutputFormat = Text then
            logo ()

        logger.Information("Found {Count} puzzles", puzzles.Length)

        let mutable correctCount = 0
        let mutable totalTimeMs = 0L
        let jsonResults = System.Collections.Generic.List<PuzzleJsonResult>()

        for puzzle in puzzles do
            try
                if options.BenchmarkRuns > 0 then
                    // Benchmark mode
                    logger.Information("🧩 Benchmarking: {Name} ({Runs} runs)", puzzle.Name, options.BenchmarkRuns)
                    let! (result, stats) = runPuzzleBenchmark logger puzzle options.Verbose options.BenchmarkRuns
                    let (_, correct, answer, latency) = result

                    if correct then
                        correctCount <- correctCount + 1

                    totalTimeMs <- totalTimeMs + latency

                    jsonResults.Add(
                        { PuzzleName = puzzle.Name
                          Difficulty = puzzle.Difficulty
                          Description = puzzle.Description
                          Success = correct
                          LatencyMs = latency
                          Answer = answer
                          BenchmarkStats = stats }
                    )

                    // Show benchmark stats
                    match stats with
                    | Some s ->
                        logger.Information(
                            "  Stats: Min={Min}ms, Max={Max}ms, Avg={Avg:F1}ms, StdDev={StdDev:F1}ms",
                            s.MinLatencyMs,
                            s.MaxLatencyMs,
                            s.AvgLatencyMs,
                            s.StdDevMs
                        )
                    | None -> ()
                else
                    // Normal mode
                    let! (_, correct, answer, latency) = runPuzzle logger puzzle options.Verbose

                    if correct then
                        correctCount <- correctCount + 1

                    totalTimeMs <- totalTimeMs + latency

                    jsonResults.Add(
                        { PuzzleName = puzzle.Name
                          Difficulty = puzzle.Difficulty
                          Description = puzzle.Description
                          Success = correct
                          LatencyMs = latency
                          Answer = answer
                          BenchmarkStats = None }
                    )
            with ex ->
                logger.Error(ex, "Error running puzzle {Name}", puzzle.Name)

                jsonResults.Add(
                    { PuzzleName = puzzle.Name
                      Difficulty = puzzle.Difficulty
                      Description = puzzle.Description
                      Success = false
                      LatencyMs = 0L
                      Answer = $"Error: {ex.Message}"
                      BenchmarkStats = None }
                )

        // Create final result
        let demoResult =
            { Timestamp = DateTime.UtcNow
              TotalPuzzles = puzzles.Length
              SuccessfulPuzzles = correctCount
              FailedPuzzles = puzzles.Length - correctCount
              TotalSolveTimeMs = totalTimeMs
              Results = jsonResults |> Seq.toList }

        // Handle output format
        match options.OutputFormat with
        | Json ->
            let jsonOptions = JsonSerializerOptions(WriteIndented = true)
            let json = JsonSerializer.Serialize(demoResult, jsonOptions)
            printfn "%s" json
        | Text ->
            // Summary
            AnsiConsole.Write(new Rule("[bold]SUMMARY[/]"))
            let table = new Table()
            table.AddColumn("Puzzle") |> ignore
            table.AddColumn("Result") |> ignore
            table.AddColumn("Latency (ms)") |> ignore

            for result in jsonResults do
                let color = if result.Success then "green" else "red"
                let status = if result.Success then "Pass" else "Fail"

                table.AddRow(result.PuzzleName, $"[{color}]{status}[/]", result.LatencyMs.ToString())
                |> ignore

            AnsiConsole.Write(table)

            if puzzles.Length > 0 then
                let score = float correctCount / float puzzles.Length * 100.0

                logger.Information(
                    "🏁 Finished. Score: {Correct}/{Total} ({Score:F1}%)",
                    correctCount,
                    puzzles.Length,
                    score
                )

        // Export if requested
        match options.ExportPath with
        | Some path ->
            try
                let jsonOptions = JsonSerializerOptions(WriteIndented = true)
                let json = JsonSerializer.Serialize(demoResult, jsonOptions)
                File.WriteAllText(path, json)
                logger.Information("📁 Results exported to: {Path}", path)
            with ex ->
                logger.Error(ex, "Failed to export results to {Path}", path)
        | None -> ()

        return if correctCount = puzzles.Length then 0 else 1
    }

/// Run all puzzles
let runAll (logger: ILogger) (verbose: bool) = runPuzzles logger allPuzzles verbose

/// Run puzzles by difficulty
let runByDifficulty (logger: ILogger) (difficulty: int) (verbose: bool) =
    let puzzles = allPuzzles |> List.filter (fun p -> p.Difficulty <= difficulty)
    runPuzzles logger puzzles verbose

/// Run a specific puzzle by name
let runByName (logger: ILogger) (name: string) (verbose: bool) =
    let puzzle =
        allPuzzles
        |> List.tryFind (fun p -> p.Name.Equals(name, StringComparison.InvariantCultureIgnoreCase))

    match puzzle with
    | Some p -> runPuzzles logger [ p ] verbose
    | None ->
        logger.Error("Puzzle '{Name}' not found.", name)
        task { return 1 }

/// Run with options (new entry point)
let runWithOptions (logger: ILogger) (options: PuzzleOptions) =
    let puzzles =
        if options.All then
            allPuzzles
        elif options.Difficulty.IsSome then
            allPuzzles |> List.filter (fun p -> p.Difficulty <= options.Difficulty.Value)
        elif options.PuzzleName.IsSome then
            match
                allPuzzles
                |> List.tryFind (fun p ->
                    p.Name.Equals(options.PuzzleName.Value, StringComparison.InvariantCultureIgnoreCase))
            with
            | Some p -> [ p ]
            | None ->
                logger.Error("Puzzle '{Name}' not found.", options.PuzzleName.Value)
                []
        else
            allPuzzles

    if puzzles.IsEmpty then
        task { return 1 }
    else
        runPuzzlesWithOptions logger puzzles options

/// Run (legacy entry point, mapped to Difficulty logic)
let run (logger: ILogger) (args: string array) =
    let difficulty =
        if args.Length > 0 then
            let mutable d = 0
            if Int32.TryParse(args.[0], &d) then d else 5
        else
            5

    runByDifficulty logger difficulty false
