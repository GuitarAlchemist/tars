module Tars.Interface.Cli.Commands.PuzzleDemo

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Core.Puzzles // Use shared puzzles
open Tars.Core.InternetPuzzles
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Interface.Cli
open Tars.Interface.Cli.ConsoleHelpers
open Spectre.Console
open Tars.Cortex
open Tars.Cortex.Patterns

/// Expose all puzzles for testing and external access
let allPuzzles = Puzzles.all


// ============================================================================
// LLM SERVICE FACTORY (FOR DEMO)
// ============================================================================

// LLM Service Factory moved to Tars.Interface.Cli.LlmFactory


// ============================================================================
// COMMAND-LINE OPTIONS
// ============================================================================

/// Output format for puzzle results
type OutputFormat =
    | Text
    | Json

/// Reasoning strategy for puzzles
type ReasoningStrategy =
    | Standard
    | ChainOfThought
    | TreeOfThoughts
    | GraphOfThoughts
    | WorkflowOfThoughts

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
        /// Fetch puzzles from internet sources
        Internet: bool
        /// Number of internet puzzles to fetch per source
        InternetCount: int
        /// Reasoning strategy (Standard, CoT, ToT, GoT, WoT)
        Strategy: ReasoningStrategy
        /// Branching factor for ToT/GoT/WoT
        BranchingFactor: int option
        /// Maximum depth for ToT/GoT/WoT
        MaxDepth: int option
    }

/// Default options
let defaultOptions =
    { All = false
      Difficulty = None
      PuzzleName = None
      Verbose = false
      OutputFormat = Text
      BenchmarkRuns = 0
      ExportPath = None
      Internet = false
      InternetCount = 3
      Strategy = Standard
      BranchingFactor = None
      MaxDepth = None }

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

// ============================================================================
// PATTERN INFRASTRUCTURE (STUBS FOR DEMO)
// ============================================================================

let private stubRegistry = 
    { new IAgentRegistry with
        member _.GetAgent _ = async { return None }
        member _.FindAgents _ = async { return [] }
        member _.GetAllAgents () = async { return [] } }

let private stubExecutor =
    { new IAgentExecutor with
        member _.Execute(_, _) = async { return Failure [ PartialFailure.Error "Executor not available in demo" ] } }

let private createPuzzleContext (logger: ILogger) =
    { Self = 
        { Id = AgentId(Guid.NewGuid())
          Name = "PuzzleSolver"
          Version = "2.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "default"
          SystemPrompt = "Expert Solver"
          Tools = []
          Capabilities = []
          State = AgentState.Idle
          Memory = [] }
      Registry = stubRegistry
      Executor = stubExecutor
      Logger = fun msg -> logger.Information("[Pattern] {Message}", msg)
      Budget = None
      Epistemic = None
      SemanticMemory = None
      KnowledgeGraph = None
      CapabilityStore = None
      Audit = None
      CancellationToken = System.Threading.CancellationToken.None }

// ============================================================================
// PUZZLE RUNNER
// ============================================================================

/// Run a single puzzle and measure latency
let runPuzzle (logger: ILogger) (puzzle: Puzzle) (options: PuzzleOptions) =
    task {
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let verbose = options.Verbose

        // Log puzzle details
        logger.Information("🧩 Solving: {Name} (Difficulty: {Difficulty}) - Strategy: {Strategy}", 
            puzzle.Name, puzzle.Difficulty, options.Strategy)

        if verbose then
            AnsiConsole.MarkupLine($"[grey]📝 Description: {Markup.Escape(puzzle.Description)}[/]")
            AnsiConsole.MarkupLine($"[grey]❓ Prompt: {Markup.Escape(puzzle.Prompt)}[/]")

        // Initialize LLM Service
        let llmService = LlmFactory.create (logger)
        
        // 1. Map Puzzle Type to Model Hint
        let modelHint =
            match puzzle.Type with
            | PuzzleType.MathWord -> Some "math"
            | PuzzleType.TheoryOfMind when puzzle.Difficulty >= 4 -> Some "complex reasoning"
            | _ -> Some "reasoning"
        
        let! result =
            match options.Strategy with
            | Standard ->
                task {
                    if verbose then
                        let hint = modelHint |> Option.defaultValue "default"
                        AnsiConsole.MarkupLine($"[bold]🧠 Routing via Hint:[/] [cyan]{hint}[/]")

                    // 2. Setup Request
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
                        { LlmRequest.Default with
                            ModelHint = modelHint
                            SystemPrompt = Some systemPrompt
                            Temperature = Some 0.2
                            Messages = [ { Role = Role.User; Content = userPrompt } ]
                            Stream = options.Verbose }

                    // 2. Routing
                    let! routed = llmService.RouteAsync request
                    let modelName = sprintf "%A" routed.Backend

                    if verbose then
                        AnsiConsole.MarkupLine($"[bold blue]🔌 Provider:[/] [yellow]{routed.Backend}[/]")
                        AnsiConsole.MarkupLine($"[bold blue]📡 Endpoint:[/] [cyan]{routed.Endpoint}[/]")
                        AnsiConsole.MarkupLine($"[bold]🧠 Model Selection:[/] [cyan]{modelName}[/]")
                        AnsiConsole.WriteLine()

                    // 3. Optional Warmup (only in verbose)
                    if verbose then
                        AnsiConsole.MarkupLine("[grey]� Warming up model...[/]")
                        let warmupReq = { request with MaxTokens = Some 5; Messages = [{ Role = Role.User; Content = "hi" }] }
                        let! _ = llmService.CompleteAsync warmupReq
                        AnsiConsole.MarkupLine("[grey]⏱️  Running speed test...[/]")
                        let! preflightResp = llmService.CompleteAsync request
                        match preflightResp.Usage with
                        | Some u ->
                            let rate = float u.CompletionTokens / sw.Elapsed.TotalSeconds
                            AnsiConsole.MarkupLine($"[bold yellow]⚡ {rate:F1} tok/s[/]")
                        | None -> ()

                    // 4. Execute
                    let! answer =
                        if verbose then
                            task {
                                AnsiConsole.MarkupLine($"[bold purple]🚀 EXECUTING LLM REQUEST[/]")
                                let fullTextBuilder = System.Text.StringBuilder()
                                let tokenCountRef = ref 0
                                let! _ = llmService.CompleteStreamAsync(request, 
                                    fun token -> 
                                        tokenCountRef := !tokenCountRef + 1
                                        fullTextBuilder.Append(token) |> ignore
                                        Console.Write(token))
                                AnsiConsole.WriteLine()
                                return fullTextBuilder.ToString()
                            }
                        else
                            task {
                                let! resp = llmService.CompleteAsync request
                                return resp.Text
                            }

                    let answer = answer.Trim()
                    let isCorrect = puzzle.Validator answer

                    if isCorrect then
                        logger.Information("✅ {Name}: SOLVED", puzzle.Name)
                    else
                        logger.Warning("❌ {Name}: FAILED", puzzle.Name)

                    sw.Stop()
                    let metrics = 
                        { LatencyMs = sw.ElapsedMilliseconds
                          TokenCount = 0 
                          ModelName = modelName
                          CostEstimate = 0.0 }

                    if isCorrect then
                        return PuzzleRunResult.Success(puzzle, answer, metrics)
                    else
                        return PuzzleRunResult.Failure(puzzle, answer, metrics, "Validation failed")
                }

            | ChainOfThought ->
                task {
                    let ctx = createPuzzleContext logger
                    
                    // Multi-step reasoning: Decompose -> Solve -> Verify
                    let steps = [
                        // Step 1: Analyze and Decompose
                        (fun (input: string) -> agent {
                            let req = { ModelHint = Some "reasoning"; Model = None; SystemPrompt = Some "Decompose the following puzzle into its logical constraints."; Messages = [{ Role = Role.User; Content = input }]; MaxTokens = Some 1000; Temperature = Some 0.1; Stop = []; Tools = []; ToolChoice = None; ResponseFormat = None; Stream = false; JsonMode = false; Seed = None; ContextWindow = None }
                            let! resp = llmService.CompleteAsync req |> Async.AwaitTask
                            return resp.Text
                        })
                        // Step 2: Solve step-by-step
                        (fun (analysis: string) -> agent {
                            let prompt = $"Puzzle: {puzzle.Description}\n\nAnalysis of constraints:\n{analysis}\n\nPlease solve it."
                            let req = { ModelHint = Some "reasoning"; Model = None; SystemPrompt = Some "Solve the puzzle based on the provided analysis."; Messages = [{ Role = Role.User; Content = prompt }]; MaxTokens = Some 2000; Temperature = Some 0.1; Stop = []; Tools = []; ToolChoice = None; ResponseFormat = None; Stream = false; JsonMode = false; Seed = None; ContextWindow = None }
                            let! resp = llmService.CompleteAsync req |> Async.AwaitTask
                            return resp.Text
                        })
                    ]
                    
                    let workflow = Patterns.chainOfThought steps puzzle.Description
                    let! outcome = workflow ctx |> Async.StartAsTask
                    
                    let answer = 
                        match outcome with
                        | ExecutionOutcome.Success a -> a
                        | ExecutionOutcome.PartialSuccess(a, _) -> a
                        | ExecutionOutcome.Failure e -> sprintf "Pattern failure: %A" e

                    sw.Stop()
                    let metrics = { LatencyMs = sw.ElapsedMilliseconds; TokenCount = 0; ModelName = "ChainOfThought"; CostEstimate = 0.0 }
                    let isCorrect = puzzle.Validator answer
                    
                    if isCorrect then return PuzzleRunResult.Success(puzzle, answer, metrics)
                    else return PuzzleRunResult.Failure(puzzle, answer, metrics, "Chain of Thought failed")
                }

            | TreeOfThoughts ->
                task {
                    let ctx = createPuzzleContext logger
                    let config = { defaultGoTConfig with 
                                    BranchingFactor = options.BranchingFactor |> Option.defaultValue 2
                                    MaxDepth = options.MaxDepth |> Option.defaultValue 2 }
                    let workflow = Patterns.treeOfThoughts llmService config puzzle.Description
                    let! outcome = workflow ctx |> Async.StartAsTask
                    
                    let answer = 
                        match outcome with
                        | ExecutionOutcome.Success a -> a
                        | ExecutionOutcome.PartialSuccess(a, _) -> a
                        | ExecutionOutcome.Failure e -> sprintf "Pattern failure: %A" e

                    sw.Stop()
                    let metrics = { LatencyMs = sw.ElapsedMilliseconds; TokenCount = 0; ModelName = "TreeOfThoughts"; CostEstimate = 0.0 }
                    if puzzle.Validator answer then return PuzzleRunResult.Success(puzzle, answer, metrics)
                    else return PuzzleRunResult.Failure(puzzle, answer, metrics, "Tree of Thoughts failed")
                }

            | GraphOfThoughts ->
                task {
                    let ctx = createPuzzleContext logger
                    let config = { defaultGoTConfig with 
                                    BranchingFactor = options.BranchingFactor |> Option.defaultValue 2
                                    MaxDepth = options.MaxDepth |> Option.defaultValue 2 }
                    let workflow = Patterns.graphOfThoughts llmService config puzzle.Description
                    let! outcome = workflow ctx |> Async.StartAsTask
                    
                    let answer = 
                        match outcome with
                        | ExecutionOutcome.Success a -> a
                        | ExecutionOutcome.PartialSuccess(a, _) -> a
                        | ExecutionOutcome.Failure e -> sprintf "Pattern failure: %A" e

                    sw.Stop()
                    let metrics = { LatencyMs = sw.ElapsedMilliseconds; TokenCount = 0; ModelName = "GraphOfThoughts"; CostEstimate = 0.0 }
                    if puzzle.Validator answer then return PuzzleRunResult.Success(puzzle, answer, metrics)
                    else return PuzzleRunResult.Failure(puzzle, answer, metrics, "Graph of Thoughts failed")
                }

            | WorkflowOfThoughts ->
                task {
                    let ctx = createPuzzleContext logger
                    let config = { defaultWoTConfig with 
                                    BaseConfig = { defaultGoTConfig with BranchingFactor = 2; MaxDepth = 2 } }
                    let workflow = Patterns.workflowOfThought llmService config puzzle.Description
                    let! outcome = workflow ctx |> Async.StartAsTask
                    
                    let answer = 
                        match outcome with
                        | ExecutionOutcome.Success a -> a
                        | ExecutionOutcome.PartialSuccess(a, _) -> a
                        | ExecutionOutcome.Failure e -> sprintf "Pattern failure: %A" e

                    sw.Stop()
                    let metrics = { LatencyMs = sw.ElapsedMilliseconds; TokenCount = 0; ModelName = "WorkflowOfThoughts"; CostEstimate = 0.0 }
                    if puzzle.Validator answer then return PuzzleRunResult.Success(puzzle, answer, metrics)
                    else return PuzzleRunResult.Failure(puzzle, answer, metrics, "Workflow of Thoughts failed")
                }
                
        return result
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

    for p in Puzzles.all do
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
        let mutable lastResult = 
            PuzzleRunResult.Failure(puzzle, "", { LatencyMs = 0L; TokenCount = 0; ModelName = "init"; CostEstimate = 0.0 }, "Initial state")

        for i in 1..runs do
            if verbose then
                logger.Information("  Run {Current}/{Total}...", i, runs)

            let! result = runPuzzle logger puzzle { defaultOptions with Verbose = false }
            let latency = 
                match result with
                | PuzzleRunResult.Success(_, _, m) -> m.LatencyMs
                | PuzzleRunResult.Failure(_, _, m, _) -> m.LatencyMs
            
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
                let! result = runPuzzle logger puzzle { defaultOptions with Verbose = verbose }
                let (correct, status) = 
                    match result with
                    | PuzzleRunResult.Success _ -> 
                        correctCount <- correctCount + 1
                        (true, "Pass")
                    | PuzzleRunResult.Failure _ -> (false, "Fail")

                results.Add((puzzle.Name, correct, status))
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
                    
                    let (correct, answer, latency) = 
                        match result with
                        | PuzzleRunResult.Success(_, a, m) -> (true, a, m.LatencyMs)
                        | PuzzleRunResult.Failure(_, a, m, _) -> (false, a, m.LatencyMs)

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
                    let! result = runPuzzle logger puzzle options
                    
                    let (correct, answer, latency) = 
                        match result with
                        | PuzzleRunResult.Success(_, a, m) -> (true, a, m.LatencyMs)
                        | PuzzleRunResult.Failure(_, a, m, _) -> (false, a, m.LatencyMs)

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
let runAll (logger: ILogger) (verbose: bool) = runPuzzles logger Puzzles.all verbose

/// Run puzzles by difficulty
let runByDifficulty (logger: ILogger) (difficulty: int) (verbose: bool) =
    let puzzles = Puzzles.all |> List.filter (fun p -> p.Difficulty <= difficulty)
    runPuzzles logger puzzles verbose

/// Run a specific puzzle by name
let runByName (logger: ILogger) (name: string) (verbose: bool) =
    let puzzle =
        Puzzles.all
        |> List.tryFind (fun p -> p.Name.Equals(name, StringComparison.InvariantCultureIgnoreCase))

    match puzzle with
    | Some p -> runPuzzles logger [ p ] verbose
    | None ->
        logger.Error("Puzzle '{Name}' not found.", name)
        task { return 1 }

/// Run with options (new entry point)
let runWithOptions (logger: ILogger) (options: PuzzleOptions) =
    task {
        // Handle internet puzzles
        if options.Internet then
            logger.Information($"{Symbols.globe} Fetching puzzles from internet sources...")
            use http = new System.Net.Http.HttpClient()
            http.Timeout <- TimeSpan.FromSeconds(30.0)
            
            let sources = [
                InternetPuzzles.GSM8K          // Multi-step math word problems
                InternetPuzzles.HuggingFaceARC // Science reasoning
            ]
            
            let! internetPuzzles = InternetPuzzles.fetchPuzzles http sources options.InternetCount
            
            if internetPuzzles.IsEmpty then
                logger.Error("Failed to fetch any internet puzzles. Check your internet connection.")
                return 1
            else
                logger.Information($"{Symbols.checkmark} Fetched {{Count}} puzzles from internet sources", internetPuzzles.Length)
                return! runPuzzlesWithOptions logger internetPuzzles options
        else
            // Standard local puzzles
            let puzzles =
                if options.All then
                    Puzzles.all
                elif options.Difficulty.IsSome then
                    Puzzles.all |> List.filter (fun p -> p.Difficulty <= options.Difficulty.Value)
                elif options.PuzzleName.IsSome then
                    match
                        Puzzles.all
                        |> List.tryFind (fun p ->
                            p.Name.Equals(options.PuzzleName.Value, StringComparison.InvariantCultureIgnoreCase))
                    with
                    | Some p -> [ p ]
                    | None ->
                        logger.Error("Puzzle '{Name}' not found.", options.PuzzleName.Value)
                        []
                else
                    Puzzles.all

            if puzzles.IsEmpty then
                return 1
            else
                return! runPuzzlesWithOptions logger puzzles options
    }

/// Run (legacy entry point, mapped to Difficulty logic)
let run (logger: ILogger) (args: string array) =
    let difficulty =
        if args.Length > 0 then
            let mutable d = 0
            if Int32.TryParse(args.[0], &d) then d else 5
        else
            5

    runByDifficulty logger difficulty false
