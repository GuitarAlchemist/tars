namespace Tars.Interface.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open Serilog
open System.Text.Json
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot
open Tars.Evolution
open Tars.Interface.Cli
open Tars.Interface.Cli.Reasoning
open Tars.Tools
open Tars.Cortex.WoTTypes

module WotCommand =

    // Tool invocation goes through Tars.Tools.ToolInvoker, which owns resilience
    // (circuit breaker + recording) and surfaces typed ToolOutcomes.

    type WotAction =
        | RunFile of path: string * options: WotExecution.RunOptions
        | Diff of runA: string * runB: string
        | CiCheck of workflow: string * baseline: string
        | Train of iterations: int * useLlm: bool
        | Help

    let parseArgs (args: string list) : WotAction =
        let rec parseOptions (acc: WotExecution.RunOptions) (rest: string list) =
            match rest with
            | [] -> acc
            | "--reason" :: "llm" :: tail -> parseOptions { acc with Mode = ReasonStepMode.Llm } tail
            | "--reason" :: "stub" :: tail -> parseOptions { acc with Mode = ReasonStepMode.Stub } tail
            | "--reason" :: "replay" :: tail ->
                parseOptions
                    { acc with
                        Mode = ReasonStepMode.Replay }
                    tail
            | "--replay-run" :: runId :: tail -> parseOptions { acc with ReplayRunId = Some runId } tail
            | "--model" :: name :: tail -> parseOptions { acc with Model = Some name } tail
            | "--model-hint" :: hint :: tail -> parseOptions { acc with ModelHint = Some hint } tail
            | "--temp" :: t :: tail ->
                match Double.TryParse(t) with
                | true, v -> parseOptions { acc with Temperature = Some v } tail
                | _ -> parseOptions acc tail
            | "--max-tokens" :: n :: tail ->
                match Int32.TryParse(n) with
                | true, v -> parseOptions { acc with MaxTokens = Some v } tail
                | _ -> parseOptions acc tail
            | "--deterministic" :: tail ->
                parseOptions
                    { acc with
                        Deterministic = true
                        Temperature = Some 0.0 }
                    tail
            | "--seed" :: s :: tail ->
                match Int32.TryParse(s) with
                | true, v -> parseOptions { acc with Seed = Some v } tail
                | _ -> parseOptions acc tail
            | "--cortex" :: tail -> parseOptions { acc with UseCortex = true } tail
            | _ :: tail -> parseOptions acc tail

        match args with
        | "run" :: path :: tail ->
            let opts = parseOptions WotExecution.RunOptions.Default tail
            RunFile(path, opts)
        | "diff" :: runA :: runB :: _ -> Diff(runA, runB)
        | "ci-check" :: flow :: baseLine :: _ -> CiCheck(flow, baseLine)
        | "curriculum" :: "train" :: rest ->
            let hasLlm = rest |> List.exists (fun s -> s = "--llm")
            let numStr = rest |> List.tryFind (fun s -> s <> "--llm")

            let iterations =
                match numStr with
                | Some n ->
                    match Int32.TryParse(n) with
                    | true, v -> v
                    | _ -> 10
                | None -> 10

            Train(iterations, hasLlm)
        | [ "help" ] -> Help
        | _ -> Help

    // Helper for running logic (Shared between RunFile and CiCheck)
    let private runWorkflow (path: string) (mode: ReasonStepMode) : Async<Result<CanonicalGolden * string, string>> =
        async {
            match Workflow.load path with
            | Result.Error err -> return Result.Error(String.concat "; " (Workflow.describe err))
            | Result.Ok plan ->
                // Create run directory for journaling
                let startTime = DateTime.UtcNow
                let runId = startTime.ToString("yyyyMMdd-HHmmss-fff")
                let runDir = System.IO.Path.Combine(".wot", "runs", runId)
                System.IO.Directory.CreateDirectory(runDir) |> ignore

                // Phase 14: Load Constitution
                let constitution =
                    if System.IO.File.Exists "constitution.json" then
                        match Tars.Core.ConstitutionLoader.load "constitution.json" with
                        | Result.Ok c -> Some c
                        | Result.Error e ->
                            AnsiConsole.MarkupLine($"[yellow]Warning: Failed to load constitution: {e}[/]")
                            None
                    else
                        None

                let reasoner, reflector =
                    match mode with
                    | ReasonStepMode.Llm ->
                        let logger = Log.Logger
                        let llm = LlmFactory.create logger

                        let reasonerSettings: ReasonerSettings =
                            { Model = None
                              ModelHint = Some "thought"
                              Temperature = None
                              MaxTokens = Some 2048
                              Deterministic = false
                              Seed = None
                              ContextWindow = None
                              AgentHint = None
                              GrammarConstraint = None }

                        // Phase 15: Symbolic Reflector
                        // We pass null graph service for now as it's not fully wired
                        let agentId = AgentId(Guid.NewGuid())

                        let ref =
                            new Tars.Evolution.SymbolicReflector(
                                llm,
                                Unchecked.defaultof<IGraphService>,
                                agentId
                            )
                            :> ISymbolicReflector

                        (CliReasoner(llm, runDir, reasonerSettings, logger) :> IReasoner), Some ref
                    | ReasonStepMode.Stub ->
                        { new IReasoner with
                            member _.Reason(_, _, _, _, _) =
                                async {
                                    return
                                        Result.Ok
                                            { Content = "<reason-step-stub>"
                                              Usage = None }
                                } },
                        None
                    | ReasonStepMode.Replay ->
                        // Replay mode not supported in training loop
                        { new IReasoner with
                            member _.Reason(_, _, _, _, _) =
                                async {
                                    return
                                        Result.Ok
                                            { Content = "<replay-not-supported>"
                                              Usage = None }
                                } },
                        None

                let! result =
                    WotExecution.runV0 (WotExecution.v0Tools ()) reasoner reflector constitution mode plan

                match result with
                | Result.Error(e, traces) -> return Result.Error e
                | Result.Ok(ctx, verifyC, traces) ->
                    return Result.Ok(WotExecution.toGolden mode ctx verifyC traces, path)
        }


    let execute (args: string list) : Task<int> =
        async {
            match parseArgs args with
            | Help ->
                printfn "TARS WoT Runner (v0.7)"
                printfn "Usage:"
                printfn "  tars wot run <file.wot.trsx> [OPTIONS]"
                printfn ""
                printfn "Options:"
                printfn "  --reason stub|llm|replay  Reasoning mode (default: stub)"
                printfn "  --replay-run <runId>      Run ID to replay (for --reason replay)"
                printfn "  --model <name>            Specific model to use"
                printfn "  --model-hint <hint>       Model routing hint (e.g. reasoning, code)"
                printfn "  --temp <float>            Temperature (0.0-2.0)"
                printfn "  --max-tokens <n>          Maximum tokens to generate"
                printfn "  --deterministic           Force temp=0, seed=42"
                printfn "  --seed <n>                Random seed for reproducibility"
                printfn "  --cortex                  Use Cortex WoT Executor (LLM-backed, full tracing)"
                printfn ""
                printfn "Other Commands:"
                printfn "  tars wot diff <runA> <runB>"
                printfn "  tars wot ci-check <workflow> <baseline>"
                printfn "  tars wot curriculum train [n] [--llm]  Run n training iterations"
                return 0

            | Train(iterations, useLlm) ->
                let modeStr = if useLlm then "LLM" else "Stub"

                AnsiConsole.MarkupLine(
                    $"[bold purple]🏋️ TARS Curriculum Trainer[/] (Iterations: {iterations}, Mode: {modeStr})"
                )

                let reasonMode = if useLlm then ReasonStepMode.Llm else ReasonStepMode.Stub
                let allProblems = Tars.Evolution.ProblemIngestor.getAllProblems "curriculum"

                let stateFile = ".tars/curriculum.json"
                let options = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
                options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())

                // Ensure .tars dir exists
                if not (System.IO.Directory.Exists(".tars")) then
                    System.IO.Directory.CreateDirectory(".tars") |> ignore

                let loadState () =
                    if System.IO.File.Exists(stateFile) then
                        try
                            let json = System.IO.File.ReadAllText(stateFile)
                            System.Text.Json.JsonSerializer.Deserialize<Tars.Evolution.CurriculumState>(json, options)
                        with _ ->
                            Tars.Evolution.CurriculumManager.init ()
                    else
                        Tars.Evolution.CurriculumManager.init ()

                let saveState (state: Tars.Evolution.CurriculumState) =
                    let json = System.Text.Json.JsonSerializer.Serialize(state, options)
                    System.IO.File.WriteAllText(stateFile, json)

                let results =
                    System.Collections.Generic.List<
                        {| Round: int
                           Id: string
                           Title: string
                           Status: string
                           Verify: bool
                           Time: string |}
                     >()

                let rec trainLoop count =
                    async {
                        if count <= 0 then
                            if results.Count > 0 then
                                AnsiConsole.WriteLine()
                                let table = Table().Border(TableBorder.Rounded)
                                table.AddColumn("[grey]Round[/]") |> ignore
                                table.AddColumn("[bold]Problem[/]") |> ignore
                                table.AddColumn("[bold]Status[/]") |> ignore
                                table.AddColumn("[bold]Verify[/]") |> ignore
                                table.AddColumn("[bold]Time[/]") |> ignore

                                for r in results do
                                    let statusColor = if r.Status = "SUCCESS" then "green" else "yellow"
                                    let verifyIcon = if r.Verify then "[green]PASS[/]" else "[red]FAIL[/]"

                                    table.AddRow(
                                        string r.Round,
                                        $"[dim]{r.Id}[/] {r.Title}",
                                        $"[{statusColor}]{r.Status}[/]",
                                        verifyIcon,
                                        r.Time
                                    )
                                    |> ignore

                                AnsiConsole.Write(table)

                            return 0
                        else
                            let state = loadState ()

                            match Tars.Evolution.CurriculumManager.getNextProblem state allProblems with
                            | None ->
                                AnsiConsole.MarkupLine("[bold green]Mastery Achieved! 🎉 No more problems.[/]")
                                return 0
                            | Some problem ->
                                let timer = System.Diagnostics.Stopwatch.StartNew()

                                AnsiConsole.MarkupLine(
                                    $"\n[bold]Round {iterations - count + 1}:[/] Solving '{problem.Title}'..."
                                )

                                // Try to find a workflow file for this problem
                                let problemIdStr =
                                    match problem.Id with
                                    | Tars.Evolution.ProblemId s -> s

                                // Map problem IDs to known puzzle workflow files
                                let puzzleMapping = Map.ofList [ "LOG-001", "puzzles/river_crossing.wot.trsx" ]

                                let possiblePaths =
                                    [ // Check mapped puzzles first
                                      yield! puzzleMapping |> Map.tryFind problemIdStr |> Option.toList
                                      // Then curriculum workflows
                                      $"curriculum/workflows/{problemIdStr}.wot.trsx"
                                      $"curriculum/{problemIdStr}.wot.trsx"
                                      $"workflows/{problemIdStr}.wot.trsx"
                                      $"puzzles/{problemIdStr}.wot.trsx"
                                      $".tars/workflows/{problemIdStr}.wot.trsx" ]

                                let workflowPath = possiblePaths |> List.tryFind System.IO.File.Exists

                                match workflowPath with
                                | Some path ->
                                    AnsiConsole.MarkupLine($"[dim]Found workflow: {path}[/]")
                                    // Run the actual workflow with the selected reasoning mode
                                    match! runWorkflow path reasonMode with
                                    | Result.Error e ->
                                        AnsiConsole.MarkupLine($"[red]Workflow Error:[/] {Markup.Escape(e)}")

                                        if useLlm then
                                            AnsiConsole.MarkupLine(
                                                "[bold purple]🧬 Triggering Self-Improvement analyze...[/]"
                                            )

                                            let llm = LlmFactory.create Log.Logger

                                            let! improvement =
                                                SelfImprovement.analyzeAndPropose llm path e "" (Guid.NewGuid())
                                                |> Async.AwaitTask

                                            match improvement with
                                            | SelfImprovement.Success(proposal, variant) ->
                                                AnsiConsole.MarkupLine(
                                                    $"[bold green]Mutation PROPOSED:[/] {Markup.Escape(proposal.Rationale)}"
                                                )

                                                AnsiConsole.MarkupLine($"[dim]Variant created at: {variant}[/]")
                                            | SelfImprovement.Failure err ->
                                                AnsiConsole.MarkupLine($"[dim]Self-improvement failed: {err}[/]")

                                        let newState = Tars.Evolution.CurriculumManager.recordFailure state problem.Id
                                        saveState newState
                                        return! trainLoop (count - 1)
                                    | Result.Ok(golden, _) ->
                                        timer.Stop()
                                        // Check verification result first, then fall back to execution success
                                        let hasOutput = golden.Summary.OutputKeys.Length > 0

                                        let success =
                                            match golden.Summary.VerifyPassed with
                                            | Some passed -> passed && golden.Summary.FirstError.IsNone
                                            | None -> hasOutput && golden.Summary.FirstError.IsNone

                                        AnsiConsole.MarkupLine(
                                            $"[dim]  Outputs: {golden.Summary.OutputKeys.Length}, VerifyPassed: {golden.Summary.VerifyPassed}[/]"
                                        )

                                        if success then
                                            AnsiConsole.MarkupLine("[green]SUCCESS![/] Problem solved.")

                                            // Phase 15.6: Pattern Compilation
                                            if useLlm then
                                                AnsiConsole.MarkupLine(
                                                    "[bold blue]🧠 Compiling Pattern from Success...[/]"
                                                )

                                                let llm = LlmFactory.create Log.Logger
                                                let runId = Guid.NewGuid() // Ideally we should get the real runId from somewhere, but golden doesn't have it

                                                try
                                                    let! patternRes =
                                                        TraceCompiler.compileFromTrace
                                                            llm
                                                            runId
                                                            golden.Steps
                                                            problem.Title

                                                    match patternRes with
                                                    | Result.Ok p ->
                                                        AnsiConsole.MarkupLine(
                                                            $"[green]Pattern Compiled:[/] {p.Name} (generic template extracted)"
                                                        )
                                                        // Save pattern
                                                        if not (System.IO.Directory.Exists(".tars/patterns")) then
                                                            System.IO.Directory.CreateDirectory(".tars/patterns")
                                                            |> ignore

                                                        let pJson =
                                                            System.Text.Json.JsonSerializer.Serialize(
                                                                p,
                                                                JsonSerializerOptions(WriteIndented = true)
                                                            )

                                                        System.IO.File.WriteAllText(
                                                            $".tars/patterns/{p.Name}.json",
                                                            pJson
                                                        )
                                                    | Result.Error e ->
                                                        AnsiConsole.MarkupLine(
                                                            $"[yellow]Pattern Compile Failed:[/] {e}"
                                                        )
                                                with ex ->
                                                    AnsiConsole.MarkupLine(
                                                        $"[yellow]Pattern Compile Error:[/] {ex.Message}"
                                                    )

                                            let newState =
                                                Tars.Evolution.CurriculumManager.recordSuccess state problem.Id

                                            saveState newState

                                            results.Add(
                                                {| Round = iterations - count + 1
                                                   Id = problemIdStr
                                                   Title = problem.Title
                                                   Status = "SUCCESS"
                                                   Verify = golden.Summary.VerifyPassed |> Option.defaultValue false
                                                   Time = $"{timer.Elapsed.TotalSeconds:F1}s" |}
                                            )
                                        else
                                            AnsiConsole.MarkupLine("[yellow]FAILURE.[/] Solution incorrect.")

                                            if useLlm then
                                                let firstError =
                                                    golden.Summary.FirstError
                                                    |> Option.defaultValue "Verification failed"

                                                AnsiConsole.MarkupLine(
                                                    "[bold purple]🧬 Triggering Neuro-Symbolic Self-Improvement...[/]"
                                                )

                                                let llm = LlmFactory.create Log.Logger

                                                let trace =
                                                    golden.Steps
                                                    |> List.collect (fun t -> t.Outputs)
                                                    |> String.concat "\n"

                                                let! improvement =
                                                    SelfImprovement.analyzeAndPropose
                                                        llm
                                                        path
                                                        firstError
                                                        trace
                                                        (Guid.NewGuid())
                                                    |> Async.AwaitTask

                                                match improvement with
                                                | SelfImprovement.Success(proposal, variant) ->
                                                    AnsiConsole.MarkupLine(
                                                        $"[bold green]Mutation PROPOSED:[/] {Markup.Escape(proposal.Rationale)}"
                                                    )

                                                    AnsiConsole.MarkupLine($"[dim]Variant created at: {variant}[/]")

                                                    // Log to symbolic memory
                                                    do!
                                                        SelfImprovement.logImprovement proposal variant true
                                                        |> Async.StartAsTask
                                                        |> Async.AwaitTask

                                                    AnsiConsole.MarkupLine("[bold blue]🧪 Verifying fix...[/]")
                                                    let! verifyResult = runWorkflow variant reasonMode

                                                    match verifyResult with
                                                    | Result.Ok(vGolden, _) ->
                                                        let vSuccess =
                                                            match vGolden.Summary.VerifyPassed with
                                                            | Some passed -> passed && vGolden.Summary.FirstError.IsNone
                                                            | None ->
                                                                vGolden.Summary.OutputKeys.Length > 0
                                                                && vGolden.Summary.FirstError.IsNone

                                                        if vSuccess then
                                                            AnsiConsole.MarkupLine(
                                                                "[bold green]✅ FIX VERIFIED! The validated variant solves the problem.[/]"
                                                            )
                                                        // In a real autonomous loop, we would overwrite the original file here.
                                                        // File.Copy(variant, path, true)
                                                        else
                                                            AnsiConsole.MarkupLine(
                                                                "[yellow]⚠️ Fix failed verification.[/]"
                                                            )
                                                    | Result.Error e ->
                                                        AnsiConsole.MarkupLine($"[red]❌ Fix failed execution: {e}[/]")
                                                | SelfImprovement.Failure err ->
                                                    AnsiConsole.MarkupLine($"[red]Self-improvement failed:[/] {err}")

                                            let newState =
                                                Tars.Evolution.CurriculumManager.recordFailure state problem.Id

                                            saveState newState

                                            results.Add(
                                                {| Round = iterations - count + 1
                                                   Id = problemIdStr
                                                   Title = problem.Title
                                                   Status = "FAILURE"
                                                   Verify = golden.Summary.VerifyPassed |> Option.defaultValue false
                                                   Time = $"{timer.Elapsed.TotalSeconds:F1}s" |}
                                            )

                                        return! trainLoop (count - 1)
                                | None ->
                                    // No workflow file found - simulate training with random success
                                    timer.Stop()

                                    results.Add(
                                        {| Round = iterations - count + 1
                                           Id = problemIdStr
                                           Title = problem.Title
                                           Status = "SIMULATED"
                                           Verify = false
                                           Time = "0.0s" |}
                                    )

                                    // Phase 16: Context Engineering & Validation
                                    if useLlm then
                                        AnsiConsole.MarkupLine(
                                            $"[dim]No workflow file. Searching patterns for '{problem.Title}'...[/]"
                                        )

                                        let llm = LlmFactory.create Log.Logger
                                        let patterns = PatternLibrary.loadAll ()

                                        let! bestMatch = PatternLibrary.findMatch llm problem.Description patterns

                                        match bestMatch with
                                        | Some(p: PatternDefinition) ->
                                            AnsiConsole.MarkupLine(
                                                $"[bold green]Found Matching Reasoning Pattern:[/] {p.Name}"
                                            )

                                            AnsiConsole.MarkupLine($"[dim]Hydrating pattern context...[/]")

                                            // 16.2 Context Hydration
                                            let hydratedTemplate =
                                                PatternLibrary.hydrate p (Map.ofList [ "input", problem.Description ])

                                            AnsiConsole.MarkupLine($"[dim]Executing pattern...[/]")

                                            // 16.3 Pattern Execution
                                            let! execResult =
                                                PatternLibrary.executePattern llm p hydratedTemplate problem.Description

                                            let patternSuccess: bool =
                                                match execResult with
                                                | Result.Ok result ->
                                                    AnsiConsole.MarkupLine($"[dim]Validating result...[/]")
                                                    PatternLibrary.validateResult result problem.ValidationCriteria
                                                | Result.Error err ->
                                                    AnsiConsole.MarkupLine($"[red]Pattern execution error:[/] {err}")
                                                    false


                                            if patternSuccess then
                                                AnsiConsole.MarkupLine(
                                                    $"[bold green]PATTERN EXECUTION SUCCESS![/] Validation passed."
                                                )

                                                let newState =
                                                    Tars.Evolution.CurriculumManager.recordSuccess state problem.Id

                                                saveState newState
                                            else
                                                AnsiConsole.MarkupLine($"[yellow]Pattern failed validation.[/]")

                                                let newState =
                                                    Tars.Evolution.CurriculumManager.recordFailure state problem.Id

                                                saveState newState
                                        | None ->
                                            AnsiConsole.MarkupLine($"[dim]No matching pattern found. Simulating...[/]")
                                            // Fallback to simulation
                                            let rng = System.Random()

                                            let successRate =
                                                match problem.Difficulty with
                                                | Tars.Evolution.ProblemDifficulty.Beginner -> 0.95
                                                | Tars.Evolution.ProblemDifficulty.Intermediate -> 0.80
                                                | Tars.Evolution.ProblemDifficulty.Advanced -> 0.60
                                                | Tars.Evolution.ProblemDifficulty.Expert -> 0.40
                                                | Tars.Evolution.ProblemDifficulty.Unascertained -> 0.65

                                            let success = rng.NextDouble() < successRate

                                            if success then
                                                AnsiConsole.MarkupLine("[green]SUCCESS![/] (Simulated)")

                                                let newState =
                                                    Tars.Evolution.CurriculumManager.recordSuccess state problem.Id

                                                saveState newState
                                            else
                                                AnsiConsole.MarkupLine("[yellow]FAILURE.[/] (Simulated)")

                                                let newState =
                                                    Tars.Evolution.CurriculumManager.recordFailure state problem.Id

                                                saveState newState

                                    else
                                        // Standard simulation
                                        AnsiConsole.MarkupLine(
                                            $"[dim]No workflow file found for '{problemIdStr}', simulating...[/]"
                                        )

                                        // Simulate success based on problem difficulty (easier = more likely success)
                                        let rng = System.Random()

                                        let successRate =
                                            match problem.Difficulty with
                                            | Tars.Evolution.ProblemDifficulty.Beginner -> 0.95
                                            | Tars.Evolution.ProblemDifficulty.Intermediate -> 0.80
                                            | Tars.Evolution.ProblemDifficulty.Advanced -> 0.60
                                            | Tars.Evolution.ProblemDifficulty.Expert -> 0.40
                                            | Tars.Evolution.ProblemDifficulty.Unascertained -> 0.65

                                        let success = rng.NextDouble() < successRate

                                        if success then
                                            AnsiConsole.MarkupLine("[green]SUCCESS![/] (Simulated)")

                                            let newState =
                                                Tars.Evolution.CurriculumManager.recordSuccess state problem.Id

                                            saveState newState
                                        else
                                            AnsiConsole.MarkupLine("[yellow]FAILURE.[/] (Simulated)")

                                            let newState =
                                                Tars.Evolution.CurriculumManager.recordFailure state problem.Id

                                            saveState newState

                                    return! trainLoop (count - 1)
                    }

                return! trainLoop iterations

            | CiCheck(flow, baseline) ->
                if not (System.IO.File.Exists baseline) then
                    AnsiConsole.MarkupLine($"[red]Baseline not found: {baseline}[/]")
                    return 1
                else
                    // Run "Stub" mode for CI check usually
                    match! runWorkflow flow ReasonStepMode.Stub with
                    | Result.Error e ->
                        AnsiConsole.MarkupLine($"[red]Run Failed:[/] {e}")
                        return 1
                    | Result.Ok(currentGolden, _) ->
                        let options =
                            System.Text.Json.JsonSerializerOptions(PropertyNameCaseInsensitive = true)

                        options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())

                        try
                            let baselineGolden =
                                System.Text.Json.JsonSerializer.Deserialize<CanonicalGolden>(
                                    System.IO.File.ReadAllText baseline,
                                    options
                                )

                            let diff = GoldenDiff.compute baselineGolden currentGolden

                            if diff.HasChanges then
                                AnsiConsole.MarkupLine("[bold red]CI Check Failed: Regression detected[/]")
                                AnsiConsole.MarkupLine("Differences found between baseline and current run.")
                                return 1
                            else
                                AnsiConsole.MarkupLine("[bold green]CI Check Passed: No regression.[/]")
                                return 0
                        with ex ->
                            AnsiConsole.MarkupLine($"[red]Error loading baseline: {ex.Message}[/]")
                            return 1

            | Diff(runA, runB) ->
                let resolvePath (p: string) =
                    if System.IO.File.Exists p then
                        p
                    else
                        let runPath = System.IO.Path.Combine(".wot", "runs", p, "golden.json")
                        if System.IO.File.Exists runPath then runPath else p

                let pathA = resolvePath runA
                let pathB = resolvePath runB

                if not (System.IO.File.Exists pathA) then
                    AnsiConsole.MarkupLine($"[red]File not found: {pathA}[/]")
                    return 1
                elif not (System.IO.File.Exists pathB) then
                    AnsiConsole.MarkupLine($"[red]File not found: {pathB}[/]")
                    return 1
                else
                    let options =
                        System.Text.Json.JsonSerializerOptions(PropertyNameCaseInsensitive = true)

                    options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())

                    try
                        let g1 =
                            System.Text.Json.JsonSerializer.Deserialize<CanonicalGolden>(
                                System.IO.File.ReadAllText pathA,
                                options
                            )

                        let g2 =
                            System.Text.Json.JsonSerializer.Deserialize<CanonicalGolden>(
                                System.IO.File.ReadAllText pathB,
                                options
                            )

                        let diff = GoldenDiff.compute g1 g2

                        if diff.HasChanges then
                            AnsiConsole.MarkupLine("[bold yellow]Differences Found:[/]")

                            if not diff.SummaryChanges.IsEmpty then
                                AnsiConsole.MarkupLine("[bold]Summary Changes:[/]")

                                for kvp in diff.SummaryChanges do
                                    let oldV, newV = kvp.Value

                                    AnsiConsole.MarkupLine(
                                        $"  {kvp.Key}: [red]{Markup.Escape(oldV)}[/] -> [green]{Markup.Escape(newV)}[/]"
                                    )

                            if not diff.StepChanges.IsEmpty then
                                AnsiConsole.MarkupLine("[bold]Step Changes:[/]")

                                for kvp in diff.StepChanges do
                                    let sId = kvp.Key

                                    match kvp.Value with
                                    | MissingInNew ->
                                        AnsiConsole.MarkupLine($"  Step [bold]{sId}[/]: [red]Missing in B[/]")
                                    | ExtraInNew ->
                                        AnsiConsole.MarkupLine($"  Step [bold]{sId}[/]: [green]Added in B[/]")
                                    | Changed changes ->
                                        AnsiConsole.MarkupLine($"  Step [bold]{sId}[/]:")

                                        for ch in changes do
                                            let fName = ch.Key
                                            let oVal, nVal = ch.Value

                                            AnsiConsole.MarkupLine(
                                                $"    {fName}: [red]{Markup.Escape(oVal)}[/] -> [green]{Markup.Escape(nVal)}[/]"
                                            )

                            return 1
                        else
                            AnsiConsole.MarkupLine("[bold green]No logical differences found.[/]")
                            return 0
                    with ex ->
                        AnsiConsole.MarkupLine($"[red]Error deserializing or comparing: {ex.Message}[/]")
                        return 1

            | RunFile(path, opts) ->
                if not (System.IO.File.Exists path) then
                    AnsiConsole.MarkupLine($"[red]File not found: {path}[/]")
                    return 1
                else
                    AnsiConsole.MarkupLine($"[bold blue]Running WoT Workflow:[/] {path} (Mode: {opts.Mode})")

                    match Workflow.load path with
                    | Result.Error err ->
                        AnsiConsole.MarkupLine("[red]Workflow failed to load:[/]")

                        for line in Workflow.describe err do
                            AnsiConsole.MarkupLine($"  {Markup.Escape(line)}")

                        return 1
                    | Result.Ok plan ->
                        AnsiConsole.MarkupLine(
                            $"[green]Parsed & Compiled OK.[/] Goal: [bold]{Markup.Escape(plan.Goal)}[/] ({plan.Steps.Length} steps)"
                        )

                        let executor =
                            if opts.UseCortex then
                                let cortexTools = ToolRegistry()
                                cortexTools.RegisterAssembly(typeof<Tars.Tools.TarsToolAttribute>.Assembly)
                                WotExecution.CortexExecutor(LlmFactory.create Log.Logger, cortexTools)
                                :> WotExecution.IWorkflowExecutor
                            else
                                WotExecution.V0Executor(opts) :> WotExecution.IWorkflowExecutor

                        return! executor.Execute plan
        }
        |> Async.StartAsTask
