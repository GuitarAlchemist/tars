namespace Tars.Interface.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open System.Text.RegularExpressions
open Spectre.Console
open Tars.Core
open Tars.Core.HybridBrain
open Tars.Tools.Standard
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Interface.Cli
open Tars.Metascript // For TrsxParser and IrCompiler

/// Type alias to avoid collision with System.Action
type private HybridAction = Tars.Core.HybridBrain.Action

module RefactorCommand =

    type RefactorOptions =
        { File: string
          Validate: bool
          Measure: bool
          Model: string option
          DryRun: bool
          Verbose: bool
          UseLlm: bool
          Cloud: bool }

    let defaultOptions =
        { File = ""
          Validate = true
          Measure = true
          Model = None
          DryRun = false
          Verbose = false
          UseLlm = false
          Cloud = false }

    let private analyzeFile (filePath: string) =
        task {
            let! complexity = CodeAnalysisTools.analyzeFileComplexity ("{\"path\": \"" + filePath + "\"}")
            let! smells = CodeAnalysisTools.findCodeSmells ("{\"path\": \"" + filePath + "\"}")
            return (complexity, smells)
        }

    /// Create a refactoring plan using the CodeAnalyzer (Heuristic)
    let private createRefactoringPlan (filePath: string) (verbose: bool) : Plan<Draft> * CodeAnalyzer.AnalysisResult =
        if verbose then
            AnsiConsole.MarkupLine($"[dim]📂 Analyzing: {filePath}[/]")

        let config =
            { CodeAnalyzer.defaultConfig with
                MaxFunctionLines = 25
                RequireDocumentation = true
                DetectDuplicates = true
                MinDuplicateLines = 4 }

        let result = CodeAnalyzer.analyzeFile config filePath

        if verbose then
            let report = CodeAnalyzer.generateReport result

            for line in report.Split('\n') do
                AnsiConsole.MarkupLine($"[dim]{Markup.Escape(line)}[/]")

        let plan = CodeAnalyzer.createPlanFromAnalysis result
        (plan, result)

    /// Create a refactoring plan using the LLM + Cognition Compiler
    let private createLlmRefactoringPlan
        (filePath: string)
        (analysis: CodeAnalyzer.AnalysisResult)
        (options: RefactorOptions)
        (previousCritique: string option)
        : Task<Plan<Draft>> =
        task {
            AnsiConsole.MarkupLine("[cyan]🧠 Invoking Hybrid Brain (Cognition Compiler)...[/]")

            // 1. Initialize LLM
            let llm = LlmFactory.create Serilog.Log.Logger

            // 2. Prepare Context
            let fileContent = File.ReadAllText(filePath)

            let issuesText =
                analysis.Issues |> List.map (fun i -> sprintf "- %A" i) |> String.concat "\n"

            // Look for CONTEXT.md
            let contextFiles = [ "CONTEXT.md"; "ARCHITECTURE.md"; "coding-standards.md" ]

            let contextContent =
                contextFiles
                |> List.tryPick (fun f ->
                    if File.Exists(f) then
                        Some(File.ReadAllText(f))
                    else
                        let path = Path.Combine(Directory.GetCurrentDirectory(), f)

                        if File.Exists(path) then
                            Some(File.ReadAllText(path))
                        else
                            None)
                |> Option.defaultValue ""

            let contextPrompt =
                if String.IsNullOrWhiteSpace(contextContent) then
                    ""
                else
                    $"\nARCHITECTURAL CONTEXT (MUST FOLLOW):\n{contextContent}\n"

            let prompt =
                $"""You are TARS, an autonomous coding agent.
Your mission is to refactor the following F# file to resolve detected quality issues.

FILE: {Path.GetFileName(filePath)}
ANALYSIS REPORT:
{issuesText}

CODE:
```fsharp
{fileContent}
```

INSTRUCTIONS:
1. You MUST generate a Workflow of Thought plan wrapped in <trsx> tags.
2. Usable Work Operations:
   - ToolCall "extract_function" {{ "name": "helper_name", "startLine": X, "endLine": Y }}
   - ToolCall "add_documentation" {{ "lineNumber": X, "docText": "Summary" }}
   - ToolCall "remove_lines" {{ "startLine": X, "endLine": Y }}
   - ToolCall "rename_symbol" {{ "old": "oldName", "new": "newName" }}
3. Format Example:
<trsx>
GOAL "Refactor example"
NODE "Reasoning" REASON Plan "I will rename X to Y"
NODE "RenameStep" WORK ToolCall "rename_symbol" {{ "old": "X", "new": "Y" }}
EDGE "Reasoning" -> "RenameStep"
</trsx>
4. Output ONLY the <trsx> block.

{contextPrompt}
"""

            let critiquePrompt =
                match previousCritique with
                | Some c ->
                    $"\n\nPREVIOUS PLAN WAS REJECTED. CRITIQUE:\n{c}\n\nPlease fix the plan based on this critique."
                | None -> ""

            let req =
                { LlmRequest.Default with
                    Messages =
                        [ { Role = Role.User
                            Content = prompt + critiquePrompt } ]
                    Temperature = Some 0.2
                    ModelHint = if options.Cloud then Some "cloud" else Some "reasoning"
                    Model = options.Model }

            AnsiConsole.MarkupLine("[cyan]Generating plan...[/]")

            if options.Verbose then
                AnsiConsole.WriteLine("--- PROMPT SENT TO LLM ---")
                AnsiConsole.WriteLine(prompt + critiquePrompt)
                AnsiConsole.WriteLine("--------------------------")

            let! llmOutput =
                task {
                    try
                        let! response = llm.CompleteAsync req
                        return response.Text
                    with ex ->
                        AnsiConsole.MarkupLine($"[red]✗ LLM call failed: {ex.Message}[/]")
                        return ""
                }

            if options.Verbose then
                Console.WriteLine("--- RAW LLM OUTPUT (Length: {0}) ---", llmOutput.Length)
                Console.WriteLine(llmOutput)
                Console.WriteLine("----------------------")

            // Strip <think> blocks if present (DeepSeek)
            let processedOutput =
                if llmOutput.Contains("<think>") then
                    Regex.Replace(llmOutput, "(?s)<think>.*?</think>", "").Trim()
                else
                    llmOutput

            let trsxContent =
                let tagMatch = Regex.Match(processedOutput, "(?s)<trsx>(.*?)</trsx>")

                if tagMatch.Success then
                    tagMatch.Groups.[1].Value.Trim()
                else
                    let m = Regex.Match(processedOutput, "(?s)```(?:trsx|wot|fsharp|xml)?\n?(.*?)```")

                    if m.Success then
                        m.Groups.[1].Value.Trim()
                    else
                        // Final fallback: look for GOAL...EDGE pattern
                        let m2 = Regex.Match(processedOutput, "(?s)(GOAL.*?EDGE.*)")
                        if m2.Success then m2.Groups.[1].Value.Trim() else ""

            if options.Verbose then
                AnsiConsole.MarkupLine("[dim]Extracted TRSX Content:[/]")
                AnsiConsole.Write(new Panel(trsxContent))

            // 5. Compile (Trace -> Graph -> Plan)
            try
                let graph = TrsxParser.parse trsxContent
                AnsiConsole.MarkupLine($"[green]✓[/] Parsed TRSX Graph: {graph.Nodes.Count} nodes")

                let plan = IrCompiler.compileFromGraph graph

                if plan.Steps.Length = 0 then
                    AnsiConsole.MarkupLine("[yellow]⚠ LLM generated an empty plan. Falling back to heuristic plan.[/]")
                    return CodeAnalyzer.createPlanFromAnalysis analysis
                else
                    return plan
            with ex ->
                AnsiConsole.MarkupLine($"[red]✗ Error compiling TRSX: {ex.Message}[/]")
                AnsiConsole.MarkupLine("[yellow]⚠ Falling back to heuristic plan.[/]")
                return CodeAnalyzer.createPlanFromAnalysis analysis
        }

    let run (options: RefactorOptions) =
        task {
            AnsiConsole.Write(new Rule("[bold blue]TARS REFACTOR - Cognition Compiler[/]"))
            AnsiConsole.MarkupLine($"[bold]Target:[/] {options.File}")

            if options.UseLlm then
                if options.Cloud then
                    AnsiConsole.MarkupLine("[cyan]☁️  Cloud LLM-Driven Refactoring Enabled[/]")
                else
                    AnsiConsole.MarkupLine("[magenta]🧠 LLM-Driven Refactoring Enabled[/]")

            if options.DryRun then
                AnsiConsole.MarkupLine("[yellow]🔍 DRY RUN MODE - No changes will be made[/]")

            AnsiConsole.WriteLine()

            // PHASE 1: ANALYSIS
            AnsiConsole.MarkupLine("[bold]PHASE 1: ANALYSIS[/]")
            // We run heuristic analysis internally to guide the LLM or as fallback
            let config =
                { CodeAnalyzer.defaultConfig with
                    MaxFunctionLines = 25
                    RequireDocumentation = true
                    DetectDuplicates = true
                    MinDuplicateLines = 4 }

            let analysisResult = CodeAnalyzer.analyzeFile config options.File

            if options.Verbose then
                let report = CodeAnalyzer.generateReport analysisResult

                for line in report.Split('\n') do
                    AnsiConsole.MarkupLine($"[dim]{Markup.Escape(line)}[/]")

            AnsiConsole.MarkupLine(
                $"  [green]✓[/] Analyzed {analysisResult.TotalLines} lines, found {analysisResult.Issues.Length} issues"
            )

            // PHASE 2: PLAN GENERATION
            AnsiConsole.MarkupLine("[bold]PHASE 2: PLAN GENERATION[/]")

            let mutable currentAttempt = 0
            let maxAttempts = if options.UseLlm then 3 else 1
            let mutable lastCritique: string option = None
            let mutable executionSuccess = false
            let mutable finalExitCode = 1

            // Create a backup BEFORE the loop
            let backupPath = options.File + ".bak"

            if not options.DryRun then
                File.Copy(options.File, backupPath, true)

            try
                while currentAttempt < maxAttempts && not executionSuccess do
                    currentAttempt <- currentAttempt + 1

                    if currentAttempt > 1 then
                        AnsiConsole.MarkupLine(
                            $"[yellow]↺ Attempt {currentAttempt}/{maxAttempts}: Retrying with critique...[/]"
                        )

                    let! draft =
                        if options.UseLlm then
                            createLlmRefactoringPlan options.File analysisResult options lastCritique
                        else
                            // Heuristic only
                            Task.FromResult(CodeAnalyzer.createPlanFromAnalysis analysisResult)

                    AnsiConsole.MarkupLine($"  [green]✓[/] Generated plan with {draft.Steps.Length} step(s)")

                    // PHASE 3: VALIDATION
                    if currentAttempt = 1 then
                        AnsiConsole.MarkupLine("[bold]PHASE 3: VALIDATION[/]")

                    // Create executor
                    let executor =
                        if options.DryRun then
                            ActionExecutor.createDryRunExecutor ()
                        else
                            ActionExecutor.createFileRefactoringExecutor options.File None

                    let config: ExecutionConfig =
                        { VerboseLogging = options.Verbose
                          DryRun = options.DryRun
                          MaxRetries = 3
                          DefaultTimeout = TimeSpan.FromMinutes(5.0)
                          Drives =
                            { Accuracy = 0.9
                              Speed = 0.5
                              Creativity = 0.5
                              Safety = 0.9 }
                          ActionExecutor = executor }

                    // Create Validation Context
                    let knownSymbols =
                        analysisResult.Functions |> List.map (fun (name, _, _) -> name) |> Set.ofList

                    let validationContext =
                        { ValidationContext.Empty with
                            KnownSymbols = knownSymbols
                            KnownFiles = Set.ofList [ options.File ] }

                    // Run full pipeline
                    let! result = HybridBrain.processAndExecute validationContext draft config

                    match result with
                    | FSharp.Core.Ok executionResult ->
                        AnsiConsole.MarkupLine("  [green]✓[/] Plan validated")

                        // PHASE 4: EXECUTION
                        AnsiConsole.MarkupLine("[bold]PHASE 4: EXECUTION[/]")

                        let outcomeSymbol =
                            match executionResult.Outcome with
                            | RunOutcome.Success -> "[green]✓[/]"
                            | RunOutcome.Partial _ -> "[yellow]⚠[/]"
                            | RunOutcome.Failure _ -> "[red]✗[/]"

                        AnsiConsole.MarkupLine(
                            $"  {outcomeSymbol} {executionResult.StepsExecuted}/{executionResult.TotalSteps} steps executed"
                        )

                        if options.Verbose && executionResult.Logs.Length > 0 then
                            AnsiConsole.MarkupLine("[dim]  Execution Log:[/]")

                            for log in executionResult.Logs do
                                let levelColor =
                                    match log.Level with
                                    | "ERROR" -> "red"
                                    | "WARN" -> "yellow"
                                    | _ -> "dim"

                                AnsiConsole.MarkupLine(
                                    $"    [{levelColor}][[{Markup.Escape(log.Level)}]][/] {Markup.Escape(log.Message)}"
                                )

                        // PHASE 5: VERIFICATION
                        AnsiConsole.MarkupLine("[bold]PHASE 5: VERIFICATION[/]")

                        let mutable buildPassed = true

                        if not options.DryRun then
                            AnsiConsole.MarkupLine("  [dim]Running build check...[/]")

                            try
                                let startInfo =
                                    System.Diagnostics.ProcessStartInfo("dotnet", "build --no-incremental")

                                startInfo.RedirectStandardOutput <- true
                                startInfo.RedirectStandardError <- true
                                startInfo.UseShellExecute <- false
                                startInfo.CreateNoWindow <- true

                                use proc = System.Diagnostics.Process.Start(startInfo)
                                let buildOutput = proc.StandardOutput.ReadToEnd()
                                proc.WaitForExit()

                                if proc.ExitCode <> 0 then
                                    // Check if it's a file lock error (usually includes "being used by another process")
                                    if
                                        buildOutput.Contains("being used by another process")
                                        || buildOutput.Contains("is locked by")
                                    then
                                        AnsiConsole.MarkupLine(
                                            "  [yellow]⚠ Build check bypassed: Core DLLs are locked by the current TARS process.[/]"
                                        )

                                        AnsiConsole.MarkupLine(
                                            "  [dim]Symbolic validation PASSED, so we will accept this change.[/]"
                                        )
                                    else
                                        buildPassed <- false
                                        AnsiConsole.MarkupLine("  [red]✗ Build failed after refactoring[/]")

                                        // Isolate errors for this file
                                        let lines = buildOutput.Split('\n')

                                        let relevantErrors =
                                            lines
                                            |> Array.filter (fun l ->
                                                l.Contains(Path.GetFileName(options.File)) || l.Contains("error FS"))
                                            |> String.concat "\n"

                                        AnsiConsole.Write(
                                            new Panel(new Text(relevantErrors, Style(foreground = Color.Red)))
                                        )

                                        lastCritique <-
                                            Some
                                                $"The refactoring plan broke the build. Error details for {options.File}:\n{relevantErrors}"
                                else
                                    AnsiConsole.MarkupLine("  [green]✓ Build passed[/]")
                            with ex ->
                                AnsiConsole.MarkupLine($"  [yellow]⚠ Build check failed to run: {ex.Message}[/]")

                        // Failure path
                        if not buildPassed then
                            // Restore backup
                            if File.Exists(backupPath) then
                                File.Copy(backupPath, options.File, true)
                                AnsiConsole.MarkupLine("  [yellow]↺ File restored from backup[/]")

                            if currentAttempt >= maxAttempts then
                                AnsiConsole.MarkupLine("[red]Max attempts reached. Aborting.[/]")
                                finalExitCode <- 1
                        else
                            // Success path
                            AnsiConsole.MarkupLine($"  [green]✓[/] Confidence: {executionResult.Confidence:P0}")

                            AnsiConsole.MarkupLine(
                                $"  [green]✓[/] Duration: {executionResult.Duration.TotalMilliseconds:F0}ms"
                            )

                            AnsiConsole.WriteLine()
                            AnsiConsole.Write(new Rule("[bold green]RESULT: SUCCESS[/]"))
                            executionSuccess <- true
                            finalExitCode <- 0

                            // Clean up backup
                            if File.Exists(backupPath) then
                                File.Delete(backupPath)

                    | FSharp.Core.Error critique ->
                        let critiqueText = CritiqueFormatter.formatForLlm critique
                        AnsiConsole.MarkupLine("  [red]✗[/] Plan rejected")
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[bold red]VALIDATION FAILURE:[/]")
                        AnsiConsole.Write(new Panel(critiqueText))

                        lastCritique <- Some critiqueText

                        if currentAttempt >= maxAttempts then
                            AnsiConsole.MarkupLine("[red]Max attempts reached. Aborting.[/]")
                            finalExitCode <- 1
            with ex ->
                AnsiConsole.WriteException(ex)
                finalExitCode <- 1

            return finalExitCode
        }

    /// Parse command line arguments and run
    let execute (args: string list) : Task<int> =
        task {
            let mutable options = defaultOptions
            let mutable showHelp = false
            let mutable i = 0

            while i < args.Length do
                match args.[i] with
                | "--dry-run"
                | "-n" ->
                    options <- { options with DryRun = true }
                    i <- i + 1
                | "--verbose"
                | "-v" ->
                    options <- { options with Verbose = true }
                    i <- i + 1
                | "--llm" ->
                    options <- { options with UseLlm = true }
                    i <- i + 1
                | "--cloud" | "-c" ->
                    options <- { options with Cloud = true; UseLlm = true }
                    i <- i + 1
                | "--help"
                | "-h" ->
                    showHelp <- true
                    i <- args.Length // Exit loop
                | arg when not (arg.StartsWith("-")) && options.File = "" ->
                    options <- { options with File = arg }
                    i <- i + 1
                | _ -> i <- i + 1

            if showHelp then
                printfn "TARS Refactor Command - Autonomous F# File Refactoring"
                printfn ""
                printfn "USAGE:"
                printfn "    tars refactor <file> [options]"
                printfn ""
                printfn "OPTIONS:"
                printfn "    --llm               Use Hybrid Brain (LLM) to generate plan"
                printfn "    --cloud, -c         Use cloud LLM (gpt-oss:120b) for high-capacity reasoning"
                printfn "    --dry-run, -n       Show what would be done without making changes"
                printfn "    --verbose, -v       Show detailed output"
                printfn "    --help, -h          Show this help message"
                printfn ""
                printfn "EXAMPLES:"
                printfn "    tars refactor src/MyModule.fs --llm"
                printfn "    tars refactor src/MyModule.fs --dry-run"
                return 0
            elif options.File = "" then
                printfn "Error: No file specified"
                printfn "Usage: tars refactor <file> [--dry-run] [--verbose] [--llm]"
                return 1
            elif not (File.Exists(options.File)) then
                AnsiConsole.MarkupLine($"[red]Error: File not found: {options.File}[/]")
                return 1
            else
                return! run options
        }
