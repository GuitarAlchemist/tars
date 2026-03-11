module Tars.Interface.Cli.Commands.PromoteCommand

open System
open Spectre.Console
open Tars.Evolution

/// CLI command: tars promote
/// Subcommands:
///   status   — Show promotion pipeline status (recurrence records, levels)
///   lineage  — Show lineage records (promotion history)
///   run      — Run the promotion pipeline on synthetic test data
///   report   — Generate a full JSON audit report
///
/// Flags:
///   --json   — Output strict JSON only (no Spectre markup). For CI/headless use.

/// When true, output strict JSON to stdout with no ANSI markup.
let mutable private jsonMode = false

let private printHeader (text: string) =
    if not jsonMode then
        AnsiConsole.MarkupLine($"[bold cyan]{text}[/]")

let private printDim (text: string) =
    if not jsonMode then
        AnsiConsole.MarkupLine($"[dim]{text}[/]")

let private levelColor = function
    | Implementation -> "grey"
    | Helper -> "green"
    | Builder -> "yellow"
    | DslClause -> "blue"
    | GrammarRule -> "magenta"

/// Show all recurrence records and their current promotion levels.
let status () =
    let records = PromotionPipeline.getRecurrenceRecords ()

    if jsonMode then
        let outputs = records |> List.map StructuredOutput.fromRecurrence
        printfn "%s" (StructuredOutput.toJson outputs)
        0
    else
        printHeader "TARS Promotion Pipeline — Status"
        AnsiConsole.WriteLine()

        if records.IsEmpty then
            printDim "  No patterns observed yet. Run 'tars evolve' or 'tars promote run' to populate."
            0
        else
            let table = Table()
            table.Border <- TableBorder.Rounded
            table.AddColumn("Pattern") |> ignore
            table.AddColumn("Level") |> ignore
            table.AddColumn("Occurrences") |> ignore
            table.AddColumn("Avg Score") |> ignore
            table.AddColumn("Tasks") |> ignore
            table.AddColumn("First Seen") |> ignore

            for r in records |> List.sortByDescending (fun r -> r.OccurrenceCount) do
                let color = levelColor r.CurrentLevel
                table.AddRow(
                    r.PatternName,
                    $"[{color}]{PromotionLevel.label r.CurrentLevel}[/]",
                    string r.OccurrenceCount,
                    $"{r.AverageScore:F2}",
                    string r.TaskIds.Length,
                    r.FirstSeen.ToString("yyyy-MM-dd HH:mm")
                ) |> ignore

            AnsiConsole.Write(table)

            let byLevel =
                records
                |> List.groupBy (fun r -> r.CurrentLevel)
                |> List.sortBy (fun (level, _) -> PromotionLevel.rank level)

            AnsiConsole.WriteLine()
            printHeader "Level Distribution"
            for (level, group) in byLevel do
                let color = levelColor level
                AnsiConsole.MarkupLine($"  [{color}]{PromotionLevel.label level,-15}[/] {group.Length} patterns")

            0

/// Show lineage records (promotion history).
let lineage () =
    let records = PromotionPipeline.getLineageRecords ()

    if jsonMode then
        let outputs =
            records
            |> List.sortByDescending (fun r -> r.PromotedAt)
            |> List.map (fun r ->
                let decisionStr, reason =
                    match r.Decision with
                    | Approve reason -> "approve", reason
                    | Reject reason -> "reject", reason
                    | Defer reason -> "defer", reason
                {| Id = r.Id; PatternId = r.PatternId
                   FromLevel = PromotionLevel.label r.FromLevel
                   ToLevel = PromotionLevel.label r.ToLevel
                   Decision = decisionStr; Reason = reason
                   Confidence = r.Confidence
                   PromotedAt = r.PromotedAt.ToString("o")
                   PromotedBy = r.PromotedBy |})
        printfn "%s" (StructuredOutput.toJson outputs)
        0
    else
        printHeader "TARS Promotion Pipeline — Lineage"
        AnsiConsole.WriteLine()

        if records.IsEmpty then
            printDim "  No promotion decisions recorded yet."
            0
        else
            let table = Table()
            table.Border <- TableBorder.Rounded
            table.AddColumn("ID") |> ignore
            table.AddColumn("Pattern") |> ignore
            table.AddColumn("From") |> ignore
            table.AddColumn("To") |> ignore
            table.AddColumn("Decision") |> ignore
            table.AddColumn("Confidence") |> ignore
            table.AddColumn("Promoted At") |> ignore

            for r in records |> List.sortByDescending (fun r -> r.PromotedAt) do
                let decisionStr, decisionColor =
                    match r.Decision with
                    | Approve reason -> $"APPROVE: {reason}", "green"
                    | Reject reason -> $"REJECT: {reason}", "red"
                    | Defer reason -> $"DEFER: {reason}", "yellow"

                let fromColor = levelColor r.FromLevel
                let toColor = levelColor r.ToLevel

                table.AddRow(
                    r.Id,
                    r.PatternId,
                    $"[{fromColor}]{PromotionLevel.label r.FromLevel}[/]",
                    $"[{toColor}]{PromotionLevel.label r.ToLevel}[/]",
                    $"[{decisionColor}]{decisionStr}[/]",
                    $"{r.Confidence:F2}",
                    r.PromotedAt.ToString("yyyy-MM-dd HH:mm")
                ) |> ignore

            AnsiConsole.Write(table)
            0

/// Run the promotion pipeline on synthetic test artifacts (for demo/testing).
let runPipeline (minOccurrences: int) =
    printHeader $"TARS Promotion Pipeline — Running (minOccurrences={minOccurrences})"
    AnsiConsole.WriteLine()

    // Create synthetic artifacts to demonstrate the pipeline.
    // We feed these through the pipeline twice — first to establish recurrence,
    // then a second pass adds rollback expansions so the Governor can approve.
    let now = DateTime.UtcNow
    let artifacts : PromotionPipeline.TraceArtifact list =
        let decomposeRollback = Some "step: analyze (reason)\n  goal: Break problem into sub-problems\n  output: Identify independent components\nstep: solve_each (reason)\n  goal: Solve each sub-problem independently\n  output: Individual solutions for each component\nstep: merge_results (reason)\n  goal: Merge sub-problem solutions into final result\n  output: Combined solution addressing full problem"

        let hypothesisRollback = Some "step: form_hypothesis (reason)\n  goal: Form hypothesis about root cause\n  output: Candidate explanation for observed behavior\nstep: design_test (reason)\n  goal: Design test to validate or refute hypothesis\n  output: Test procedure with expected outcomes\nstep: execute_test (tool)\n  tool: run_tests\n  output: Test results with pass/fail\nstep: analyze_results (reason)\n  goal: Analyze test results against hypothesis\n  output: Confirmation or refutation with evidence\nstep: refine (reason)\n  goal: Refine hypothesis based on evidence\n  output: Updated hypothesis or confirmed solution"

        [ { TaskId = "task_001"; PatternName = "decompose_and_solve"
            PatternTemplate = "Break problem into sub-problems, solve each, merge results"
            Context = "coding_challenge"; Score = 0.85; Timestamp = now
            RollbackExpansion = decomposeRollback }
          { TaskId = "task_002"; PatternName = "decompose_and_solve"
            PatternTemplate = "Break problem into sub-problems, solve each, merge results"
            Context = "architecture_review"; Score = 0.78; Timestamp = now
            RollbackExpansion = decomposeRollback }
          { TaskId = "task_003"; PatternName = "decompose_and_solve"
            PatternTemplate = "Break problem into sub-problems, solve each, merge results"
            Context = "bug_triage"; Score = 0.92; Timestamp = now
            RollbackExpansion = decomposeRollback }
          { TaskId = "task_004"; PatternName = "verify_then_commit"
            PatternTemplate = "Run validation checks before persisting changes"
            Context = "code_generation"; Score = 0.71; Timestamp = now
            RollbackExpansion = None }
          { TaskId = "task_005"; PatternName = "verify_then_commit"
            PatternTemplate = "Run validation checks before persisting changes"
            Context = "schema_migration"; Score = 0.68; Timestamp = now
            RollbackExpansion = None }
          { TaskId = "task_006"; PatternName = "verify_then_commit"
            PatternTemplate = "Run validation checks before persisting changes"
            Context = "data_ingestion"; Score = 0.75; Timestamp = now
            RollbackExpansion = None }
          { TaskId = "task_007"; PatternName = "hypothesis_test_loop"
            PatternTemplate = "Form hypothesis, design test, execute, analyze, refine"
            Context = "debugging"; Score = 0.88; Timestamp = now
            RollbackExpansion = hypothesisRollback }
          { TaskId = "task_008"; PatternName = "hypothesis_test_loop"
            PatternTemplate = "Form hypothesis, design test, execute, analyze, refine"
            Context = "performance_tuning"; Score = 0.82; Timestamp = now
            RollbackExpansion = hypothesisRollback }
          { TaskId = "task_009"; PatternName = "hypothesis_test_loop"
            PatternTemplate = "Form hypothesis, design test, execute, analyze, refine"
            Context = "root_cause_analysis"; Score = 0.90; Timestamp = now
            RollbackExpansion = hypothesisRollback }
          { TaskId = "task_010"; PatternName = "hypothesis_test_loop"
            PatternTemplate = "Form hypothesis, design test, execute, analyze, refine"
            Context = "security_audit"; Score = 0.77; Timestamp = now
            RollbackExpansion = hypothesisRollback } ]

    if not jsonMode then
        AnsiConsole.MarkupLine($"[dim]  Feeding {artifacts.Length} trace artifacts into pipeline...[/]")
        AnsiConsole.WriteLine()

    let results = PromotionPipeline.run minOccurrences artifacts

    if jsonMode then
        // Headless: strict JSON only, no markup
        printfn "%s" (StructuredOutput.pipelineRunToJson results artifacts.Length)
    else
        if results.IsEmpty then
            printDim "  No promotions triggered. Patterns may need more occurrences."
        else
            for r in results do
                AnsiConsole.MarkupLine("[bold]───────────────────────────────────────────────[/]")
                AnsiConsole.Write(Markup.Escape(r.AuditReport) |> Markup)
                AnsiConsole.WriteLine()

                match r.RoundtripValidation with
                | Some rt when rt.Passed ->
                    AnsiConsole.MarkupLine($"  [green]Round-trip: PASSED (similarity={rt.SemanticMatch:F2})[/]")
                | Some rt ->
                    AnsiConsole.MarkupLine($"  [red]Round-trip: FAILED (similarity={rt.SemanticMatch:F2})[/]")
                    for issue in rt.Issues do
                        AnsiConsole.MarkupLine($"    [red]- {Markup.Escape(issue)}[/]")
                | None -> ()

        AnsiConsole.WriteLine()
        let jsonOutput = StructuredOutput.pipelineRunToJson results artifacts.Length
        AnsiConsole.MarkupLine("[bold cyan]Structured JSON Output:[/]")
        AnsiConsole.WriteLine(jsonOutput)

    0

/// Generate a full JSON report of current pipeline state.
let report () =
    let records = PromotionPipeline.getRecurrenceRecords ()
    let lineageRecords = PromotionPipeline.getLineageRecords ()

    if jsonMode then
        let fullReport =
            {| Recurrence = records |> List.map StructuredOutput.fromRecurrence
               LineageCount = lineageRecords.Length
               TotalPatterns = records.Length
               PromotedCount = records |> List.filter (fun r -> r.CurrentLevel <> Implementation) |> List.length
               AvgScore =
                   if records.IsEmpty then 0.0
                   else records |> List.averageBy (fun r -> r.AverageScore) |> fun x -> System.Math.Round(x, 3) |}
        printfn "%s" (StructuredOutput.toJson fullReport)
    else
        printHeader "TARS Promotion Pipeline — Full Report"
        AnsiConsole.WriteLine()

        let recurrenceJson =
            records
            |> List.map StructuredOutput.fromRecurrence
            |> StructuredOutput.toJson

        AnsiConsole.MarkupLine("[bold]Recurrence Records:[/]")
        AnsiConsole.WriteLine(recurrenceJson)
        AnsiConsole.WriteLine()

        AnsiConsole.MarkupLine($"[bold]Lineage Records:[/] {lineageRecords.Length} entries")
        AnsiConsole.MarkupLine($"[bold]Total Patterns:[/] {records.Length}")
        AnsiConsole.MarkupLine($"[bold]Promoted (above Implementation):[/] {records |> List.filter (fun r -> r.CurrentLevel <> Implementation) |> List.length}")

    0

/// Entry point for the CLI command.
let run (args: string list) =
    // Extract --json flag and filter it from args
    let hasJson = args |> List.contains "--json"
    let filteredArgs = args |> List.filter (fun a -> a <> "--json")
    jsonMode <- hasJson

    match filteredArgs with
    | [] | [ "status" ] -> status ()
    | [ "lineage" ] -> lineage ()
    | [ "run" ] -> runPipeline 3
    | [ "run"; "--min"; n ] ->
        match Int32.TryParse(n) with
        | true, v -> runPipeline v
        | _ ->
            if jsonMode then
                printfn """{"error": "Invalid --min value"}"""
            else
                AnsiConsole.MarkupLine("[red]Invalid --min value[/]")
            1
    | [ "report" ] -> report ()
    | [ "help" ] | _ ->
        AnsiConsole.MarkupLine("[bold cyan]TARS Promotion Pipeline[/]")
        AnsiConsole.MarkupLine("  The 7-step CompoundCore loop: Inspect > Extract > Classify > Propose > Validate > Persist > Govern")
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("  [bold]tars promote[/]                    Show current promotion status")
        AnsiConsole.MarkupLine("  [bold]tars promote status[/]             Show recurrence records and level distribution")
        AnsiConsole.MarkupLine("  [bold]tars promote lineage[/]            Show promotion history and governance decisions")
        AnsiConsole.MarkupLine("  [bold]tars promote run[/]                Run the pipeline on synthetic test data")
        AnsiConsole.MarkupLine("  [bold]tars promote run --min N[/]        Set minimum occurrences for promotion (default: 3)")
        AnsiConsole.MarkupLine("  [bold]tars promote report[/]             Generate a full JSON audit report")
        AnsiConsole.MarkupLine("  [bold]tars promote <cmd> --json[/]       Output strict JSON only (for CI/headless)")
        0
