namespace Tars.Interface.Cli.Commands

open System
open Spectre.Console
open Tars.Llm
open Tars.Llm.ClaudeCodeService
open Tars.Core.MetaCognition
open Tars.Cortex
open Tars.Evolution

/// CLI command for running TARS meta-cognitive analysis.
/// Analyzes execution history to identify capability gaps, generate
/// targeted curriculum, and produce actionable recommendations.
module MetaCommand =

    let private printHelp () =
        AnsiConsole.MarkupLine("[bold cyan]TARS Meta-Cognition[/] - Self-aware gap detection and learning")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [bold]tars meta analyze[/]      Run full meta-cognitive cycle on execution history")
        AnsiConsole.MarkupLine("  [bold]tars meta gaps[/]         Detect capability gaps only")
        AnsiConsole.MarkupLine("  [bold]tars meta clusters[/]     Show failure clusters")
        AnsiConsole.MarkupLine("  [bold]tars meta curriculum[/]   Generate targeted learning tasks")
        AnsiConsole.MarkupLine("  [bold]tars meta stats[/]        Show execution history statistics")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [dim]Options:[/]")
        AnsiConsole.MarkupLine("    --use-claude            Use Claude Code for LLM-enhanced analysis")
        AnsiConsole.MarkupLine("    --threshold <0.0-1.0>   Gap detection threshold (default 0.5)")
        0

    let private loadOutcomes () =
        PatternOutcomeStore.loadAll ()

    let private showStats (outcomes: PatternOutcomeStore.PatternOutcome list) =
        AnsiConsole.MarkupLine("[bold cyan]TARS Execution History[/]")
        AnsiConsole.MarkupLine("")

        let total = outcomes.Length
        let successes = outcomes |> List.filter (fun o -> o.Success) |> List.length
        let failures = total - successes
        let successRate = if total > 0 then float successes / float total * 100.0 else 0.0

        AnsiConsole.MarkupLine(sprintf "  Total executions: [bold]%d[/]" total)
        AnsiConsole.MarkupLine(sprintf "  Successes: [green]%d[/]  Failures: [red]%d[/]  Rate: [bold]%.1f%%[/]" successes failures successRate)
        AnsiConsole.MarkupLine("")

        // By pattern
        let byPattern =
            outcomes
            |> List.groupBy (fun o -> sprintf "%A" o.PatternKind)
            |> List.sortByDescending (fun (_, items) -> items.Length)

        if not byPattern.IsEmpty then
            let table = Table()
            table.AddColumn("Pattern") |> ignore
            table.AddColumn("Total") |> ignore
            table.AddColumn("Success") |> ignore
            table.AddColumn("Failure") |> ignore
            table.AddColumn("Rate") |> ignore
            table.AddColumn("Avg Duration") |> ignore

            for (pattern, items) in byPattern do
                let s = items |> List.filter (fun o -> o.Success) |> List.length
                let f = items.Length - s
                let rate = if items.Length > 0 then float s / float items.Length * 100.0 else 0.0
                let avgMs = items |> List.averageBy (fun o -> float o.DurationMs)
                let rateColor = if rate >= 70.0 then "green" elif rate >= 40.0 then "yellow" else "red"
                table.AddRow(
                    pattern,
                    string items.Length,
                    sprintf "[green]%d[/]" s,
                    sprintf "[red]%d[/]" f,
                    sprintf "[%s]%.0f%%[/]" rateColor rate,
                    sprintf "%.0fms" avgMs) |> ignore

            AnsiConsole.Write(table)

    let private showClusters (outcomes: PatternOutcomeStore.PatternOutcome list) (threshold: float) =
        let failures =
            outcomes
            |> List.filter (fun o -> not o.Success)
            |> List.map (fun o ->
                { RunId = Guid.NewGuid().ToString("N").Substring(0, 8)
                  Goal = o.Goal
                  PatternUsed = sprintf "%A" o.PatternKind
                  ErrorMessage = "Execution failed"
                  TraceStepCount = 0
                  FailedAtStep = None
                  Timestamp = o.Timestamp
                  Tags = GapDetection.extractDomainTags o.Goal
                  Score = 0.0 })

        let clusters = FailureClustering.buildClusters threshold failures

        AnsiConsole.MarkupLine("[bold cyan]Failure Clusters[/]")
        AnsiConsole.MarkupLine(sprintf "  Found [bold]%d[/] clusters from [bold]%d[/] failures" clusters.Length failures.Length)
        AnsiConsole.MarkupLine("")

        if clusters.IsEmpty then
            AnsiConsole.MarkupLine("  [green]No failure clusters detected![/]")
        else
            for cluster in clusters |> List.sortByDescending (fun c -> c.Frequency) do
                let causeStr =
                    match cluster.RootCause with
                    | FailureRootCause.MissingTool t -> sprintf "Missing tool: %s" t
                    | FailureRootCause.WrongPattern(u, s) -> sprintf "Wrong pattern: %s -> %s" u s
                    | FailureRootCause.KnowledgeGap d -> sprintf "Knowledge gap: %s" d
                    | FailureRootCause.InsufficientContext i -> sprintf "Insufficient context: %s" i
                    | FailureRootCause.ModelLimitation d -> sprintf "Model limitation: %s" d
                    | FailureRootCause.BadPrompt i -> sprintf "Bad prompt: %s" i
                    | FailureRootCause.ExternalFailure s -> sprintf "External failure: %s" s
                    | FailureRootCause.Unknown d -> sprintf "Unknown: %s" d

                AnsiConsole.MarkupLine(sprintf "  [bold red]%s[/] (%d failures)" cluster.ClusterId cluster.Frequency)
                AnsiConsole.MarkupLine(sprintf "    Root cause: [yellow]%s[/]" causeStr)
                AnsiConsole.MarkupLine(sprintf "    Patterns affected: %s" (String.Join(", ", cluster.AffectedGoalPatterns)))
                AnsiConsole.MarkupLine(sprintf "    Sample goal: %s" cluster.Representative.Goal)
                AnsiConsole.MarkupLine("")

    let private showGaps (outcomes: PatternOutcomeStore.PatternOutcome list) (threshold: float) =
        let failures =
            outcomes
            |> List.filter (fun o -> not o.Success)
            |> List.map (fun o ->
                { RunId = Guid.NewGuid().ToString("N").Substring(0, 8)
                  Goal = o.Goal
                  PatternUsed = sprintf "%A" o.PatternKind
                  ErrorMessage = "Execution failed"
                  TraceStepCount = 0
                  FailedAtStep = None
                  Timestamp = o.Timestamp
                  Tags = GapDetection.extractDomainTags o.Goal
                  Score = 0.0 })

        let successes =
            outcomes
            |> List.filter (fun o -> o.Success)
            |> List.map (fun o -> o.Goal, GapDetection.extractDomainTags o.Goal)

        let clusters = FailureClustering.buildClusters 0.4 failures
        let gaps = GapDetection.detectGaps threshold clusters successes failures |> GapDetection.rankGaps

        AnsiConsole.MarkupLine("[bold cyan]Capability Gaps[/]")
        AnsiConsole.MarkupLine("")

        if gaps.IsEmpty then
            AnsiConsole.MarkupLine("  [green]No significant capability gaps detected![/]")
        else
            let table = Table()
            table.AddColumn("Gap") |> ignore
            table.AddColumn("Domain") |> ignore
            table.AddColumn("Failure Rate") |> ignore
            table.AddColumn("Samples") |> ignore
            table.AddColumn("Confidence") |> ignore
            table.AddColumn("Suggested Remedy") |> ignore

            for gap in gaps do
                let remedyStr =
                    match gap.SuggestedRemedy with
                    | GapRemedy.LearnPattern d -> sprintf "Learn: %s" d
                    | GapRemedy.AcquireTool(t, _) -> sprintf "Tool: %s" t
                    | GapRemedy.IngestKnowledge(d, _) -> sprintf "Study: %s" d
                    | GapRemedy.ImprovePrompt(_, s) -> sprintf "Prompt: %s" s
                    | GapRemedy.ComposePatterns ps -> sprintf "Compose: %s" (String.Join("+", ps))

                let rateColor = if gap.FailureRate >= 0.7 then "red" elif gap.FailureRate >= 0.5 then "yellow" else "dim"

                table.AddRow(
                    gap.GapId,
                    gap.Domain,
                    sprintf "[%s]%.0f%%[/]" rateColor (gap.FailureRate * 100.0),
                    string gap.SampleSize,
                    sprintf "%.0f%%" (gap.Confidence * 100.0),
                    remedyStr) |> ignore

            AnsiConsole.Write(table)

    let private showCurriculum (outcomes: PatternOutcomeStore.PatternOutcome list) (threshold: float) (useClaude: bool) =
        let failures =
            outcomes
            |> List.filter (fun o -> not o.Success)
            |> List.map (fun o ->
                { RunId = Guid.NewGuid().ToString("N").Substring(0, 8)
                  Goal = o.Goal
                  PatternUsed = sprintf "%A" o.PatternKind
                  ErrorMessage = "Execution failed"
                  TraceStepCount = 0
                  FailedAtStep = None
                  Timestamp = o.Timestamp
                  Tags = GapDetection.extractDomainTags o.Goal
                  Score = 0.0 })

        let successes =
            outcomes
            |> List.filter (fun o -> o.Success)
            |> List.map (fun o -> o.Goal, GapDetection.extractDomainTags o.Goal)

        let clusters = FailureClustering.buildClusters 0.4 failures
        let gaps = GapDetection.detectGaps threshold clusters successes failures |> GapDetection.rankGaps

        let tasks =
            if useClaude && ClaudeCodeService.isAvailable () then
                AnsiConsole.MarkupLine("[dim]Using Claude Code for curriculum generation...[/]")
                let llm = ClaudeCodeService.create None
                CurriculumPlanner.generateTasksWithLlm llm gaps 5
                |> fun t -> t.Result
            else
                CurriculumPlanner.generateTasksFromTemplates gaps 5

        AnsiConsole.MarkupLine("[bold cyan]Targeted Curriculum[/]")
        AnsiConsole.MarkupLine(sprintf "  Generated [bold]%d[/] tasks to address [bold]%d[/] gaps" tasks.Length gaps.Length)
        AnsiConsole.MarkupLine("")

        if tasks.IsEmpty then
            AnsiConsole.MarkupLine("  [green]No gaps to address - all capabilities nominal![/]")
        else
            let table = Table()
            table.AddColumn("Task") |> ignore
            table.AddColumn("Gap") |> ignore
            table.AddColumn("Description") |> ignore
            table.AddColumn("Difficulty") |> ignore
            table.AddColumn("Priority") |> ignore

            for task in tasks do
                let diffStr = String.replicate task.Difficulty "*"
                table.AddRow(
                    task.TaskId,
                    task.GapId,
                    task.Description.Substring(0, min 60 task.Description.Length),
                    diffStr,
                    sprintf "%.2f" task.Priority) |> ignore

            AnsiConsole.Write(table)

    let private runFullAnalysis (outcomes: PatternOutcomeStore.PatternOutcome list) (threshold: float) (useClaude: bool) =
        AnsiConsole.MarkupLine("[bold cyan]TARS Meta-Cognitive Analysis[/]")
        AnsiConsole.MarkupLine("[dim]Analyzing execution history for gaps, patterns, and learning opportunities...[/]")
        AnsiConsole.MarkupLine("")

        let llm =
            if useClaude && ClaudeCodeService.isAvailable () then
                AnsiConsole.MarkupLine("[dim]LLM Backend: Claude Code[/]")
                Some (ClaudeCodeService.create None)
            else
                AnsiConsole.MarkupLine("[dim]LLM Backend: heuristic only (use --use-claude for enhanced analysis)[/]")
                None

        let config =
            { MetaCognitionConfig.defaults with
                FailureClusterThreshold = 0.4
                GapDetectionThreshold = threshold }

        let result =
            MetaCognitionOrchestrator.runCycle
                llm config outcomes [] []
            |> fun t -> t.Result

        // Display results
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine(sprintf "[bold]Results:[/] %d clusters, %d gaps, %d tasks, %d reflections"
            result.FailureClusters.Length
            result.DetectedGaps.Length
            result.GeneratedTasks.Length
            result.Reflections.Length)
        AnsiConsole.MarkupLine("")

        // Clusters
        if not result.FailureClusters.IsEmpty then
            AnsiConsole.MarkupLine("[bold yellow]Failure Clusters:[/]")
            for c in result.FailureClusters |> List.sortByDescending (fun c -> c.Frequency) |> List.truncate 5 do
                AnsiConsole.MarkupLine(sprintf "  [red]%d failures[/] - %A" c.Frequency c.RootCause)
            AnsiConsole.MarkupLine("")

        // Gaps
        if not result.DetectedGaps.IsEmpty then
            AnsiConsole.MarkupLine("[bold yellow]Capability Gaps:[/]")
            for g in result.DetectedGaps |> List.truncate 5 do
                AnsiConsole.MarkupLine(sprintf "  [red]%s[/]: %.0f%% failure rate (%d samples) - %A"
                    g.Domain (g.FailureRate * 100.0) g.SampleSize g.SuggestedRemedy)
            AnsiConsole.MarkupLine("")

        // Tasks
        if not result.GeneratedTasks.IsEmpty then
            AnsiConsole.MarkupLine("[bold green]Generated Curriculum:[/]")
            for t in result.GeneratedTasks |> List.truncate 5 do
                AnsiConsole.MarkupLine(sprintf "  [green]+[/] %s (difficulty %d, priority %.2f)" t.Description t.Difficulty t.Priority)
            AnsiConsole.MarkupLine("")

        // Recommendations
        if not result.Recommendations.IsEmpty then
            AnsiConsole.MarkupLine("[bold cyan]Recommendations:[/]")
            for r in result.Recommendations do
                AnsiConsole.MarkupLine(sprintf "  > %s" r)
            AnsiConsole.MarkupLine("")

    let run (args: string list) =
        let mutable subCmd = ""
        let mutable useClaude = false
        let mutable threshold = 0.5
        let mutable remaining = args

        while not remaining.IsEmpty do
            match remaining with
            | "--use-claude" :: rest ->
                useClaude <- true
                remaining <- rest
            | "--threshold" :: v :: rest ->
                threshold <- try float v with _ -> 0.5
                remaining <- rest
            | cmd :: rest when subCmd = "" ->
                subCmd <- cmd
                remaining <- rest
            | _ :: rest ->
                remaining <- rest
            | [] -> ()

        let outcomes = loadOutcomes ()

        if outcomes.IsEmpty then
            AnsiConsole.MarkupLine("[yellow]No execution history found.[/]")
            AnsiConsole.MarkupLine("[dim]Run some tasks first: tars agent run \"your goal\"[/]")
            1
        else
            match subCmd with
            | "stats" -> showStats outcomes; 0
            | "clusters" -> showClusters outcomes threshold; 0
            | "gaps" -> showGaps outcomes threshold; 0
            | "curriculum" -> showCurriculum outcomes threshold useClaude; 0
            | "analyze" -> runFullAnalysis outcomes threshold useClaude; 0
            | "" -> runFullAnalysis outcomes threshold useClaude; 0
            | _ -> printHelp ()
