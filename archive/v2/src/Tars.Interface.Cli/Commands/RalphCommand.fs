namespace Tars.Interface.Cli.Commands

open System
open Spectre.Console
open Tars.Cortex
open Tars.Core.MetaCognition
open Tars.Evolution

/// CLI command for Ralph Loop integration.
/// Bridges TARS meta-cognition with Ralph's iterative improvement mechanism.
module RalphCommand =

    let private projectDir =
        Environment.CurrentDirectory

    let private printHelp () =
        AnsiConsole.MarkupLine("[bold cyan]TARS Ralph Loop Integration[/] - Iterative self-improvement via Ralph loops")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [bold]tars ralph status[/]                Check if a Ralph loop is active")
        AnsiConsole.MarkupLine("  [bold]tars ralph start[/]                 Start a self-improvement Ralph loop")
        AnsiConsole.MarkupLine("  [bold]tars ralph start --focus <area>[/]  Focus on a specific area")
        AnsiConsole.MarkupLine("  [bold]tars ralph start --goal <goal>[/]   Start a task-specific Ralph loop")
        AnsiConsole.MarkupLine("  [bold]tars ralph stop[/]                  Stop the active Ralph loop")
        AnsiConsole.MarkupLine("  [bold]tars ralph prompt[/]                Generate and display the Ralph prompt")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [dim]Options:[/]")
        AnsiConsole.MarkupLine("    --max-iterations N      Maximum iterations (default: 10)")
        AnsiConsole.MarkupLine("    --focus <area>          Focus improvement on a specific area")
        AnsiConsole.MarkupLine("    --goal <goal>           Custom goal instead of gap-driven improvement")
        AnsiConsole.MarkupLine("    --promise <text>        Custom completion promise (default: TARS GAPS RESOLVED)")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [dim]How it works:[/]")
        AnsiConsole.MarkupLine("    1. Runs meta-cognitive analysis to find capability gaps")
        AnsiConsole.MarkupLine("    2. Generates a targeted Ralph prompt from the gaps")
        AnsiConsole.MarkupLine("    3. Creates .claude/ralph-loop.local.md state file")
        AnsiConsole.MarkupLine("    4. Claude Code's stop hook picks up the loop")
        AnsiConsole.MarkupLine("    5. Each iteration: build -> test -> fix gap -> verify")
        AnsiConsole.MarkupLine("    6. Loop ends when gaps resolved or max iterations reached")
        0

    let private showStatus () =
        match RalphBridge.readState projectDir with
        | Some state ->
            AnsiConsole.MarkupLine("[bold cyan]Ralph Loop Status[/]")
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine(sprintf "  Active: [bold green]Yes[/]")
            AnsiConsole.MarkupLine(sprintf "  Iteration: [bold]%d[/]" state.Iteration)
            match state.MaxIterations with
            | Some max -> AnsiConsole.MarkupLine(sprintf "  Max iterations: [bold]%d[/]" max)
            | None -> AnsiConsole.MarkupLine("  Max iterations: [dim]unlimited[/]")
            match state.CompletionPromise with
            | Some p -> AnsiConsole.MarkupLine(sprintf "  Completion promise: [bold]%s[/]" p)
            | None -> AnsiConsole.MarkupLine("  Completion promise: [dim]none[/]")
            match state.StartedAt with
            | Some dt ->
                let elapsed = DateTime.UtcNow - dt
                AnsiConsole.MarkupLine(sprintf "  Running for: [bold]%s[/]" (elapsed.ToString(@"hh\:mm\:ss")))
            | None -> ()
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine("[dim]Prompt:[/]")
            AnsiConsole.MarkupLine(sprintf "  %s" (state.Prompt.Replace("\n", "\n  ")))
            0
        | None ->
            AnsiConsole.MarkupLine("[dim]No active Ralph loop.[/]")
            AnsiConsole.MarkupLine("[dim]Start one with: tars ralph start[/]")
            0

    let private generatePrompt (focus: string option) =
        let outcomes = PatternOutcomeStore.loadAll ()
        if outcomes.IsEmpty then
            AnsiConsole.MarkupLine("[yellow]No execution history. Using generic improvement prompt.[/]")
            RalphBridge.generateTarsPrompt [] focus
        else
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
            let gaps =
                GapDetection.detectGaps 0.3 clusters successes failures
                |> GapDetection.rankGaps

            AnsiConsole.MarkupLine(sprintf "[dim]Found %d capability gaps from %d executions[/]" gaps.Length outcomes.Length)
            RalphBridge.generateTarsPrompt gaps focus

    let private startLoop
        (focus: string option)
        (goal: string option)
        (maxIterations: int)
        (promise: string option)
        =
        if RalphBridge.isActive projectDir then
            AnsiConsole.MarkupLine("[red]A Ralph loop is already active![/]")
            AnsiConsole.MarkupLine("[dim]Stop it first with: tars ralph stop[/]")
            1
        else
            let prompt, completionPromise =
                match goal with
                | Some g ->
                    let p = promise |> Option.defaultValue "TASK COMPLETE"
                    let criteria = "The goal described above is fully achieved and all tests pass."
                    RalphBridge.generateTaskPrompt g criteria p, p
                | None ->
                    let p = promise |> Option.defaultValue "TARS GAPS RESOLVED"
                    generatePrompt focus, p

            RalphBridge.startLoop projectDir prompt (Some maxIterations) (Some completionPromise)

            AnsiConsole.MarkupLine("[bold green]Ralph loop started![/]")
            AnsiConsole.MarkupLine(sprintf "  Max iterations: [bold]%d[/]" maxIterations)
            AnsiConsole.MarkupLine(sprintf "  Completion promise: [bold]%s[/]" completionPromise)
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine("[dim]The loop will activate when this Claude Code session ends.[/]")
            AnsiConsole.MarkupLine("[dim]Each iteration will:[/]")
            AnsiConsole.MarkupLine("[dim]  1. Receive the same prompt[/]")
            AnsiConsole.MarkupLine("[dim]  2. See its previous work in files/git[/]")
            AnsiConsole.MarkupLine("[dim]  3. Build, test, fix one gap, verify[/]")
            AnsiConsole.MarkupLine("[dim]  4. Repeat until done[/]")
            0

    let private stopLoop () =
        if RalphBridge.stopLoop projectDir then
            AnsiConsole.MarkupLine("[green]Ralph loop stopped.[/]")
            0
        else
            AnsiConsole.MarkupLine("[dim]No active Ralph loop to stop.[/]")
            0

    let private showPrompt (focus: string option) =
        let prompt = generatePrompt focus
        AnsiConsole.MarkupLine("[bold cyan]Generated Ralph Prompt:[/]")
        AnsiConsole.MarkupLine("")
        printfn "%s" prompt
        0

    let run (args: string list) =
        let mutable subCmd = ""
        let mutable focus = None
        let mutable goal = None
        let mutable maxIterations = 10
        let mutable promise = None
        let mutable remaining = args

        while not remaining.IsEmpty do
            match remaining with
            | "--focus" :: v :: rest ->
                focus <- Some v
                remaining <- rest
            | "--goal" :: v :: rest ->
                goal <- Some v
                remaining <- rest
            | "--max-iterations" :: v :: rest ->
                maxIterations <- try int v with _ -> 10
                remaining <- rest
            | "--promise" :: v :: rest ->
                promise <- Some v
                remaining <- rest
            | cmd :: rest when subCmd = "" ->
                subCmd <- cmd
                remaining <- rest
            | _ :: rest ->
                remaining <- rest
            | [] -> ()

        match subCmd with
        | "status" -> showStatus ()
        | "start" -> startLoop focus goal maxIterations promise
        | "stop" -> stopLoop ()
        | "prompt" -> showPrompt focus
        | "" -> printHelp ()
        | _ -> printHelp ()
