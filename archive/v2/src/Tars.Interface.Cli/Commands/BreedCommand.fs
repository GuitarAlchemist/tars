namespace Tars.Interface.Cli.Commands

open System
open Spectre.Console
open Tars.Cortex
open Tars.Evolution

/// CLI command for evolutionary pattern breeding via genetic algorithms.
/// Uses ix's Rust GA when available, falls back to built-in F# GA.
module BreedCommand =

    let private machinDeOufDir =
        let candidate = IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "ix")
        if IO.Directory.Exists(candidate) then Some candidate else None

    let private printHelp () =
        AnsiConsole.MarkupLine("[bold cyan]TARS Breed[/] - Evolutionary pattern optimization via genetic algorithms")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [bold]tars breed[/]                     Run GA breeding on execution history")
        AnsiConsole.MarkupLine("  [bold]tars breed --generations N[/]     Set GA generations (default 50)")
        AnsiConsole.MarkupLine("  [bold]tars breed --show-genome[/]       Display the evolved genome details")
        AnsiConsole.MarkupLine("  [bold]tars breed status[/]              Check ix availability")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [dim]Uses ix's Rust-based GA when available, otherwise built-in F# GA.[/]")
        0

    let private showStatus () =
        AnsiConsole.MarkupLine("[bold cyan]ix Bridge Status[/]")
        AnsiConsole.MarkupLine("")

        match machinDeOufDir with
        | Some dir ->
            AnsiConsole.MarkupLine(sprintf "  ix repo: [green]found[/] at %s" dir)
            let config = { MachinBridge.defaultConfig with WorkingDir = Some dir }
            if MachinBridge.isAvailable config then
                AnsiConsole.MarkupLine("  ix: [green]available[/]")
            else
                AnsiConsole.MarkupLine("  ix: [yellow]not built[/] (run `cargo build -p ix`)")
        | None ->
            AnsiConsole.MarkupLine("  ix repo: [dim]not found[/]")
            AnsiConsole.MarkupLine("  [dim]Expected at ~/source/repos/ix[/]")

        AnsiConsole.MarkupLine("  F# fallback GA: [green]always available[/]")

        let outcomes = PatternOutcomeStore.loadAll ()
        AnsiConsole.MarkupLine(sprintf "  Execution history: [bold]%d[/] outcomes" outcomes.Length)
        0

    let private runBreeding (generations: int) (showGenome: bool) =
        let outcomes = PatternOutcomeStore.loadAll ()

        if outcomes.IsEmpty then
            AnsiConsole.MarkupLine("[yellow]No execution history found.[/]")
            AnsiConsole.MarkupLine("[dim]Run some tasks first: tars agent run \"your goal\"[/]")
            1
        else
            AnsiConsole.MarkupLine("[bold cyan]Evolutionary Pattern Breeding[/]")
            AnsiConsole.MarkupLine(sprintf "  Outcomes: [bold]%d[/]  Generations: [bold]%d[/]" outcomes.Length generations)

            let machinConfig =
                machinDeOufDir
                |> Option.map (fun dir ->
                    { MachinBridge.defaultConfig with WorkingDir = Some dir })

            let sw = Diagnostics.Stopwatch.StartNew()
            let result = EvolutionaryPatternBreeder.breed machinConfig outcomes generations
            sw.Stop()

            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine(sprintf "  Backend: [bold]%s[/]" (if result.UsedMachinDeOuf then "ix Rust GA" else "Built-in F# GA"))
            AnsiConsole.MarkupLine(sprintf "  Population: [bold]%d[/]  Generations: [bold]%d[/]" result.PopulationSize result.Generations)
            AnsiConsole.MarkupLine(sprintf "  Best fitness: [bold green]%.4f[/]" result.BestFitness)
            AnsiConsole.MarkupLine(sprintf "  Duration: [bold]%dms[/]" sw.ElapsedMilliseconds)
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine(sprintf "  [bold]Recommendation:[/] %s" result.Recommendation)

            if showGenome then
                let g = result.BestGenome
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("[bold]Evolved Genome:[/]")

                let table = Table()
                table.AddColumn("Parameter") |> ignore
                table.AddColumn("Value") |> ignore
                table.AddColumn("Interpretation") |> ignore

                table.AddRow("CoT Weight", sprintf "%.3f" g.CotWeight,
                    if g.CotWeight > 0.5 then "[green]Strong[/]" else "[dim]Weak[/]") |> ignore
                table.AddRow("ReAct Weight", sprintf "%.3f" g.ReactWeight,
                    if g.ReactWeight > 0.5 then "[green]Strong[/]" else "[dim]Weak[/]") |> ignore
                table.AddRow("ToT Weight", sprintf "%.3f" g.TotWeight,
                    if g.TotWeight > 0.5 then "[green]Strong[/]" else "[dim]Weak[/]") |> ignore
                table.AddRow("GoT Weight", sprintf "%.3f" g.GotWeight,
                    if g.GotWeight > 0.5 then "[green]Strong[/]" else "[dim]Weak[/]") |> ignore
                table.AddRow("Step Multiplier", sprintf "%.2f" g.StepMultiplier,
                    sprintf "%d steps" (int (5.0 * g.StepMultiplier))) |> ignore
                table.AddRow("Temperature", sprintf "%.2f" g.Temperature,
                    if g.Temperature > 0.8 then "Creative" elif g.Temperature < 0.4 then "Precise" else "Balanced") |> ignore
                table.AddRow("Confidence Threshold", sprintf "%.2f" g.ConfidenceThreshold,
                    if g.ConfidenceThreshold > 0.7 then "Conservative" else "Exploratory") |> ignore
                table.AddRow("Branching Factor", sprintf "%.1f" g.BranchingFactor,
                    sprintf "%.0f branches" g.BranchingFactor) |> ignore

                AnsiConsole.Write(table)

                // Show pattern suggestion for common goals
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("[bold]Pattern Suggestions (from evolved genome):[/]")
                let goals =
                    [ "Analyze code for security vulnerabilities"
                      "Generate a REST API implementation"
                      "Design a database schema"
                      "Write unit tests for the parser" ]
                for goal in goals do
                    let suggested = EvolutionaryPatternBreeder.suggestPattern g goal
                    AnsiConsole.MarkupLine(sprintf "  %A <- \"%s\"" suggested goal)

            0

    let run (args: string list) =
        let mutable subCmd = ""
        let mutable generations = 50
        let mutable showGenome = false
        let mutable remaining = args

        while not remaining.IsEmpty do
            match remaining with
            | "--generations" :: v :: rest ->
                generations <- try int v with _ -> 50
                remaining <- rest
            | "--show-genome" :: rest ->
                showGenome <- true
                remaining <- rest
            | cmd :: rest when subCmd = "" ->
                subCmd <- cmd
                remaining <- rest
            | _ :: rest ->
                remaining <- rest
            | [] -> ()

        match subCmd with
        | "status" -> showStatus ()
        | "help" -> printHelp ()
        | "" -> runBreeding generations showGenome
        | _ -> printHelp ()
