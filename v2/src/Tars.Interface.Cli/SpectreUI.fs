/// Enhanced CLI Experience using Spectre.Console
/// Provides rich progress bars, tables, and live updates
module Tars.Interface.Cli.SpectreUI

open System
open Spectre.Console

/// Rich console output helpers using Spectre.Console
module RichOutput =

    /// Print the TARS banner
    let printBanner () =
        let banner = FigletText("TARS v2")
        banner.Color <- Color.Cyan1
        AnsiConsole.Write(banner)

        AnsiConsole.MarkupLine("[cyan]Self-Improving Agent System[/]")
        AnsiConsole.WriteLine()

    /// Print a section header
    let header (text: string) =
        let rule = Rule(text)
        rule.Style <- Style.Parse("cyan bold")
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()

    /// Print info message
    let info (message: string) =
        AnsiConsole.MarkupLine("[blue]ℹ[/] " + Markup.Escape(message))

    /// Print success message
    let success (message: string) =
        AnsiConsole.MarkupLine("[green]✓[/] " + Markup.Escape(message))

    /// Print warning message
    let warning (message: string) =
        AnsiConsole.MarkupLine("[yellow]⚠[/] " + Markup.Escape(message))

    /// Print error message
    let error (message: string) =
        AnsiConsole.MarkupLine("[red]✗[/] " + Markup.Escape(message))

    /// Print dim/secondary text
    let dim (message: string) =
        AnsiConsole.MarkupLine("[dim]" + Markup.Escape(message) + "[/]")

/// LLM information display
module LlmDisplay =

    /// Print current LLM configuration
    let printConfig (model: string) (endpoint: string) =
        let table = Table()
        table.Border <- TableBorder.Rounded
        table.AddColumn("[bold]Setting[/]") |> ignore
        table.AddColumn("[bold]Value[/]") |> ignore

        table.AddRow("[cyan]🤖 Model[/]", "[green]" + Markup.Escape(model) + "[/]")
        |> ignore

        table.AddRow("[cyan]🔗 Endpoint[/]", "[blue]" + Markup.Escape(endpoint) + "[/]")
        |> ignore

        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()

    /// Print model being used with style
    let printModel (model: string) =
        AnsiConsole.MarkupLine("[bold cyan]🤖 LLM Model:[/] [green]" + Markup.Escape(model) + "[/]")

/// Task display helpers
module TaskDisplay =

    /// Print a new task box
    let printTask (goal: string) (constraints: string list) =
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[yellow bold]╔══════════════════════════════════════════════════════════════╗[/]")
        AnsiConsole.MarkupLine("[yellow bold]║[/] [white bold]📋 NEW TASK[/]")
        AnsiConsole.MarkupLine("[yellow bold]╟──────────────────────────────────────────────────────────────╢[/]")
        AnsiConsole.MarkupLine("[yellow bold]║[/] " + Markup.Escape(goal))

        if not (List.isEmpty constraints) then
            AnsiConsole.MarkupLine("[yellow bold]║[/] [dim]Constraints:[/]")

            for c in constraints do
                AnsiConsole.MarkupLine("[yellow bold]║[/]   [yellow]•[/] " + Markup.Escape(c))

        AnsiConsole.MarkupLine("[yellow bold]╚══════════════════════════════════════════════════════════════╝[/]")
        AnsiConsole.WriteLine()

    /// Print task success
    let printSuccess (result: string) (duration: TimeSpan) (verified: bool) =
        let status =
            if verified then
                "[green]✓ VERIFIED[/]"
            else
                "[yellow]⚠ UNVERIFIED[/]"

        let preview =
            let s = result.Replace("\n", " ").Replace("\r", "")
            if s.Length > 100 then s.Substring(0, 97) + "..." else s

        AnsiConsole.WriteLine()

        AnsiConsole.MarkupLine(
            sprintf "[green bold]╔══════════════════════════════════════════════════════════════╗[/]"
        )

        AnsiConsole.MarkupLine(
            sprintf "[green bold]║[/] [white bold]✅ TASK SUCCESS[/] [dim](%.1fs)[/] %s" duration.TotalSeconds status
        )

        AnsiConsole.MarkupLine(
            sprintf "[green bold]╟──────────────────────────────────────────────────────────────╢[/]"
        )

        AnsiConsole.MarkupLine(sprintf "[green bold]║[/] %s" (Markup.Escape(preview)))

        AnsiConsole.MarkupLine(
            sprintf "[green bold]╚══════════════════════════════════════════════════════════════╝[/]"
        )

        AnsiConsole.WriteLine()

    /// Print task failure
    let printFailure (reason: string) (duration: TimeSpan) =
        let preview =
            let s = reason.Replace("\n", " ").Replace("\r", "")
            if s.Length > 80 then s.Substring(0, 77) + "..." else s

        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[red bold]╔══════════════════════════════════════════════════════════════╗[/]")

        AnsiConsole.MarkupLine(
            sprintf "[red bold]║[/] [white bold]❌ TASK FAILED[/] [dim](%.1fs)[/]" duration.TotalSeconds
        )

        AnsiConsole.MarkupLine("[red bold]╟──────────────────────────────────────────────────────────────╢[/]")
        AnsiConsole.MarkupLine("[red bold]║[/] " + Markup.Escape(preview))
        AnsiConsole.MarkupLine("[red bold]╚══════════════════════════════════════════════════════════════╝[/]")
        AnsiConsole.WriteLine()

/// Progress display helpers
module Progress =

    /// Create a progress context with spinner
    let withSpinner (message: string) (action: unit -> 'T) : 'T =
        AnsiConsole
            .Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Style.Parse("cyan"))
            .Start(message, fun _ -> action ())

    /// Create a progress context with spinner async
    let withSpinnerAsync
        (message: string)
        (action: unit -> System.Threading.Tasks.Task<'T>)
        : System.Threading.Tasks.Task<'T> =
        AnsiConsole
            .Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Style.Parse("cyan"))
            .StartAsync(message, fun _ -> action ())

    /// Print a progress step
    let step (current: int) (total: int) (message: string) =
        AnsiConsole.MarkupLine(sprintf "[dim][[%d/%d]][/] %s" current total (Markup.Escape(message)))

/// Generation/Evolution display
module Evolution =

    /// Print generation header
    let printGeneration (gen: int) (completed: int) (total: int option) =
        let totalStr = total |> Option.map (sprintf "/%d") |> Option.defaultValue ""
        let rule = Rule(sprintf "Generation %d (%d%s tasks)" gen completed totalStr)
        rule.Style <- Style.Parse("cyan bold")
        AnsiConsole.Write(rule)

    /// Print task queue
    let printQueue (tasks: (float * string * string) list) =
        if not (List.isEmpty tasks) then
            let table = Table()
            table.Border <- TableBorder.Simple
            table.AddColumn("[bold]Score[/]") |> ignore
            table.AddColumn("[bold]Cost[/]") |> ignore
            table.AddColumn("[bold]Task[/]") |> ignore

            for (score, cost, goal) in tasks |> List.truncate 5 do
                let goalPreview =
                    if goal.Length > 50 then
                        goal.Substring(0, 47) + "..."
                    else
                        goal

                table.AddRow(sprintf "[cyan]%.2f[/]" score, sprintf "[yellow]%s[/]" cost, Markup.Escape(goalPreview))
                |> ignore

            AnsiConsole.Write(table)
            AnsiConsole.WriteLine()

    /// Print agent communication
    let printSpeechAct (performative: string) (fromAgent: string) (toAgent: string) =
        let color =
            match performative.ToLower() with
            | "request" -> "blue"
            | "inform" -> "green"
            | "refuse" -> "red"
            | "query" -> "purple"
            | _ -> "grey"

        AnsiConsole.MarkupLine(
            sprintf "[%s][%s][/] %s → %s" color performative (Markup.Escape(fromAgent)) (Markup.Escape(toAgent))
        )

    /// Print reflection step
    let printReflection (step: int) (total: int) (feedback: string) =
        let bar = String.replicate step "█" + String.replicate (total - step) "░"
        AnsiConsole.MarkupLine(sprintf "[cyan]🔄 REFLECTION[/] [%s] %d/%d" bar step total)

        if not (String.IsNullOrWhiteSpace feedback) then
            let preview =
                if feedback.Length > 80 then
                    feedback.Substring(0, 77) + "..."
                else
                    feedback

            AnsiConsole.MarkupLine("   [dim]" + Markup.Escape(preview) + "[/]")

    /// Print thinking indicator
    let printThinking (agent: string) =
        AnsiConsole.MarkupLine("[dim]💭 " + Markup.Escape(agent) + " is thinking...[/]")

    /// Print tool usage
    let printToolUse (tool: string) (inputPreview: string) =
        let preview =
            if inputPreview.Length > 60 then
                inputPreview.Substring(0, 57) + "..."
            else
                inputPreview

        AnsiConsole.MarkupLine(
            "[purple]🔧 "
            + Markup.Escape(tool)
            + "[/] [dim]"
            + Markup.Escape(preview)
            + "[/]"
        )

/// Summary display
module Summary =

    /// Print evolution summary
    let printEvolution (generations: int) (completed: int) (tokensUsed: int) (duration: TimeSpan) =
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[cyan bold]╔══════════════════════════════════════════════════════════════╗[/]")
        AnsiConsole.MarkupLine("[cyan bold]║[/] [white bold]📊 Evolution Summary[/]")
        AnsiConsole.MarkupLine("[cyan bold]╟──────────────────────────────────────────────────────────────╢[/]")
        AnsiConsole.MarkupLine(sprintf "[cyan bold]║[/] [bold]Generations:[/] %d" generations)
        AnsiConsole.MarkupLine(sprintf "[cyan bold]║[/] [bold]Tasks Completed:[/] %d" completed)
        AnsiConsole.MarkupLine(sprintf "[cyan bold]║[/] [bold]Tokens Used:[/] %s" (tokensUsed.ToString("N0")))
        AnsiConsole.MarkupLine(sprintf "[cyan bold]║[/] [bold]Duration:[/] %.1f minutes" duration.TotalMinutes)
        AnsiConsole.MarkupLine("[cyan bold]╚══════════════════════════════════════════════════════════════╝[/]")

    /// Print health check summary
    let printHealth (checks: (string * bool * float) list) =
        let table = Table()
        table.Border <- TableBorder.Rounded
        table.AddColumn("[bold]Check[/]") |> ignore
        table.AddColumn("[bold]Status[/]") |> ignore
        table.AddColumn("[bold]Duration[/]") |> ignore

        for (name, healthy, ms) in checks do
            let status =
                if healthy then
                    "[green]✓ Healthy[/]"
                else
                    "[red]✗ Unhealthy[/]"

            table.AddRow(Markup.Escape(name), status, sprintf "%.1fms" ms) |> ignore

        AnsiConsole.Write(table)
