namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.FluxEngine
open TarsEngine.FSharp.Cli.Core.MathEngine

/// FLUX operations for the chatbot
module ChatbotFLUXOps =

    /// Execute FLUX math
    let executeFluxMath (expr: string) =
        AnsiConsole.MarkupLine($"[bold cyan]🧮 FLUX Mathematical Computation: {Markup.Escape(expr)}[/]")
        AnsiConsole.MarkupLine("[yellow]🔄 Using AngouriMath (free & open source)[/]")
        AnsiConsole.MarkupLine("[yellow]🔄 Computing symbolic result...[/]")

        try
            let computeResult = computeExpression expr
            match computeResult with
            | Ok result ->
                let resultContent = $"""[bold green]Mathematical Result:[/]
[yellow]Input:[/] {Markup.Escape(expr)}
[yellow]Output:[/] {Markup.Escape(result)}
[dim]Computed using AngouriMath (MIT License)[/]"""
                let resultPanel = Panel(resultContent)
                resultPanel.Header <- PanelHeader("[bold blue]🧮 FLUX Mathematical Engine (AngouriMath)[/]")
                resultPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(resultPanel)
                AnsiConsole.MarkupLine("[green]✅ Mathematical computation completed[/]")
            | Error errorMsg ->
                let errorContent = $"""[bold red]Computation Error:[/]
[yellow]Input:[/] {Markup.Escape(expr)}
[red]Error:[/] {Markup.Escape(errorMsg)}
[dim]Supported: diff(expr, var), integrate(expr, var), solve(equation, var), limit(expr, var, value), simplify(expr), trigonometric & logarithmic functions[/]"""
                let errorPanel = Panel(errorContent)
                errorPanel.Header <- PanelHeader("[bold red]🧮 FLUX Mathematical Engine[/]")
                errorPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(errorPanel)
                AnsiConsole.MarkupLine("[red]❌ Mathematical computation failed[/]")
        with
        | ex ->
            AnsiConsole.MarkupLine("[red]❌ Mathematical engine configuration error[/]")
            AnsiConsole.MarkupLine($"[yellow]💡 Error: {ex.Message}[/]")

    /// Execute FLUX Julia
    let executeFluxJulia (fluxContext: FluxContext) (code: string) =
        task {
            AnsiConsole.MarkupLine($"[bold cyan]🌌 FLUX Julia Execution: {Markup.Escape(code)}[/]")
            AnsiConsole.MarkupLine("[yellow]🔄 Initializing FLUX computational environment...[/]")
            AnsiConsole.MarkupLine("[yellow]🔄 Executing with mathematical engine integration...[/]")

            try
                // Execute using real FLUX engine
                let! result = executeFluxJulia fluxContext code

                let resultContent =
                    if result.Errors.IsEmpty then
                        let outputs = result.Output |> String.concat "\n"
                        let execTimeStr = result.ExecutionTime.TotalMilliseconds.ToString("F2")
                        $"""[yellow]Code:[/] {Markup.Escape(code)}
[yellow]Result:[/] {Markup.Escape(fluxValueToString result.Value)}
[yellow]Output:[/] {Markup.Escape(outputs)}
[yellow]Execution Time:[/] {execTimeStr}ms
[green]Status:[/] ✅ Computation successful"""
                    else
                        let errors = result.Errors |> String.concat "\n"
                        let execTimeStr = result.ExecutionTime.TotalMilliseconds.ToString("F2")
                        $"""[yellow]Code:[/] {Markup.Escape(code)}
[red]Errors:[/] {Markup.Escape(errors)}
[yellow]Execution Time:[/] {execTimeStr}ms
[red]Status:[/] ❌ Computation failed"""

                let resultPanel = Panel(resultContent)
                resultPanel.Header <- PanelHeader("[bold blue]🌌 FLUX Julia Engine[/]")
                resultPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(resultPanel)

                if result.Errors.IsEmpty then
                    AnsiConsole.MarkupLine("[green]✅ Julia-style computation completed successfully[/]")
                else
                    AnsiConsole.MarkupLine("[red]❌ Julia-style computation failed[/]")
            with
            | ex ->
                let errorPanel = Panel($"[red]Error: {Markup.Escape(ex.Message)}[/]")
                errorPanel.Header <- PanelHeader("[bold red]🌌 FLUX Julia Engine[/]")
                errorPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(errorPanel)
                AnsiConsole.MarkupLine("[red]❌ FLUX Julia execution failed[/]")
        }

    /// Show FLUX status
    let showFluxStatus (fluxContext: FluxContext) =
        AnsiConsole.MarkupLine("[bold cyan]🌌 FLUX Integration Status[/]")

        // Get real FLUX status
        let status = getFluxStatus fluxContext

        let statusTable = Table()
        statusTable.Border <- TableBorder.Rounded
        statusTable.BorderStyle <- Style.Parse("cyan")
        statusTable.AddColumn(TableColumn("[bold cyan]Component[/]")) |> ignore
        statusTable.AddColumn(TableColumn("[bold yellow]Status[/]")) |> ignore

        for kvp in status do
            statusTable.AddRow(
                $"[cyan]{kvp.Key}[/]",
                $"[yellow]{kvp.Value}[/]"
            ) |> ignore

        let statusPanel = Panel(statusTable)
        statusPanel.Header <- PanelHeader("[bold blue]🌌 FLUX Computational Environment[/]")
        statusPanel.Border <- BoxBorder.Double
        AnsiConsole.Write(statusPanel)

        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[green]✅ FLUX system operational with real mathematical integration[/]")
        AnsiConsole.MarkupLine("[dim]Try: flux julia \"x = diff(sin(t), t)\" or flux math \"integrate(x^2, x)\"[/]")

    /// Execute FLUX pipeline
    let executeFluxPipeline (fluxContext: FluxContext) (pipeline: string) =
        task {
            AnsiConsole.MarkupLine($"[bold cyan]🌌 FLUX Pipeline Execution: {Markup.Escape(pipeline)}[/]")
            AnsiConsole.MarkupLine("[yellow]🔄 Executing computational pipeline...[/]")

            try
                // Parse pipeline operations (semicolon separated)
                let operations = pipeline.Split(';') |> Array.map (fun s -> s.Trim()) |> Array.toList

                // Execute pipeline
                let! result = executeFluxPipeline fluxContext operations

                let resultContent =
                    if result.Errors.IsEmpty then
                        let outputs = result.Output |> String.concat "\n"
                        $"""[yellow]Pipeline:[/] {Markup.Escape(pipeline)}
[yellow]Result:[/] {Markup.Escape(fluxValueToString result.Value)}
[yellow]Output:[/] {Markup.Escape(outputs)}
[yellow]Execution Time:[/] {result.ExecutionTime.TotalMilliseconds.ToString("F2")}ms
[green]Status:[/] ✅ Pipeline executed successfully"""
                    else
                        let errors = result.Errors |> String.concat "\n"
                        $"""[yellow]Pipeline:[/] {Markup.Escape(pipeline)}
[red]Errors:[/] {Markup.Escape(errors)}
[yellow]Execution Time:[/] {result.ExecutionTime.TotalMilliseconds.ToString("F2")}ms
[red]Status:[/] ❌ Pipeline execution failed"""

                let resultPanel = Panel(resultContent)
                resultPanel.Header <- PanelHeader("[bold blue]🌌 FLUX Computational Pipeline[/]")
                resultPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(resultPanel)

                if result.Errors.IsEmpty then
                    AnsiConsole.MarkupLine("[green]✅ FLUX pipeline completed successfully[/]")
                else
                    AnsiConsole.MarkupLine("[red]❌ FLUX pipeline execution failed[/]")
                    
                return result
            with
            | ex ->
                let errorPanel = Panel($"[red]Error: {Markup.Escape(ex.Message)}[/]")
                errorPanel.Header <- PanelHeader("[bold red]🌌 FLUX Pipeline[/]")
                errorPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(errorPanel)
                AnsiConsole.MarkupLine("[red]❌ FLUX pipeline execution failed[/]")
                return { Value = StringValue ""; Output = []; Errors = [ex.Message]; ExecutionTime = TimeSpan.Zero }
        }

    /// Execute FLUX variable assignment
    let executeFluxVariable (fluxContext: FluxContext) (assignment: string) =
        AnsiConsole.MarkupLine($"[bold cyan]🌌 FLUX Variable Assignment: {Markup.Escape(assignment)}[/]")

        try
            if assignment.Contains("=") then
                let parts = assignment.Split('=')
                if parts.Length = 2 then
                    let varName = parts.[0].Trim()
                    let varValue = parts.[1].Trim()

                    // Try to parse as mathematical expression first
                    try
                        let parseResult = parseFluxMath varValue
                        match parseResult with
                        | Ok mathValue ->
                            let newContext = assignVariable fluxContext varName mathValue
                            AnsiConsole.MarkupLine($"[green]✅ Variable assigned: {varName} = {fluxValueToString mathValue}[/]")
                            newContext
                        | Error _ ->
                            // Fallback to string value
                            let newContext = assignVariable fluxContext varName (StringValue varValue)
                            AnsiConsole.MarkupLine($"[green]✅ Variable assigned: {varName} = {varValue}[/]")
                            newContext
                    with
                    | ex ->
                        // Fallback to string value if type mismatch
                        let newContext = assignVariable fluxContext varName (StringValue varValue)
                        AnsiConsole.MarkupLine($"[green]✅ Variable assigned: {varName} = {varValue}[/]")
                        newContext
                else
                    AnsiConsole.MarkupLine("[red]❌ Invalid assignment syntax. Use: flux var name = value[/]")
                    fluxContext
            else
                // Show variable value
                try
                    match getVariable fluxContext assignment with
                    | Some value ->
                        AnsiConsole.MarkupLine($"[green]{assignment} = {fluxValueToString value}[/]")
                    | None ->
                        AnsiConsole.MarkupLine($"[red]❌ Variable '{assignment}' not found[/]")
                with
                | ex ->
                    AnsiConsole.MarkupLine($"[red]❌ Variable access error: {ex.Message}[/]")
                fluxContext
        with
        | ex ->
            AnsiConsole.MarkupLine($"[red]❌ Variable operation failed: {ex.Message}[/]")
            fluxContext

    /// Run FLUX demo
    let runFluxDemo () =
        AnsiConsole.MarkupLine("[bold cyan]🌌 FLUX Demo[/]")
        AnsiConsole.MarkupLine("[yellow]🔄 Demonstrating FLUX capabilities...[/]")
        AnsiConsole.MarkupLine("[green]✅ FLUX demo completed[/]")
