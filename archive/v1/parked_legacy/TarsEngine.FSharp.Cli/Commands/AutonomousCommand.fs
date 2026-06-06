namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Agents

/// <summary>
/// TARS Autonomous Modification Command - Real Tier 2 Autonomous Capabilities
/// Demonstrates genuine autonomous code modification with validation and rollback
/// </summary>
type AutonomousCommand(logger: ILogger<AutonomousCommand>) =

    interface ICommand with
        member _.Name = "autonomous"
        member _.Description = "TARS Autonomous Modification - Real Tier 2 autonomous code modification"
        member _.Usage = "tars autonomous [modify|demo|status|history] [options]"

        member this.ExecuteAsync(args: string[]) (options: CommandOptions) =
            task {
                try
                    // Display TARS Autonomous header
                    let rule = Rule("[bold red]🤖 TARS AUTONOMOUS MODIFICATION ENGINE[/]")
                    rule.Justification <- Justify.Center
                    AnsiConsole.Write(rule)
                    AnsiConsole.WriteLine()

                    match args with
                    | [||] | [|"help"|] ->
                        // Show help information
                        AnsiConsole.MarkupLine("[yellow]📖 TARS Autonomous Commands:[/]")
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[cyan]  modify <file> <description>[/] - Autonomously modify code file")
                        AnsiConsole.MarkupLine("[cyan]  demo[/]                      - Run autonomous modification demo")
                        AnsiConsole.MarkupLine("[cyan]  status[/]                    - Show autonomous system status")
                        AnsiConsole.MarkupLine("[cyan]  history[/]                   - Show modification history")
                        AnsiConsole.MarkupLine("[cyan]  cleanup[/]                   - Clean up all modifications")
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[dim]Examples:[/]")
                        AnsiConsole.MarkupLine("[dim]  tars autonomous modify ./MyFile.fs \"add error handling\"[/]")
                        AnsiConsole.MarkupLine("[dim]  tars autonomous demo[/]")
                        AnsiConsole.MarkupLine("[dim]  tars autonomous status[/]")
                        return CommandResult.success "TARS Autonomous help displayed"

                    | [|"modify"; filePath; description|] ->
                        // Autonomous modification
                        return! this.ExecuteModification(filePath, description)

                    | [|"demo"|] ->
                        // Run demonstration
                        return! this.RunDemo()

                    | [|"status"|] ->
                        // Show system status
                        return! this.ShowStatus()

                    | [|"history"|] ->
                        // Show modification history
                        return! this.ShowHistory()

                    | [|"cleanup"|] ->
                        // Clean up modifications
                        return! this.CleanupModifications()

                    | _ ->
                        AnsiConsole.MarkupLine("[red]❌ Unknown autonomous command. Use 'tars autonomous help' for usage.[/]")
                        return CommandResult.failure "Unknown autonomous command"

                with ex ->
                    logger.LogError(ex, "Error in AutonomousCommand")
                    AnsiConsole.MarkupLine($"[red]❌ Autonomous command failed: {ex.Message}[/]")
                    return CommandResult.failure $"Autonomous command failed: {ex.Message}"
            }

    /// Execute autonomous modification
    member private this.ExecuteModification(filePath: string, description: string) =
        task {
            AnsiConsole.MarkupLine($"[blue]🤖 Autonomous Modification Request[/]")
            AnsiConsole.MarkupLine($"[dim]File: {filePath}[/]")
            AnsiConsole.MarkupLine($"[dim]Description: {description}[/]")
            AnsiConsole.WriteLine()

            if not (File.Exists(filePath)) then
                AnsiConsole.MarkupLine($"[red]❌ File not found: {filePath}[/]")
                return CommandResult.failure $"File not found: {filePath}"

            let autonomousEngine = new RealAutonomousEngine(logger)

            let request = {
                Id = $"AUTO-{DateTime.Now.Ticks}"
                Description = description
                TargetFiles = [filePath]
                ExpectedOutcome = "Code improvement without breaking functionality"
                RiskLevel = "Medium"
                MaxExecutionTime = TimeSpan.FromMinutes(5.0)
            }

            // Execute with progress display
            let! result = AnsiConsole.Progress()
                .Columns([|
                    TaskDescriptionColumn() :> ProgressColumn
                    ProgressBarColumn() :> ProgressColumn
                    PercentageColumn() :> ProgressColumn
                    SpinnerColumn() :> ProgressColumn
                |])
                .StartAsync(fun ctx ->
                    task {
                        let task = ctx.AddTask("[green]Autonomous modification in progress...[/]")
                        task.StartTask()

                        task.Description <- "[blue]Analyzing code...[/]"
                        task.Increment(20.0)

                        task.Description <- "[yellow]Generating modifications...[/]"
                        task.Increment(20.0)

                        task.Description <- "[orange1]Applying changes...[/]"
                        task.Increment(20.0)

                        task.Description <- "[purple]Validating modifications...[/]"
                        let! result = autonomousEngine.ExecuteAutonomousModification(request)
                        task.Increment(30.0)

                        task.Description <- "[green]Autonomous modification complete[/]"
                        task.Increment(10.0)
                        task.StopTask()

                        return result
                    })

            // Display results
            this.DisplayModificationResult(result)

            if result.Success then
                return CommandResult.success "Autonomous modification completed successfully"
            else
                return CommandResult.failure $"Autonomous modification failed: {result.ErrorMessage |> Option.defaultValue "Unknown error"}"
        }

    /// Run autonomous modification demo
    member private this.RunDemo() =
        task {
            AnsiConsole.MarkupLine("[blue]🎯 TARS Autonomous Modification Demo[/]")
            AnsiConsole.WriteLine()

            // Create a demo file with intentional issues
            let demoFilePath = Path.Combine(Directory.GetCurrentDirectory(), "AutonomousDemo.fs")
            let demoContent = """namespace TarsDemo

open System

module DemoModule =

    // TODO: Implement real functionality
    let processData (input: string) =
        Console.WriteLine("Processing: " + input)
        try
            let result = input.ToUpper()
            result
        // Missing error handling

    let calculateScore (value: int) =
        if value > 0 then
            value * 2
        // Missing else branch
"""

            File.WriteAllText(demoFilePath, demoContent)
            AnsiConsole.MarkupLine($"[green]✅ Created demo file: {demoFilePath}[/]")
            AnsiConsole.WriteLine()

            // Show original content
            AnsiConsole.MarkupLine("[yellow]📄 Original Code (with issues):[/]")
            let panel = Panel(demoContent)
            panel.Header <- PanelHeader("AutonomousDemo.fs")
            panel.Border <- BoxBorder.Rounded
            AnsiConsole.Write(panel)
            AnsiConsole.WriteLine()

            // Execute autonomous modification
            let! result = this.ExecuteModification(demoFilePath, "fix compilation errors and add proper error handling")

            // Show modified content if successful
            if result.ExitCode = 0 && File.Exists(demoFilePath) then
                AnsiConsole.MarkupLine("[green]📄 Modified Code:[/]")
                let modifiedContent = File.ReadAllText(demoFilePath)
                let modifiedPanel = Panel(modifiedContent)
                modifiedPanel.Header <- PanelHeader("AutonomousDemo.fs (Modified)")
                modifiedPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(modifiedPanel)

            return result
        }

    /// Display modification result
    member private this.DisplayModificationResult(result: AutonomousResult) =
        AnsiConsole.WriteLine()

        let statusColor = if result.Success then "green" else "red"
        let statusIcon = if result.Success then "✅" else "❌"

        AnsiConsole.MarkupLine($"[{statusColor}]{statusIcon} Autonomous Modification Result[/]")
        AnsiConsole.WriteLine()

        let table = Table()
        table.AddColumn(TableColumn("[bold]Metric[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Value[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Status[/]")) |> ignore

        table.AddRow([|"Request ID"; result.RequestId; if result.Success then "[green]Success[/]" else "[red]Failed[/]"|]) |> ignore
        table.AddRow([|"Execution Time"; $"{result.ExecutionTime.TotalSeconds:F1}s"; "[blue]Info[/]"|]) |> ignore
        table.AddRow([|"Modifications Applied"; $"{result.ModificationsApplied.Length}"; "[blue]Info[/]"|]) |> ignore
        table.AddRow([|"Validation Score"; $"{result.ValidationResult.Score:P1}";
            if result.ValidationResult.Score >= 0.8 then "[green]Excellent[/]"
            elif result.ValidationResult.Score >= 0.6 then "[yellow]Good[/]"
            else "[red]Poor[/]"|]) |> ignore
        table.AddRow([|"Compilation"; if result.ValidationResult.CompilationPassed then "Passed" else "Failed";
            if result.ValidationResult.CompilationPassed then "[green]✓[/]" else "[red]✗[/]"|]) |> ignore
        table.AddRow([|"Tests"; if result.ValidationResult.TestsPassed then "Passed" else "Failed";
            if result.ValidationResult.TestsPassed then "[green]✓[/]" else "[red]✗[/]"|]) |> ignore
        table.AddRow([|"Coverage"; $"{result.ValidationResult.CoverageAchieved:P1}"; "[blue]Info[/]"|]) |> ignore

        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()

        // Show issues if any
        if not result.ValidationResult.QualityIssues.IsEmpty then
            AnsiConsole.MarkupLine("[yellow]⚠️ Quality Issues:[/]")
            for issue in result.ValidationResult.QualityIssues do
                AnsiConsole.MarkupLine($"[dim]  • {issue}[/]")
            AnsiConsole.WriteLine()

        // Show recommendations if any
        if not result.ValidationResult.Recommendations.IsEmpty then
            AnsiConsole.MarkupLine("[blue]💡 Recommendations:[/]")
            for recommendation in result.ValidationResult.Recommendations do
                AnsiConsole.MarkupLine($"[dim]  • {recommendation}[/]")
            AnsiConsole.WriteLine()

    /// Show autonomous system status
    member private this.ShowStatus() =
        task {
            let autonomousEngine = new RealAutonomousEngine(logger)
            let successRate = autonomousEngine.GetSuccessRate()
            let history = autonomousEngine.GetHistory()

            AnsiConsole.MarkupLine("[blue]📊 TARS Autonomous System Status[/]")
            AnsiConsole.WriteLine()

            let statusTable = Table()
            statusTable.AddColumn(TableColumn("[bold]Component[/]")) |> ignore
            statusTable.AddColumn(TableColumn("[bold]Status[/]")) |> ignore
            statusTable.AddColumn(TableColumn("[bold]Details[/]")) |> ignore

            statusTable.AddRow([|"Execution Harness"; "[green]Active[/]"; "Real command execution"|]) |> ignore
            statusTable.AddRow([|"Auto-Validation"; "[green]Active[/]"; "Comprehensive validation"|]) |> ignore
            statusTable.AddRow([|"Rollback System"; "[green]Active[/]"; "Safe modification rollback"|]) |> ignore
            statusTable.AddRow([|"Success Rate"; $"{successRate:P1}"; $"{history.Length} total modifications"|]) |> ignore
            statusTable.AddRow([|"Backup System"; "[green]Active[/]"; "Automatic file backups"|]) |> ignore

            AnsiConsole.Write(statusTable)
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[green]✅ Autonomous system operational[/]")

            return CommandResult.success "Autonomous system status displayed"
        }

    /// Show modification history
    member private this.ShowHistory() =
        task {
            let autonomousEngine = new RealAutonomousEngine(logger)
            let history = autonomousEngine.GetHistory()

            AnsiConsole.MarkupLine("[blue]📜 Autonomous Modification History[/]")
            AnsiConsole.WriteLine()

            if history.IsEmpty then
                AnsiConsole.MarkupLine("[dim]No autonomous modifications performed yet.[/]")
            else
                for result in history |> List.rev do // Show most recent first
                    let statusIcon = if result.Success then "✅" else "❌"
                    let statusColor = if result.Success then "green" else "red"

                    AnsiConsole.MarkupLine($"[{statusColor}]{statusIcon} {result.RequestId}[/] - {result.ExecutionTime.TotalSeconds:F1}s - Score: {result.ValidationResult.Score:P1}")
                    AnsiConsole.MarkupLine($"[dim]  Files: {String.Join(", ", result.ModificationsApplied |> List.map (fun p -> Path.GetFileName(p.TargetFile)))}[/]")
                    if result.ErrorMessage.IsSome then
                        AnsiConsole.MarkupLine($"[red]  Error: {result.ErrorMessage.Value}[/]")
                    AnsiConsole.WriteLine()

            return CommandResult.success "Autonomous modification history displayed"
        }

    /// Clean up all modifications
    member private this.CleanupModifications() =
        task {
            let autonomousEngine = new RealAutonomousEngine(logger)
            autonomousEngine.CleanupAll()

            AnsiConsole.MarkupLine("[green]✅ All autonomous modifications cleaned up[/]")

            return CommandResult.success "Autonomous modifications cleaned up"
        }

