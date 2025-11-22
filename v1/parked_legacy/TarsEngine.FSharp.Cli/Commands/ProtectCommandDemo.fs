namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core

/// <summary>
/// TARS Code Protection Demo - Simplified standalone version
/// Demonstrates the code protection system without CLI dependencies
/// </summary>
module ProtectCommandDemo =
    
    /// Simple scan result type
    type ScanResult = {
        Path: string
        FilesScanned: int
        SecurityIssues: int
        QualityIssues: int
        SecurityScore: int
        QualityScore: int
        HighRisk: int
        MediumRisk: int
        LowRisk: int
    }
    
    /// Generate demo scan results
    let generateDemoResults (path: string) =
        {
            Path = path
            FilesScanned = 42
            SecurityIssues = 3
            QualityIssues = 7
            SecurityScore = 78
            QualityScore = 85
            HighRisk = 1
            MediumRisk = 2
            LowRisk = 4
        }
    
    /// Display scan results in a beautiful table
    let displayResults (results: ScanResult) =
        AnsiConsole.WriteLine()
        
        // Create header
        let rule = Rule("[bold blue]🛡️ TARS PROTECTION SCAN RESULTS[/]")
        rule.Justification <- Justify.Center
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()
        
        // Create results table
        let table = Table()
        table.AddColumn(TableColumn("[bold]Metric[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Value[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Status[/]")) |> ignore
        
        // Add rows with color-coded status
        table.AddRow([|"Files Scanned"; $"{results.FilesScanned}"; "[green]✓[/]"|]) |> ignore
        table.AddRow([|"Security Score"; $"{results.SecurityScore}%%";
            (if results.SecurityScore >= 80 then "[green]Good[/]"
             elif results.SecurityScore >= 60 then "[yellow]Fair[/]"
             else "[red]Poor[/]")|]) |> ignore
        table.AddRow([|"Quality Score"; $"{results.QualityScore}%%";
            (if results.QualityScore >= 80 then "[green]Good[/]"
             elif results.QualityScore >= 60 then "[yellow]Fair[/]"
             else "[red]Poor[/]")|]) |> ignore
        table.AddRow([|"High Risk Issues"; $"{results.HighRisk}";
            (if results.HighRisk = 0 then "[green]None[/]" else "[red]Action Required[/]")|]) |> ignore
        table.AddRow([|"Medium Risk Issues"; $"{results.MediumRisk}";
            (if results.MediumRisk = 0 then "[green]None[/]" else "[yellow]Review[/]")|]) |> ignore
        table.AddRow([|"Low Risk Issues"; $"{results.LowRisk}";
            (if results.LowRisk = 0 then "[green]None[/]" else "[blue]Monitor[/]")|]) |> ignore
        
        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()
        
        // Summary message
        if results.HighRisk > 0 then
            AnsiConsole.MarkupLine("[red]⚠️ High-risk vulnerabilities detected! Immediate action required.[/]")
        elif results.MediumRisk > 0 then
            AnsiConsole.MarkupLine("[yellow]⚠️ Medium-risk issues found. Review recommended.[/]")
        else
            AnsiConsole.MarkupLine("[green]✅ No critical security issues detected.[/]")
        
        AnsiConsole.WriteLine()
    
    /// Run a demo scan with progress animation
    let runDemoScan (path: string) =
        task {
            // Display header
            let rule = Rule("[bold blue]🛡️ TARS CODE PROTECTION DEMO[/]")
            rule.Justification <- Justify.Center
            AnsiConsole.Write(rule)
            AnsiConsole.WriteLine()
            
            AnsiConsole.MarkupLine($"[green]🔍 Scanning: {path}[/]")
            AnsiConsole.WriteLine()
            
            // Run progress animation
            let! result = 
                AnsiConsole.Progress()
                    .Columns([|
                        TaskDescriptionColumn() :> ProgressColumn
                        ProgressBarColumn() :> ProgressColumn
                        PercentageColumn() :> ProgressColumn
                        SpinnerColumn() :> ProgressColumn
                    |])
                    .StartAsync(fun ctx ->
                        task {
                            let scanTask = ctx.AddTask("[green]Analyzing code security...[/]")
                            scanTask.StartTask()
                            
                            // Phase 1: File discovery
                            scanTask.Description <- "[blue]Discovering files...[/]"
                            do! Task.Delay(800)
                            scanTask.Increment(25.0)
                            
                            // Phase 2: Security analysis
                            scanTask.Description <- "[yellow]Analyzing security patterns...[/]"
                            do! Task.Delay(1200)
                            scanTask.Increment(30.0)
                            
                            // Phase 3: Vulnerability detection
                            scanTask.Description <- "[orange1]Detecting vulnerabilities...[/]"
                            do! Task.Delay(1000)
                            scanTask.Increment(25.0)
                            
                            // Phase 4: Quality assessment
                            scanTask.Description <- "[purple]Assessing code quality...[/]"
                            do! Task.Delay(800)
                            scanTask.Increment(20.0)
                            
                            scanTask.StopTask()
                            return generateDemoResults(path)
                        })
            
            // Display results
            displayResults(result)
            
            return result
        }
    
    /// Show protection system status
    let showStatus () =
        AnsiConsole.WriteLine()
        
        let rule = Rule("[bold blue]🛡️ TARS PROTECTION SYSTEM STATUS[/]")
        rule.Justification <- Justify.Center
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()
        
        let statusTable = Table()
        statusTable.AddColumn(TableColumn("[bold]Component[/]")) |> ignore
        statusTable.AddColumn(TableColumn("[bold]Status[/]")) |> ignore
        statusTable.AddColumn(TableColumn("[bold]Version[/]")) |> ignore
        
        statusTable.AddRow([|"RAG Code Analyzer"; "[green]Active[/]"; "v1.0.0"|]) |> ignore
        statusTable.AddRow([|"Security Scanner"; "[green]Active[/]"; "v1.0.0"|]) |> ignore
        statusTable.AddRow([|"Vulnerability Database"; "[green]Updated[/]"; "2024-09-07"|]) |> ignore
        statusTable.AddRow([|"Code Quality Engine"; "[green]Active[/]"; "v1.0.0"|]) |> ignore
        statusTable.AddRow([|"Autonomous Validator"; "[yellow]Partial[/]"; "v0.9.0"|]) |> ignore
        
        AnsiConsole.Write(statusTable)
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[green]✅ Protection system operational[/]")
        AnsiConsole.WriteLine()
    
    /// Generate a demo security report
    let generateReport () =
        task {
            AnsiConsole.MarkupLine("[blue]📊 Generating TARS security report...[/]")
            AnsiConsole.WriteLine()
            
            // Simulate report generation with progress
            do! AnsiConsole.Status()
                .Spinner(Spinner.Known.Star)
                .StartAsync("[yellow]Generating report...[/]", fun ctx ->
                    task {
                        ctx.Status("[blue]Analyzing scan data...[/]")
                        do! Task.Delay(1000)
                        ctx.Status("[yellow]Formatting report...[/]")
                        do! Task.Delay(800)
                        ctx.Status("[green]Finalizing...[/]")
                        do! Task.Delay(500)
                    })
            
            let reportPath = Path.Combine(Directory.GetCurrentDirectory(), "tars-security-report.md")
            let reportContent = sprintf """# TARS Security Report
Generated: %s

## Summary
- **Overall Security Score**: 78%%
- **Code Quality Score**: 85%%
- **Files Analyzed**: 42
- **Total Issues**: 10

## High-Risk Vulnerabilities (1)
1. **SQL Injection Risk** - Line 45 in DatabaseHelper.cs
   - Severity: High
   - Description: Unsanitized user input in SQL query
   - Recommendation: Use parameterized queries

## Medium-Risk Issues (2)
1. **Weak Cryptography** - Line 123 in CryptoUtils.cs
2. **Information Disclosure** - Line 67 in ErrorHandler.cs

## Recommendations
- Implement input validation
- Update cryptographic algorithms
- Add security headers
- Enable security logging

---
Generated by TARS Code Protection System""" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
            
            File.WriteAllText(reportPath, reportContent)
            AnsiConsole.MarkupLine($"[green]✅ Report generated: {reportPath}[/]")
            AnsiConsole.WriteLine()
            
            return reportPath
        }
    
    /// Main demo function
    let runDemo (args: string[]) =
        task {
            try
                match args with
                | [||] | [|"help"|] ->
                    // Show help
                    let rule = Rule("[bold blue]🛡️ TARS PROTECTION DEMO[/]")
                    rule.Justification <- Justify.Center
                    AnsiConsole.Write(rule)
                    AnsiConsole.WriteLine()
                    
                    AnsiConsole.MarkupLine("[yellow]📖 Available Demo Commands:[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[cyan]  scan[/]     - Demo security scan of current directory")
                    AnsiConsole.MarkupLine("[cyan]  scan <path>[/] - Demo security scan of specific path")
                    AnsiConsole.MarkupLine("[cyan]  status[/]   - Show protection system status")
                    AnsiConsole.MarkupLine("[cyan]  report[/]   - Generate demo security report")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[dim]Examples:[/]")
                    AnsiConsole.MarkupLine("[dim]  dotnet run scan[/]")
                    AnsiConsole.MarkupLine("[dim]  dotnet run scan ./src[/]")
                    AnsiConsole.MarkupLine("[dim]  dotnet run status[/]")
                    AnsiConsole.WriteLine()
                    
                | [|"scan"|] ->
                    let currentDir = Directory.GetCurrentDirectory()
                    let! _ = runDemoScan(currentDir)
                    ()
                    
                | [|"scan"; path|] ->
                    if Directory.Exists(path) || File.Exists(path) then
                        let! _ = runDemoScan(path)
                        ()
                    else
                        AnsiConsole.MarkupLine($"[red]❌ Path not found: {path}[/]")
                        
                | [|"status"|] ->
                    showStatus()
                    
                | [|"report"|] ->
                    let! _ = generateReport()
                    ()
                    
                | _ ->
                    AnsiConsole.MarkupLine("[red]❌ Unknown command. Use 'help' for usage.[/]")
                    
            with ex ->
                AnsiConsole.MarkupLine($"[red]❌ Demo failed: {ex.Message}[/]")
        }
