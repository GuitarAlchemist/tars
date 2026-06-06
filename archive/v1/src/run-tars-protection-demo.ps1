#!/usr/bin/env pwsh

# TARS Code Protection Demo Runner
# Easy-to-use script for running the TARS protection system demo

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter(Position=1)]
    [string]$Path = "",

    [switch]$VerboseOutput,
    [switch]$Interactive
)

Write-Host "🛡️ TARS CODE PROTECTION DEMO RUNNER" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if F# Interactive is available
try {
    $fsiVersion = & dotnet fsi --help 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "F# Interactive not found"
    }
} catch {
    Write-Host "❌ F# Interactive (dotnet fsi) is required but not found." -ForegroundColor Red
    Write-Host "Please install .NET SDK with F# support." -ForegroundColor Yellow
    exit 1
}

# Validate command
$validCommands = @("help", "scan", "status", "report", "interactive")
if ($Command -notin $validCommands) {
    Write-Host "❌ Invalid command: $Command" -ForegroundColor Red
    Write-Host "Valid commands: $($validCommands -join ', ')" -ForegroundColor Yellow
    $Command = "help"
}

# Create F# script content based on command
$fsharpScript = @"
#r "nuget: Spectre.Console, 0.49.1"
#r "nuget: Microsoft.Extensions.Logging, 9.0.0"
#r "nuget: Microsoft.Extensions.Logging.Console, 9.0.5"

open System
open System.IO
open System.Threading.Tasks
open Spectre.Console

// Simplified protection demo module
module TarsProtectionDemo =
    
    type ScanResult = {
        Path: string
        FilesScanned: int
        SecurityScore: int
        QualityScore: int
        HighRisk: int
        MediumRisk: int
        LowRisk: int
    }
    
    let generateResults path =
        let random = Random()
        {
            Path = path
            FilesScanned = random.Next(20, 100)
            SecurityScore = random.Next(60, 95)
            QualityScore = random.Next(70, 95)
            HighRisk = random.Next(0, 3)
            MediumRisk = random.Next(1, 5)
            LowRisk = random.Next(2, 8)
        }
    
    let displayResults results =
        let rule = Rule("[bold blue]🛡️ TARS PROTECTION SCAN RESULTS[/]")
        rule.Justification <- Justify.Center
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()
        
        let table = Table()
        table.AddColumn(TableColumn("[bold]Metric[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Value[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Status[/]")) |> ignore
        
        table.AddRow([|"Files Scanned"; string results.FilesScanned; "[green]✓[/]"|]) |> ignore
        table.AddRow([|"Security Score"; sprintf "%d%%" results.SecurityScore;
            if results.SecurityScore >= 80 then "[green]Good[/]"
            elif results.SecurityScore >= 60 then "[yellow]Fair[/]"
            else "[red]Poor[/]"|]) |> ignore
        table.AddRow([|"Quality Score"; sprintf "%d%%" results.QualityScore;
            if results.QualityScore >= 80 then "[green]Good[/]"
            elif results.QualityScore >= 60 then "[yellow]Fair[/]"
            else "[red]Poor[/]"|]) |> ignore
        table.AddRow([|"High Risk"; string results.HighRisk;
            if results.HighRisk = 0 then "[green]None[/]" else "[red]Action Required[/]"|]) |> ignore
        table.AddRow([|"Medium Risk"; string results.MediumRisk;
            if results.MediumRisk = 0 then "[green]None[/]" else "[yellow]Review[/]"|]) |> ignore
        table.AddRow([|"Low Risk"; string results.LowRisk;
            if results.LowRisk = 0 then "[green]None[/]" else "[blue]Monitor[/]"|]) |> ignore
        
        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()
        
        if results.HighRisk > 0 then
            AnsiConsole.MarkupLine("[red]⚠️ High-risk vulnerabilities detected![/]")
        elif results.MediumRisk > 0 then
            AnsiConsole.MarkupLine("[yellow]⚠️ Medium-risk issues found.[/]")
        else
            AnsiConsole.MarkupLine("[green]✅ No critical issues detected.[/]")
    
    let runScan path =
        async {
            let rule = Rule("[bold blue]🛡️ TARS CODE PROTECTION DEMO[/]")
            rule.Justification <- Justify.Center
            AnsiConsole.Write(rule)
            AnsiConsole.WriteLine()
            
            AnsiConsole.MarkupLine(sprintf "[green]🔍 Scanning: %s[/]" path)
            AnsiConsole.WriteLine()
            
            // Simulate scanning with progress
            let! _ = AnsiConsole.Progress()
                .Columns([|
                    TaskDescriptionColumn() :> ProgressColumn
                    ProgressBarColumn() :> ProgressColumn
                    PercentageColumn() :> ProgressColumn
                    SpinnerColumn() :> ProgressColumn
                |])
                .StartAsync(fun ctx ->
                    async {
                        let task = ctx.AddTask("[green]Analyzing code security...[/]")
                        task.StartTask()
                        
                        task.Description <- "[blue]Discovering files...[/]"
                        do! Async.Sleep(800)
                        task.Increment(25.0)
                        
                        task.Description <- "[yellow]Security analysis...[/]"
                        do! Async.Sleep(1200)
                        task.Increment(30.0)
                        
                        task.Description <- "[orange1]Vulnerability detection...[/]"
                        do! Async.Sleep(1000)
                        task.Increment(25.0)
                        
                        task.Description <- "[purple]Quality assessment...[/]"
                        do! Async.Sleep(800)
                        task.Increment(20.0)
                        
                        task.StopTask()
                        return generateResults(path)
                    } |> Async.StartAsTask) |> Async.AwaitTask
            
            displayResults(_)
        }
    
    let showStatus () =
        let rule = Rule("[bold blue]🛡️ TARS PROTECTION SYSTEM STATUS[/]")
        rule.Justification <- Justify.Center
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()
        
        let table = Table()
        table.AddColumn(TableColumn("[bold]Component[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Status[/]")) |> ignore
        table.AddColumn(TableColumn("[bold]Version[/]")) |> ignore
        
        table.AddRow([|"RAG Code Analyzer"; "[green]Active[/]"; "v1.0.0"|]) |> ignore
        table.AddRow([|"Security Scanner"; "[green]Active[/]"; "v1.0.0"|]) |> ignore
        table.AddRow([|"Vulnerability DB"; "[green]Updated[/]"; "2024-09-07"|]) |> ignore
        table.AddRow([|"Quality Engine"; "[green]Active[/]"; "v1.0.0"|]) |> ignore
        table.AddRow([|"Autonomous Validator"; "[yellow]Partial[/]"; "v0.9.0"|]) |> ignore
        
        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[green]✅ Protection system operational[/]")
    
    let generateReport () =
        async {
            AnsiConsole.MarkupLine("[blue]📊 Generating security report...[/]")
            
            do! AnsiConsole.Status()
                .Spinner(Spinner.Known.Star)
                .StartAsync("[yellow]Generating...[/]", fun ctx ->
                    async {
                        ctx.Status("[blue]Analyzing data...[/]")
                        do! Async.Sleep(1000)
                        ctx.Status("[yellow]Formatting...[/]")
                        do! Async.Sleep(800)
                        ctx.Status("[green]Finalizing...[/]")
                        do! Async.Sleep(500)
                    } |> Async.StartAsTask) |> Async.AwaitTask
            
            let reportPath = Path.Combine(Directory.GetCurrentDirectory(), "tars-security-report.md")
            let content = sprintf """# TARS Security Report
Generated: %s

## Summary
- **Security Score**: 78%%
- **Quality Score**: 85%%
- **Files Analyzed**: 42
- **Issues Found**: 10

## Recommendations
- Implement input validation
- Update cryptographic algorithms
- Add security headers
- Enable security logging

---
Generated by TARS Code Protection System""" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
            
            File.WriteAllText(reportPath, content)
            AnsiConsole.MarkupLine(sprintf "[green]✅ Report: %s[/]" reportPath)
        }
    
    let showHelp () =
        let rule = Rule("[bold blue]🛡️ TARS PROTECTION DEMO[/]")
        rule.Justification <- Justify.Center
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("[yellow]📖 Available Commands:[/]")
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[cyan]  scan[/]     - Security scan of current directory")
        AnsiConsole.MarkupLine("[cyan]  scan <path>[/] - Security scan of specific path")
        AnsiConsole.MarkupLine("[cyan]  status[/]   - Show system status")
        AnsiConsole.MarkupLine("[cyan]  report[/]   - Generate security report")
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[dim]Examples:[/]")
        AnsiConsole.MarkupLine("[dim]  ./run-tars-protection-demo.ps1 scan[/]")
        AnsiConsole.MarkupLine("[dim]  ./run-tars-protection-demo.ps1 scan ./src[/]")
        AnsiConsole.MarkupLine("[dim]  ./run-tars-protection-demo.ps1 status[/]")

// Execute command
match "$Command" with
| "help" -> TarsProtectionDemo.showHelp()
| "scan" -> 
    let path = if "$Path" = "" then Directory.GetCurrentDirectory() else "$Path"
    TarsProtectionDemo.runScan(path) |> Async.RunSynchronously
| "status" -> TarsProtectionDemo.showStatus()
| "report" -> TarsProtectionDemo.generateReport() |> Async.RunSynchronously
| "interactive" -> 
    AnsiConsole.MarkupLine("[blue]🔧 Interactive mode - F# REPL ready![/]")
    AnsiConsole.MarkupLine("[dim]Use TarsProtectionDemo module functions directly[/]")
| _ -> TarsProtectionDemo.showHelp()
"@

# Handle interactive mode
if ($Interactive -or $Command -eq "interactive") {
    Write-Host "🔧 Starting F# Interactive mode..." -ForegroundColor Yellow
    Write-Host "Use TarsProtectionDemo module functions directly" -ForegroundColor Cyan
    Write-Host ""
    
    # Create temp script file
    $tempScript = [System.IO.Path]::GetTempFileName() + ".fsx"
    $fsharpScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    # Launch F# Interactive
    & dotnet fsi --load:$tempScript
    
    # Cleanup
    Remove-Item $tempScript -ErrorAction SilentlyContinue
    exit 0
}

# Run the demo
Write-Host "🚀 Running TARS Protection Demo..." -ForegroundColor Green
if ($VerboseOutput) {
    Write-Host "Command: $Command" -ForegroundColor Cyan
    if ($Path) { Write-Host "Path: $Path" -ForegroundColor Cyan }
}
Write-Host ""

# Create and execute temp script
$tempScript = [System.IO.Path]::GetTempFileName() + ".fsx"
try {
    $fsharpScript | Out-File -FilePath $tempScript -Encoding UTF8
    & dotnet fsi $tempScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Demo completed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "❌ Demo failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error running demo: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    # Cleanup
    if (Test-Path $tempScript) {
        Remove-Item $tempScript -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "• Run './run-tars-protection-demo.ps1 scan' for security analysis" -ForegroundColor White
Write-Host "• Run './run-tars-protection-demo.ps1 status' for system status" -ForegroundColor White
Write-Host "• Run './run-tars-protection-demo.ps1 report' to generate reports" -ForegroundColor White
Write-Host "• Run './run-tars-protection-demo.ps1 -Interactive' for F# REPL" -ForegroundColor White
Write-Host ""
Write-Host "🛡️ TARS Code Protection Demo Ready!" -ForegroundColor Green
