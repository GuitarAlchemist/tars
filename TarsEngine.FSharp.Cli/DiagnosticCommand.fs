namespace TarsEngine.FSharp.CLI

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open Spectre.Console
open TarsEngine.FSharp.Diagnostics

/// CLI command for TARS metascript diagnostics
module DiagnosticCommand =
    
    /// Diagnostic command options
    type DiagnosticOptions = {
        MetascriptPath: string option
        ProjectPath: string option
        TraceFile: string option
        EnableSemanticAnalysis: bool
        ListTraces: bool
        ShowSummary: bool
        OutputFormat: string
        TestCryptographicProof: bool
        VerifyTrace: string option
    }
    
    /// Execute diagnostic command
    let execute (options: DiagnosticOptions) (serviceProvider: IServiceProvider) =
        task {
            let logger = serviceProvider.GetRequiredService<ILogger<obj>>()
            let diagnosticRunner = serviceProvider.GetRequiredService<MetascriptDiagnosticRunner>()
            
            // Create a fancy header
            let rule = Rule("[bold green]TARS Metascript Diagnostic Engine[/]")
            rule.Alignment <- Justify.Center
            AnsiConsole.Write(rule)
            AnsiConsole.WriteLine()
            
            match options with
            | { ListTraces = true } ->
                do! executeListTraces diagnosticRunner logger
                
            | { ShowSummary = true } ->
                do! executeShowSummary diagnosticRunner logger
                
            | { TraceFile = Some traceFile } ->
                do! executeAnalyzeTrace traceFile diagnosticRunner logger
                
            | { MetascriptPath = Some metascriptPath } ->
                do! executeRunDiagnostics metascriptPath options.ProjectPath options.EnableSemanticAnalysis diagnosticRunner logger
                
            | _ ->
                showHelp()
        }
    
    /// Execute list traces command
    let private executeListTraces (diagnosticRunner: MetascriptDiagnosticRunner) (logger: ILogger<obj>) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]📊 Available Diagnostic Traces[/]")
            AnsiConsole.WriteLine()
            
            let traceFiles = diagnosticRunner.ListTraceFiles()
            
            if traceFiles.IsEmpty then
                AnsiConsole.MarkupLine("[yellow]No diagnostic traces found.[/]")
                AnsiConsole.MarkupLine("[dim]Run a metascript with diagnostics to generate traces.[/]")
            else
                let table = Table()
                table.AddColumn("Trace ID") |> ignore
                table.AddColumn("Created") |> ignore
                table.AddColumn("Size") |> ignore
                table.AddColumn("File Path") |> ignore
                
                for trace in traceFiles do
                    table.AddRow(
                        trace.TraceId,
                        trace.CreatedTime.ToString("yyyy-MM-dd HH:mm:ss"),
                        sprintf "%.1f KB" (float trace.Size / 1024.0),
                        trace.FileName
                    ) |> ignore
                
                AnsiConsole.Write(table)
                AnsiConsole.WriteLine()
                
                AnsiConsole.MarkupLine("[dim]Use --trace-file <file> to analyze a specific trace[/]")
        }
    
    /// Execute show summary command
    let private executeShowSummary (diagnosticRunner: MetascriptDiagnosticRunner) (logger: ILogger<obj>) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]📈 Diagnostic Summary[/]")
            AnsiConsole.WriteLine()
            
            let! summary = diagnosticRunner.GetDiagnosticSummary()
            
            // Overall statistics
            let statsPanel = Panel(sprintf """
[bold]Total Diagnostic Runs:[/] %d
[bold]Recent Runs:[/] %d
[bold]Common Issues Found:[/] %d
""" summary.TotalRuns summary.RecentRuns.Length summary.CommonIssues.Length)
            statsPanel.Header <- PanelHeader("📊 Statistics")
            statsPanel.Border <- BoxBorder.Rounded
            AnsiConsole.Write(statsPanel)
            AnsiConsole.WriteLine()
            
            // Recent runs
            if not summary.RecentRuns.IsEmpty then
                AnsiConsole.MarkupLine("[bold yellow]🕒 Recent Runs[/]")
                let recentTable = Table()
                recentTable.AddColumn("Trace ID") |> ignore
                recentTable.AddColumn("Metascript") |> ignore
                recentTable.AddColumn("Duration") |> ignore
                recentTable.AddColumn("Components") |> ignore
                recentTable.AddColumn("Errors") |> ignore
                recentTable.AddColumn("Status") |> ignore
                
                for run in summary.RecentRuns do
                    let status = if run.Success then "[green]✅ Success[/]" else "[red]❌ Failed[/]"
                    let metascriptName = Path.GetFileName(run.MetascriptPath)
                    
                    recentTable.AddRow(
                        run.TraceId,
                        metascriptName,
                        sprintf "%.1fs" run.ExecutionTime,
                        run.ComponentsGenerated.ToString(),
                        run.ErrorCount.ToString(),
                        status
                    ) |> ignore
                
                AnsiConsole.Write(recentTable)
                AnsiConsole.WriteLine()
            
            // Common issues
            if not summary.CommonIssues.IsEmpty then
                AnsiConsole.MarkupLine("[bold red]⚠️ Common Issues[/]")
                for issue in summary.CommonIssues do
                    AnsiConsole.MarkupLine(sprintf "• [red]%s[/] ([dim]%d occurrences[/])" issue.IssueType issue.Count)
                AnsiConsole.WriteLine()
            
            // Recommendations
            if not summary.Recommendations.IsEmpty then
                AnsiConsole.MarkupLine("[bold green]💡 Recommendations[/]")
                for recommendation in summary.Recommendations do
                    AnsiConsole.MarkupLine(sprintf "• [green]%s[/]" recommendation)
                AnsiConsole.WriteLine()
        }
    
    /// Execute analyze trace command
    let private executeAnalyzeTrace (traceFile: string) (diagnosticRunner: MetascriptDiagnosticRunner) (logger: ILogger<obj>) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]🔍 Analyzing Trace: %s[/]" (Path.GetFileName(traceFile)))
            AnsiConsole.WriteLine()
            
            let! result = 
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("green"))
                    .StartAsync("Analyzing trace file...", fun ctx ->
                        task {
                            ctx.Status <- "Loading trace data..."
                            do! Task.Delay(500)
                            
                            ctx.Status <- "Performing semantic analysis..."
                            let! result = diagnosticRunner.AnalyzeExistingTrace(traceFile)
                            
                            ctx.Status <- "Generating report..."
                            do! Task.Delay(300)
                            
                            return result
                        })
            
            match result with
            | Ok analysis ->
                AnsiConsole.MarkupLine("[green]✅ Analysis completed successfully![/]")
                AnsiConsole.WriteLine()
                
                let infoPanel = Panel(sprintf """
[bold]Trace ID:[/] %s
[bold]Original Trace:[/] %s
[bold]Semantic Report:[/] %s
""" analysis.TraceId (Path.GetFileName(analysis.OriginalTraceFile)) (Path.GetFileName(analysis.SemanticReportFile)))
                infoPanel.Header <- PanelHeader("📋 Analysis Results")
                infoPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(infoPanel)
                
                // Show issue analysis if available
                match analysis.IssueAnalysis with
                | Some issue ->
                    AnsiConsole.WriteLine()
                    let issuePanel = Panel(sprintf """
[bold red]Issue Type:[/] %s
[bold red]Severity:[/] %s
[bold red]Description:[/] %s
[bold red]Root Cause:[/] %s
""" issue.IssueType (issue.Severity.ToString()) issue.Description issue.RootCause)
                    issuePanel.Header <- PanelHeader("⚠️ Issue Analysis")
                    issuePanel.Border <- BoxBorder.Rounded
                    AnsiConsole.Write(issuePanel)
                | None ->
                    AnsiConsole.MarkupLine("[green]No critical issues identified.[/]")
                
                // Show recommended fixes
                if not analysis.RecommendedFixes.IsEmpty then
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold yellow]🔧 Recommended Fixes:[/]")
                    for fix in analysis.RecommendedFixes |> List.take 3 do
                        AnsiConsole.MarkupLine(sprintf "• [yellow]%s[/] ([dim]%s priority, %s effort[/])" 
                                             fix.Description (fix.Priority.ToString()) fix.EstimatedEffort)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]Full report available at: %s[/]" analysis.SemanticReportFile)
                
            | Error errorMsg ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Analysis failed: %s[/]" errorMsg)
        }
    
    /// Execute run diagnostics command
    let private executeRunDiagnostics (metascriptPath: string) (projectPath: string option) (enableSemanticAnalysis: bool) 
                                     (diagnosticRunner: MetascriptDiagnosticRunner) (logger: ILogger<obj>) =
        task {
            let metascriptName = Path.GetFileName(metascriptPath)
            AnsiConsole.MarkupLine(sprintf "[bold cyan]🚀 Running Metascript with Diagnostics: %s[/]" metascriptName)
            AnsiConsole.WriteLine()
            
            if not (File.Exists(metascriptPath)) then
                AnsiConsole.MarkupLine(sprintf "[red]❌ Metascript file not found: %s[/]" metascriptPath)
                return ()
            
            let! result = 
                AnsiConsole.Progress()
                    .Columns([|
                        TaskDescriptionColumn()
                        ProgressBarColumn()
                        PercentageColumn()
                        SpinnerColumn()
                    |])
                    .StartAsync(fun ctx ->
                        task {
                            let task1 = ctx.AddTask("Initializing diagnostics...")
                            task1.Increment(20.0)
                            
                            let task2 = ctx.AddTask("Executing metascript...")
                            let task3 = ctx.AddTask("Generating trace...")
                            let task4 = ctx.AddTask("Semantic analysis...")
                            
                            let! result = diagnosticRunner.RunWithDiagnostics(metascriptPath, ?projectPath = projectPath, enableSemanticAnalysis)
                            
                            task1.Increment(80.0)
                            task2.Increment(100.0)
                            task3.Increment(100.0)
                            
                            if enableSemanticAnalysis then
                                task4.Increment(100.0)
                            
                            return result
                        })
            
            match result with
            | Ok diagnosticResult ->
                AnsiConsole.MarkupLine("[green]✅ Metascript execution with diagnostics completed![/]")
                AnsiConsole.WriteLine()
                
                let resultPanel = Panel(sprintf """
[bold]Trace ID:[/] %s
[bold]Trace File:[/] %s
[bold]Report File:[/] %s
%s
""" diagnosticResult.TraceId 
    (Path.GetFileName(diagnosticResult.TraceFile))
    (Path.GetFileName(diagnosticResult.ReportFile))
    (match diagnosticResult.SemanticReportFile with
     | Some file -> sprintf "[bold]Semantic Report:[/] %s" (Path.GetFileName(file))
     | None -> "[dim]Semantic analysis disabled[/]"))
                
                resultPanel.Header <- PanelHeader("📊 Diagnostic Results")
                resultPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(resultPanel)
                
                // Show issue analysis
                match diagnosticResult.IssueAnalysis with
                | Some issue ->
                    AnsiConsole.WriteLine()
                    let issuePanel = Panel(sprintf """
[bold red]Issue Detected:[/] %s
[bold red]Severity:[/] %s
[bold red]Root Cause:[/] %s
""" issue.IssueType (issue.Severity.ToString()) issue.RootCause)
                    issuePanel.Header <- PanelHeader("⚠️ Issues Found")
                    issuePanel.Border <- BoxBorder.Rounded
                    AnsiConsole.Write(issuePanel)
                    
                    // Show top recommended fixes
                    if not diagnosticResult.RecommendedFixes.IsEmpty then
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[bold yellow]🔧 Top Recommended Fixes:[/]")
                        for fix in diagnosticResult.RecommendedFixes |> List.take 2 do
                            AnsiConsole.MarkupLine(sprintf "• [yellow]%s[/]" fix.Description)
                            AnsiConsole.MarkupLine(sprintf "  [dim]Priority: %s | Effort: %s | Risk: %s[/]" 
                                                 (fix.Priority.ToString()) fix.EstimatedEffort (fix.RiskLevel.ToString()))
                
                | None ->
                    AnsiConsole.MarkupLine("[green]✅ No critical issues detected.[/]")
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[dim]Files saved to .tars/traces/ directory[/]")
                
            | Error errorMsg ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Diagnostic execution failed: %s[/]" errorMsg)
        }
    
    /// Show help information
    let private showHelp() =
        let helpPanel = Panel("""
[bold yellow]TARS Metascript Diagnostic Engine[/]

[bold]Usage:[/]
  tars diagnose --metascript <path>           Run metascript with full diagnostics
  tars diagnose --trace-file <path>          Analyze existing trace file
  tars diagnose --list-traces                List all available traces
  tars diagnose --summary                    Show diagnostic summary

[bold]Options:[/]
  --metascript <path>        Path to metascript file (.trsx)
  --project <path>           Path to project directory (optional)
  --trace-file <path>        Path to existing trace file (.json)
  --no-semantic              Disable semantic analysis
  --list-traces              List all available trace files
  --summary                  Show summary of recent diagnostic runs
  --output-format <format>   Output format (json, yaml, markdown)

[bold]Examples:[/]
  tars diagnose --metascript .tars/ui-builder.trsx
  tars diagnose --trace-file .tars/traces/trace_abc123_2024-12-19_14-30-15.json
  tars diagnose --summary
""")
        helpPanel.Header <- PanelHeader("Help")
        helpPanel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(helpPanel)
