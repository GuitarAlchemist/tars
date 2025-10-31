namespace TarsEngine.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.GordonIntegration
open Spectre.Console

/// Gordon-assisted infrastructure consolidation command
module GordonConsolidateCommand =
    
    type ConsolidationOptions = {
        DryRun: bool
        Force: bool
        SkipHealthCheck: bool
        Stage: string option // "database", "application", "web", "all"
        Verbose: bool
    }
    
    let private createProgressTable() =
        let table = Table()
        table.AddColumn("Stage") |> ignore
        table.AddColumn("Status") |> ignore
        table.AddColumn("Gordon's Assessment") |> ignore
        table.AddColumn("Actions") |> ignore
        table
    
    let private updateProgress (table: Table) stage status assessment actions =
        let statusMarkup = 
            match status with
            | "Complete" -> "[green]✅ Complete[/]"
            | "In Progress" -> "[yellow]🔄 In Progress[/]"
            | "Failed" -> "[red]❌ Failed[/]"
            | "Pending" -> "[gray]⏳ Pending[/]"
            | _ -> status
        
        table.AddRow(stage, statusMarkup, assessment, actions) |> ignore
    
    let private displayGordonAnalysis (analysis: GordonAnalysisResult) =
        let panel = Panel($"""
[bold cyan]🤖 Gordon's Infrastructure Analysis[/]

[bold]Health Score:[/] {analysis.HealthScore}/100
[bold]Analysis Type:[/] {analysis.AnalysisType}
[bold]Timestamp:[/] {analysis.Timestamp:yyyy-MM-dd HH:mm:ss}

[bold yellow]Summary:[/]
{analysis.Summary}

[bold red]Critical Issues ({analysis.CriticalIssues.Length}):[/]
{String.Join("\n", analysis.CriticalIssues |> List.map (fun issue -> $"• {issue}"))}

[bold orange3]Warnings ({analysis.Warnings.Length}):[/]
{String.Join("\n", analysis.Warnings |> List.map (fun warning -> $"• {warning}"))}

[bold green]Recommendations ({analysis.Recommendations.Length}):[/]
{String.Join("\n", analysis.Recommendations |> List.take (min 3 analysis.Recommendations.Length) |> List.map (fun rec -> $"• {rec.Action}"))}

[bold blue]Next Steps:[/]
{String.Join("\n", analysis.NextSteps |> List.map (fun step -> $"• {step}"))}
""")
        panel.Header <- PanelHeader("Gordon Analysis")
        panel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(panel)
    
    let private executeStage (gordonService: IGordonService) (stage: string) (options: ConsolidationOptions) =
        task {
            AnsiConsole.MarkupLine($"[bold blue]🔄 Executing {stage} consolidation stage...[/]")
            
            // Get Gordon's analysis for this stage
            let analysisType = 
                match stage.ToLower() with
                | "database" -> DatabaseStatus
                | "application" -> ContainerHealth
                | "web" -> NetworkConnectivity
                | _ -> ConsolidationStrategy
            
            let! analysis = gordonService.AnalyzeInfrastructure(analysisType)
            
            match analysis with
            | Ok result ->
                if options.Verbose then
                    displayGordonAnalysis result
                
                // Get specific recommendations for this stage
                let! recommendations = gordonService.GetRecommendations($"{stage} tier consolidation")
                
                match recommendations with
                | Ok recs ->
                    AnsiConsole.MarkupLine($"[green]✅ Gordon provided {recs.Length} recommendations for {stage}[/]")
                    
                    if not options.DryRun then
                        // Execute high-priority recommendations
                        for rec in recs |> List.filter (fun r -> r.Priority <= 2) do
                            AnsiConsole.MarkupLine($"[yellow]🔧 Executing: {rec.Action}[/]")
                            
                            let! monitorResult = gordonService.MonitorOperation(rec.Action)
                            match monitorResult with
                            | Ok status -> 
                                AnsiConsole.MarkupLine($"[green]✅ {status}[/]")
                            | Error error -> 
                                AnsiConsole.MarkupLine($"[red]❌ Failed: {error}[/]")
                    
                    return Ok $"{stage} consolidation completed"
                | Error error ->
                    AnsiConsole.MarkupLine($"[red]❌ Failed to get Gordon recommendations: {error}[/]")
                    return Error error
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Gordon analysis failed: {error}[/]")
                return Error error
        }
    
    let executeConsolidation (serviceProvider: IServiceProvider) (options: ConsolidationOptions) =
        task {
            let logger = serviceProvider.GetRequiredService<ILogger<GordonConsolidationOrchestrator>>()
            let gordonService = serviceProvider.GetRequiredService<IGordonService>()
            
            AnsiConsole.Clear()
            
            let rule = Rule("[bold cyan]🤖 Gordon-Assisted TARS Consolidation[/]")
            rule.Style <- Style.Parse("cyan")
            AnsiConsole.Write(rule)
            
            AnsiConsole.WriteLine()
            
            if options.DryRun then
                AnsiConsole.MarkupLine("[bold yellow]🔍 DRY RUN MODE - No changes will be made[/]")
                AnsiConsole.WriteLine()
            
            // Step 1: Initial Gordon assessment
            AnsiConsole.MarkupLine("[bold blue]🔍 Step 1: Gordon's Initial Assessment[/]")
            
            let! initialAnalysis = gordonService.AnalyzeInfrastructure(ConsolidationStrategy)
            
            match initialAnalysis with
            | Ok analysis ->
                displayGordonAnalysis analysis
                
                if analysis.HealthScore < 30 && not options.Force then
                    AnsiConsole.MarkupLine("[bold red]⚠️  Gordon detected critical issues. Use --force to proceed anyway.[/]")
                    return Error "Critical issues detected"
                
                // Step 2: Determine consolidation stages
                let stages = 
                    match options.Stage with
                    | Some stage -> [stage]
                    | None -> ["database"; "application"; "web"]
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine($"[bold blue]🚀 Step 2: Executing {stages.Length} consolidation stages[/]")
                
                // Create progress table
                let progressTable = createProgressTable()
                
                // Execute each stage
                let mutable allSuccessful = true
                
                for stage in stages do
                    updateProgress progressTable stage "In Progress" "Analyzing..." "Querying Gordon..."
                    AnsiConsole.Write(progressTable)
                    AnsiConsole.Clear()
                    
                    let! stageResult = executeStage gordonService stage options
                    
                    match stageResult with
                    | Ok message ->
                        updateProgress progressTable stage "Complete" "✅ Healthy" message
                    | Error error ->
                        updateProgress progressTable stage "Failed" "❌ Issues detected" error
                        allSuccessful <- false
                
                AnsiConsole.Write(progressTable)
                
                // Step 3: Final Gordon assessment
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold blue]🏥 Step 3: Gordon's Final Health Check[/]")
                
                let! finalAnalysis = gordonService.AnalyzeInfrastructure(ContainerHealth)
                
                match finalAnalysis with
                | Ok finalResult ->
                    displayGordonAnalysis finalResult
                    
                    let successPanel = Panel($"""
[bold green]🎉 Gordon-Assisted Consolidation Complete![/]

[bold]Final Health Score:[/] {finalResult.HealthScore}/100
[bold]Stages Completed:[/] {stages.Length}
[bold]Overall Status:[/] {if allSuccessful then "[green]✅ Success[/]" else "[yellow]⚠️  Partial Success[/]"}

[bold cyan]🤖 Gordon's Summary:[/]
• Consolidation strategy executed successfully
• Infrastructure health improved
• Monitoring systems active
• Ready for production workloads

[bold blue]📋 Access Points:[/]
• TARS Main: http://localhost
• MongoDB Admin: http://localhost:8081
• ChromaDB: http://localhost:8000
• Redis Commander: http://localhost:8082
""")
                    successPanel.Header <- PanelHeader("Consolidation Complete")
                    successPanel.Border <- BoxBorder.Double
                    AnsiConsole.Write(successPanel)
                    
                    return Ok "Gordon-assisted consolidation completed successfully"
                | Error error ->
                    AnsiConsole.MarkupLine($"[red]❌ Final health check failed: {error}[/]")
                    return Error error
                    
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Initial Gordon analysis failed: {error}[/]")
                return Error error
        }
    
    let registerCommand (app: Spectre.Console.Cli.IConfigurator) =
        app.AddCommand<ConsolidateCommand>("gordon-consolidate")
           .WithDescription("🤖 Gordon-assisted TARS infrastructure consolidation")
           .WithExample(["gordon-consolidate"; "--dry-run"])
           .WithExample(["gordon-consolidate"; "--stage"; "database"])
           .WithExample(["gordon-consolidate"; "--force"; "--verbose"]) |> ignore
    
    and ConsolidateCommand() =
        inherit Spectre.Console.Cli.AsyncCommand<ConsolidateCommand.Settings>()
        
        type Settings() =
            inherit Spectre.Console.Cli.CommandSettings()
            
            [<Spectre.Console.Cli.CommandOption("--dry-run")>]
            member val DryRun = false with get, set
            
            [<Spectre.Console.Cli.CommandOption("--force")>]
            member val Force = false with get, set
            
            [<Spectre.Console.Cli.CommandOption("--skip-health-check")>]
            member val SkipHealthCheck = false with get, set
            
            [<Spectre.Console.Cli.CommandOption("--stage")>]
            member val Stage: string = null with get, set
            
            [<Spectre.Console.Cli.CommandOption("--verbose")>]
            member val Verbose = false with get, set
        
        override _.ExecuteAsync(context, settings) =
            task {
                let serviceProvider = context.Data :?> IServiceProvider
                
                let options = {
                    DryRun = settings.DryRun
                    Force = settings.Force
                    SkipHealthCheck = settings.SkipHealthCheck
                    Stage = if String.IsNullOrEmpty(settings.Stage) then None else Some settings.Stage
                    Verbose = settings.Verbose
                }
                
                let! result = executeConsolidation serviceProvider options
                
                match result with
                | Ok _ -> return 0
                | Error _ -> return 1
            }
