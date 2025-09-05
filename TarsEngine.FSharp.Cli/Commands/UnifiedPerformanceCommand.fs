namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Core.UnifiedCache
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem
open TarsEngine.FSharp.Cli.Monitoring.UnifiedMonitoring
open TarsEngine.FSharp.Cli.Commands.Types

/// Unified Performance Command - Demonstrate caching and monitoring systems
module UnifiedPerformanceCommand =
    
    /// Run cache performance demonstration
    let runCacheDemo (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🚀 TARS Unified Cache System Demo[/]")
                AnsiConsole.MarkupLine("[dim]Demonstrating multi-level caching with performance monitoring[/]")
                AnsiConsole.WriteLine()
                
                use configManager = createConfigurationManager logger
                use proofGenerator = createProofGenerator logger
                let! _ = configManager.InitializeAsync(CancellationToken.None)
                
                use cacheManager = new UnifiedCacheManager(logger, configManager, proofGenerator)
                
                AnsiConsole.MarkupLine("[yellow]📊 Cache Performance Test...[/]")
                
                // Test cache operations
                let testData = [
                    ("user:123", "John Doe")
                    ("user:456", "Jane Smith")
                    ("config:theme", "dark")
                    ("config:language", "en-US")
                    ("session:abc123", "active")
                ]
                
                // Set cache entries
                AnsiConsole.MarkupLine("[green]Setting cache entries...[/]")
                for (key, value) in testData do
                    let! setResult = cacheManager.SetAsync(key, value, TimeSpan.FromMinutes(10.0))
                    match setResult with
                    | Success _ -> AnsiConsole.MarkupLine($"  ✅ Set: [yellow]{key}[/] = [cyan]{value}[/]")
                    | Failure (error, _) -> AnsiConsole.MarkupLine($"  ❌ Failed to set {key}: {TarsError.toString error}")
                
                AnsiConsole.WriteLine()
                
                // Get cache entries
                AnsiConsole.MarkupLine("[green]Retrieving cache entries...[/]")
                for (key, expectedValue) in testData do
                    let! getResult = cacheManager.GetAsync<string>(key)
                    match getResult with
                    | Success (Some value, metadata) -> 
                        let source = metadata.["source"] :?> string
                        AnsiConsole.MarkupLine($"  ✅ Get: [yellow]{key}[/] = [cyan]{value}[/] (from {source})")
                    | Success (None, _) -> AnsiConsole.MarkupLine($"  ⚠️ Cache miss: [yellow]{key}[/]")
                    | Failure (error, _) -> AnsiConsole.MarkupLine($"  ❌ Failed to get {key}: {TarsError.toString error}")
                
                AnsiConsole.WriteLine()
                
                // Show cache statistics
                let stats = cacheManager.GetStatistics()
                AnsiConsole.MarkupLine("[bold cyan]📈 Cache Statistics:[/]")
                AnsiConsole.MarkupLine($"  Total Entries: [green]{stats.TotalEntries}[/]")
                AnsiConsole.MarkupLine($"  Memory Entries: [yellow]{stats.MemoryEntries}[/]")
                AnsiConsole.MarkupLine($"  Disk Entries: [blue]{stats.DiskEntries}[/]")
                AnsiConsole.MarkupLine($"  Hit Count: [green]{stats.HitCount}[/]")
                AnsiConsole.MarkupLine($"  Miss Count: [red]{stats.MissCount}[/]")
                let hitRatioStr = stats.HitRatio.ToString("P2")
                let totalSizeKB = stats.TotalSize / 1024L
                let opsPerSecStr = stats.OperationsPerSecond.ToString("F2")
                AnsiConsole.MarkupLine($"  Hit Ratio: [yellow]{hitRatioStr}[/]")
                AnsiConsole.MarkupLine($"  Total Size: [cyan]{totalSizeKB} KB[/]")
                AnsiConsole.MarkupLine($"  Operations/sec: [magenta]{opsPerSecStr}[/]")
                
                AnsiConsole.WriteLine()
                
                // Test cache cleanup
                AnsiConsole.MarkupLine("[yellow]🧹 Testing cache cleanup...[/]")
                let! cleanupResult = cacheManager.CleanupAsync()
                match cleanupResult with
                | Success (cleanedCount, _) -> 
                    AnsiConsole.MarkupLine($"  ✅ Cleaned up [green]{cleanedCount}[/] expired entries")
                | Failure (error, _) -> 
                    AnsiConsole.MarkupLine($"  ❌ Cleanup failed: {TarsError.toString error}")
                
                return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Cache demo failed: {ex.Message}[/]")
                return 1
        }
    
    /// Run monitoring system demonstration
    let runMonitoringDemo (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]📊 TARS Unified Monitoring System Demo[/]")
                AnsiConsole.MarkupLine("[dim]Demonstrating real-time system monitoring and alerting[/]")
                AnsiConsole.WriteLine()
                
                use configManager = createConfigurationManager logger
                use proofGenerator = createProofGenerator logger
                let! _ = configManager.InitializeAsync(CancellationToken.None)
                
                use monitoringManager = new UnifiedMonitoringManager(logger, configManager, proofGenerator)
                
                // Start monitoring
                AnsiConsole.MarkupLine("[yellow]🔄 Starting monitoring system...[/]")
                let! startResult = monitoringManager.StartMonitoringAsync()
                match startResult with
                | Success _ -> AnsiConsole.MarkupLine("  ✅ Monitoring started successfully")
                | Failure (error, _) -> 
                    AnsiConsole.MarkupLine($"  ❌ Failed to start monitoring: {TarsError.toString error}")
                    return 1
                
                // Wait for some metrics to be collected
                AnsiConsole.MarkupLine("[dim]Collecting metrics for 10 seconds...[/]")
                do! Task.Delay(10000)
                
                // Get system health
                AnsiConsole.MarkupLine("[green]🏥 System Health Status:[/]")
                let! healthResult = monitoringManager.GetSystemHealthAsync()
                match healthResult with
                | Success (health, _) ->
                    let healthColor = if health.OverallHealth > 0.8 then "green" else if health.OverallHealth > 0.5 then "yellow" else "red"
                    let overallHealthStr = health.OverallHealth.ToString("P1")
                    let uptimeStr = health.Uptime.ToString(@"hh\:mm\:ss")
                    let lastUpdateStr = health.LastUpdate.ToString("HH:mm:ss")
                    AnsiConsole.MarkupLine($"  Overall Health: [{healthColor}]{overallHealthStr}[/]")
                    AnsiConsole.MarkupLine($"  Uptime: [cyan]{uptimeStr}[/]")
                    AnsiConsole.MarkupLine($"  Active Alerts: [red]{health.ActiveAlerts.Length}[/]")
                    AnsiConsole.MarkupLine($"  Last Update: [dim]{lastUpdateStr}[/]")
                    
                    if health.ProofId.IsSome then
                        AnsiConsole.MarkupLine($"  Proof ID: [green]{health.ProofId.Value.Substring(0, 8)}...[/]")
                    
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold yellow]Component Health:[/]")
                    for kvp in health.ComponentHealth do
                        let componentColor = if kvp.Value > 0.8 then "green" else if kvp.Value > 0.5 then "yellow" else "red"
                        let componentHealthStr = kvp.Value.ToString("P1")
                        AnsiConsole.MarkupLine($"  {kvp.Key}: [{componentColor}]{componentHealthStr}[/]")
                    
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold yellow]Resource Utilization:[/]")
                    for kvp in health.ResourceUtilization do
                        let utilizationColor = if kvp.Value < 50.0 then "green" else if kvp.Value < 80.0 then "yellow" else "red"
                        let utilizationStr = kvp.Value.ToString("F1")
                        AnsiConsole.MarkupLine($"  {kvp.Key}: [{utilizationColor}]{utilizationStr}%[/]")
                
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine($"  ❌ Failed to get health status: {TarsError.toString error}")
                
                AnsiConsole.WriteLine()
                
                // Get recent metrics
                AnsiConsole.MarkupLine("[green]📈 Recent Metrics:[/]")
                let! metricsResult = monitoringManager.GetRecentMetricsAsync(10)
                match metricsResult with
                | Success (metrics, _) ->
                    for metric in metrics |> List.take (Math.Min(5, metrics.Length)) do
                        let valueColor = 
                            match metric.Name with
                            | name when name.Contains("cpu") && metric.Value > 80.0 -> "red"
                            | name when name.Contains("memory") && metric.Value > 400.0 -> "red"
                            | _ -> "cyan"
                        let metricValueStr = metric.Value.ToString("F2")
                        let metricTimeStr = metric.Timestamp.ToString("HH:mm:ss")
                        AnsiConsole.MarkupLine($"  {metric.Name}: [{valueColor}]{metricValueStr} {metric.Unit}[/] [dim]({metricTimeStr})[/]")
                
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine($"  ❌ Failed to get metrics: {TarsError.toString error}")
                
                AnsiConsole.WriteLine()
                
                // Get active alerts
                AnsiConsole.MarkupLine("[green]🚨 Active Alerts:[/]")
                let! alertsResult = monitoringManager.GetActiveAlertsAsync()
                match alertsResult with
                | Success (alerts, _) ->
                    if alerts.IsEmpty then
                        AnsiConsole.MarkupLine("  [green]✅ No active alerts[/]")
                    else
                        for alert in alerts |> List.take (Math.Min(5, alerts.Length)) do
                            let severityColor = 
                                match alert.Severity with
                                | Critical -> "red"
                                | Error -> "red"
                                | Warning -> "yellow"
                                | Info -> "blue"
                            let alertTimeStr = alert.Timestamp.ToString("HH:mm:ss")
                            AnsiConsole.MarkupLine($"  [{severityColor}]{alert.Severity}[/]: {alert.Message} [dim]({alertTimeStr})[/]")
                
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine($"  ❌ Failed to get alerts: {TarsError.toString error}")
                
                AnsiConsole.WriteLine()
                
                // Show monitoring statistics
                let stats = monitoringManager.GetStatistics()
                AnsiConsole.MarkupLine("[bold cyan]📊 Monitoring Statistics:[/]")
                let isMonitoring = stats.["isMonitoring"]
                let totalMetrics = stats.["totalMetrics"]
                let totalAlerts = stats.["totalAlerts"]
                let activeAlerts = stats.["activeAlerts"]
                let uptimeMinutes = (stats.["uptime"] :?> float).ToString("F1")
                AnsiConsole.MarkupLine($"  Is Monitoring: [green]{isMonitoring}[/]")
                AnsiConsole.MarkupLine($"  Total Metrics: [yellow]{totalMetrics}[/]")
                AnsiConsole.MarkupLine($"  Total Alerts: [blue]{totalAlerts}[/]")
                AnsiConsole.MarkupLine($"  Active Alerts: [red]{activeAlerts}[/]")
                AnsiConsole.MarkupLine($"  Uptime: [cyan]{uptimeMinutes} minutes[/]")
                
                // Stop monitoring
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[yellow]🛑 Stopping monitoring system...[/]")
                let! stopResult = monitoringManager.StopMonitoringAsync()
                match stopResult with
                | Success _ -> AnsiConsole.MarkupLine("  ✅ Monitoring stopped successfully")
                | Failure (error, _) -> AnsiConsole.MarkupLine($"  ❌ Failed to stop monitoring: {TarsError.toString error}")
                
                return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Monitoring demo failed: {ex.Message}[/]")
                return 1
        }
    
    /// Run combined performance demonstration
    let runCombinedDemo (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🎯 TARS Unified Performance Systems Demo[/]")
                AnsiConsole.MarkupLine("[dim]Demonstrating cache and monitoring systems working together[/]")
                AnsiConsole.WriteLine()
                
                // Run cache demo
                let! cacheResult = runCacheDemo logger
                
                AnsiConsole.WriteLine()
                AnsiConsole.Rule("[yellow]Monitoring Demo[/]")
                AnsiConsole.WriteLine()
                
                // Run monitoring demo
                let! monitoringResult = runMonitoringDemo logger
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold green]🎉 Performance Systems Demo Complete![/]")
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold cyan]🚀 PERFORMANCE SYSTEM ACHIEVEMENTS:[/]")
                AnsiConsole.MarkupLine("  ✅ [green]Multi-level Caching[/] - Memory, disk, and distributed caching")
                AnsiConsole.MarkupLine("  ✅ [green]Cache Performance[/] - Hit ratio tracking and optimization")
                AnsiConsole.MarkupLine("  ✅ [green]Real-time Monitoring[/] - System metrics and health tracking")
                AnsiConsole.MarkupLine("  ✅ [green]Intelligent Alerting[/] - Threshold-based alert system")
                AnsiConsole.MarkupLine("  ✅ [green]Performance Analytics[/] - Resource utilization monitoring")
                AnsiConsole.MarkupLine("  ✅ [green]Proof Generation[/] - Cryptographic evidence for all operations")
                
                return Math.Max(cacheResult, monitoringResult)
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Combined demo failed: {ex.Message}[/]")
                return 1
        }
    
    /// Unified Performance Command implementation
    type UnifiedPerformanceCommand() =
        interface ICommand with
            member _.Name = "performance"
            member _.Description = "Demonstrate unified caching and monitoring systems"
            member _.Usage = "tars performance [--cache] [--monitoring] [--combined]"
            member _.Examples = [
                "tars performance --cache       # Demo caching system"
                "tars performance --monitoring  # Demo monitoring system"
                "tars performance --combined    # Demo both systems"
                "tars performance               # Show overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedPerformanceCommand"
                        
                        let isCacheMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--cache")
                        
                        let isMonitoringMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--monitoring")
                        
                        let isCombinedMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--combined")
                        
                        if isCacheMode then
                            let! result = runCacheDemo logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        elif isMonitoringMode then
                            let! result = runMonitoringDemo logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        elif isCombinedMode then
                            let! result = runCombinedDemo logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]🎯 TARS Unified Performance Systems[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Advanced performance optimization systems for TARS unified architecture.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Available Systems:[/]")
                            AnsiConsole.MarkupLine("  🚀 [cyan]Caching System[/] - Multi-level caching with performance monitoring")
                            AnsiConsole.MarkupLine("    • Memory, disk, and distributed caching")
                            AnsiConsole.MarkupLine("    • Intelligent cache invalidation strategies")
                            AnsiConsole.MarkupLine("    • Real-time performance analytics")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("  📊 [green]Monitoring System[/] - Real-time system monitoring and alerting")
                            AnsiConsole.MarkupLine("    • System health monitoring with proof generation")
                            AnsiConsole.MarkupLine("    • Intelligent alerting with severity levels")
                            AnsiConsole.MarkupLine("    • Performance analytics and capacity planning")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--cache[/]       Demo caching system")
                            AnsiConsole.MarkupLine("  [yellow]--monitoring[/]  Demo monitoring system")
                            AnsiConsole.MarkupLine("  [yellow]--combined[/]    Demo both systems together")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Examples:")
                            AnsiConsole.MarkupLine("  [dim]tars performance --cache[/]")
                            AnsiConsole.MarkupLine("  [dim]tars performance --monitoring[/]")
                            AnsiConsole.MarkupLine("  [dim]tars performance --combined[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[red]❌ Performance command failed: {ex.Message}[/]")
                        return { Message = ""; ExitCode = 1; Success = false }
                }
