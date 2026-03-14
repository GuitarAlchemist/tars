namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Diagnostics.TarsRealDiagnostics
open TarsEngine.FSharp.Cli.UI.TarsElmishDiagnostics
open TarsEngine.FSharp.Cli.UI.ElmishRuntime

/// Enhanced Diagnostics Module - Helper functions
module EnhancedDiagnosticsModule =

    /// Display comprehensive diagnostics in console format
    let displayConsoleReport (diagnostics: ComprehensiveDiagnostics) (logger: ILogger) =
        logger.LogInformation("╔══════════════════════════════════════════════════════════════╗")
        logger.LogInformation("║                 TARS ENHANCED DIAGNOSTICS                   ║")
        logger.LogInformation("╚══════════════════════════════════════════════════════════════╝")
        logger.LogInformation("")

        // Overall Health
        let healthEmoji =
            if diagnostics.OverallHealthScore > 90.0 then "🟢"
            elif diagnostics.OverallHealthScore > 75.0 then "🟡"
            else "🔴"

        logger.LogInformation("🏥 OVERALL SYSTEM HEALTH: {HealthEmoji} {Health:F1}%", healthEmoji, diagnostics.OverallHealthScore)
        logger.LogInformation("📅 Timestamp: {Timestamp}", diagnostics.Timestamp.ToString("yyyy-MM-dd HH:mm:ss UTC"))
        logger.LogInformation("")

        // Git Health
        logger.LogInformation("📁 GIT REPOSITORY:")
        logger.LogInformation("   Repository: {IsRepo}", if diagnostics.GitHealth.IsRepository then "✅" else "❌")
        logger.LogInformation("   Clean: {IsClean}", if diagnostics.GitHealth.IsClean then "✅" else "❌")
        logger.LogInformation("   Commits: {Commits}", diagnostics.GitHealth.Commits)
        logger.LogInformation("")

        // GPU Information
        logger.LogInformation("🎮 GPU INFORMATION:")
        for gpu in diagnostics.GpuInfo do
            logger.LogInformation("   GPU: {Name}", gpu.Name)
            logger.LogInformation("   CUDA: {Cuda}", if gpu.CudaSupported then "✅" else "❌")
            logger.LogInformation("   Memory: {Memory} MB", gpu.MemoryTotal / (1024L * 1024L))
        logger.LogInformation("")

        // Network Diagnostics
        logger.LogInformation("🌐 NETWORK:")
        logger.LogInformation("   Connected: {Connected}", if diagnostics.NetworkDiagnostics.IsConnected then "✅" else "❌")
        logger.LogInformation("   DNS Resolution: {DNS}ms", diagnostics.NetworkDiagnostics.DnsResolutionTime)
        logger.LogInformation("")

    /// Display real-time update in compact format
    let displayRealTimeUpdate (diagnostics: ComprehensiveDiagnostics) (logger: ILogger) =
        let timestamp = diagnostics.Timestamp.ToString("HH:mm:ss")
        let healthEmoji =
            if diagnostics.OverallHealthScore > 90.0 then "🟢"
            elif diagnostics.OverallHealthScore > 75.0 then "🟡"
            else "🔴"

        logger.LogInformation("[{Timestamp}] {HealthEmoji} Health: {Health:F1}%",
            timestamp, healthEmoji, diagnostics.OverallHealthScore)

    /// Generate HTML report
    // FUNCTIONAL ELMISH TARS DIAGNOSTICS - Real Interactive UI
    let generateFunctionalElmishReport () =
        let (initialHtml, updateFunction, initialModel) = TarsEngine.FSharp.Cli.UI.ElmishRuntime.runElmishProgram ()
        initialHtml

    let generateHtmlReport (diagnostics: ComprehensiveDiagnostics) =
        let timestamp = diagnostics.Timestamp.ToString("yyyy-MM-dd HH:mm:ss UTC")
        let healthScore = diagnostics.OverallHealthScore.ToString("F1")

        let componentsHtml = "<div>No component analysis available</div>" // Simplified for now

        sprintf """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARS Enhanced Diagnostics Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .health-score { font-size: 3em; font-weight: bold; text-align: center; margin: 20px 0; }
        .section { padding: 20px; border-bottom: 1px solid #eee; }
        .component { background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric { display: inline-block; margin: 5px 10px; padding: 5px 10px; background: #e9ecef; border-radius: 3px; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 TARS Enhanced Diagnostics Report</h1>
            <p>Generated: %s</p>
        </div>

        <div class="health-score">
            Overall Health: %s%%
        </div>

        <div class="section">
            <h2>💻 System Resources</h2>
            <div class="metric">System Health Available</div>
        </div>

        <div class="section">
            <h2>🔍 Component Analysis</h2>
            %s
        </div>
    </div>
</body>
</html>""" timestamp healthScore componentsHtml

    /// Format uptime in human-readable format
    let formatUptime (uptime: TimeSpan) =
        if uptime.TotalDays >= 1.0 then
            sprintf "%dd %dh %dm" uptime.Days uptime.Hours uptime.Minutes
        elif uptime.TotalHours >= 1.0 then
            sprintf "%dh %dm" uptime.Hours uptime.Minutes
        else
            sprintf "%dm %ds" uptime.Minutes uptime.Seconds

/// Enhanced Diagnostics Command - Real-time system monitoring with NO FAKE DATA
type EnhancedDiagnosticsCommand(logger: ILogger<EnhancedDiagnosticsCommand>) =
    interface ICommand with
        member _.Name = "enhanced-diagnostics"
        member _.Description = "Run enhanced real-time TARS diagnostics with live monitoring and visualizations"
        member _.Usage = "tars enhanced-diagnostics [options]"
        member _.Examples = [
            "tars enhanced-diagnostics"
            "tars enhanced-diagnostics --format yaml"
            "tars enhanced-diagnostics --real-time false"
            "tars enhanced-diagnostics --interval 10"
        ]
        member _.ValidateOptions(_) = true

        member _.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    logger.LogInformation("🚀 Starting Enhanced TARS Diagnostics...")

                    let repositoryPath =
                        match options.Options.TryFind("repository-path") with
                        | Some path -> path
                        | None -> Environment.CurrentDirectory

                    let realTime =
                        match options.Options.TryFind("real-time") with
                        | Some value -> Boolean.Parse(value)
                        | None -> true

                    let outputFormat =
                        match options.Options.TryFind("format") with
                        | Some format -> format
                        | None -> "console"

                    let interval =
                        match options.Options.TryFind("interval") with
                        | Some value ->
                            match Int32.TryParse(value) with
                            | true, i -> i
                            | false, _ -> 5
                        | None -> 5
                    
                    logger.LogInformation("📊 Configuration:")
                    logger.LogInformation("   Repository: {RepositoryPath}", repositoryPath)
                    logger.LogInformation("   Real-time: {RealTime}", realTime)
                    logger.LogInformation("   Format: {Format}", outputFormat)
                    logger.LogInformation("   Interval: {Interval}s", interval)
                    logger.LogInformation("")
                    
                    // Get comprehensive diagnostics
                    logger.LogInformation("🔍 Gathering comprehensive system diagnostics...")
                    let! diagnostics = getComprehensiveDiagnostics repositoryPath
                    
                    // Display results based on format
                    match outputFormat.ToLower() with
                    | "json" ->
                        let json = System.Text.Json.JsonSerializer.Serialize(diagnostics, System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                        Console.WriteLine(json)
                        
                    | "yaml" ->
                        // Simple YAML-like output
                        Console.WriteLine("# TARS Enhanced Diagnostics Report")
                        Console.WriteLine(sprintf "timestamp: %s" (diagnostics.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")))
                        Console.WriteLine(sprintf "overall_health_score: %.1f" diagnostics.OverallHealthScore)
                        Console.WriteLine("")
                        Console.WriteLine("git_health:")
                        Console.WriteLine(sprintf "  is_repository: %b" diagnostics.GitHealth.IsRepository)
                        Console.WriteLine(sprintf "  is_clean: %b" diagnostics.GitHealth.IsClean)
                        Console.WriteLine(sprintf "  commits: %d" diagnostics.GitHealth.Commits)
                        Console.WriteLine("")
                        Console.WriteLine("gpu_info:")
                        for gpu in diagnostics.GpuInfo do
                            Console.WriteLine(sprintf "  - name: %s" gpu.Name)
                            Console.WriteLine(sprintf "    cuda_supported: %b" gpu.CudaSupported)
                            Console.WriteLine(sprintf "    memory_total: %d" gpu.MemoryTotal)
                        
                    | "html" ->
                        // Generate Functional Elmish HTML report
                        let html = EnhancedDiagnosticsModule.generateFunctionalElmishReport ()
                        let htmlFile = sprintf "tars-elmish-diagnostics-%s.html" (DateTime.Now.ToString("yyyyMMdd-HHmmss"))
                        do! System.IO.File.WriteAllTextAsync(htmlFile, html)
                        logger.LogInformation("🧠 Functional Elmish TARS report saved to: {HtmlFile}", htmlFile)
                        logger.LogInformation("🚀 Open the file in your browser for interactive TARS diagnostics!")

                        // Try to open in browser automatically
                        try
                            let fullPath = System.IO.Path.GetFullPath(htmlFile)
                            let psi = System.Diagnostics.ProcessStartInfo(fullPath)
                            psi.UseShellExecute <- true
                            System.Diagnostics.Process.Start(psi) |> ignore
                            logger.LogInformation("🌐 Opened in default browser")
                        with
                        | ex -> logger.LogWarning("⚠️ Could not auto-open browser: {Error}", ex.Message)
                        
                    | _ -> // Default console output
                        EnhancedDiagnosticsModule.displayConsoleReport diagnostics logger
                    
                    // Start real-time monitoring if requested
                    if realTime then
                        logger.LogInformation("")
                        logger.LogInformation("🔄 Starting real-time monitoring (Press Ctrl+C to stop)...")
                        logger.LogInformation("📡 Updates every {Interval} seconds", interval)
                        logger.LogInformation("")
                        
                        let mutable running = true
                        Console.CancelKeyPress.Add(fun _ -> 
                            running <- false
                            logger.LogInformation("🛑 Stopping real-time monitoring...")
                        )
                        
                        while running do
                            try
                                do! Task.Delay(interval * 1000)
                                if running then
                                    let! newDiagnostics = getComprehensiveDiagnostics repositoryPath
                                    EnhancedDiagnosticsModule.displayRealTimeUpdate newDiagnostics logger
                            with
                            | ex -> 
                                logger.LogError(ex, "❌ Error during real-time update")
                                do! Task.Delay(1000) // Wait before retrying
                    
                    return { Success = true; Message = "Enhanced diagnostics completed successfully"; ExitCode = 0 }

                with
                | ex ->
                    logger.LogError(ex, "❌ Enhanced diagnostics failed")
                    return { Success = false; Message = sprintf "Enhanced diagnostics failed: %s" ex.Message; ExitCode = 1 }
            }


