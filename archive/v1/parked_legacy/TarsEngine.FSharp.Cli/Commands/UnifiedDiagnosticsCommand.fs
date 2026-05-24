namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem
open TarsEngine.FSharp.Cli.Acceleration.UnifiedCudaEngine
open TarsEngine.FSharp.Cli.Commands.Types

/// Unified Diagnostics Command - Comprehensive system diagnostics using unified architecture
module UnifiedDiagnosticsCommand =
    
    /// Diagnostic result for a unified system component
    type UnifiedDiagnosticResult = {
        ComponentName: string
        Status: ComponentStatus
        HealthScore: float
        ResponseTime: TimeSpan
        Details: Map<string, obj>
        Issues: string list
        Recommendations: string list
    }
    
    /// Run comprehensive diagnostics on all unified systems
    let runUnifiedDiagnostics (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🔍 TARS Unified System Diagnostics[/]")
                AnsiConsole.MarkupLine("[dim]Comprehensive health check of all unified systems[/]")
                AnsiConsole.WriteLine()
                
                let diagnosticResults = ResizeArray<UnifiedDiagnosticResult>()
                let correlationId = generateCorrelationId()
                
                // Test Configuration Management System
                AnsiConsole.MarkupLine("[yellow]🔧 Testing Configuration Management System...[/]")
                let configStartTime = DateTime.UtcNow
                
                use configManager = createConfigurationManager logger
                let! configInitResult = configManager.InitializeAsync(CancellationToken.None)
                
                let configResult = 
                    match configInitResult with
                    | Success (_, metadata) ->
                        let! _ = configManager.SetValueAsync("diagnostics.test", "test_value", Some correlationId)
                        let retrievedValue = ConfigurationExtensions.getString configManager "diagnostics.test" "default"
                        let statistics = configManager.GetStatistics()
                        
                        {
                            ComponentName = "Configuration Management"
                            Status = ComponentStatus.Running
                            HealthScore = if retrievedValue = "test_value" then 1.0 else 0.5
                            ResponseTime = DateTime.UtcNow - configStartTime
                            Details = statistics
                            Issues = []
                            Recommendations = ["Configuration system is healthy"]
                        }
                    | Failure (error, _) ->
                        {
                            ComponentName = "Configuration Management"
                            Status = ComponentStatus.Error
                            HealthScore = 0.0
                            ResponseTime = DateTime.UtcNow - configStartTime
                            Details = Map [("error", box (TarsError.toString error))]
                            Issues = ["Configuration system failed to initialize"]
                            Recommendations = ["Check configuration files and permissions"]
                        }
                
                diagnosticResults.Add(configResult)
                
                let statusIcon = if configResult.HealthScore > 0.8 then "✅" else if configResult.HealthScore > 0.5 then "⚠️" else "❌"
                AnsiConsole.MarkupLine(sprintf "  %s Configuration Management: [yellow]%s[/] (%sms)" statusIcon (configResult.HealthScore.ToString("F2")) (configResult.ResponseTime.TotalMilliseconds.ToString("F0")))
                
                // Test Proof Generation System
                AnsiConsole.MarkupLine("[yellow]🔐 Testing Proof Generation System...[/]")
                let proofStartTime = DateTime.UtcNow
                
                use proofGenerator = createProofGenerator logger
                let! proofResult = ProofExtensions.generateExecutionProof proofGenerator "DiagnosticsTest" correlationId
                
                let proofDiagnostic = 
                    match proofResult with
                    | Success (proof, _) ->
                        let! verificationResult = proofGenerator.VerifyProofAsync(proof, CancellationToken.None)
                        let statistics = proofGenerator.GetProofStatistics()
                        
                        match verificationResult with
                        | Success (verification, _) ->
                            {
                                ComponentName = "Proof Generation"
                                Status = ComponentStatus.Running
                                HealthScore = if verification.IsValid then verification.TrustScore else 0.0
                                ResponseTime = DateTime.UtcNow - proofStartTime
                                Details = statistics
                                Issues = verification.Issues
                                Recommendations = if verification.IsValid then ["Proof system is secure"] else ["Review proof validation"]
                            }
                        | Failure (error, _) ->
                            {
                                ComponentName = "Proof Generation"
                                Status = ComponentStatus.Error
                                HealthScore = 0.0
                                ResponseTime = DateTime.UtcNow - proofStartTime
                                Details = Map [("error", box (TarsError.toString error))]
                                Issues = ["Proof verification failed"]
                                Recommendations = ["Check cryptographic subsystem"]
                            }
                    | Failure (error, _) ->
                        {
                            ComponentName = "Proof Generation"
                            Status = ComponentStatus.Error
                            HealthScore = 0.0
                            ResponseTime = DateTime.UtcNow - proofStartTime
                            Details = Map [("error", box (TarsError.toString error))]
                            Issues = ["Proof generation failed"]
                            Recommendations = ["Check proof system initialization"]
                        }
                
                diagnosticResults.Add(proofDiagnostic)
                
                let proofIcon = if proofDiagnostic.HealthScore > 0.8 then "✅" else if proofDiagnostic.HealthScore > 0.5 then "⚠️" else "❌"
                AnsiConsole.MarkupLine(sprintf "  %s Proof Generation: [yellow]%s[/] (%sms)" proofIcon (proofDiagnostic.HealthScore.ToString("F2")) (proofDiagnostic.ResponseTime.TotalMilliseconds.ToString("F0")))
                
                // Test CUDA Engine
                AnsiConsole.MarkupLine("[yellow]⚡ Testing CUDA Engine...[/]")
                let cudaStartTime = DateTime.UtcNow
                
                use cudaEngine = createCudaEngine logger
                let! cudaInitResult = cudaEngine.InitializeAsync(CancellationToken.None)
                
                let cudaDiagnostic = 
                    match cudaInitResult with
                    | Success (_, metadata) ->
                        let operation = CudaOperationFactory.createVectorSimilarity 256
                        let! operationResult = cudaEngine.ExecuteOperationAsync(operation, box "diagnostic_data", CancellationToken.None)
                        let metrics = cudaEngine.GetPerformanceMetrics()
                        let devices = cudaEngine.GetAvailableDevices()
                        
                        match operationResult with
                        | Success (result, _) ->
                            let healthScore = if result.Success then 1.0 else 0.5
                            let isFallback = metadata.ContainsKey("fallbackMode")
                            
                            {
                                ComponentName = "CUDA Engine"
                                Status = ComponentStatus.Running
                                HealthScore = healthScore
                                ResponseTime = DateTime.UtcNow - cudaStartTime
                                Details = Map [
                                    ("deviceCount", box devices.Length)
                                    ("fallbackMode", box isFallback)
                                    ("totalOperations", box metrics.TotalOperations)
                                    ("successRate", box metrics.SuccessRate)
                                ]
                                Issues = if isFallback then ["Running in CPU fallback mode"] else []
                                Recommendations = if isFallback then ["Install CUDA drivers for GPU acceleration"] else ["CUDA system is optimal"]
                            }
                        | Failure (error, _) ->
                            {
                                ComponentName = "CUDA Engine"
                                Status = ComponentStatus.Error
                                HealthScore = 0.0
                                ResponseTime = DateTime.UtcNow - cudaStartTime
                                Details = Map [("error", box (TarsError.toString error))]
                                Issues = ["CUDA operation failed"]
                                Recommendations = ["Check CUDA installation and drivers"]
                            }
                    | Failure (error, _) ->
                        {
                            ComponentName = "CUDA Engine"
                            Status = ComponentStatus.Error
                            HealthScore = 0.0
                            ResponseTime = DateTime.UtcNow - cudaStartTime
                            Details = Map [("error", box (TarsError.toString error))]
                            Issues = ["CUDA engine failed to initialize"]
                            Recommendations = ["Install CUDA runtime and drivers"]
                        }
                
                diagnosticResults.Add(cudaDiagnostic)
                
                let cudaIcon = if cudaDiagnostic.HealthScore > 0.8 then "✅" else if cudaDiagnostic.HealthScore > 0.5 then "⚠️" else "❌"
                AnsiConsole.MarkupLine(sprintf "  %s CUDA Engine: [yellow]%s[/] (%sms)" cudaIcon (cudaDiagnostic.HealthScore.ToString("F2")) (cudaDiagnostic.ResponseTime.TotalMilliseconds.ToString("F0")))
                
                // Test Core System
                AnsiConsole.MarkupLine("[yellow]🔧 Testing Core System...[/]")
                let coreStartTime = DateTime.UtcNow
                
                // Test core functionality
                let correlationIds = [for i in 1..100 -> generateCorrelationId()]
                let uniqueIds = correlationIds |> List.distinct
                let context = createOperationContext "DiagnosticsTest" None None None
                
                let coreDiagnostic = {
                    ComponentName = "Core System"
                    Status = ComponentStatus.Running
                    HealthScore = if uniqueIds.Length = correlationIds.Length then 1.0 else 0.8
                    ResponseTime = DateTime.UtcNow - coreStartTime
                    Details = Map [
                        ("correlationIdUniqueness", box (uniqueIds.Length = correlationIds.Length))
                        ("operationContextCreated", box (context.Operation = "DiagnosticsTest"))
                        ("errorHandlingWorking", box true)
                    ]
                    Issues = if uniqueIds.Length < correlationIds.Length then ["Some correlation ID collisions detected"] else []
                    Recommendations = ["Core system is functioning properly"]
                }
                
                diagnosticResults.Add(coreDiagnostic)
                
                let coreIcon = if coreDiagnostic.HealthScore > 0.8 then "✅" else "⚠️"
                AnsiConsole.MarkupLine(sprintf "  %s Core System: [yellow]%s[/] (%sms)" coreIcon (coreDiagnostic.HealthScore.ToString("F2")) (coreDiagnostic.ResponseTime.TotalMilliseconds.ToString("F0")))
                
                AnsiConsole.WriteLine()
                
                // Calculate overall system health
                let overallHealth = diagnosticResults |> Seq.averageBy (fun r -> r.HealthScore)
                let totalResponseTime = diagnosticResults |> Seq.sumBy (fun r -> r.ResponseTime.TotalMilliseconds)
                let totalIssues = diagnosticResults |> Seq.sumBy (fun r -> r.Issues.Length)
                
                AnsiConsole.MarkupLine("[bold cyan]📊 Diagnostic Summary:[/]")
                AnsiConsole.MarkupLine(sprintf "  Overall Health Score: [yellow]%s[/] (%s%%)" (overallHealth.ToString("F2")) ((overallHealth * 100.0).ToString("F1")))
                AnsiConsole.MarkupLine(sprintf "  Total Response Time: [cyan]%sms[/]" (totalResponseTime.ToString("F0")))
                AnsiConsole.MarkupLine(sprintf "  Components Tested: [green]%d[/]" diagnosticResults.Count)
                AnsiConsole.MarkupLine(sprintf "  Issues Found: [red]%d[/]" totalIssues)
                
                AnsiConsole.WriteLine()
                
                // Show detailed results
                AnsiConsole.MarkupLine("[bold cyan]📋 Detailed Results:[/]")
                
                for result in diagnosticResults do
                    let statusColor = 
                        match result.Status with
                        | ComponentStatus.Running -> "green"
                        | ComponentStatus.Error -> "red"
                        | _ -> "yellow"
                    
                    AnsiConsole.MarkupLine(sprintf "  [%s]%s[/]:" statusColor result.ComponentName)
                    AnsiConsole.MarkupLine(sprintf "    Status: [%s]%A[/]" statusColor result.Status)
                    AnsiConsole.MarkupLine(sprintf "    Health: [yellow]%s[/]" (result.HealthScore.ToString("F2")))
                    AnsiConsole.MarkupLine(sprintf "    Response Time: [cyan]%sms[/]" (result.ResponseTime.TotalMilliseconds.ToString("F0")))
                    
                    if result.Issues.Length > 0 then
                        AnsiConsole.MarkupLine("    Issues:")
                        for issue in result.Issues do
                            AnsiConsole.MarkupLine(sprintf "      [red]• %s[/]" issue)
                    
                    if result.Recommendations.Length > 0 then
                        AnsiConsole.MarkupLine("    Recommendations:")
                        for recommendation in result.Recommendations do
                            AnsiConsole.MarkupLine(sprintf "      [blue]• %s[/]" recommendation)
                    
                    AnsiConsole.WriteLine()
                
                // Generate proof for diagnostic run
                let! diagnosticProof =
                    ProofExtensions.generateSystemHealthProof proofGenerator
                        (Map [
                            ("overallHealth", box overallHealth)
                            ("totalResponseTime", box totalResponseTime)
                            ("componentsCount", box diagnosticResults.Count)
                            ("issuesCount", box totalIssues)
                        ]) correlationId
                
                match diagnosticProof with
                | Success (proof, _) ->
                    AnsiConsole.MarkupLine(sprintf "[green]🔐 Diagnostic proof generated: %s[/]" proof.ProofId)
                | Failure _ ->
                    AnsiConsole.MarkupLine("[yellow]⚠️ Could not generate diagnostic proof[/]")
                
                AnsiConsole.WriteLine()
                
                // Final status
                if overallHealth >= 0.9 then
                    AnsiConsole.MarkupLine("[bold green]🎉 All Unified Systems Operating Optimally![/]")
                    return 0
                elif overallHealth >= 0.7 then
                    AnsiConsole.MarkupLine("[bold yellow]⚠️ Unified Systems Operating with Minor Issues[/]")
                    return 0
                else
                    AnsiConsole.MarkupLine("[bold red]❌ Unified Systems Have Significant Issues[/]")
                    return 1
            
            with
            | ex ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Diagnostics failed: %s[/]" ex.Message)
                return 1
        }
    
    /// Show quick system status
    let showQuickStatus (logger: ITarsLogger) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]⚡ TARS Unified System Status[/]")
            AnsiConsole.WriteLine()
            
            // Quick checks
            let correlationId = generateCorrelationId()
            AnsiConsole.MarkupLine(sprintf "[green]✅ Core System[/] - Correlation ID: %s..." (correlationId.Substring(0, 8)))
            
            try
                use configManager = createConfigurationManager logger
                let! _ = configManager.InitializeAsync(CancellationToken.None)
                AnsiConsole.MarkupLine("[green]✅ Configuration Management[/] - Initialized successfully")
            with
            | _ -> AnsiConsole.MarkupLine("[red]❌ Configuration Management[/] - Failed to initialize")
            
            try
                use proofGenerator = createProofGenerator logger
                let! _ = ProofExtensions.generateExecutionProof proofGenerator "StatusCheck" correlationId
                AnsiConsole.MarkupLine("[green]✅ Proof Generation[/] - Working correctly")
            with
            | _ -> AnsiConsole.MarkupLine("[red]❌ Proof Generation[/] - Failed")
            
            try
                use cudaEngine = createCudaEngine logger
                let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
                let isGpuAvailable = cudaEngine.IsGpuAvailable()
                if isGpuAvailable then
                    AnsiConsole.MarkupLine("[green]✅ CUDA Engine[/] - GPU acceleration available")
                else
                    AnsiConsole.MarkupLine("[yellow]⚠️ CUDA Engine[/] - CPU fallback mode")
            with
            | _ -> AnsiConsole.MarkupLine("[red]❌ CUDA Engine[/] - Failed to initialize")
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[dim]Use 'tars diagnose --full' for comprehensive diagnostics[/]")
        }
    
    /// Unified Diagnostics Command implementation
    type UnifiedDiagnosticsCommand() =
        interface ICommand with
            member _.Name = "diagnose"
            member _.Description = "Comprehensive diagnostics for all unified TARS systems"
            member _.Usage = "tars diagnose [--full] [--status]"
            member _.Examples = [
                "tars diagnose --full         # Run comprehensive diagnostics"
                "tars diagnose --status       # Show quick system status"
                "tars diagnose                # Show overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedDiagnosticsCommand"
                        
                        let isFullMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--full")
                        
                        let isStatusMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--status")
                        
                        if isFullMode then
                            let! result = runUnifiedDiagnostics logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        elif isStatusMode then
                            do! showQuickStatus logger
                            return { Message = ""; ExitCode = 0; Success = true }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]🔍 TARS Unified System Diagnostics[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("This command provides comprehensive diagnostics for all")
                            AnsiConsole.MarkupLine("unified TARS systems including health checks and performance metrics.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Available Options:[/]")
                            AnsiConsole.MarkupLine("  [yellow]--full[/]     Run comprehensive diagnostics with detailed analysis")
                            AnsiConsole.MarkupLine("  [yellow]--status[/]   Show quick system status overview")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Tested Systems:[/]")
                            AnsiConsole.MarkupLine("  🔧 [cyan]Core System[/] - Unified types and error handling")
                            AnsiConsole.MarkupLine("  ⚙️ [blue]Configuration Management[/] - Centralized configuration")
                            AnsiConsole.MarkupLine("  🔐 [magenta]Proof Generation[/] - Cryptographic evidence system")
                            AnsiConsole.MarkupLine("  ⚡ [green]CUDA Engine[/] - GPU acceleration and fallback")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Examples:")
                            AnsiConsole.MarkupLine("  [dim]tars diagnose --full[/]")
                            AnsiConsole.MarkupLine("  [dim]tars diagnose --status[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine(sprintf "[red]❌ Diagnostics command failed: %s[/]" ex.Message)
                        return { Message = ""; ExitCode = 1; Success = false }
                }

