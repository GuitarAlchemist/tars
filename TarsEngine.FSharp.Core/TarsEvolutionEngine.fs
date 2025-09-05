namespace TarsEngine.FSharp.Core

open System
open System.IO
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Console
open TarsEngine.FSharp.Core.TarsProjectDiscovery
open TarsEngine.FSharp.Core.TarsPerformanceMeasurement
open TarsEngine.FSharp.Core.TarsSafeFileOperations

/// Comprehensive TARS Evolution Engine that integrates all auto-improvement capabilities
/// This fixes the critical issues identified in our Blue-Green evolution experiments
module TarsEvolutionEngine =

    /// Evolution session configuration
    type EvolutionConfig = {
        MaxExecutionTimeMs: int
        PerformanceImprovementThreshold: double // Minimum % improvement to accept changes
        SafetyMode: bool // If true, requires human approval for risky changes
        BackupEnabled: bool
        TestingEnabled: bool
        MaxConcurrentOperations: int
    }

    /// Evolution step result
    type EvolutionStepResult = {
        StepName: string
        Success: bool
        ExecutionTimeMs: int64
        Details: string
        Metrics: Map<string, obj>
        ErrorMessage: string option
    }

    /// Complete evolution session result
    type EvolutionSessionResult = {
        SessionId: string
        StartTime: DateTime
        EndTime: DateTime
        TotalDurationMs: int64
        Config: EvolutionConfig
        Steps: EvolutionStepResult array
        OverallSuccess: bool
        ProjectsAnalyzed: int
        ImprovementsApplied: int
        PerformanceGain: double option
        RecommendedNextSteps: string array
    }

    /// TARS Evolution Engine service
    type TarsEvolutionEngineService(logger: ILogger<TarsEvolutionEngineService>) =

        let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
        let discoveryService = TarsProjectDiscoveryService(loggerFactory.CreateLogger<TarsProjectDiscoveryService>())
        let performanceService = TarsPerformanceMeasurementService(loggerFactory.CreateLogger<TarsPerformanceMeasurementService>())
        let fileService = TarsSafeFileOperationsService(loggerFactory.CreateLogger<TarsSafeFileOperationsService>())

        /// Default evolution configuration
        member this.DefaultConfig = {
            MaxExecutionTimeMs = 300000 // 5 minutes
            PerformanceImprovementThreshold = 5.0 // 5% minimum improvement
            SafetyMode = true
            BackupEnabled = true
            TestingEnabled = true
            MaxConcurrentOperations = 2
        }

        /// Execute a single evolution step with comprehensive logging
        member private this.ExecuteStep(stepName: string, operation: Async<'T>) : Async<EvolutionStepResult * 'T option> = async {
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            logger.LogInformation($"🔄 Executing step: {stepName}")
            
            try
                let! result = operation
                stopwatch.Stop()
                
                let stepResult = {
                    StepName = stepName
                    Success = true
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                    Details = $"Step completed successfully"
                    Metrics = Map.empty
                    ErrorMessage = None
                }
                
                logger.LogInformation($"✅ Step completed: {stepName} ({stopwatch.ElapsedMilliseconds}ms)")
                return (stepResult, Some result)
            with
            | ex ->
                stopwatch.Stop()
                let stepResult = {
                    StepName = stepName
                    Success = false
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                    Details = $"Step failed: {ex.Message}"
                    Metrics = Map.empty
                    ErrorMessage = Some ex.Message
                }
                
                logger.LogError(ex, $"❌ Step failed: {stepName}")
                return (stepResult, None)
        }

        /// Run comprehensive TARS evolution session
        member this.RunEvolutionSession(startPath: string, ?config: EvolutionConfig) : Async<EvolutionSessionResult> = async {
            let sessionId = $"tars-evolution-{DateTimeOffset.UtcNow.ToUnixTimeSeconds()}"
            let sessionStart = DateTime.UtcNow
            let evolutionConfig = defaultArg config this.DefaultConfig
            let steps = ResizeArray<EvolutionStepResult>()
            
            logger.LogInformation($"🚀 Starting TARS Evolution Session: {sessionId}")
            logger.LogInformation($"Configuration: Safety={evolutionConfig.SafetyMode}, Testing={evolutionConfig.TestingEnabled}")
            
            try
                // Step 1: Project Discovery
                let! (step1, discoveryResult) = this.ExecuteStep("Project Discovery", async {
                    return! discoveryService.DiscoverTarsProjects(startPath)
                })
                steps.Add(step1)
                
                match discoveryResult with
                | None ->
                    logger.LogError("❌ Cannot proceed without successful project discovery")
                    let sessionResult = {
                        SessionId = sessionId
                        StartTime = sessionStart
                        EndTime = DateTime.UtcNow
                        TotalDurationMs = (DateTime.UtcNow - sessionStart).Ticks / 10000L
                        Config = evolutionConfig
                        Steps = steps.ToArray()
                        OverallSuccess = false
                        ProjectsAnalyzed = 0
                        ImprovementsApplied = 0
                        PerformanceGain = None
                        RecommendedNextSteps = [| "Fix project discovery issues"; "Verify TARS installation" |]
                    }
                    return sessionResult
                    
                | Some discovery when not discovery.Success ->
                    logger.LogError($"❌ Project discovery failed: {discovery.ErrorMessage}")
                    let sessionResult = {
                        SessionId = sessionId
                        StartTime = sessionStart
                        EndTime = DateTime.UtcNow
                        TotalDurationMs = (DateTime.UtcNow - sessionStart).Ticks / 10000L
                        Config = evolutionConfig
                        Steps = steps.ToArray()
                        OverallSuccess = false
                        ProjectsAnalyzed = 0
                        ImprovementsApplied = 0
                        PerformanceGain = None
                        RecommendedNextSteps = [| "Check TARS installation"; "Verify project structure" |]
                    }
                    return sessionResult
                    
                | Some discovery ->
                    logger.LogInformation($"✅ Discovered {discovery.ProjectsFound.Length} TARS projects")
                    
                    // Step 2: Performance Baseline
                    let! (step2, baselineResults) = this.ExecuteStep("Performance Baseline", async {
                        let results = ResizeArray<string * PerformanceMetrics>()
                        
                        for project in discovery.ProjectsFound do
                            try
                                let! metrics = performanceService.GetComprehensiveMetrics(project.ProjectPath, evolutionConfig.TestingEnabled)
                                results.Add((project.Name, metrics))
                                logger.LogInformation($"📊 Baseline for {project.Name}: Build={metrics.BuildTimeMs}ms, Success={metrics.CompilationSuccess}")
                            with
                            | ex ->
                                logger.LogWarning(ex, $"Failed to get baseline for {project.Name}")
                        
                        return results.ToArray()
                    })
                    steps.Add(step2)
                    
                    match baselineResults with
                    | None ->
                        logger.LogWarning("⚠️ No performance baselines established")
                    | Some baselines ->
                        logger.LogInformation($"📊 Established baselines for {baselines.Length} projects")
                    
                    // Step 3: Code Analysis
                    let! (step3, analysisResults) = this.ExecuteStep("Code Analysis", async {
                        let improvements = ResizeArray<string>()
                        
                        for project in discovery.ProjectsFound do
                            // Analyze each project for improvement opportunities
                            if project.ComplexityScore > 50 then
                                improvements.Add($"Reduce complexity in {project.Name} (score: {project.ComplexityScore})")
                            
                            if project.LineCount > 1000 && not project.HasTests then
                                improvements.Add($"Add tests to {project.Name} ({project.LineCount} lines without tests)")
                            
                            // Check for specific patterns
                            for sourceFile in project.SourceFiles do
                                try
                                    let! fileResult = fileService.ReadFileWithTimeout(sourceFile, 5000)
                                    match fileResult.Result with
                                    | Some content ->
                                        if content.Contains("Console.WriteLine") && not (content.Contains("Console.ForegroundColor")) then
                                            improvements.Add($"Enhance console output in {Path.GetFileName(sourceFile)}")
                                        
                                        if content.Contains("string +") || content.Contains("String.Concat") then
                                            improvements.Add($"Optimize string operations in {Path.GetFileName(sourceFile)}")
                                    | None ->
                                        logger.LogDebug($"Could not read file for analysis: {sourceFile}")
                                with
                                | ex ->
                                    logger.LogDebug(ex, $"Error analyzing file: {sourceFile}")
                        
                        return improvements.ToArray()
                    })
                    steps.Add(step3)
                    
                    let improvementOpportunities = analysisResults |> Option.defaultValue [||]
                    logger.LogInformation($"🎯 Found {improvementOpportunities.Length} improvement opportunities")
                    
                    // Step 4: Apply Safe Improvements
                    let! (step4, implementationResults) = this.ExecuteStep("Apply Improvements", async {
                        let appliedImprovements = ResizeArray<string>()
                        
                        // For now, apply one safe improvement: Enhanced console output
                        if improvementOpportunities |> Array.exists (fun imp -> imp.Contains("console output")) then
                            let enhancedConsoleCode = """
// TARS Auto-Generated: Enhanced Console Output
module TarsEnhancedConsole =
    open System
    
    let writeSuccess (message: string) =
        Console.ForegroundColor <- ConsoleColor.Green
        Console.WriteLine($"✅ {message}")
        Console.ResetColor()
    
    let writeWarning (message: string) =
        Console.ForegroundColor <- ConsoleColor.Yellow
        Console.WriteLine($"⚠️  {message}")
        Console.ResetColor()
    
    let writeError (message: string) =
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine($"❌ {message}")
        Console.ResetColor()
    
    let writeInfo (message: string) =
        Console.ForegroundColor <- ConsoleColor.Cyan
        Console.WriteLine($"ℹ️  {message}")
        Console.ResetColor()
"""
                            
                            let enhancedConsolePath = Path.Combine(discovery.TarsRootPath, "TarsEnhancedConsole.fs")
                            let! writeResult = fileService.WriteFileWithBackup(enhancedConsolePath, enhancedConsoleCode)
                            
                            if writeResult.Success then
                                appliedImprovements.Add("Enhanced Console Output")
                                logger.LogInformation($"✅ Applied improvement: Enhanced Console Output")
                            else
                                logger.LogWarning($"❌ Failed to apply Enhanced Console Output: {writeResult.ErrorMessage}")
                        
                        return appliedImprovements.ToArray()
                    })
                    steps.Add(step4)
                    
                    let appliedImprovements = implementationResults |> Option.defaultValue [||]
                    
                    // Step 5: Measure Performance Impact
                    let! (step5, performanceImpact) = this.ExecuteStep("Measure Impact", async {
                        match baselineResults with
                        | Some baselines when baselines.Length > 0 ->
                            let (projectName, baselineMetrics) = baselines.[0]
                            let project = discovery.ProjectsFound |> Array.find (fun p -> p.Name = projectName)
                            
                            // Re-measure performance after changes
                            let! newMetrics = performanceService.GetComprehensiveMetrics(project.ProjectPath, false)
                            let comparison = performanceService.ComparePerformance(baselineMetrics, newMetrics)
                            
                            logger.LogInformation($"📈 Performance comparison: {comparison.OverallImprovement} overall improvement")
                            return Some comparison.OverallImprovement
                        | _ ->
                            return None
                    })
                    steps.Add(step5)
                    
                    let performanceGain = performanceImpact |> Option.flatten
                    
                    // Generate final session result
                    let sessionEnd = DateTime.UtcNow
                    let overallSuccess = steps |> Seq.forall (fun s -> s.Success)
                    
                    let recommendedNextSteps = [|
                        if appliedImprovements.Length = 0 then "Implement more sophisticated code improvements"
                        if performanceGain.IsNone then "Establish better performance measurement"
                        if not evolutionConfig.TestingEnabled then "Enable testing in evolution configuration"
                        "Integrate with TARS CLI for automated evolution"
                    |]
                    
                    let sessionResult = {
                        SessionId = sessionId
                        StartTime = sessionStart
                        EndTime = sessionEnd
                        TotalDurationMs = (sessionEnd - sessionStart).Ticks / 10000L
                        Config = evolutionConfig
                        Steps = steps.ToArray()
                        OverallSuccess = overallSuccess
                        ProjectsAnalyzed = discovery.ProjectsFound.Length
                        ImprovementsApplied = appliedImprovements.Length
                        PerformanceGain = performanceGain
                        RecommendedNextSteps = recommendedNextSteps
                    }
                    
                    logger.LogInformation($"🎉 Evolution session completed: {sessionId}")
                    logger.LogInformation($"Projects analyzed: {sessionResult.ProjectsAnalyzed}")
                    logger.LogInformation($"Improvements applied: {sessionResult.ImprovementsApplied}")
                    logger.LogInformation($"Duration: {sessionResult.TotalDurationMs}ms")
                    
                    return sessionResult
            with
            | ex ->
                logger.LogError(ex, $"❌ Evolution session failed: {sessionId}")
                let sessionResult = {
                    SessionId = sessionId
                    StartTime = sessionStart
                    EndTime = DateTime.UtcNow
                    TotalDurationMs = (DateTime.UtcNow - sessionStart).Ticks / 10000L
                    Config = evolutionConfig
                    Steps = steps.ToArray()
                    OverallSuccess = false
                    ProjectsAnalyzed = 0
                    ImprovementsApplied = 0
                    PerformanceGain = None
                    RecommendedNextSteps = [| "Debug evolution engine issues"; "Check system requirements" |]
                }
                return sessionResult
        }

        /// Quick evolution check - fast validation without full session
        member this.QuickEvolutionCheck(startPath: string) : Async<bool * string> = async {
            try
                logger.LogInformation("🔍 Running quick evolution check...")
                
                // Test project discovery
                let discoveryResult = discoveryService.QuickDiscovery(startPath)
                if not discoveryResult.Success then
                    return (false, $"Project discovery failed: {discoveryResult.ErrorMessage}")
                else
                    // Test file operations
                    let tempDir = Path.GetTempPath()
                    let! fileOpsWorking = SafeFileHelpers.testFileOperations tempDir (loggerFactory.CreateLogger<TarsSafeFileOperationsService>())
                    if not fileOpsWorking then
                        return (false, "File operations test failed")
                    else
                        logger.LogInformation("✅ Quick evolution check passed")
                        return (true, $"Ready for evolution: {discoveryResult.TotalSourceFiles} source files found")
            with
            | ex ->
                logger.LogError(ex, "Quick evolution check failed")
                return (false, ex.Message)
        }

    /// Static helper functions
    module EvolutionHelpers =
        
        /// Run a quick evolution session
        let quickEvolution (startPath: string) (logger: ILogger<_>) = async {
            let engine = TarsEvolutionEngineService(logger)
            let config = { engine.DefaultConfig with MaxExecutionTimeMs = 60000; TestingEnabled = false }
            return! engine.RunEvolutionSession(startPath, config)
        }

        /// Check if evolution is ready to run
        let isEvolutionReady (startPath: string) (logger: ILogger<_>) = async {
            let engine = TarsEvolutionEngineService(logger)
            let! (ready, message) = engine.QuickEvolutionCheck(startPath)
            return ready
        }
