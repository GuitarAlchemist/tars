namespace TarsEngine.FSharp.Core

open System
open System.Diagnostics
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Real performance measurement for TARS auto-improvement system
/// Fixes the critical issue where no real performance metrics were captured
module TarsPerformanceMeasurement =

    /// Comprehensive performance metrics
    type PerformanceMetrics = {
        BuildTimeMs: int64
        MemoryUsageMB: int64
        ResponseTimeMs: double
        CompilationSuccess: bool
        TestExecutionTimeMs: int64 option
        TestSuccess: bool option
        CpuUsagePercent: double option
        DiskIOBytes: int64 option
        Timestamp: DateTime
    }

    /// Performance baseline for comparison
    type PerformanceBaseline = {
        ProjectName: string
        BaselineMetrics: PerformanceMetrics
        MeasurementDate: DateTime
        TarsVersion: string option
        Environment: string
    }

    /// Performance comparison result
    type PerformanceComparison = {
        Before: PerformanceMetrics
        After: PerformanceMetrics
        BuildTimeImprovement: double // Percentage improvement (positive = better)
        MemoryImprovement: double
        ResponseTimeImprovement: double
        OverallImprovement: double
        SignificantImprovement: bool
    }

    /// Command execution result with timeout support
    type CommandResult = {
        ExitCode: int
        Output: string
        Error: string
        ExecutionTimeMs: int64
        TimedOut: bool
    }

    /// TARS performance measurement service
    type TarsPerformanceMeasurementService(logger: ILogger<TarsPerformanceMeasurementService>) =

        /// Execute command with timeout and performance tracking
        member this.ExecuteCommandWithTimeout(command: string, arguments: string, workingDirectory: string, timeoutMs: int) : Async<CommandResult> = async {
            let stopwatch = Stopwatch.StartNew()
            
            try
                use cts = new CancellationTokenSource(timeoutMs)
                
                let processStartInfo = ProcessStartInfo(
                    FileName = command,
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )

                use proc = new Process(StartInfo = processStartInfo)
                
                let outputBuilder = System.Text.StringBuilder()
                let errorBuilder = System.Text.StringBuilder()
                
                proc.OutputDataReceived.Add(fun args -> 
                    if not (isNull args.Data) then
                        outputBuilder.AppendLine(args.Data) |> ignore)
                
                proc.ErrorDataReceived.Add(fun args -> 
                    if not (isNull args.Data) then
                        errorBuilder.AppendLine(args.Data) |> ignore)

                proc.Start() |> ignore
                proc.BeginOutputReadLine()
                proc.BeginErrorReadLine()

                let! completed = proc.WaitForExitAsync(cts.Token) |> Async.AwaitTask
                stopwatch.Stop()

                if cts.Token.IsCancellationRequested then
                    try proc.Kill() with | _ -> ()
                    return {
                        ExitCode = -1
                        Output = outputBuilder.ToString()
                        Error = "Command timed out"
                        ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                        TimedOut = true
                    }
                else
                    return {
                        ExitCode = proc.ExitCode
                        Output = outputBuilder.ToString()
                        Error = errorBuilder.ToString()
                        ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                        TimedOut = false
                    }
            with
            | ex ->
                stopwatch.Stop()
                logger.LogError(ex, $"Command execution failed: {command} {arguments}")
                return {
                    ExitCode = -1
                    Output = ""
                    Error = ex.Message
                    ExecutionTimeMs = stopwatch.ElapsedMilliseconds
                    TimedOut = false
                }
        }

        /// Measure build performance for a TARS project
        member this.MeasureBuildPerformance(projectPath: string) : Async<PerformanceMetrics> = async {
            logger.LogInformation($"📊 Measuring build performance for: {Path.GetFileName(projectPath)}")
            
            let beforeMemory = GC.GetTotalMemory(true)
            let beforeTime = DateTime.UtcNow

            try
                // Get the directory containing the project file
                let projectDirectory = Path.GetDirectoryName(projectPath)
                let projectFileName = Path.GetFileName(projectPath)

                // Clean before build
                let! cleanResult = this.ExecuteCommandWithTimeout("dotnet", $"clean {projectFileName}", projectDirectory, 30000)

                // Measure build time
                let! buildResult = this.ExecuteCommandWithTimeout("dotnet", $"build --no-restore {projectFileName}", projectDirectory, 120000)
                
                let afterMemory = GC.GetTotalMemory(false)
                let memoryUsed = (afterMemory - beforeMemory) / 1024L / 1024L

                let metrics = {
                    BuildTimeMs = buildResult.ExecutionTimeMs
                    MemoryUsageMB = Math.Max(memoryUsed, 0L)
                    ResponseTimeMs = 0.0 // Will be measured separately
                    CompilationSuccess = buildResult.ExitCode = 0 && not buildResult.TimedOut
                    TestExecutionTimeMs = None
                    TestSuccess = None
                    CpuUsagePercent = None
                    DiskIOBytes = None
                    Timestamp = DateTime.UtcNow
                }

                if metrics.CompilationSuccess then
                    logger.LogInformation($"✅ Build successful: {metrics.BuildTimeMs}ms, {metrics.MemoryUsageMB}MB")
                else
                    logger.LogWarning($"❌ Build failed: {buildResult.Error}")

                return metrics
            with
            | ex ->
                logger.LogError(ex, "Build performance measurement failed")
                return {
                    BuildTimeMs = 0L
                    MemoryUsageMB = 0L
                    ResponseTimeMs = 0.0
                    CompilationSuccess = false
                    TestExecutionTimeMs = None
                    TestSuccess = None
                    CpuUsagePercent = None
                    DiskIOBytes = None
                    Timestamp = DateTime.UtcNow
                }
        }

        /// Measure CLI response time
        member this.MeasureCliResponseTime(cliProjectPath: string, command: string) : Async<double> = async {
            try
                logger.LogDebug($"📊 Measuring CLI response time for: {command}")
                
                let responseTimes = ResizeArray<int64>()
                
                // Run command multiple times for average
                for i in 1..3 do
                    let cliProjectDirectory = Path.GetDirectoryName(cliProjectPath)
                    let cliProjectFileName = Path.GetFileName(cliProjectPath)
                    let! result = this.ExecuteCommandWithTimeout("dotnet", $"run --project {cliProjectFileName} -- {command}", cliProjectDirectory, 30000)
                    if result.ExitCode = 0 && not result.TimedOut then
                        responseTimes.Add(result.ExecutionTimeMs)

                if responseTimes.Count > 0 then
                    let avgResponseTime = responseTimes |> Seq.map double |> Seq.average
                    logger.LogDebug($"CLI response time: {avgResponseTime:F1}ms")
                    return avgResponseTime
                else
                    logger.LogWarning("No successful CLI responses measured")
                    return 0.0
            with
            | ex ->
                logger.LogError(ex, "CLI response measurement failed")
                return 0.0
        }

        /// Measure test execution performance
        member this.MeasureTestPerformance(projectPath: string) : Async<int64 * bool> = async {
            try
                logger.LogDebug($"📊 Measuring test performance for: {Path.GetFileName(projectPath)}")
                
                let projectDirectory = Path.GetDirectoryName(projectPath)
                let projectFileName = Path.GetFileName(projectPath)
                let! testResult = this.ExecuteCommandWithTimeout("dotnet", $"test --no-build {projectFileName}", projectDirectory, 180000)
                
                let testSuccess = testResult.ExitCode = 0 && not testResult.TimedOut
                logger.LogDebug($"Test execution: {testResult.ExecutionTimeMs}ms, Success: {testSuccess}")
                
                return (testResult.ExecutionTimeMs, testSuccess)
            with
            | ex ->
                logger.LogError(ex, "Test performance measurement failed")
                return (0L, false)
        }

        /// Get comprehensive performance metrics for a TARS project
        member this.GetComprehensiveMetrics(projectPath: string, includeTests: bool) : Async<PerformanceMetrics> = async {
            logger.LogInformation($"📊 Getting comprehensive performance metrics for: {Path.GetFileName(projectPath)}")
            
            // Measure build performance
            let! buildMetrics = this.MeasureBuildPerformance(projectPath)
            
            if not buildMetrics.CompilationSuccess then
                return buildMetrics
            else
                // Measure CLI response time if it's a CLI project
                let! responseTime = 
                    if Path.GetFileName(projectPath).Contains("Cli") then
                        this.MeasureCliResponseTime(projectPath, "help")
                    else
                        async { return 0.0 }

                // Measure test performance if requested
                let! (testTime, testSuccess) = 
                    if includeTests then
                        this.MeasureTestPerformance(projectPath)
                    else
                        async { return (None |> Option.toNullable |> Option.ofNullable |> Option.defaultValue 0L, None |> Option.toNullable |> Option.ofNullable |> Option.defaultValue false) }

                return {
                    buildMetrics with
                        ResponseTimeMs = responseTime
                        TestExecutionTimeMs = if includeTests then Some testTime else None
                        TestSuccess = if includeTests then Some testSuccess else None
                }
        }

        /// Compare performance metrics
        member this.ComparePerformance(before: PerformanceMetrics, after: PerformanceMetrics) : PerformanceComparison =
            let calculateImprovement (beforeValue: double) (afterValue: double) =
                if beforeValue > 0.0 then
                    ((beforeValue - afterValue) / beforeValue) * 100.0
                else
                    0.0

            let buildImprovement = calculateImprovement (double before.BuildTimeMs) (double after.BuildTimeMs)
            let memoryImprovement = calculateImprovement (double before.MemoryUsageMB) (double after.MemoryUsageMB)
            let responseImprovement = calculateImprovement before.ResponseTimeMs after.ResponseTimeMs

            let overallImprovement = (buildImprovement + memoryImprovement + responseImprovement) / 3.0
            let significantImprovement = overallImprovement > 5.0 // 5% improvement threshold

            {
                Before = before
                After = after
                BuildTimeImprovement = buildImprovement
                MemoryImprovement = memoryImprovement
                ResponseTimeImprovement = responseImprovement
                OverallImprovement = overallImprovement
                SignificantImprovement = significantImprovement
            }

        /// Save performance baseline
        member this.SaveBaseline(projectName: string, metrics: PerformanceMetrics, filePath: string) : Async<unit> = async {
            try
                let baseline = {
                    ProjectName = projectName
                    BaselineMetrics = metrics
                    MeasurementDate = DateTime.UtcNow
                    TarsVersion = None // TODO: Get from assembly
                    Environment = Environment.MachineName
                }

                let json = System.Text.Json.JsonSerializer.Serialize(baseline, System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                do! File.WriteAllTextAsync(filePath, json) |> Async.AwaitTask
                
                logger.LogInformation($"💾 Performance baseline saved: {filePath}")
            with
            | ex ->
                logger.LogError(ex, $"Failed to save performance baseline: {filePath}")
        }

        /// Load performance baseline
        member this.LoadBaseline(filePath: string) : Async<PerformanceBaseline option> = async {
            try
                if File.Exists(filePath) then
                    let! json = File.ReadAllTextAsync(filePath) |> Async.AwaitTask
                    let baseline = System.Text.Json.JsonSerializer.Deserialize<PerformanceBaseline>(json)
                    logger.LogInformation($"📂 Performance baseline loaded: {baseline.ProjectName}")
                    return Some baseline
                else
                    logger.LogWarning($"Performance baseline file not found: {filePath}")
                    return None
            with
            | ex ->
                logger.LogError(ex, $"Failed to load performance baseline: {filePath}")
                return None
        }

    /// Static helper functions for quick performance checks
    module PerformanceHelpers =
        
        /// Quick build time check
        let quickBuildTimeCheck (projectPath: string) (logger: ILogger<_>) = async {
            let service = TarsPerformanceMeasurementService(logger)
            let! metrics = service.MeasureBuildPerformance(projectPath)
            return metrics.BuildTimeMs
        }
        
        /// Check if performance is acceptable
        let isPerformanceAcceptable (metrics: PerformanceMetrics) =
            metrics.CompilationSuccess && 
            metrics.BuildTimeMs < 60000L && // Less than 1 minute
            metrics.MemoryUsageMB < 1000L    // Less than 1GB
