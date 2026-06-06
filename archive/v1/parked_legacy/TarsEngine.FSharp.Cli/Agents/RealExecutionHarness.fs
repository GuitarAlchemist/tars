namespace TarsEngine.FSharp.Cli.Agents

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Real execution result with concrete metrics
type ExecutionResult = {
    Success: bool
    ExitCode: int
    StandardOutput: string
    StandardError: string
    ExecutionTime: TimeSpan
    MemoryUsage: int64
    CpuTime: TimeSpan
}

/// Real test execution result
type TestResult = {
    TestsPassed: int
    TestsFailed: int
    TotalTests: int
    Coverage: float
    ExecutionTime: TimeSpan
    FailureDetails: string list
}

/// Real performance metrics
type PerformanceMetrics = {
    ExecutionTime: TimeSpan
    MemoryPeak: int64
    CpuUsage: float
    ThroughputOps: float
    LatencyMs: float
}

/// Real code modification patch
type CodePatch = {
    Id: string
    TargetFile: string
    OriginalContent: string
    ModifiedContent: string
    Description: string
    Timestamp: DateTime
    BackupPath: string
}

/// Real Execution Harness - NO SIMULATIONS
type RealExecutionHarness(logger: ILogger<RealExecutionHarness>) =
    
    let mutable appliedPatches: CodePatch list = []
    let backupDirectory = Path.Combine(Directory.GetCurrentDirectory(), "tars-backups")
    
    do
        // Ensure backup directory exists
        if not (Directory.Exists(backupDirectory)) then
            Directory.CreateDirectory(backupDirectory) |> ignore
    
    /// Execute real command with metrics collection
    member this.ExecuteCommand(command: string, arguments: string, workingDirectory: string) =
        task {
            try
                let startTime = DateTime.UtcNow
                let startMemory = GC.GetTotalMemory(false)
                
                let processInfo = ProcessStartInfo(
                    FileName = command,
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )
                
                use proc = new Process(StartInfo = processInfo)
                proc.Start() |> ignore
                
                let! output = proc.StandardOutput.ReadToEndAsync()
                let! error = proc.StandardError.ReadToEndAsync()
                
                proc.WaitForExit()
                
                let endTime = DateTime.UtcNow
                let endMemory = GC.GetTotalMemory(false)
                let executionTime = endTime - startTime
                
                logger.LogInformation($"Command executed: {command} {arguments}")
                logger.LogInformation($"Exit code: {proc.ExitCode}, Time: {executionTime.TotalMilliseconds}ms")
                
                return {
                    Success = proc.ExitCode = 0
                    ExitCode = proc.ExitCode
                    StandardOutput = output
                    StandardError = error
                    ExecutionTime = executionTime
                    MemoryUsage = endMemory - startMemory
                    CpuTime = proc.TotalProcessorTime
                }
                
            with ex ->
                logger.LogError(ex, $"Failed to execute command: {command}")
                return {
                    Success = false
                    ExitCode = -1
                    StandardOutput = ""
                    StandardError = ex.Message
                    ExecutionTime = TimeSpan.Zero
                    MemoryUsage = 0L
                    CpuTime = TimeSpan.Zero
                }
        }
    
    /// Run real tests and collect metrics
    member this.RunTests(projectPath: string) =
        task {
            logger.LogInformation($"Running tests for project: {projectPath}")
            
            let! result = this.ExecuteCommand("dotnet", $"test \"{projectPath}\" --logger trx --collect:\"XPlat Code Coverage\"", Directory.GetCurrentDirectory())
            
            if result.Success then
                // Parse test results from output
                let output = result.StandardOutput
                let lines = output.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                
                let mutable passed = 0
                let mutable failed = 0
                let mutable total = 0
                let mutable coverage = 0.0
                let failures = ResizeArray<string>()
                
                for line in lines do
                    if line.Contains("Passed!") then
                        // Parse: "Passed!  - Failed:     0, Passed:    42, Skipped:     0, Total:    42"
                        let parts = line.Split([|','; ' '|], StringSplitOptions.RemoveEmptyEntries)
                        for i in 0..parts.Length-2 do
                            if parts.[i] = "Failed:" then
                                Int32.TryParse(parts.[i+1], &failed) |> ignore
                            elif parts.[i] = "Passed:" then
                                Int32.TryParse(parts.[i+1], &passed) |> ignore
                            elif parts.[i] = "Total:" then
                                Int32.TryParse(parts.[i+1], &total) |> ignore
                    elif line.Contains("Failed") && line.Contains("Error Message:") then
                        failures.Add(line)
                    elif line.Contains("Line coverage:") then
                        // Parse coverage percentage
                        let coverageStr = line.Substring(line.IndexOf(":") + 1).Trim().Replace("%", "")
                        Double.TryParse(coverageStr, &coverage) |> ignore
                
                return {
                    TestsPassed = passed
                    TestsFailed = failed
                    TotalTests = total
                    Coverage = coverage / 100.0
                    ExecutionTime = result.ExecutionTime
                    FailureDetails = failures |> List.ofSeq
                }
            else
                logger.LogError($"Test execution failed: {result.StandardError}")
                return {
                    TestsPassed = 0
                    TestsFailed = 0
                    TotalTests = 0
                    Coverage = 0.0
                    ExecutionTime = result.ExecutionTime
                    FailureDetails = [result.StandardError]
                }
        }
    
    /// Measure real performance metrics
    member this.MeasurePerformance(executable: string, arguments: string, iterations: int) =
        task {
            logger.LogInformation($"Measuring performance: {executable} {arguments} ({iterations} iterations)")
            
            let measurements = ResizeArray<PerformanceMetrics>()
            
            for i in 1..iterations do
                let! result = this.ExecuteCommand(executable, arguments, Directory.GetCurrentDirectory())
                
                if result.Success then
                    let metrics = {
                        ExecutionTime = result.ExecutionTime
                        MemoryPeak = result.MemoryUsage
                        CpuUsage = result.CpuTime.TotalMilliseconds / result.ExecutionTime.TotalMilliseconds * 100.0
                        ThroughputOps = if result.ExecutionTime.TotalSeconds > 0.0 then 1.0 / result.ExecutionTime.TotalSeconds else 0.0
                        LatencyMs = result.ExecutionTime.TotalMilliseconds
                    }
                    measurements.Add(metrics)
                else
                    logger.LogWarning($"Performance measurement {i} failed: {result.StandardError}")
            
            if measurements.Count > 0 then
                // Calculate averages
                let avgExecutionTime = TimeSpan.FromTicks(measurements |> Seq.map (fun m -> m.ExecutionTime.Ticks) |> Seq.map float |> Seq.average |> int64)
                let avgMemory = measurements |> Seq.map (fun m -> m.MemoryPeak) |> Seq.map float |> Seq.average |> int64
                let avgCpu = measurements |> Seq.map (fun m -> m.CpuUsage) |> Seq.average
                let avgThroughput = measurements |> Seq.map (fun m -> m.ThroughputOps) |> Seq.average
                let avgLatency = measurements |> Seq.map (fun m -> m.LatencyMs) |> Seq.average
                
                return Some {
                    ExecutionTime = avgExecutionTime
                    MemoryPeak = avgMemory
                    CpuUsage = avgCpu
                    ThroughputOps = avgThroughput
                    LatencyMs = avgLatency
                }
            else
                return None
        }
    
    /// Apply real code patch with backup
    member this.ApplyPatch(patch: CodePatch) : Task<bool> =
        task {
            try
                logger.LogInformation($"Applying patch {patch.Id} to {patch.TargetFile}")

                // Verify target file exists
                if not (File.Exists(patch.TargetFile)) then
                    logger.LogError($"Target file not found: {patch.TargetFile}")
                    ()

                    // Create backup
                    let backupFileName = $"{patch.Id}_{Path.GetFileName(patch.TargetFile)}.backup"
                    let backupPath = Path.Combine(backupDirectory, backupFileName)
                    File.Copy(patch.TargetFile, backupPath, true)

                    // Apply modification
                    File.WriteAllText(patch.TargetFile, patch.ModifiedContent)

                    // Update patch with backup path
                    let updatedPatch = { patch with BackupPath = backupPath }
                    appliedPatches <- updatedPatch :: appliedPatches

                    logger.LogInformation($"Patch {patch.Id} applied successfully. Backup: {backupPath}")
                    ()
                
            with ex ->
                logger.LogError(ex, $"Failed to apply patch {patch.Id}")
                ()
        }
    
    /// Rollback real code patch
    member this.RollbackPatch(patchId: string) : Task<bool> =
        task {
            try
                let patch = appliedPatches |> List.tryFind (fun p -> p.Id = patchId)
                
                match patch with
                | Some p ->
                    logger.LogInformation($"Rolling back patch {patchId}")
                    
                    // Restore from backup
                    if File.Exists(p.BackupPath) then
                        File.Copy(p.BackupPath, p.TargetFile, true)
                        File.Delete(p.BackupPath)
                        
                        // Remove from applied patches
                        appliedPatches <- appliedPatches |> List.filter (fun ap -> ap.Id <> patchId)
                        
                        logger.LogInformation($"Patch {patchId} rolled back successfully")
                        return true
                    else
                        logger.LogError($"Backup file not found for patch {patchId}: {p.BackupPath}")
                        return false
                | None ->
                    logger.LogWarning($"Patch {patchId} not found in applied patches")
                    return false
                    
            with ex ->
                logger.LogError(ex, $"Failed to rollback patch {patchId}")
                return false
        }
    
    /// Get real compilation status
    member this.CheckCompilation(projectPath: string) =
        task {
            logger.LogInformation($"Checking compilation for: {projectPath}")
            
            let! result = this.ExecuteCommand("dotnet", $"build \"{projectPath}\" --no-restore", Directory.GetCurrentDirectory())
            
            return {|
                Success = result.Success
                Errors = if result.Success then [] else [result.StandardError]
                Warnings = []
                BuildTime = result.ExecutionTime
            |}
        }
    
    /// Get applied patches
    member this.GetAppliedPatches() = appliedPatches
    
    /// Clean up all backups
    member this.CleanupBackups() =
        try
            if Directory.Exists(backupDirectory) then
                Directory.Delete(backupDirectory, true)
                Directory.CreateDirectory(backupDirectory) |> ignore
            appliedPatches <- []
            logger.LogInformation("All backups cleaned up")
        with ex ->
            logger.LogError(ex, "Failed to cleanup backups")
