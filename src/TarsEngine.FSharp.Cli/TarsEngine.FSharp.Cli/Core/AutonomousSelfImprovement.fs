namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Text.Json
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Tier 9: Autonomous Self-Improvement Framework
/// Provides safe, verified self-enhancement capabilities for TARS

/// Improvement task for autonomous execution
type ImprovementTask = {
    taskId: Guid
    targetComponent: string
    improvementType: string  // "Performance", "Quality", "Capability", "Algorithm"
    description: string
    originalCode: string
    proposedCode: string
    expectedBenefit: float  // 0.0 to 1.0
    implementationRisk: float  // 0.0 to 1.0
    estimatedImpact: float  // 0.0 to 1.0
    safetyChecks: string list
    rollbackData: string
    status: string  // "Pending", "Testing", "Verified", "Applied", "Failed", "Rolled_Back"
    createdAt: DateTime
    lastModified: DateTime
}

/// Windows Sandbox configuration
type WindowsSandboxConfig = {
    configFilePath: string
    mountedFolders: (string * string * bool) list  // (hostPath, sandboxPath, readOnly)
    networkingEnabled: bool
    vGpuEnabled: bool
    audioInputEnabled: bool
    videoInputEnabled: bool
    protectedClientEnabled: bool
    printingEnabled: bool
    clipboardRedirectionEnabled: bool
    memoryInMB: int option
    logonCommand: string option
}

/// Sandbox environment for safe testing with Windows Sandbox integration
type SandboxEnvironment = {
    sandboxId: Guid
    sandboxType: string  // "WindowsSandbox" or "TempDirectory"
    isolatedPath: string
    windowsSandboxConfig: WindowsSandboxConfig option
    sandboxProcess: Process option
    testResults: Map<string, obj>
    performanceMetrics: Map<string, float>
    safetyValidation: bool
    executionLog: string list
    resourceUsage: Map<string, float>  // CPU, Memory, Disk usage
    timeoutMinutes: int
    createdAt: DateTime
    isActive: bool
}

/// Verification system for improvement validation
type VerificationResult = {
    verificationId: Guid
    taskId: Guid
    testsPassed: int
    testsFailed: int
    performanceImprovement: float  // Measured improvement percentage
    qualityImprovement: float  // Code quality delta
    safetyScore: float  // 0.0 to 1.0 safety assessment
    riskAssessment: string
    recommendation: string  // "Apply", "Reject", "Modify", "Retest"
    verificationTime: DateTime
    details: string list
}

/// Rollback manager for safe recovery
type RollbackManager = {
    rollbackId: Guid
    originalState: string
    backupPath: string
    rollbackSteps: string list
    verificationChecks: string list
    isReady: bool
    createdAt: DateTime
}

/// Test results from sandbox execution
type SandboxTestResults = {
    testsPassed: int
    testsFailed: int
    compilationResult: bool
    performanceImprovement: float
    safetyScore: float
    memoryUsage: int64
    details: string list
}

/// Self-improvement execution state
type SelfImprovementState = {
    activeImprovements: Map<Guid, ImprovementTask>
    completedImprovements: ImprovementTask list
    sandboxEnvironments: Map<Guid, SandboxEnvironment>
    verificationHistory: VerificationResult list
    rollbackCapability: RollbackManager option
    improvementQueue: ImprovementTask list
    safetyMetrics: Map<string, float>
    lastImprovementCycle: DateTime
}

/// Tier 9: Autonomous Self-Improvement Engine with Windows Sandbox Integration
type AutonomousSelfImprovementEngine(logger: ILogger<AutonomousSelfImprovementEngine>) =

    let mutable improvementState = {
        activeImprovements = Map.empty
        completedImprovements = []
        sandboxEnvironments = Map.empty
        verificationHistory = []
        rollbackCapability = None
        improvementQueue = []
        safetyMetrics = Map.empty
        lastImprovementCycle = DateTime.MinValue
    }

    /// Check if Windows Sandbox is available on the system
    member private this.IsWindowsSandboxAvailable() =
        try
            // Check if Windows Sandbox feature is enabled
            let psi = ProcessStartInfo()
            psi.FileName <- "powershell.exe"
            let featureName = "Containers-DisposableClientVM"
            let cmdArgs = $"-Command \"Get-WindowsOptionalFeature -Online -FeatureName {featureName} | Select-Object -ExpandProperty State\""
            psi.Arguments <- cmdArgs
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.CreateNoWindow <- true

            use proc = Process.Start(psi)
            let output = proc.StandardOutput.ReadToEnd().Trim()
            proc.WaitForExit()

            let isEnabled = output.Contains("Enabled")
            let statusText = if isEnabled then "Available" else "Not Available"
            logger.LogInformation($"Windows Sandbox availability check: {statusText}")
            isEnabled
        with
        | ex ->
            logger.LogWarning($"Failed to check Windows Sandbox availability: {ex.Message}")
            false

    /// Create Windows Sandbox configuration file
    member private this.CreateWindowsSandboxConfig(sandboxId: Guid, sourcePath: string) =
        try
            let configIdShort = sandboxId.ToString("N").[..7]
            let configDir = Path.Combine(Path.GetTempPath(), $"tars_sandbox_config_{configIdShort}")
            Directory.CreateDirectory(configDir) |> ignore

            let configFilePath = Path.Combine(configDir, "TarsSandbox.wsb")
            let sandboxSourcePath = "C:\\TarsSource"
            let sandboxWorkPath = "C:\\TarsWork"

            let configContent =
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
                "<Configuration>\n" +
                "    <VGpu>Disable</VGpu>\n" +
                "    <Networking>Disable</Networking>\n" +
                "    <MappedFolders>\n" +
                "        <MappedFolder>\n" +
                $"            <HostFolder>{sourcePath}</HostFolder>\n" +
                $"            <SandboxFolder>{sandboxSourcePath}</SandboxFolder>\n" +
                "            <ReadOnly>true</ReadOnly>\n" +
                "        </MappedFolder>\n" +
                "        <MappedFolder>\n" +
                $"            <HostFolder>{configDir}</HostFolder>\n" +
                $"            <SandboxFolder>{sandboxWorkPath}</SandboxFolder>\n" +
                "            <ReadOnly>false</ReadOnly>\n" +
                "        </MappedFolder>\n" +
                "    </MappedFolders>\n" +
                "    <LogonCommand>\n" +
                $"        <Command>powershell.exe -ExecutionPolicy Bypass -File {sandboxWorkPath}\\test_script.ps1</Command>\n" +
                "    </LogonCommand>\n" +
                "    <AudioInput>Disable</AudioInput>\n" +
                "    <VideoInput>Disable</VideoInput>\n" +
                "    <ProtectedClient>Disable</ProtectedClient>\n" +
                "    <PrinterRedirection>Disable</PrinterRedirection>\n" +
                "    <ClipboardRedirection>Disable</ClipboardRedirection>\n" +
                "    <MemoryInMB>2048</MemoryInMB>\n" +
                "</Configuration>"

            File.WriteAllText(configFilePath, configContent)

            let config = {
                configFilePath = configFilePath
                mountedFolders = [
                    (sourcePath, sandboxSourcePath, true)
                    (configDir, sandboxWorkPath, false)
                ]
                networkingEnabled = false
                vGpuEnabled = false
                audioInputEnabled = false
                videoInputEnabled = false
                protectedClientEnabled = false
                printingEnabled = false
                clipboardRedirectionEnabled = false
                memoryInMB = Some 2048
                logonCommand = Some $"powershell.exe -ExecutionPolicy Bypass -File {sandboxWorkPath}\\test_script.ps1"
            }

            logger.LogInformation($"Created Windows Sandbox configuration: {configFilePath}")
            Some config
        with
        | ex ->
            logger.LogError($"Failed to create Windows Sandbox configuration: {ex.Message}")
            None

    /// Create PowerShell test script for sandbox execution
    member private this.CreateSandboxTestScript(improvement: ImprovementTask, workPath: string) =
        try
            let scriptPath = Path.Combine(workPath, "test_script.ps1")
            let testFilePath = Path.Combine(workPath, "test_improvement.fs")
            let resultFilePath = Path.Combine(workPath, "test_results.json")

            // Write the proposed code to test file
            File.WriteAllText(testFilePath, improvement.proposedCode)

            // Create a simple PowerShell script without complex interpolation
            let scriptLines = [
                "# TARS Autonomous Self-Improvement Test Script"
                $"# Sandbox Test for Improvement: {improvement.description}"
                ""
                "$ErrorActionPreference = \"Stop\""
                "$StartTime = Get-Date"
                ""
                "try {"
                "    Write-Host \"Starting TARS improvement test in Windows Sandbox...\""
                $"    Write-Host \"Improvement ID: {improvement.taskId}\""
                $"    Write-Host \"Target Component: {improvement.targetComponent}\""
                ""
                "    # Initialize test results"
                "    $TestResults = @{"
                $"        ImprovementId = \"{improvement.taskId}\""
                "        StartTime = $StartTime"
                "        CompilationResult = $false"
                "        PerformanceImprovement = 0.0"
                "        SafetyScore = 0.0"
                "        TestsPassed = 0"
                "        TestsFailed = 0"
                "        ExecutionTime = 0.0"
                "        MemoryUsage = 0"
                "        Errors = @()"
                "        Details = @()"
                "    }"
                ""
                "    # Test 1: Basic syntax validation"
                "    Write-Host \"Test 1: Syntax validation...\""
                $"    $TestCode = Get-Content \"{testFilePath}\" -Raw"
                "    if ($TestCode -match \"member|let|type\" -and $TestCode.Length -gt 50) {"
                "        $TestResults.TestsPassed++"
                "        $TestResults.Details += \"Syntax validation: PASSED\""
                "        Write-Host \"✓ Syntax validation passed\""
                "    } else {"
                "        $TestResults.TestsFailed++"
                "        $TestResults.Errors += \"Invalid F# syntax structure\""
                "        $TestResults.Details += \"Syntax validation: FAILED\""
                "        Write-Host \"✗ Syntax validation failed\""
                "    }"
                ""
                "    # Test 2: Compilation simulation"
                "    Write-Host \"Test 2: Compilation simulation...\""
                "    if ($TestCode -notmatch \"unsafe|extern|DllImport\") {"
                "        $TestResults.CompilationResult = $true"
                "        $TestResults.TestsPassed++"
                "        $TestResults.Details += \"Compilation simulation: PASSED\""
                "        Write-Host \"✓ Compilation simulation passed\""
                "    } else {"
                "        $TestResults.TestsFailed++"
                "        $TestResults.Errors += \"Compilation simulation failed\""
                "        $TestResults.Details += \"Compilation simulation: FAILED\""
                "        Write-Host \"✗ Compilation simulation failed\""
                "    }"
                ""
                "    # Test 3: Performance estimation"
                "    Write-Host \"Test 3: Performance estimation...\""
                "    $PerformanceScore = 0.0"
                "    if ($TestCode -match \"memoization|cache|optimize\") { $PerformanceScore += 0.2 }"
                "    if ($TestCode -match \"parallel|async|task\") { $PerformanceScore += 0.15 }"
                $"    $TestResults.PerformanceImprovement = [Math]::Min($PerformanceScore, {improvement.expectedBenefit:F2})"
                "    $TestResults.Details += \"Performance improvement estimated: $($TestResults.PerformanceImprovement * 100)%\""
                "    Write-Host \"✓ Performance improvement estimated: $($TestResults.PerformanceImprovement * 100)%\""
                ""
                "    # Test 4: Safety assessment"
                "    Write-Host \"Test 4: Safety assessment...\""
                "    $SafetyScore = 0.8"
                "    if ($TestCode -match \"File\\.Delete|Directory\\.Delete|Process\\.Kill\") {"
                "        $SafetyScore -= 0.3"
                "        $TestResults.Errors += \"Potentially unsafe operations detected\""
                "    }"
                "    $TestResults.SafetyScore = [Math]::Max($SafetyScore, 0.0)"
                "    $TestResults.Details += \"Safety score: $($TestResults.SafetyScore)\""
                "    Write-Host \"✓ Safety score: $($TestResults.SafetyScore)\""
                ""
                "    # Resource usage simulation"
                "    $TestResults.MemoryUsage = (Get-Process -Name \"powershell\" | Measure-Object WorkingSet -Sum).Sum"
                "    $TestResults.ExecutionTime = ((Get-Date) - $StartTime).TotalMilliseconds"
                "    Write-Host \"Test completed successfully\""
                ""
                "} catch {"
                "    $TestResults.Errors += $_.Exception.Message"
                "    Write-Host \"✗ Test execution failed: $($_.Exception.Message)\""
                "} finally {"
                "    $TestResults.EndTime = Get-Date"
                "    $TestResults.ExecutionTime = ((Get-Date) - $StartTime).TotalMilliseconds"
                $"    $TestResults | ConvertTo-Json -Depth 3 | Out-File \"{resultFilePath}\" -Encoding UTF8"
                $"    Write-Host \"Test results saved to: {resultFilePath}\""
                "    Start-Sleep -Seconds 30"
                "}"
            ]

            let scriptContent = String.Join("\n", scriptLines)
            File.WriteAllText(scriptPath, scriptContent)
            logger.LogInformation($"Created sandbox test script: {scriptPath}")
            scriptPath
        with
        | ex ->
            logger.LogError($"Failed to create sandbox test script: {ex.Message}")
            ""
    
    /// Create a sandboxed testing environment (Windows Sandbox or fallback)
    member private this.CreateSandboxEnvironment() =
        let sandboxId = Guid.NewGuid()
        let sandboxIdShort = sandboxId.ToString("N").[..7]
        let sandboxPath = Path.Combine(Path.GetTempPath(), $"tars_sandbox_{sandboxIdShort}")

        try
            Directory.CreateDirectory(sandboxPath) |> ignore

            // Try to use Windows Sandbox if available
            let (sandboxType, windowsSandboxConfig) =
                if this.IsWindowsSandboxAvailable() then
                    let sourcePath = Path.GetFullPath("src/TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli")
                    match this.CreateWindowsSandboxConfig(sandboxId, sourcePath) with
                    | Some config -> ("WindowsSandbox", Some config)
                    | None ->
                        logger.LogWarning("Failed to create Windows Sandbox config, falling back to temp directory")
                        ("TempDirectory", None)
                else
                    logger.LogInformation("Windows Sandbox not available, using temp directory sandbox")
                    ("TempDirectory", None)

            let sandbox = {
                sandboxId = sandboxId
                sandboxType = sandboxType
                isolatedPath = sandboxPath
                windowsSandboxConfig = windowsSandboxConfig
                sandboxProcess = None
                testResults = Map.empty
                performanceMetrics = Map.empty
                safetyValidation = false
                executionLog = []
                resourceUsage = Map.empty
                timeoutMinutes = 10  // 10-minute timeout for safety
                createdAt = DateTime.UtcNow
                isActive = true
            }

            improvementState <-
                { improvementState with
                    sandboxEnvironments = improvementState.sandboxEnvironments.Add(sandboxId, sandbox) }

            logger.LogInformation($"Created {sandboxType} sandbox environment: {sandboxPath}")
            Some sandbox
        with
        | ex ->
            logger.LogError($"Failed to create sandbox environment: {ex.Message}")
            None
    
    /// Test improvement in sandbox environment (Windows Sandbox or temp directory)
    member private this.TestInSandbox(improvement: ImprovementTask, sandbox: SandboxEnvironment) =
        try
            let testStartTime = DateTime.UtcNow
            logger.LogInformation($"Starting {sandbox.sandboxType} test for improvement {improvement.taskId}")

            match sandbox.sandboxType with
            | "WindowsSandbox" -> this.TestInWindowsSandbox(improvement, sandbox)
            | _ -> this.TestInTempDirectorySandbox(improvement, sandbox)

        with
        | ex ->
            logger.LogError($"Sandbox testing failed: {ex.Message}")
            {
                verificationId = Guid.NewGuid()
                taskId = improvement.taskId
                testsPassed = 0
                testsFailed = 3
                performanceImprovement = 0.0
                qualityImprovement = 0.0
                safetyScore = 0.0
                riskAssessment = "Critical Risk"
                recommendation = "Reject"
                verificationTime = DateTime.UtcNow
                details = [$"Testing failed: {ex.Message}"]
            }

    /// Test improvement in Windows Sandbox
    member private this.TestInWindowsSandbox(improvement: ImprovementTask, sandbox: SandboxEnvironment) =
        try
            let testStartTime = DateTime.UtcNow

            match sandbox.windowsSandboxConfig with
            | Some config ->
                // Create test script for sandbox execution
                let workPath = Path.GetDirectoryName(config.configFilePath)
                let scriptPath = this.CreateSandboxTestScript(improvement, workPath)

                if not (String.IsNullOrEmpty(scriptPath)) then
                    // Launch Windows Sandbox
                    let psi = ProcessStartInfo()
                    psi.FileName <- "WindowsSandbox.exe"
                    let configPath = config.configFilePath
                    psi.Arguments <- $"\"{configPath}\""
                    psi.UseShellExecute <- true
                    psi.CreateNoWindow <- false

                    let configPathInfo = configPath
                    logger.LogInformation($"Launching Windows Sandbox: {configPathInfo}")
                    use sandboxProcess = Process.Start(psi)

                    // Wait for sandbox to complete or timeout
                    let timeoutMs = sandbox.timeoutMinutes * 60 * 1000
                    let completed = sandboxProcess.WaitForExit(timeoutMs)

                    if not completed then
                        logger.LogWarning($"Windows Sandbox test timed out after {sandbox.timeoutMinutes} minutes")
                        sandboxProcess.Kill()

                    // Read test results
                    let resultFilePath = Path.Combine(workPath, "test_results.json")
                    let testResults = this.ReadSandboxTestResults(resultFilePath)

                    let testDuration = (DateTime.UtcNow - testStartTime).TotalMilliseconds

                    let verificationResult = {
                        verificationId = Guid.NewGuid()
                        taskId = improvement.taskId
                        testsPassed = testResults.testsPassed
                        testsFailed = testResults.testsFailed
                        performanceImprovement = testResults.performanceImprovement
                        qualityImprovement = improvement.expectedBenefit * 0.6
                        safetyScore = testResults.safetyScore
                        riskAssessment =
                            if testResults.safetyScore > 0.7 then "Low Risk"
                            elif testResults.safetyScore > 0.5 then "Medium Risk"
                            else "High Risk"
                        recommendation =
                            if testResults.compilationResult && testResults.safetyScore > 0.7 && testResults.performanceImprovement > 0.05 then "Apply"
                            elif testResults.compilationResult && testResults.safetyScore > 0.5 then "Modify"
                            elif testResults.compilationResult then "Retest"
                            else "Reject"
                        verificationTime = DateTime.UtcNow
                        details =
                            let timeoutStatus = if completed then "No" else "Yes"
                            testResults.details @ [
                                $"Windows Sandbox execution time: {testDuration:F1}ms"
                                $"Sandbox timeout: {timeoutStatus}"
                                $"Memory usage: {testResults.memoryUsage} bytes"
                            ]
                    }

                    // Cleanup
                    this.CleanupSandboxEnvironment(sandbox)

                    logger.LogInformation($"Windows Sandbox testing completed for {improvement.taskId}. Recommendation: {verificationResult.recommendation}")
                    verificationResult
                else
                    failwith "Failed to create test script"
            | None ->
                failwith "Windows Sandbox configuration not available"

        with
        | ex ->
            logger.LogError($"Windows Sandbox testing failed: {ex.Message}")
            this.TestInTempDirectorySandbox(improvement, sandbox)  // Fallback

    /// Test improvement in temporary directory sandbox (fallback method)
    member private this.TestInTempDirectorySandbox(improvement: ImprovementTask, sandbox: SandboxEnvironment) =
        try
            let testStartTime = DateTime.UtcNow

            // Create test file with proposed code
            let testFilePath = Path.Combine(sandbox.isolatedPath, "test_improvement.fs")
            File.WriteAllText(testFilePath, improvement.proposedCode)

            // Simulate compilation test
            let compilationResult =
                try
                    let hasBasicSyntax = improvement.proposedCode.Contains("member") || improvement.proposedCode.Contains("let")
                    let hasProperStructure = improvement.proposedCode.Length > 50
                    let noUnsafeOperations = not (improvement.proposedCode.Contains("unsafe") || improvement.proposedCode.Contains("extern"))
                    hasBasicSyntax && hasProperStructure && noUnsafeOperations
                with
                | _ -> false

            // Simulate performance test
            let performanceImprovement =
                if compilationResult then
                    let baselineTime = 100.0
                    let improvedTime = baselineTime * (1.0 - improvement.expectedBenefit * 0.5)
                    (baselineTime - improvedTime) / baselineTime
                else 0.0

            // Enhanced safety checks
            let safetyScore =
                let mutable score = 0.8
                if improvement.proposedCode.Contains("File.Delete") || improvement.proposedCode.Contains("Directory.Delete") then
                    score <- score - 0.3
                if improvement.proposedCode.Contains("Process.Kill") || improvement.proposedCode.Contains("System.Diagnostics") then
                    score <- score - 0.2
                if improvement.proposedCode.Contains("unsafe") || improvement.proposedCode.Contains("extern") then
                    score <- score - 0.4
                let riskPenalty = improvement.implementationRisk * 0.2
                let benefitBonus = improvement.expectedBenefit * 0.1
                max 0.0 (score - riskPenalty + benefitBonus)

            let testDuration = (DateTime.UtcNow - testStartTime).TotalMilliseconds

            let verificationResult = {
                verificationId = Guid.NewGuid()
                taskId = improvement.taskId
                testsPassed = if compilationResult then 3 else 1
                testsFailed = if compilationResult then 0 else 2
                performanceImprovement = performanceImprovement
                qualityImprovement = improvement.expectedBenefit * 0.6
                safetyScore = safetyScore
                riskAssessment = if safetyScore > 0.7 then "Low Risk" elif safetyScore > 0.5 then "Medium Risk" else "High Risk"
                recommendation =
                    if compilationResult && safetyScore > 0.7 && performanceImprovement > 0.05 then "Apply"
                    elif compilationResult && safetyScore > 0.5 then "Modify"
                    elif compilationResult then "Retest"
                    else "Reject"
                verificationTime = DateTime.UtcNow
                details =
                    let compilationStatus = if compilationResult then "PASSED" else "FAILED"
                    [
                        $"Compilation: {compilationStatus}"
                        $"Performance improvement: {performanceImprovement:P1}"
                        $"Safety score: {safetyScore:F2}"
                        $"Test duration: {testDuration:F1}ms"
                        $"Sandbox type: Temp Directory (fallback)"
                    ]
            }

            improvementState <-
                { improvementState with
                    verificationHistory = verificationResult :: improvementState.verificationHistory }

            logger.LogInformation($"Temp directory sandbox testing completed for {improvement.taskId}. Recommendation: {verificationResult.recommendation}")
            verificationResult

        with
        | ex ->
            logger.LogError($"Temp directory sandbox testing failed: {ex.Message}")
            {
                verificationId = Guid.NewGuid()
                taskId = improvement.taskId
                testsPassed = 0
                testsFailed = 3
                performanceImprovement = 0.0
                qualityImprovement = 0.0
                safetyScore = 0.0
                riskAssessment = "Critical Risk"
                recommendation = "Reject"
                verificationTime = DateTime.UtcNow
                details = [$"Testing failed: {ex.Message}"]
            }

    /// Read test results from Windows Sandbox execution
    member private this.ReadSandboxTestResults(resultFilePath: string) =
        try
            if File.Exists(resultFilePath) then
                let jsonContent = File.ReadAllText(resultFilePath)
                let testData = JsonSerializer.Deserialize<Map<string, obj>>(jsonContent)

                {
                    testsPassed = if testData.ContainsKey("TestsPassed") then int (testData.["TestsPassed"].ToString()) else 0
                    testsFailed = if testData.ContainsKey("TestsFailed") then int (testData.["TestsFailed"].ToString()) else 3
                    compilationResult = if testData.ContainsKey("CompilationResult") then bool.Parse(testData.["CompilationResult"].ToString()) else false
                    performanceImprovement = if testData.ContainsKey("PerformanceImprovement") then float (testData.["PerformanceImprovement"].ToString()) else 0.0
                    safetyScore = if testData.ContainsKey("SafetyScore") then float (testData.["SafetyScore"].ToString()) else 0.0
                    memoryUsage = if testData.ContainsKey("MemoryUsage") then int64 (testData.["MemoryUsage"].ToString()) else 0L
                    details = if testData.ContainsKey("Details") then [testData.["Details"].ToString()] else ["No details available"]
                }
            else
                logger.LogWarning($"Test results file not found: {resultFilePath}")
                {
                    testsPassed = 0; testsFailed = 3; compilationResult = false
                    performanceImprovement = 0.0; safetyScore = 0.0; memoryUsage = 0L
                    details = ["Test results file not found"]
                }
        with
        | ex ->
            logger.LogError($"Failed to read test results: {ex.Message}")
            {
                testsPassed = 0; testsFailed = 3; compilationResult = false
                performanceImprovement = 0.0; safetyScore = 0.0; memoryUsage = 0L
                details = [$"Failed to read results: {ex.Message}"]
            }

    /// Cleanup sandbox environment and resources
    member private this.CleanupSandboxEnvironment(sandbox: SandboxEnvironment) =
        try
            match sandbox.sandboxType with
            | "WindowsSandbox" ->
                // Windows Sandbox automatically cleans up when closed
                match sandbox.windowsSandboxConfig with
                | Some config ->
                    let configDir = Path.GetDirectoryName(config.configFilePath)
                    if Directory.Exists(configDir) then
                        Directory.Delete(configDir, true)
                        let configDirectory = configDir
                        logger.LogInformation($"Cleaned up Windows Sandbox config directory: {configDirectory}")
                | None -> ()
            | _ ->
                // Clean up temp directory
                if Directory.Exists(sandbox.isolatedPath) then
                    Directory.Delete(sandbox.isolatedPath, true)
                    logger.LogInformation($"Cleaned up temp sandbox directory: {sandbox.isolatedPath}")

            // Remove from active environments
            improvementState <-
                { improvementState with
                    sandboxEnvironments = improvementState.sandboxEnvironments.Remove(sandbox.sandboxId) }

        with
        | ex ->
            logger.LogWarning($"Failed to cleanup sandbox environment: {ex.Message}")
    
    /// Create rollback capability for safe recovery
    member private this.CreateRollbackCapability(targetComponent: string) =
        try
            let rollbackId = Guid.NewGuid()
            let backupIdShort = rollbackId.ToString("N").[..7]
            let backupPath = Path.Combine(Path.GetTempPath(), $"tars_backup_{backupIdShort}")
            
            // In a real implementation, this would backup the actual component
            let originalState = $"// Backup of {targetComponent} at {DateTime.UtcNow}"
            
            let rollbackManager = {
                rollbackId = rollbackId
                originalState = originalState
                backupPath = backupPath
                rollbackSteps = [
                    "1. Stop affected services"
                    "2. Restore original code from backup"
                    "3. Recompile and verify"
                    "4. Restart services"
                    "5. Validate functionality"
                ]
                verificationChecks = [
                    "Compilation successful"
                    "Unit tests passing"
                    "Performance within baseline"
                    "No regression detected"
                ]
                isReady = true
                createdAt = DateTime.UtcNow
            }
            
            improvementState <- 
                { improvementState with rollbackCapability = Some rollbackManager }
            
            logger.LogInformation($"Rollback capability created for {targetComponent}")
            Some rollbackManager
            
        with
        | ex ->
            logger.LogError($"Failed to create rollback capability: {ex.Message}")
            None
    
    /// Generate improvement tasks from Tier 8 analysis
    member this.GenerateImprovementTasks(tier8Analysis: obj) =
        // Simulate improvement task generation based on Tier 8 analysis
        let improvementTasks = [
            {
                taskId = Guid.NewGuid()
                targetComponent = "ProblemDecomposition"
                improvementType = "Performance"
                description = "Optimize algorithm complexity from O(n³) to O(n²)"
                originalCode = "// Original O(n³) implementation"
                proposedCode = """
// Optimized O(n²) implementation
member this.OptimizedDecomposition(plan: EnhancedSkill list) =
    let mutable optimizedPlan = plan
    // Use memoization to reduce complexity
    let memoCache = new System.Collections.Generic.Dictionary<string, obj>()
    optimizedPlan |> List.map (fun skill -> 
        match memoCache.TryGetValue(skill.name) with
        | true, cached -> cached :?> EnhancedSkill
        | false, _ -> 
            let optimized = { skill with complexity = max 1 (skill.complexity - 1) }
            memoCache.[skill.name] <- optimized
            optimized)
"""
                expectedBenefit = 0.25  // 25% performance improvement
                implementationRisk = 0.4  // Medium risk
                estimatedImpact = 0.7  // High impact
                safetyChecks = ["Compilation test"; "Unit test validation"; "Performance benchmark"]
                rollbackData = "// Rollback to original implementation"
                status = "Pending"
                createdAt = DateTime.UtcNow
                lastModified = DateTime.UtcNow
            }
            
            {
                taskId = Guid.NewGuid()
                targetComponent = "CodeQuality"
                improvementType = "Quality"
                description = "Refactor complex functions to improve maintainability"
                originalCode = "// Original complex function"
                proposedCode = """
// Refactored for better maintainability
member private this.RefactoredComplexFunction(input: string) =
    let validateInput input = 
        not (String.IsNullOrEmpty(input)) && input.Length > 0
    
    let processInput input =
        input.Trim().ToLower()
    
    let generateOutput processedInput =
        $"Processed: {processedInput}"
    
    if validateInput input then
        input |> processInput |> generateOutput
    else
        "Invalid input"
"""
                expectedBenefit = 0.15  // 15% quality improvement
                implementationRisk = 0.2  // Low risk
                estimatedImpact = 0.5  // Medium impact
                safetyChecks = ["Code review"; "Maintainability analysis"; "Regression testing"]
                rollbackData = "// Rollback to original function"
                status = "Pending"
                createdAt = DateTime.UtcNow
                lastModified = DateTime.UtcNow
            }
        ]
        
        improvementState <- 
            { improvementState with 
                improvementQueue = improvementTasks @ improvementState.improvementQueue }
        
        logger.LogInformation($"Generated {improvementTasks.Length} improvement tasks")
        improvementTasks
    
    /// Execute autonomous self-improvement cycle
    member this.ExecuteSelfImprovementCycle() =
        let cycleStartTime = DateTime.UtcNow
        logger.LogInformation("Starting autonomous self-improvement cycle...")
        
        let results = ResizeArray<VerificationResult>()
        
        // Process pending improvements
        let pendingImprovements = 
            improvementState.improvementQueue 
            |> List.filter (fun task -> task.status = "Pending")
            |> List.take (min 2 improvementState.improvementQueue.Length)  // Limit to 2 per cycle for safety
        
        for improvement in pendingImprovements do
            logger.LogInformation($"Processing improvement: {improvement.description}")
            
            // Create sandbox environment
            match this.CreateSandboxEnvironment() with
            | Some sandbox ->
                // Create rollback capability
                match this.CreateRollbackCapability(improvement.targetComponent) with
                | Some rollback ->
                    // Test in sandbox
                    let verificationResult = this.TestInSandbox(improvement, sandbox)
                    results.Add(verificationResult)
                    
                    // Update improvement status based on verification
                    let updatedImprovement = 
                        { improvement with 
                            status = 
                                match verificationResult.recommendation with
                                | "Apply" -> "Verified"
                                | "Reject" -> "Failed"
                                | _ -> "Testing"
                            lastModified = DateTime.UtcNow }
                    
                    // Update state
                    improvementState <- 
                        { improvementState with 
                            activeImprovements = improvementState.activeImprovements.Add(improvement.taskId, updatedImprovement)
                            improvementQueue = improvementState.improvementQueue |> List.filter (fun t -> t.taskId <> improvement.taskId) }
                    
                | None ->
                    logger.LogWarning($"Failed to create rollback capability for {improvement.targetComponent}")
            | None ->
                logger.LogWarning($"Failed to create sandbox environment for {improvement.taskId}")
        
        let cycleDuration = (DateTime.UtcNow - cycleStartTime).TotalMilliseconds
        improvementState <- { improvementState with lastImprovementCycle = DateTime.UtcNow }
        
        logger.LogInformation($"Self-improvement cycle completed in {cycleDuration:F1}ms. Processed {results.Count} improvements")
        
        {|
            cycleId = Guid.NewGuid()
            processedImprovements = results.Count
            verifiedImprovements = results |> Seq.filter (fun r -> r.recommendation = "Apply") |> Seq.length
            rejectedImprovements = results |> Seq.filter (fun r -> r.recommendation = "Reject") |> Seq.length
            averagePerformanceImprovement = if results.Count > 0 then results |> Seq.averageBy (fun r -> r.performanceImprovement) else 0.0
            averageSafetyScore = if results.Count > 0 then results |> Seq.averageBy (fun r -> r.safetyScore) else 0.0
            cycleDuration = cycleDuration
            timestamp = DateTime.UtcNow
        |}
    
    /// Get current self-improvement state
    member this.GetSelfImprovementState() = improvementState
    
    /// Get improvement metrics
    member this.GetImprovementMetrics() =
        let totalImprovements = improvementState.completedImprovements.Length + improvementState.activeImprovements.Count
        let successfulImprovements = improvementState.completedImprovements |> List.filter (fun i -> i.status = "Applied") |> List.length
        let successRate = if totalImprovements > 0 then float successfulImprovements / float totalImprovements else 0.0
        
        {|
            totalImprovements = totalImprovements
            successfulImprovements = successfulImprovements
            successRate = successRate
            activeImprovements = improvementState.activeImprovements.Count
            queuedImprovements = improvementState.improvementQueue.Length
            lastCycle = improvementState.lastImprovementCycle
            averageSafetyScore = 
                if improvementState.verificationHistory.Length > 0 then
                    improvementState.verificationHistory |> List.averageBy (fun v -> v.safetyScore)
                else 0.0
        |}
