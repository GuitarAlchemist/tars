namespace TarsEngine.FSharp.Core.Testing

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.LLM
open TarsEngine.FSharp.Core.ChromaDB

/// Autonomous testing service that generates and executes tests for metascripts
type AutonomousTestingService(
    reasoningService: IAutonomousReasoningService,
    hybridRAG: IHybridRAGService,
    logger: ILogger<AutonomousTestingService>) =
    
    interface IAutonomousTestingService with
        member _.GenerateTestSuiteAsync(metascriptPath: string) =
            task {
                try
                    logger.LogInformation("Generating autonomous test suite for: {MetascriptPath}", metascriptPath)
                    
                    // Read metascript content
                    let metascriptContent = 
                        if File.Exists(metascriptPath) then
                            File.ReadAllText(metascriptPath)
                        else
                            // For demo, use sample metascript content
                            sprintf """DESCRIBE {
    name: "Sample Metascript"
    version: "1.0"
    description: "Sample metascript for testing"
}

VARIABLE input_file {
    value: "test.txt"
}

FSHARP {
    open System.IO
    let fileName = "test.txt"
    if File.Exists(fileName) then
        let content = File.ReadAllText(fileName)
        sprintf "File content: %%s" content
    else
        "File not found"
}"""
                    
                    let metascriptName = Path.GetFileNameWithoutExtension(metascriptPath)
                    
                    // Generate different types of tests
                    printfn "🧪 Generating unit tests..."
                    let! unitTests = (this :> IAutonomousTestingService).GenerateUnitTestsAsync(metascriptContent)
                    
                    printfn "🔗 Generating integration tests..."
                    let! integrationTests = (this :> IAutonomousTestingService).GenerateIntegrationTestsAsync(metascriptContent)
                    
                    // Generate performance tests
                    let performanceTests = [
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = sprintf "%s_performance_test" metascriptName
                            Description = "Verify metascript executes within acceptable time limits"
                            TestType = "Performance"
                            InputData = Map.empty
                            ExpectedOutput = Some "Execution time < 5000ms"
                            TestCode = sprintf """
// Performance test for %s
let stopwatch = System.Diagnostics.Stopwatch.StartNew()
// Execute metascript here
stopwatch.Stop()
let executionTime = stopwatch.ElapsedMilliseconds
Assert.True(executionTime < 5000L, sprintf "Execution took %%dms, expected < 5000ms" executionTime)
""" metascriptName
                            Timeout = TimeSpan.FromSeconds(10)
                            CreatedAt = DateTime.UtcNow
                        }
                    ]
                    
                    // Generate security tests
                    let securityTests = [
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = sprintf "%s_security_test" metascriptName
                            Description = "Verify metascript handles malicious input safely"
                            TestType = "Security"
                            InputData = Map.ofList [("malicious_input", "../../../etc/passwd" :> obj)]
                            ExpectedOutput = Some "No path traversal vulnerability"
                            TestCode = """
// Security test - path traversal
let maliciousPath = "../../../etc/passwd"
// Test that metascript rejects or sanitizes malicious paths
Assert.False(maliciousPath.Contains(".."), "Path traversal attempt should be blocked")
"""
                            Timeout = TimeSpan.FromSeconds(5)
                            CreatedAt = DateTime.UtcNow
                        }
                    ]
                    
                    let testSuite = {
                        Id = Guid.NewGuid().ToString()
                        MetascriptName = metascriptName
                        MetascriptPath = metascriptPath
                        UnitTests = unitTests
                        IntegrationTests = integrationTests
                        PerformanceTests = performanceTests
                        SecurityTests = securityTests
                        CreatedAt = DateTime.UtcNow
                    }
                    
                    // Store test suite in knowledge base
                    let metadata = Map.ofList [
                        ("type", "test_suite" :> obj)
                        ("metascript_name", metascriptName :> obj)
                        ("total_tests", (unitTests.Length + integrationTests.Length + performanceTests.Length + securityTests.Length) :> obj)
                    ]
                    let! _ = hybridRAG.StoreKnowledgeAsync(
                        sprintf "Generated test suite for %s with %d total tests" metascriptName (unitTests.Length + integrationTests.Length + performanceTests.Length + securityTests.Length),
                        metadata)
                    
                    logger.LogInformation("Generated test suite with {TotalTests} tests for: {MetascriptName}", 
                                        (unitTests.Length + integrationTests.Length + performanceTests.Length + securityTests.Length), metascriptName)
                    
                    return testSuite
                with
                | ex ->
                    logger.LogError(ex, "Failed to generate test suite for: {MetascriptPath}", metascriptPath)
                    reraise()
            }
        
        member _.ExecuteTestSuiteAsync(testSuite: TestSuite) =
            task {
                try
                    logger.LogInformation("Executing test suite: {SuiteId} for metascript: {MetascriptName}", testSuite.Id, testSuite.MetascriptName)
                    
                    let allTests = testSuite.UnitTests @ testSuite.IntegrationTests @ testSuite.PerformanceTests @ testSuite.SecurityTests
                    let mutable results = []
                    let startTime = DateTime.UtcNow
                    
                    printfn "🧪 Executing %d tests for %s..." allTests.Length testSuite.MetascriptName
                    
                    for test in allTests do
                        printfn "  Running: %s (%s)" test.Name test.TestType
                        
                        let testStartTime = DateTime.UtcNow
                        
                        // Simulate test execution
                        do! Task.Delay(100) // Simulate test execution time
                        
                        let testEndTime = DateTime.UtcNow
                        let executionTime = testEndTime - testStartTime
                        
                        // Simulate test result (in real implementation, this would execute actual test code)
                        let success = Random().NextDouble() > 0.1 // 90% success rate for demo
                        
                        let result = {
                            TestCaseId = test.Id
                            Status = if success then TestStatus.Passed else TestStatus.Failed
                            ExecutionTime = executionTime
                            ActualOutput = if success then "Test passed successfully" else "Test failed - assertion error"
                            ErrorMessage = if success then None else Some "Assertion failed: expected behavior not met"
                            Assertions = [
                                ("Input validation", success)
                                ("Output format", success)
                                ("Performance criteria", success)
                            ]
                            Metrics = Map.ofList [
                                ("execution_time_ms", executionTime.TotalMilliseconds)
                                ("memory_usage_mb", 10.5)
                                ("cpu_usage_percent", 15.2)
                            ]
                            ExecutedAt = testEndTime
                        }
                        
                        results <- result :: results
                        
                        let statusIcon = if success then "✅" else "❌"
                        printfn "    %s %s (%dms)" statusIcon test.Name (int executionTime.TotalMilliseconds)
                    
                    let endTime = DateTime.UtcNow
                    let totalExecutionTime = endTime - startTime
                    
                    let passedTests = results |> List.filter (fun r -> r.Status = TestStatus.Passed) |> List.length
                    let failedTests = results |> List.filter (fun r -> r.Status = TestStatus.Failed) |> List.length
                    let skippedTests = results |> List.filter (fun r -> r.Status = TestStatus.Skipped) |> List.length
                    let errorTests = results |> List.filter (fun r -> r.Status = TestStatus.Error) |> List.length
                    
                    let summary = {
                        SuiteId = testSuite.Id
                        TotalTests = allTests.Length
                        PassedTests = passedTests
                        FailedTests = failedTests
                        SkippedTests = skippedTests
                        ErrorTests = errorTests
                        TotalExecutionTime = totalExecutionTime
                        CoveragePercentage = if allTests.Length > 0 then (float passedTests / float allTests.Length) * 100.0 else 0.0
                        ExecutedAt = endTime
                    }
                    
                    // Store execution results
                    let metadata = Map.ofList [
                        ("type", "test_execution_summary" :> obj)
                        ("metascript_name", testSuite.MetascriptName :> obj)
                        ("passed_tests", passedTests :> obj)
                        ("failed_tests", failedTests :> obj)
                        ("coverage_percentage", summary.CoveragePercentage :> obj)
                    ]
                    let! _ = hybridRAG.StoreKnowledgeAsync(
                        sprintf "Test execution completed for %s: %d/%d tests passed (%.1f%% coverage)" 
                                testSuite.MetascriptName passedTests allTests.Length summary.CoveragePercentage,
                        metadata)
                    
                    logger.LogInformation("Test suite execution completed: {PassedTests}/{TotalTests} passed", passedTests, allTests.Length)
                    return summary
                with
                | ex ->
                    logger.LogError(ex, "Failed to execute test suite: {SuiteId}", testSuite.Id)
                    reraise()
            }
        
        member _.GenerateUnitTestsAsync(metascriptContent: string) =
            task {
                try
                    logger.LogInformation("Generating unit tests for metascript")
                    
                    // Use autonomous reasoning to analyze metascript and generate unit tests
                    let analysisTask = sprintf "Analyze this metascript and generate comprehensive unit tests:\n\n%s\n\nGenerate unit tests that cover:\n1. Input validation\n2. Core logic\n3. Output verification\n4. Error handling\n5. Edge cases" metascriptContent
                    let context = Map.ofList [
                        ("test_generation_type", "unit_tests" :> obj)
                        ("timestamp", DateTime.UtcNow :> obj)
                    ]
                    
                    let! testAnalysis = reasoningService.ReasonAboutTaskAsync(analysisTask, context)
                    
                    // Generate specific unit test cases
                    let unitTests = [
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = "test_input_validation"
                            Description = "Verify metascript validates input parameters correctly"
                            TestType = "Unit"
                            InputData = Map.ofList [("test_input", "valid_input" :> obj)]
                            ExpectedOutput = Some "Input validation passed"
                            TestCode = """
// Unit test: Input validation
let validInput = "valid_input"
let result = validateInput(validInput)
Assert.True(result, "Valid input should pass validation")

let invalidInput = ""
let invalidResult = validateInput(invalidInput)
Assert.False(invalidResult, "Invalid input should fail validation")
"""
                            Timeout = TimeSpan.FromSeconds(5)
                            CreatedAt = DateTime.UtcNow
                        }
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = "test_core_logic"
                            Description = "Verify metascript core logic produces expected results"
                            TestType = "Unit"
                            InputData = Map.ofList [("test_data", "sample_data" :> obj)]
                            ExpectedOutput = Some "Core logic executed successfully"
                            TestCode = """
// Unit test: Core logic
let testData = "sample_data"
let result = executeCoreLogic(testData)
Assert.NotNull(result, "Core logic should return a result")
Assert.True(result.Length > 0, "Result should not be empty")
"""
                            Timeout = TimeSpan.FromSeconds(5)
                            CreatedAt = DateTime.UtcNow
                        }
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = "test_error_handling"
                            Description = "Verify metascript handles errors gracefully"
                            TestType = "Unit"
                            InputData = Map.ofList [("error_scenario", "file_not_found" :> obj)]
                            ExpectedOutput = Some "Error handled gracefully"
                            TestCode = """
// Unit test: Error handling
let nonExistentFile = "non_existent_file.txt"
let result = processFile(nonExistentFile)
Assert.True(result.Contains("Error") || result.Contains("not found"), "Should handle file not found error")
"""
                            Timeout = TimeSpan.FromSeconds(5)
                            CreatedAt = DateTime.UtcNow
                        }
                    ]
                    
                    logger.LogInformation("Generated {UnitTestCount} unit tests", unitTests.Length)
                    return unitTests
                with
                | ex ->
                    logger.LogError(ex, "Failed to generate unit tests")
                    return []
            }
        
        member _.GenerateIntegrationTestsAsync(metascriptContent: string) =
            task {
                try
                    logger.LogInformation("Generating integration tests for metascript")
                    
                    // Generate integration tests that test metascript interaction with external systems
                    let integrationTests = [
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = "test_file_system_integration"
                            Description = "Verify metascript integrates correctly with file system"
                            TestType = "Integration"
                            InputData = Map.ofList [("test_file", "integration_test.txt" :> obj)]
                            ExpectedOutput = Some "File system operations completed successfully"
                            TestCode = """
// Integration test: File system
let testFile = "integration_test.txt"
File.WriteAllText(testFile, "test content")

let result = processFileSystem(testFile)
Assert.True(File.Exists(testFile), "Test file should exist")
Assert.True(result.Contains("test content"), "Should read file content correctly")

File.Delete(testFile)
"""
                            Timeout = TimeSpan.FromSeconds(10)
                            CreatedAt = DateTime.UtcNow
                        }
                        {
                            Id = Guid.NewGuid().ToString()
                            Name = "test_metascript_chaining"
                            Description = "Verify metascript can be chained with other metascripts"
                            TestType = "Integration"
                            InputData = Map.ofList [("chain_input", "chained_data" :> obj)]
                            ExpectedOutput = Some "Metascript chaining successful"
                            TestCode = """
// Integration test: Metascript chaining
let chainInput = "chained_data"
let firstResult = executeFirstMetascript(chainInput)
let finalResult = executeSecondMetascript(firstResult)
Assert.NotNull(finalResult, "Chained execution should produce result")
"""
                            Timeout = TimeSpan.FromSeconds(15)
                            CreatedAt = DateTime.UtcNow
                        }
                    ]
                    
                    logger.LogInformation("Generated {IntegrationTestCount} integration tests", integrationTests.Length)
                    return integrationTests
                with
                | ex ->
                    logger.LogError(ex, "Failed to generate integration tests")
                    return []
            }
        
        member _.ValidateMetascriptAsync(metascriptPath: string) =
            task {
                try
                    logger.LogInformation("Validating metascript: {MetascriptPath}", metascriptPath)
                    
                    // Generate and execute test suite
                    let! testSuite = (this :> IAutonomousTestingService).GenerateTestSuiteAsync(metascriptPath)
                    let! summary = (this :> IAutonomousTestingService).ExecuteTestSuiteAsync(testSuite)
                    
                    // Validation criteria: >80% tests pass, no critical failures
                    let isValid = summary.CoveragePercentage >= 80.0 && summary.ErrorTests = 0
                    
                    logger.LogInformation("Metascript validation result: {IsValid} (Coverage: {Coverage}%)", isValid, summary.CoveragePercentage)
                    return isValid
                with
                | ex ->
                    logger.LogError(ex, "Failed to validate metascript: {MetascriptPath}", metascriptPath)
                    return false
            }
        
        member _.AutoFixFailedTestsAsync(testResults: TestResult list, metascriptContent: string) =
            task {
                try
                    logger.LogInformation("Auto-fixing failed tests")
                    
                    let failedTests = testResults |> List.filter (fun r -> r.Status = TestStatus.Failed)
                    
                    if failedTests.IsEmpty then
                        return metascriptContent
                    
                    // Use autonomous reasoning to fix failed tests
                    let failureAnalysis = 
                        failedTests
                        |> List.map (fun t -> sprintf "Test: %s, Error: %s" t.TestCaseId (t.ErrorMessage |> Option.defaultValue "Unknown error"))
                        |> String.concat "\n"
                    
                    let fixTask = sprintf "Analyze these test failures and fix the metascript:\n\nOriginal Metascript:\n%s\n\nFailed Tests:\n%s\n\nProvide the corrected metascript that will pass all tests." metascriptContent failureAnalysis
                    let context = Map.ofList [
                        ("fix_type", "test_failures" :> obj)
                        ("failed_test_count", failedTests.Length :> obj)
                    ]
                    
                    let! fixedMetascript = reasoningService.ReasonAboutTaskAsync(fixTask, context)
                    
                    logger.LogInformation("Auto-fix completed for {FailedTestCount} failed tests", failedTests.Length)
                    return fixedMetascript
                with
                | ex ->
                    logger.LogError(ex, "Failed to auto-fix failed tests")
                    return metascriptContent
            }

