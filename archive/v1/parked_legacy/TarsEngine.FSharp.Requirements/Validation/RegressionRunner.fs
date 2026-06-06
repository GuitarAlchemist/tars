namespace TarsEngine.FSharp.Requirements.Validation

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Requirements.Models
open TarsEngine.FSharp.Requirements.Repository

/// <summary>
/// Regression test run configuration
/// </summary>
type RegressionRunConfig = {
    Name: string
    Description: string
    IncludeRequirementTypes: RequirementType list option
    IncludeRequirementStatuses: RequirementStatus list option
    IncludeTestTypes: string list option
    ExcludeFailingTests: bool
    MaxConcurrency: int
    Timeout: TimeSpan
    Environment: TestEnvironment option
    NotifyOnFailure: bool
    GenerateReport: bool
}

/// <summary>
/// Regression test run result
/// </summary>
type RegressionRunResult = {
    RunId: string
    Config: RegressionRunConfig
    StartTime: DateTime
    EndTime: DateTime
    Duration: TimeSpan
    TotalTests: int
    PassedTests: int
    FailedTests: int
    SkippedTests: int
    ErrorTests: int
    PassRate: float
    TestResults: TestExecutionResult list
    RequirementsCovered: string list
    Summary: string
}

/// <summary>
/// Regression runner for automated requirement validation
/// Real implementation with comprehensive test execution
/// </summary>
type RegressionRunner(repository: IRequirementRepository, testExecutor: TestExecutor, logger: ILogger<RegressionRunner>) =
    
    /// <summary>
    /// Default regression run configuration
    /// </summary>
    let defaultConfig = {
        Name = "Default Regression Run"
        Description = "Standard regression test execution"
        IncludeRequirementTypes = None
        IncludeRequirementStatuses = None
        IncludeTestTypes = None
        ExcludeFailingTests = false
        MaxConcurrency = Environment.ProcessorCount
        Timeout = TimeSpan.FromMinutes(30.0)
        Environment = None
        NotifyOnFailure = true
        GenerateReport = true
    }
    
    /// <summary>
    /// Filter requirements based on configuration
    /// </summary>
    let filterRequirements (requirements: Requirement list) (config: RegressionRunConfig) =
        requirements
        |> List.filter (fun req ->
            match config.IncludeRequirementTypes with
            | Some types -> List.contains req.Type types
            | None -> true
        )
        |> List.filter (fun req ->
            match config.IncludeRequirementStatuses with
            | Some statuses -> List.contains req.Status statuses
            | None -> true
        )
    
    /// <summary>
    /// Filter test cases based on configuration
    /// </summary>
    let filterTestCases (testCases: TestCase list) (config: RegressionRunConfig) =
        testCases
        |> List.filter (fun tc ->
            match config.IncludeTestTypes with
            | Some types -> List.contains tc.TestType types
            | None -> true
        )
        |> List.filter (fun tc ->
            if config.ExcludeFailingTests then
                tc.Status <> TestStatus.Failed
            else
                true
        )
    
    /// <summary>
    /// Generate regression run summary
    /// </summary>
    let generateSummary (result: RegressionRunResult) =
        let passRate = result.PassRate
        let status = if passRate >= 95.0 then "EXCELLENT" elif passRate >= 80.0 then "GOOD" elif passRate >= 60.0 then "FAIR" else "POOR"
        let requirementsList = String.Join(", ", result.RequirementsCovered |> List.take (min 10 result.RequirementsCovered.Length))
        let moreRequirements = if result.RequirementsCovered.Length > 10 then $"... and {result.RequirementsCovered.Length - 10} more" else ""

        let failedTestsSection =
            if result.FailedTests > 0 then
                let failedResults = result.TestResults |> List.filter (fun tr -> tr.Status = TestStatus.Failed)
                let failedSummary = failedResults |> List.take (min 5 failedResults.Length) |> List.map (fun tr ->
                    let errorMsg = tr.ErrorMessage |> Option.defaultValue "Unknown error"
                    $"- {tr.TestCaseId}: {errorMsg}")
                let failedSummaryText = String.Join("\n", failedSummary)
                let moreFailed = if failedResults.Length > 5 then $"... and {failedResults.Length - 5} more failures" else ""
                $"""
Failed Tests:
{failedSummaryText}
{moreFailed}
"""
            else ""

        let recommendations =
            [
                if passRate < 80.0 then yield "- Review and fix failing tests before deployment"
                if result.ErrorTests > 0 then yield "- Investigate test execution errors"
                if result.SkippedTests > 0 then yield "- Review skipped tests for relevance"
                if passRate >= 95.0 then yield "- Excellent test coverage! Ready for deployment."
            ] |> String.concat "\n"

        $"""
Regression Test Run Summary
==========================
Run ID: {result.RunId}
Configuration: {result.Config.Name}
Duration: {result.Duration.TotalMinutes:F1} minutes
Status: {status}

Test Results:
- Total Tests: {result.TotalTests}
- Passed: {result.PassedTests} ({result.PassRate:F1}%)
- Failed: {result.FailedTests}
- Skipped: {result.SkippedTests}
- Errors: {result.ErrorTests}

Requirements Coverage:
- Requirements Tested: {result.RequirementsCovered.Length}
- Requirements: {requirementsList}
{moreRequirements}

{failedTestsSection}

Recommendations:
{recommendations}
"""
    
    /// <summary>
    /// Run regression tests with default configuration
    /// </summary>
    member this.RunRegressionTestsAsync() = task {
        return! this.RunRegressionTestsAsync(defaultConfig)
    }
    
    /// <summary>
    /// Run regression tests with custom configuration
    /// </summary>
    member this.RunRegressionTestsAsync(config: RegressionRunConfig) = task {
        let runId = Guid.NewGuid().ToString()
        let startTime = DateTime.UtcNow
        
        logger.LogInformation($"Starting regression test run: {config.Name} (ID: {runId})")
        
        try
            // Get all requirements
            let! requirementsResult = repository.ListRequirementsAsync()
            match requirementsResult with
            | Error error ->
                logger.LogError($"Failed to get requirements: {error}")
                return Error $"Failed to get requirements: {error}"
            | Ok requirements ->
                
                // Filter requirements based on configuration
                let filteredRequirements = filterRequirements requirements config
                logger.LogInformation($"Found {filteredRequirements.Length} requirements matching criteria")
                
                // Get test cases for filtered requirements
                let testCaseTasks = filteredRequirements |> List.map (fun req -> task {
                    let! testCasesResult = repository.GetTestCasesByRequirementAsync(req.Id)
                    return match testCasesResult with
                           | Ok testCases -> testCases
                           | Error _ -> []
                })
                
                let! allTestCasesArrays = Task.WhenAll(testCaseTasks)
                let allTestCases = allTestCasesArrays |> Array.toList |> List.collect id
                
                // Filter test cases based on configuration
                let filteredTestCases = filterTestCases allTestCases config
                logger.LogInformation($"Found {filteredTestCases.Length} test cases to execute")
                
                if filteredTestCases.IsEmpty then
                    logger.LogWarning("No test cases found matching criteria")
                    let endTime = DateTime.UtcNow
                    let emptyResult = {
                        RunId = runId
                        Config = config
                        StartTime = startTime
                        EndTime = endTime
                        Duration = endTime - startTime
                        TotalTests = 0
                        PassedTests = 0
                        FailedTests = 0
                        SkippedTests = 0
                        ErrorTests = 0
                        PassRate = 0.0
                        TestResults = []
                        RequirementsCovered = []
                        Summary = "No tests executed - no test cases found matching criteria"
                    }
                    return Ok emptyResult
                
                // Execute test cases
                logger.LogInformation($"Executing {filteredTestCases.Length} test cases with max concurrency {config.MaxConcurrency}")
                
                let! testResults = 
                    if config.MaxConcurrency > 1 then
                        testExecutor.ExecuteTestsParallelAsync(filteredTestCases, ?environment = config.Environment, maxConcurrency = config.MaxConcurrency)
                    else
                        testExecutor.ExecuteTestsAsync(filteredTestCases, ?environment = config.Environment)
                
                // Save test execution results
                for result in testResults do
                    let! saveResult = repository.SaveTestExecutionResultAsync(result)
                    match saveResult with
                    | Error error -> logger.LogWarning($"Failed to save test result for {result.TestCaseId}: {error}")
                    | Ok _ -> ()
                
                let endTime = DateTime.UtcNow
                let duration = endTime - startTime
                
                // Calculate statistics
                let totalTests = testResults.Length
                let passedTests = testResults |> List.filter (fun tr -> tr.Status = TestStatus.Passed) |> List.length
                let failedTests = testResults |> List.filter (fun tr -> tr.Status = TestStatus.Failed) |> List.length
                let skippedTests = testResults |> List.filter (fun tr -> tr.Status = TestStatus.Skipped) |> List.length
                let errorTests = testResults |> List.filter (fun tr -> tr.Status = TestStatus.Error) |> List.length
                let passRate = if totalTests > 0 then (float passedTests / float totalTests) * 100.0 else 0.0
                
                let requirementsCovered = 
                    filteredTestCases 
                    |> List.map (fun tc -> tc.RequirementId) 
                    |> List.distinct
                    |> List.sort
                
                let regressionResult = {
                    RunId = runId
                    Config = config
                    StartTime = startTime
                    EndTime = endTime
                    Duration = duration
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SkippedTests = skippedTests
                    ErrorTests = errorTests
                    PassRate = passRate
                    TestResults = testResults
                    RequirementsCovered = requirementsCovered
                    Summary = ""
                }
                
                let finalResult = { regressionResult with Summary = generateSummary regressionResult }
                
                logger.LogInformation($"Regression test run completed: {passedTests}/{totalTests} tests passed ({passRate:F1}%)")
                
                if config.NotifyOnFailure && failedTests > 0 then
                    logger.LogWarning($"Regression test run has {failedTests} failing tests!")
                
                return Ok finalResult
                
        with
        | ex ->
            logger.LogError(ex, $"Regression test run failed: {ex.Message}")
            return Error $"Regression test run failed: {ex.Message}"
    }
    
    /// <summary>
    /// Run regression tests for specific requirements
    /// </summary>
    member this.RunRegressionTestsForRequirementsAsync(requirementIds: string list, ?config: RegressionRunConfig) = task {
        let runConfig = config |> Option.defaultValue defaultConfig
        let requirementIdsText = String.Join(", ", requirementIds |> List.take 3)
        let customConfig = { runConfig with Name = $"Targeted Regression Run - {requirementIdsText}" }
        
        logger.LogInformation($"Starting targeted regression test run for {requirementIds.Length} requirements")
        
        try
            // Get test cases for specific requirements
            let testCaseTasks = requirementIds |> List.map (fun reqId -> task {
                let! testCasesResult = repository.GetTestCasesByRequirementAsync(reqId)
                return match testCasesResult with
                       | Ok testCases -> testCases
                       | Error _ -> []
            })
            
            let! allTestCasesArrays = Task.WhenAll(testCaseTasks)
            let allTestCases = allTestCasesArrays |> Array.toList |> List.collect id
            
            // Filter test cases based on configuration
            let filteredTestCases = filterTestCases allTestCases customConfig
            
            if filteredTestCases.IsEmpty then
                logger.LogWarning("No test cases found for specified requirements")
                return Error "No test cases found for specified requirements"
            
            // Execute tests using the main regression runner logic
            return! this.RunRegressionTestsAsync(customConfig)
            
        with
        | ex ->
            logger.LogError(ex, $"Targeted regression test run failed: {ex.Message}")
            return Error $"Targeted regression test run failed: {ex.Message}"
    }
    
    /// <summary>
    /// Create custom regression run configuration
    /// </summary>
    member this.CreateConfig(name: string, ?description: string, ?requirementTypes: RequirementType list, ?requirementStatuses: RequirementStatus list, ?testTypes: string list, ?maxConcurrency: int) =
        {
            Name = name
            Description = description |> Option.defaultValue $"Custom regression run: {name}"
            IncludeRequirementTypes = requirementTypes
            IncludeRequirementStatuses = requirementStatuses
            IncludeTestTypes = testTypes
            ExcludeFailingTests = false
            MaxConcurrency = maxConcurrency |> Option.defaultValue Environment.ProcessorCount
            Timeout = TimeSpan.FromMinutes(30.0)
            Environment = None
            NotifyOnFailure = true
            GenerateReport = true
        }

module RegressionRunnerHelpers =

    /// <summary>
    /// Create default regression config
    /// </summary>
    let createDefaultConfig() = {
        Name = "Default Regression Run"
        Description = "Standard regression test execution"
        IncludeRequirementTypes = None
        IncludeRequirementStatuses = None
        IncludeTestTypes = None
        ExcludeFailingTests = false
        MaxConcurrency = Environment.ProcessorCount
        Timeout = TimeSpan.FromMinutes(30.0)
        Environment = None
        NotifyOnFailure = true
        GenerateReport = true
    }

    /// <summary>
    /// Create quick regression config for functional requirements
    /// </summary>
    let functionalRequirementsConfig = {
        createDefaultConfig() with
            Name = "Functional Requirements Regression"
            Description = "Regression tests for functional requirements only"
            IncludeRequirementTypes = Some [RequirementType.Functional]
    }

    /// <summary>
    /// Create quick regression config for critical requirements
    /// </summary>
    let criticalRequirementsConfig = {
        createDefaultConfig() with
            Name = "Critical Requirements Regression"
            Description = "Regression tests for critical priority requirements"
            IncludeRequirementStatuses = Some [RequirementStatus.Implemented; RequirementStatus.Verified]
    }

    /// <summary>
    /// Create smoke test configuration
    /// </summary>
    let smokeTestConfig = {
        createDefaultConfig() with
            Name = "Smoke Test"
            Description = "Quick smoke test execution"
            IncludeTestTypes = Some ["smoke"; "critical"]
            MaxConcurrency = 1
            Timeout = TimeSpan.FromMinutes(5.0)
    }
