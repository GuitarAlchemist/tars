namespace TarsEngine.FSharp.Requirements.Models

open System
open System.Text.Json.Serialization

/// <summary>
/// Test case model for requirement validation
/// Real implementation for automated testing
/// </summary>
[<CLIMutable>]
type TestCase = {
    /// <summary>Unique identifier for the test case</summary>
    Id: string
    
    /// <summary>ID of the requirement this test validates</summary>
    RequirementId: string
    
    /// <summary>Name of the test case</summary>
    Name: string
    
    /// <summary>Description of what the test validates</summary>
    Description: string
    
    /// <summary>Test code or script to execute</summary>
    TestCode: string
    
    /// <summary>Programming language of the test code</summary>
    Language: string
    
    /// <summary>Expected result or outcome</summary>
    ExpectedResult: string
    
    /// <summary>Current test status</summary>
    Status: TestStatus
    
    /// <summary>Last execution result</summary>
    LastResult: string option
    
    /// <summary>Last execution time</summary>
    LastExecuted: DateTime option
    
    /// <summary>Execution duration in milliseconds</summary>
    ExecutionDuration: int64 option
    
    /// <summary>Test type (unit, integration, acceptance, etc.)</summary>
    TestType: string
    
    /// <summary>Priority of the test</summary>
    Priority: RequirementPriority
    
    /// <summary>Tags for categorization</summary>
    Tags: string list
    
    /// <summary>Setup code required before test execution</summary>
    SetupCode: string option
    
    /// <summary>Cleanup code required after test execution</summary>
    TeardownCode: string option
    
    /// <summary>Test data or fixtures</summary>
    TestData: string option
    
    /// <summary>Environment requirements</summary>
    Environment: string option
    
    /// <summary>Dependencies on other tests</summary>
    Dependencies: string list
    
    /// <summary>Date test was created</summary>
    CreatedAt: DateTime
    
    /// <summary>Date test was last updated</summary>
    UpdatedAt: DateTime
    
    /// <summary>User who created the test</summary>
    CreatedBy: string
    
    /// <summary>User who last updated the test</summary>
    UpdatedBy: string
    
    /// <summary>Version number for change tracking</summary>
    Version: int
    
    /// <summary>Additional metadata</summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Test execution result
/// </summary>
[<CLIMutable>]
type TestExecutionResult = {
    /// <summary>Test case ID</summary>
    TestCaseId: string
    
    /// <summary>Execution status</summary>
    Status: TestStatus
    
    /// <summary>Result message or output</summary>
    Result: string
    
    /// <summary>Error message if test failed</summary>
    ErrorMessage: string option
    
    /// <summary>Stack trace if test failed</summary>
    StackTrace: string option
    
    /// <summary>Execution start time</summary>
    StartTime: DateTime
    
    /// <summary>Execution end time</summary>
    EndTime: DateTime
    
    /// <summary>Duration in milliseconds</summary>
    Duration: int64
    
    /// <summary>Environment where test was executed</summary>
    Environment: string
    
    /// <summary>Additional execution metadata</summary>
    Metadata: Map<string, string>
}

module TestCaseHelpers =
    
    /// <summary>
    /// Create a new test case
    /// </summary>
    let create (requirementId: string) (name: string) (description: string) (testCode: string) (language: string) (createdBy: string) =
        {
            Id = Guid.NewGuid().ToString()
            RequirementId = requirementId
            Name = name
            Description = description
            TestCode = testCode
            Language = language
            ExpectedResult = ""
            Status = TestStatus.NotRun
            LastResult = None
            LastExecuted = None
            ExecutionDuration = None
            TestType = "unit"
            Priority = RequirementPriority.Medium
            Tags = []
            SetupCode = None
            TeardownCode = None
            TestData = None
            Environment = None
            Dependencies = []
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            CreatedBy = createdBy
            UpdatedBy = createdBy
            Version = 1
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Update test case
    /// </summary>
    let update (testCase: TestCase) (updatedBy: string) (updateFn: TestCase -> TestCase) =
        let updated = updateFn testCase
        { updated with 
            UpdatedAt = DateTime.UtcNow
            UpdatedBy = updatedBy
            Version = testCase.Version + 1 }
    
    /// <summary>
    /// Create test execution result
    /// </summary>
    let createExecutionResult (testCaseId: string) (status: TestStatus) (result: string) (startTime: DateTime) (endTime: DateTime) (environment: string) =
        {
            TestCaseId = testCaseId
            Status = status
            Result = result
            ErrorMessage = None
            StackTrace = None
            StartTime = startTime
            EndTime = endTime
            Duration = (endTime - startTime).Ticks / TimeSpan.TicksPerMillisecond
            Environment = environment
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Create failed test execution result
    /// </summary>
    let createFailedResult (testCaseId: string) (result: string) (errorMessage: string) (stackTrace: string option) (startTime: DateTime) (endTime: DateTime) (environment: string) =
        {
            TestCaseId = testCaseId
            Status = TestStatus.Failed
            Result = result
            ErrorMessage = Some errorMessage
            StackTrace = stackTrace
            StartTime = startTime
            EndTime = endTime
            Duration = (endTime - startTime).Ticks / TimeSpan.TicksPerMillisecond
            Environment = environment
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Check if test is passing
    /// </summary>
    let isPassing (testCase: TestCase) =
        testCase.Status = TestStatus.Passed
    
    /// <summary>
    /// Check if test needs to be run
    /// </summary>
    let needsExecution (testCase: TestCase) =
        testCase.Status = TestStatus.NotRun || testCase.Status = TestStatus.Failed
    
    /// <summary>
    /// Get test execution summary
    /// </summary>
    let getSummary (testCases: TestCase list) =
        let total = testCases.Length
        let passed = testCases |> List.filter (fun tc -> tc.Status = TestStatus.Passed) |> List.length
        let failed = testCases |> List.filter (fun tc -> tc.Status = TestStatus.Failed) |> List.length
        let notRun = testCases |> List.filter (fun tc -> tc.Status = TestStatus.NotRun) |> List.length
        let running = testCases |> List.filter (fun tc -> tc.Status = TestStatus.Running) |> List.length
        
        {|
            Total = total
            Passed = passed
            Failed = failed
            NotRun = notRun
            Running = running
            PassRate = if total > 0 then (float passed / float total) * 100.0 else 0.0
        |}
    
    /// <summary>
    /// Validate test case
    /// </summary>
    let validate (testCase: TestCase) =
        let errors = ResizeArray<string>()
        
        if String.IsNullOrWhiteSpace(testCase.Name) then
            errors.Add("Test name is required")
        
        if String.IsNullOrWhiteSpace(testCase.Description) then
            errors.Add("Test description is required")
        
        if String.IsNullOrWhiteSpace(testCase.TestCode) then
            errors.Add("Test code is required")
        
        if String.IsNullOrWhiteSpace(testCase.Language) then
            errors.Add("Test language is required")
        
        if String.IsNullOrWhiteSpace(testCase.RequirementId) then
            errors.Add("Requirement ID is required")
        
        errors |> List.ofSeq
