namespace TarsEngine.FSharp.Core.Testing

open System
open System.Collections.Generic
open System.Threading.Tasks

/// Test result status
type TestStatus =
    | Passed
    | Failed
    | Skipped
    | Error

/// Test case definition
type TestCase = {
    Id: string
    Name: string
    Description: string
    TestType: string // "Unit", "Integration", "Performance", "Security"
    InputData: Map<string, obj>
    ExpectedOutput: string option
    TestCode: string
    Timeout: TimeSpan
    CreatedAt: DateTime
}

/// Test execution result
type TestResult = {
    TestCaseId: string
    Status: TestStatus
    ExecutionTime: TimeSpan
    ActualOutput: string
    ErrorMessage: string option
    Assertions: (string * bool) list
    Metrics: Map<string, float>
    ExecutedAt: DateTime
}

/// Test suite for a metascript
type TestSuite = {
    Id: string
    MetascriptName: string
    MetascriptPath: string
    UnitTests: TestCase list
    IntegrationTests: TestCase list
    PerformanceTests: TestCase list
    SecurityTests: TestCase list
    CreatedAt: DateTime
}

/// Test execution summary
type TestExecutionSummary = {
    SuiteId: string
    TotalTests: int
    PassedTests: int
    FailedTests: int
    SkippedTests: int
    ErrorTests: int
    TotalExecutionTime: TimeSpan
    CoveragePercentage: float
    ExecutedAt: DateTime
}

/// Autonomous testing service interface
type IAutonomousTestingService =
    abstract member GenerateTestSuiteAsync: metascriptPath: string -> Task<TestSuite>
    abstract member ExecuteTestSuiteAsync: testSuite: TestSuite -> Task<TestExecutionSummary>
    abstract member GenerateUnitTestsAsync: metascriptContent: string -> Task<TestCase list>
    abstract member GenerateIntegrationTestsAsync: metascriptContent: string -> Task<TestCase list>
    abstract member ValidateMetascriptAsync: metascriptPath: string -> Task<bool>
    abstract member AutoFixFailedTestsAsync: testResults: TestResult list * metascriptContent: string -> Task<string>

/// Test generation service interface
type ITestGenerationService =
    abstract member AnalyzeMetascriptForTestingAsync: content: string -> Task<string>
    abstract member GenerateTestCaseAsync: testType: string * metascriptContent: string * scenario: string -> Task<TestCase>
    abstract member GenerateAssertionsAsync: expectedBehavior: string -> Task<string list>
    abstract member CreateTestDataAsync: inputRequirements: string -> Task<Map<string, obj>>

