namespace TarsEngine.FSharp.Core.CodeGen.Testing.Runners

open System
open System.Threading
open System.Threading.Tasks

/// <summary>
/// Represents a test result.
/// </summary>
type TestResult = {
    /// <summary>
    /// The name of the test.
    /// </summary>
    TestName: string
    
    /// <summary>
    /// Whether the test passed.
    /// </summary>
    Passed: bool
    
    /// <summary>
    /// The error message, if any.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// The stack trace, if any.
    /// </summary>
    StackTrace: string option
    
    /// <summary>
    /// The duration of the test in milliseconds.
    /// </summary>
    DurationMs: float
    
    /// <summary>
    /// The output of the test.
    /// </summary>
    Output: string option
    
    /// <summary>
    /// Additional information about the test result.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a test run result.
/// </summary>
type TestRunResult = {
    /// <summary>
    /// The results of the tests.
    /// </summary>
    Results: TestResult list
    
    /// <summary>
    /// The total number of tests.
    /// </summary>
    TotalTests: int
    
    /// <summary>
    /// The number of passed tests.
    /// </summary>
    PassedTests: int
    
    /// <summary>
    /// The number of failed tests.
    /// </summary>
    FailedTests: int
    
    /// <summary>
    /// The number of skipped tests.
    /// </summary>
    SkippedTests: int
    
    /// <summary>
    /// The total duration of the test run in milliseconds.
    /// </summary>
    TotalDurationMs: float
    
    /// <summary>
    /// Additional information about the test run.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Interface for running tests.
/// </summary>
type ITestRunner =
    /// <summary>
    /// Gets the name of the test runner.
    /// </summary>
    abstract member Name : string
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    abstract member SupportedFrameworks : string list
    
    /// <summary>
    /// Runs tests in a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The test run result.</returns>
    abstract member RunTestsInDirectoryAsync : directoryPath:string * ?filter:string * ?cancellationToken:CancellationToken -> Task<TestRunResult>
    
    /// <summary>
    /// Runs tests in an assembly.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The test run result.</returns>
    abstract member RunTestsInAssemblyAsync : assemblyPath:string * ?filter:string * ?cancellationToken:CancellationToken -> Task<TestRunResult>
    
    /// <summary>
    /// Runs a specific test.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the test.</param>
    /// <param name="testName">The name of the test to run.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The test result.</returns>
    abstract member RunTestAsync : assemblyPath:string * testName:string * ?cancellationToken:CancellationToken -> Task<TestResult>
    
    /// <summary>
    /// Discovers tests in a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The list of discovered tests.</returns>
    abstract member DiscoverTestsInDirectoryAsync : directoryPath:string * ?filter:string * ?cancellationToken:CancellationToken -> Task<string list>
    
    /// <summary>
    /// Discovers tests in an assembly.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The list of discovered tests.</returns>
    abstract member DiscoverTestsInAssemblyAsync : assemblyPath:string * ?filter:string * ?cancellationToken:CancellationToken -> Task<string list>
