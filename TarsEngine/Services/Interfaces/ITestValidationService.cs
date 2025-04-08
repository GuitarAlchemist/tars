using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the test validation service
/// </summary>
public interface ITestValidationService
{
    /// <summary>
    /// Validates improved code against existing tests
    /// </summary>
    /// <param name="originalCode">Original code</param>
    /// <param name="improvedCode">Improved code</param>
    /// <param name="testCode">Test code</param>
    /// <param name="language">Programming language</param>
    /// <returns>Validation result</returns>
    Task<TestValidationResult> ValidateImprovedCodeAsync(string originalCode, string improvedCode, string testCode, string language);

    /// <summary>
    /// Runs tests against a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to test</param>
    /// <param name="testFilePath">Path to the test file</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <returns>Test result</returns>
    Task<TestResult> RunTestsAsync(string filePath, string testFilePath, string projectPath);

    /// <summary>
    /// Compares test results before and after improvements
    /// </summary>
    /// <param name="beforeResult">Test result before improvements</param>
    /// <param name="afterResult">Test result after improvements</param>
    /// <returns>Test comparison result</returns>
    TestComparisonResult CompareTestResults(TestResult beforeResult, TestResult afterResult);

    /// <summary>
    /// Suggests fixes for failing tests
    /// </summary>
    /// <param name="testResult">Test result</param>
    /// <param name="codeFilePath">Path to the code file</param>
    /// <param name="testFilePath">Path to the test file</param>
    /// <returns>List of suggested fixes</returns>
    Task<List<TestFix>> SuggestFixesForFailingTestsAsync(TestResult testResult, string codeFilePath, string testFilePath);
}

/// <summary>
/// Represents a test validation result
/// </summary>
public class TestValidationResult
{
    /// <summary>
    /// Whether the validation was successful
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of validation issues
    /// </summary>
    public List<ValidationIssue> Issues { get; set; } = new();

    /// <summary>
    /// Test result
    /// </summary>
    public TestResult? TestResult { get; set; }
}

/// <summary>
/// Represents a validation issue
/// </summary>
public class ValidationIssue
{
    /// <summary>
    /// Description of the issue
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Severity of the issue
    /// </summary>
    public IssueSeverity Severity { get; set; } = IssueSeverity.Warning;

    /// <summary>
    /// Location of the issue
    /// </summary>
    public string Location { get; set; } = string.Empty;

    /// <summary>
    /// Suggested fix for the issue
    /// </summary>
    public string? SuggestedFix { get; set; }
}

// Using IssueSeverity from TarsEngine.Models

/// <summary>
/// Represents a test result
/// </summary>
public class TestResult
{
    /// <summary>
    /// Total number of tests
    /// </summary>
    public int TotalTests { get; set; }

    /// <summary>
    /// Number of passed tests
    /// </summary>
    public int PassedTests { get; set; }

    /// <summary>
    /// Number of failed tests
    /// </summary>
    public int FailedTests { get; set; }

    /// <summary>
    /// Number of skipped tests
    /// </summary>
    public int SkippedTests { get; set; }

    /// <summary>
    /// Test execution time in milliseconds
    /// </summary>
    public long ExecutionTimeMs { get; set; }

    /// <summary>
    /// List of test failures
    /// </summary>
    public List<TestFailure> Failures { get; set; } = new();

    /// <summary>
    /// Raw output from the test runner
    /// </summary>
    public string RawOutput { get; set; } = string.Empty;
}

/// <summary>
/// Represents a test failure
/// </summary>
public class TestFailure
{
    /// <summary>
    /// Name of the test
    /// </summary>
    public string TestName { get; set; } = string.Empty;

    /// <summary>
    /// Error message
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;

    /// <summary>
    /// Stack trace
    /// </summary>
    public string StackTrace { get; set; } = string.Empty;
}

/// <summary>
/// Represents a test comparison result
/// </summary>
public class TestComparisonResult
{
    /// <summary>
    /// Whether the comparison was successful
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Number of new passing tests
    /// </summary>
    public int NewPassingTests { get; set; }

    /// <summary>
    /// Number of new failing tests
    /// </summary>
    public int NewFailingTests { get; set; }

    /// <summary>
    /// List of new failures
    /// </summary>
    public List<TestFailure> NewFailures { get; set; } = new();

    /// <summary>
    /// List of fixed failures
    /// </summary>
    public List<string> FixedFailures { get; set; } = new();

    /// <summary>
    /// Change in execution time (negative means faster)
    /// </summary>
    public long ExecutionTimeChange { get; set; }
}

/// <summary>
/// Represents a test fix
/// </summary>
public class TestFix
{
    /// <summary>
    /// Description of the fix
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Code to fix the issue
    /// </summary>
    public string FixCode { get; set; } = string.Empty;

    /// <summary>
    /// File path where the fix should be applied
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Line number where the fix should be applied
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Confidence in the fix (0-1)
    /// </summary>
    public float Confidence { get; set; }
}
