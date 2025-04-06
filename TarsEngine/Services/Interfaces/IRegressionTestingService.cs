using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the regression testing service
/// </summary>
public interface IRegressionTestingService
{
    /// <summary>
    /// Runs regression tests for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to test</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <returns>Regression test result</returns>
    Task<RegressionTestResult> RunRegressionTestsAsync(string filePath, string projectPath);

    /// <summary>
    /// Runs regression tests for a specific project
    /// </summary>
    /// <param name="projectPath">Path to the project to test</param>
    /// <returns>Regression test result</returns>
    Task<RegressionTestResult> RunRegressionTestsForProjectAsync(string projectPath);

    /// <summary>
    /// Runs regression tests for a specific solution
    /// </summary>
    /// <param name="solutionPath">Path to the solution to test</param>
    /// <returns>Regression test result</returns>
    Task<RegressionTestResult> RunRegressionTestsForSolutionAsync(string solutionPath);

    /// <summary>
    /// Identifies potential regression issues
    /// </summary>
    /// <param name="originalCode">Original code</param>
    /// <param name="improvedCode">Improved code</param>
    /// <param name="language">Programming language</param>
    /// <returns>List of potential regression issues</returns>
    Task<List<RegressionIssue>> IdentifyPotentialRegressionIssuesAsync(string originalCode, string improvedCode, string language);

    /// <summary>
    /// Tracks test coverage
    /// </summary>
    /// <param name="projectPath">Path to the project to track</param>
    /// <returns>Test coverage result</returns>
    Task<TestCoverageResult> TrackTestCoverageAsync(string projectPath);

    /// <summary>
    /// Suggests additional tests to improve coverage
    /// </summary>
    /// <param name="coverageResult">Test coverage result</param>
    /// <param name="projectPath">Path to the project</param>
    /// <returns>List of suggested tests</returns>
    Task<List<SuggestedTest>> SuggestAdditionalTestsAsync(TestCoverageResult coverageResult, string projectPath);
}

/// <summary>
/// Represents a regression test result
/// </summary>
public class RegressionTestResult
{
    /// <summary>
    /// Whether the regression tests passed
    /// </summary>
    public bool Passed { get; set; }

    /// <summary>
    /// List of regression issues
    /// </summary>
    public List<RegressionIssue> Issues { get; set; } = new();

    /// <summary>
    /// Test results for individual test suites
    /// </summary>
    public List<TestResult> TestResults { get; set; } = new();

    /// <summary>
    /// Total execution time in milliseconds
    /// </summary>
    public long TotalExecutionTimeMs { get; set; }

    /// <summary>
    /// Test coverage result
    /// </summary>
    public TestCoverageResult? CoverageResult { get; set; }
}

/// <summary>
/// Represents a regression issue
/// </summary>
public class RegressionIssue
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
    /// Affected functionality
    /// </summary>
    public string AffectedFunctionality { get; set; } = string.Empty;

    /// <summary>
    /// Suggested fix for the issue
    /// </summary>
    public string? SuggestedFix { get; set; }

    /// <summary>
    /// Confidence in the issue (0-1)
    /// </summary>
    public float Confidence { get; set; }
}

/// <summary>
/// Represents a test coverage result
/// </summary>
public class TestCoverageResult
{
    /// <summary>
    /// Line coverage percentage
    /// </summary>
    public float LineCoverage { get; set; }

    /// <summary>
    /// Branch coverage percentage
    /// </summary>
    public float BranchCoverage { get; set; }

    /// <summary>
    /// Method coverage percentage
    /// </summary>
    public float MethodCoverage { get; set; }

    /// <summary>
    /// Class coverage percentage
    /// </summary>
    public float ClassCoverage { get; set; }

    /// <summary>
    /// List of uncovered lines
    /// </summary>
    public List<UncoveredLine> UncoveredLines { get; set; } = new();

    /// <summary>
    /// List of uncovered branches
    /// </summary>
    public List<UncoveredBranch> UncoveredBranches { get; set; } = new();

    /// <summary>
    /// List of uncovered methods
    /// </summary>
    public List<UncoveredMethod> UncoveredMethods { get; set; } = new();
}

/// <summary>
/// Represents an uncovered line
/// </summary>
public class UncoveredLine
{
    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Line number
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Line content
    /// </summary>
    public string LineContent { get; set; } = string.Empty;
}

/// <summary>
/// Represents an uncovered branch
/// </summary>
public class UncoveredBranch
{
    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Line number
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Branch content
    /// </summary>
    public string BranchContent { get; set; } = string.Empty;

    /// <summary>
    /// Method name
    /// </summary>
    public string MethodName { get; set; } = string.Empty;
}

/// <summary>
/// Represents an uncovered method
/// </summary>
public class UncoveredMethod
{
    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Method name
    /// </summary>
    public string MethodName { get; set; } = string.Empty;

    /// <summary>
    /// Class name
    /// </summary>
    public string ClassName { get; set; } = string.Empty;

    /// <summary>
    /// Line number
    /// </summary>
    public int LineNumber { get; set; }
}

/// <summary>
/// Represents a suggested test
/// </summary>
public class SuggestedTest
{
    /// <summary>
    /// Name of the test
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Description of the test
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Target method to test
    /// </summary>
    public string TargetMethod { get; set; } = string.Empty;

    /// <summary>
    /// Target class to test
    /// </summary>
    public string TargetClass { get; set; } = string.Empty;

    /// <summary>
    /// File path where the test should be added
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Test code
    /// </summary>
    public string TestCode { get; set; } = string.Empty;

    /// <summary>
    /// Priority of the test
    /// </summary>
    public TestCasePriority Priority { get; set; } = TestCasePriority.Medium;
}
