using System.Text;

namespace TarsCli.Services.Testing;

/// <summary>
/// Service for analyzing test results
/// </summary>
public class TestResultAnalyzer
{
    private readonly ILogger<TestResultAnalyzer> _logger;

    /// <summary>
    /// Initializes a new instance of the TestResultAnalyzer class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public TestResultAnalyzer(ILogger<TestResultAnalyzer> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Analyzes test results
    /// </summary>
    /// <param name="testRunResult">Test run result</param>
    /// <returns>Test analysis result</returns>
    public async Task<TestAnalysisResult> AnalyzeResultsAsync(TestRunResult testRunResult)
    {
        _logger.LogInformation($"Analyzing test results for {testRunResult.ProjectPath}");

        var result = new TestAnalysisResult
        {
            ProjectPath = testRunResult.ProjectPath,
            Success = testRunResult.Success,
            PassedCount = testRunResult.PassedCount,
            FailedCount = testRunResult.FailedCount,
            SkippedCount = testRunResult.SkippedCount,
            TotalCount = testRunResult.TotalCount
        };

        try
        {
            // Analyze failed tests
            foreach (var testResult in testRunResult.TestResults.Where(r => r.Status == TestStatus.Failed))
            {
                var issue = AnalyzeFailedTest(testResult);
                result.Issues.Add(issue);
            }

            // Generate summary
            result.Summary = GenerateSummary(testRunResult);

            // Generate detailed report
            result.DetailedReport = GenerateDetailedReport(testRunResult);

            // Generate improvement suggestions
            result.ImprovementSuggestions = GenerateImprovementSuggestions(testRunResult);

            _logger.LogInformation($"Analysis completed for {testRunResult.ProjectPath}. Found {result.Issues.Count} issues.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing test results for {testRunResult.ProjectPath}");
            result.ErrorMessage = ex.Message;
        }

        return await Task.FromResult(result);
    }

    /// <summary>
    /// Analyzes a failed test
    /// </summary>
    /// <param name="testResult">Test result</param>
    /// <returns>Test issue</returns>
    private TestIssue AnalyzeFailedTest(TestResult testResult)
    {
        var issue = new TestIssue
        {
            TestName = testResult.TestName,
            ErrorMessage = testResult.ErrorMessage,
            Severity = TestIssueSeverity.Error
        };

        // Determine the issue type based on the error message
        if (string.IsNullOrEmpty(testResult.ErrorMessage))
        {
            issue.Type = TestIssueType.Unknown;
            issue.Description = "Unknown error";
            issue.Severity = TestIssueSeverity.Critical;
        }
        else if (testResult.ErrorMessage.Contains("Assert.Fail"))
        {
            issue.Type = TestIssueType.NotImplemented;
            issue.Description = "Test not implemented";
            issue.Severity = TestIssueSeverity.Info;
        }
        else if (testResult.ErrorMessage.Contains("NullReferenceException"))
        {
            issue.Type = TestIssueType.NullReference;
            issue.Description = "Null reference exception";
        }
        else if (testResult.ErrorMessage.Contains("ArgumentException") || testResult.ErrorMessage.Contains("ArgumentNullException"))
        {
            issue.Type = TestIssueType.InvalidArgument;
            issue.Description = "Invalid argument";
        }
        else if (testResult.ErrorMessage.Contains("Expected") && testResult.ErrorMessage.Contains("Actual"))
        {
            issue.Type = TestIssueType.AssertionFailure;
            issue.Description = "Assertion failure";
        }
        else if (testResult.ErrorMessage.Contains("Timeout"))
        {
            issue.Type = TestIssueType.Timeout;
            issue.Description = "Test timeout";
            issue.Severity = TestIssueSeverity.Warning;
        }
        else if (testResult.ErrorMessage.Contains("Exception"))
        {
            issue.Type = TestIssueType.UnhandledException;
            issue.Description = "Unhandled exception";
        }
        else
        {
            issue.Type = TestIssueType.Other;
            issue.Description = "Other error";
        }

        // Generate a suggestion based on the issue type
        issue.Suggestion = GenerateSuggestion(issue);

        return issue;
    }

    /// <summary>
    /// Generates a suggestion for a test issue
    /// </summary>
    /// <param name="issue">Test issue</param>
    /// <returns>Suggestion</returns>
    private string GenerateSuggestion(TestIssue issue)
    {
        switch (issue.Type)
        {
            case TestIssueType.NotImplemented:
                return "Implement the test by replacing Assert.Fail with actual test logic";
            case TestIssueType.NullReference:
                return "Check for null values and initialize objects before using them";
            case TestIssueType.InvalidArgument:
                return "Validate input arguments and provide valid values";
            case TestIssueType.AssertionFailure:
                return "Check the expected and actual values in the assertion";
            case TestIssueType.Timeout:
                return "Optimize the test or increase the timeout value";
            case TestIssueType.UnhandledException:
                return "Add exception handling or fix the code to prevent the exception";
            case TestIssueType.Other:
                return "Review the error message and fix the underlying issue";
            case TestIssueType.Unknown:
            default:
                return "Run the test with a debugger to identify the issue";
        }
    }

    /// <summary>
    /// Generates a summary of the test results
    /// </summary>
    /// <param name="testRunResult">Test run result</param>
    /// <returns>Summary</returns>
    private string GenerateSummary(TestRunResult testRunResult)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Test Results Summary for {testRunResult.ProjectPath}");
        sb.AppendLine();
        sb.AppendLine($"- **Total Tests**: {testRunResult.TotalCount}");
        sb.AppendLine($"- **Passed**: {testRunResult.PassedCount}");
        sb.AppendLine($"- **Failed**: {testRunResult.FailedCount}");
        sb.AppendLine($"- **Skipped**: {testRunResult.SkippedCount}");
        sb.AppendLine($"- **Success Rate**: {(testRunResult.TotalCount > 0 ? (double)testRunResult.PassedCount / testRunResult.TotalCount * 100 : 0):F2}%");
        sb.AppendLine();

        if (testRunResult.FailedCount > 0)
        {
            sb.AppendLine("## Failed Tests");
            sb.AppendLine();
            foreach (var testResult in testRunResult.TestResults.Where(r => r.Status == TestStatus.Failed))
            {
                sb.AppendLine($"- **{testResult.TestName}**: {testResult.ErrorMessage}");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates a detailed report of the test results
    /// </summary>
    /// <param name="testRunResult">Test run result</param>
    /// <returns>Detailed report</returns>
    private string GenerateDetailedReport(TestRunResult testRunResult)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Detailed Test Report for {testRunResult.ProjectPath}");
        sb.AppendLine();

        // Group tests by status
        var passedTests = testRunResult.TestResults.Where(r => r.Status == TestStatus.Passed).ToList();
        var failedTests = testRunResult.TestResults.Where(r => r.Status == TestStatus.Failed).ToList();
        var skippedTests = testRunResult.TestResults.Where(r => r.Status == TestStatus.Skipped).ToList();

        // Report passed tests
        if (passedTests.Count > 0)
        {
            sb.AppendLine("## Passed Tests");
            sb.AppendLine();
            foreach (var testResult in passedTests)
            {
                sb.AppendLine($"- **{testResult.TestName}** ({testResult.Duration})");
            }
            sb.AppendLine();
        }

        // Report failed tests
        if (failedTests.Count > 0)
        {
            sb.AppendLine("## Failed Tests");
            sb.AppendLine();
            foreach (var testResult in failedTests)
            {
                sb.AppendLine($"### {testResult.TestName}");
                sb.AppendLine();
                sb.AppendLine($"- **Duration**: {testResult.Duration}");
                sb.AppendLine($"- **Error Message**: {testResult.ErrorMessage}");
                sb.AppendLine();
            }
        }

        // Report skipped tests
        if (skippedTests.Count > 0)
        {
            sb.AppendLine("## Skipped Tests");
            sb.AppendLine();
            foreach (var testResult in skippedTests)
            {
                sb.AppendLine($"- **{testResult.TestName}**");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates improvement suggestions based on the test results
    /// </summary>
    /// <param name="testRunResult">Test run result</param>
    /// <returns>Improvement suggestions</returns>
    private string GenerateImprovementSuggestions(TestRunResult testRunResult)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Test Improvement Suggestions");
        sb.AppendLine();

        // Check if there are any failed tests
        if (testRunResult.FailedCount > 0)
        {
            sb.AppendLine("## Fixing Failed Tests");
            sb.AppendLine();

            // Group failed tests by issue type
            var issueGroups = testRunResult.TestResults
                .Where(r => r.Status == TestStatus.Failed)
                .Select(r => AnalyzeFailedTest(r))
                .GroupBy(i => i.Type)
                .OrderByDescending(g => g.Count());

            foreach (var group in issueGroups)
            {
                sb.AppendLine($"### {group.Key} Issues ({group.Count()})");
                sb.AppendLine();
                sb.AppendLine($"**Suggestion**: {group.First().Suggestion}");
                sb.AppendLine();
                sb.AppendLine("**Affected Tests**:");
                foreach (var issue in group)
                {
                    sb.AppendLine($"- {issue.TestName}");
                }
                sb.AppendLine();
            }
        }

        // Check if there are any skipped tests
        if (testRunResult.SkippedCount > 0)
        {
            sb.AppendLine("## Implementing Skipped Tests");
            sb.AppendLine();
            sb.AppendLine("Consider implementing the skipped tests to improve test coverage.");
            sb.AppendLine();
        }

        // General suggestions
        sb.AppendLine("## General Improvements");
        sb.AppendLine();
        sb.AppendLine("- **Increase Test Coverage**: Add tests for untested code paths");
        sb.AppendLine("- **Add Edge Cases**: Test boundary conditions and error scenarios");
        sb.AppendLine("- **Improve Test Organization**: Group related tests together");
        sb.AppendLine("- **Use Test Data Generators**: Parameterize tests with different inputs");
        sb.AppendLine("- **Add Performance Tests**: Measure and optimize performance");
        sb.AppendLine();

        return sb.ToString();
    }
}

/// <summary>
/// Result of test analysis
/// </summary>
public class TestAnalysisResult
{
    /// <summary>
    /// Path to the project
    /// </summary>
    public string ProjectPath { get; set; }

    /// <summary>
    /// Whether the test run was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if the analysis failed
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// Number of passed tests
    /// </summary>
    public int PassedCount { get; set; }

    /// <summary>
    /// Number of failed tests
    /// </summary>
    public int FailedCount { get; set; }

    /// <summary>
    /// Number of skipped tests
    /// </summary>
    public int SkippedCount { get; set; }

    /// <summary>
    /// Total number of tests
    /// </summary>
    public int TotalCount { get; set; }

    /// <summary>
    /// List of issues found in the tests
    /// </summary>
    public List<TestIssue> Issues { get; set; } = new();

    /// <summary>
    /// Summary of the test results
    /// </summary>
    public string Summary { get; set; }

    /// <summary>
    /// Detailed report of the test results
    /// </summary>
    public string DetailedReport { get; set; }

    /// <summary>
    /// Improvement suggestions
    /// </summary>
    public string ImprovementSuggestions { get; set; }
}

/// <summary>
/// Issue found in a test
/// </summary>
public class TestIssue
{
    /// <summary>
    /// Name of the test
    /// </summary>
    public string TestName { get; set; }

    /// <summary>
    /// Type of the issue
    /// </summary>
    public TestIssueType Type { get; set; }

    /// <summary>
    /// Severity of the issue
    /// </summary>
    public TestIssueSeverity Severity { get; set; }

    /// <summary>
    /// Description of the issue
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Error message
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// Suggestion for fixing the issue
    /// </summary>
    public string Suggestion { get; set; }
}

/// <summary>
/// Type of test issue
/// </summary>
public enum TestIssueType
{
    /// <summary>
    /// Test is not implemented
    /// </summary>
    NotImplemented,

    /// <summary>
    /// Null reference exception
    /// </summary>
    NullReference,

    /// <summary>
    /// Invalid argument
    /// </summary>
    InvalidArgument,

    /// <summary>
    /// Assertion failure
    /// </summary>
    AssertionFailure,

    /// <summary>
    /// Test timeout
    /// </summary>
    Timeout,

    /// <summary>
    /// Unhandled exception
    /// </summary>
    UnhandledException,

    /// <summary>
    /// Other issue
    /// </summary>
    Other,

    /// <summary>
    /// Unknown issue
    /// </summary>
    Unknown
}

/// <summary>
/// Severity of a test issue
/// </summary>
public enum TestIssueSeverity
{
    /// <summary>
    /// Informational issue
    /// </summary>
    Info,

    /// <summary>
    /// Warning issue
    /// </summary>
    Warning,

    /// <summary>
    /// Error issue
    /// </summary>
    Error,

    /// <summary>
    /// Critical issue
    /// </summary>
    Critical
}