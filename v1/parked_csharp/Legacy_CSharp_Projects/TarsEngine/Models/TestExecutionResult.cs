namespace TarsEngine.Models;

/// <summary>
/// Represents the result of a test execution
/// </summary>
public class TestExecutionResult
{
    /// <summary>
    /// Gets or sets the project path
    /// </summary>
    public string ProjectPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the test filter
    /// </summary>
    public string? TestFilter { get; set; }

    /// <summary>
    /// Gets or sets whether the test execution was successful
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets the error message
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the output
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the test execution was started
    /// </summary>
    public DateTime StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the test execution was completed
    /// </summary>
    public DateTime CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the duration of the test execution in milliseconds
    /// </summary>
    public long DurationMs { get; set; }

    /// <summary>
    /// Gets or sets the total number of tests
    /// </summary>
    public int TotalTests { get; set; }

    /// <summary>
    /// Gets or sets the number of passed tests
    /// </summary>
    public int PassedTests { get; set; }

    /// <summary>
    /// Gets or sets the number of failed tests
    /// </summary>
    public int FailedTests { get; set; }

    /// <summary>
    /// Gets or sets the number of skipped tests
    /// </summary>
    public int SkippedTests { get; set; }

    /// <summary>
    /// Gets or sets the test failures
    /// </summary>
    public List<TestFailure> TestFailures { get; set; } = new();
}
