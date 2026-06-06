namespace TarsEngine.Models;

/// <summary>
/// Represents a test failure
/// </summary>
public class TestFailure
{
    /// <summary>
    /// Gets or sets the test name
    /// </summary>
    public string TestName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the error message
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the stack trace
    /// </summary>
    public string StackTrace { get; set; } = string.Empty;
}
