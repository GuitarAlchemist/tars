namespace TarsEngine.Models;

/// <summary>
/// Represents the result of a change validation
/// </summary>
public class ChangeValidationResult
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the project path
    /// </summary>
    public string ProjectPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the validation was successful
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets the error message
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the validation was started
    /// </summary>
    public DateTime StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the validation was completed
    /// </summary>
    public DateTime CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the duration of the validation in milliseconds
    /// </summary>
    public long DurationMs { get; set; }

    /// <summary>
    /// Gets or sets the validation results
    /// </summary>
    public List<ValidationResult> ValidationResults { get; set; } = new();

    /// <summary>
    /// Gets or sets the test result
    /// </summary>
    public TestExecutionResult? TestResult { get; set; }
}
