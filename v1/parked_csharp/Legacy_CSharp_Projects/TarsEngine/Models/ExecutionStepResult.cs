namespace TarsEngine.Models;

/// <summary>
/// Represents the result of an execution step
/// </summary>
public class ExecutionStepResult
{
    /// <summary>
    /// Gets or sets the ID of the execution step
    /// </summary>
    public string ExecutionStepId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the status of the execution step
    /// </summary>
    public ExecutionStepStatus Status { get; set; } = ExecutionStepStatus.Pending;

    /// <summary>
    /// Gets or sets whether the execution step was successful
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution step was started
    /// </summary>
    public DateTime? StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution step was completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the duration of the execution step in milliseconds
    /// </summary>
    public long? DurationMs { get; set; }

    /// <summary>
    /// Gets or sets the output of the execution step
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the error message of the execution step
    /// </summary>
    public string Error { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the exception that caused the execution step to fail
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// Gets or sets the validation results of the execution step
    /// </summary>
    public List<ValidationResult> ValidationResults { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files modified by the execution step
    /// </summary>
    public List<string> ModifiedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files created by the execution step
    /// </summary>
    public List<string> CreatedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files deleted by the execution step
    /// </summary>
    public List<string> DeletedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files backed up by the execution step
    /// </summary>
    public List<string> BackedUpFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the metrics of the execution step
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the execution step result
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
