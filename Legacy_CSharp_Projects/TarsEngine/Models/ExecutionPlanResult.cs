namespace TarsEngine.Models;

/// <summary>
/// Represents the result of an execution plan
/// </summary>
public class ExecutionPlanResult
{
    /// <summary>
    /// Gets or sets the ID of the execution plan
    /// </summary>
    public string ExecutionPlanId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the status of the execution plan
    /// </summary>
    public ExecutionPlanStatus Status { get; set; } = ExecutionPlanStatus.Created;

    /// <summary>
    /// Gets or sets whether the execution plan was successful
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution plan was started
    /// </summary>
    public DateTime? StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution plan was completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the duration of the execution plan in milliseconds
    /// </summary>
    public long? DurationMs { get; set; }

    /// <summary>
    /// Gets or sets the output of the execution plan
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the error message of the execution plan
    /// </summary>
    public string Error { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the exception that caused the execution plan to fail
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// Gets or sets the step results of the execution plan
    /// </summary>
    public List<ExecutionStepResult> StepResults { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files modified by the execution plan
    /// </summary>
    public List<string> ModifiedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files created by the execution plan
    /// </summary>
    public List<string> CreatedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files deleted by the execution plan
    /// </summary>
    public List<string> DeletedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of files backed up by the execution plan
    /// </summary>
    public List<string> BackedUpFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the metrics of the execution plan
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the execution plan result
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
