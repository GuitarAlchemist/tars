namespace TarsEngine.Models;

/// <summary>
/// Represents the result of executing a metascript
/// </summary>
public class MetascriptExecutionResult
{
    /// <summary>
    /// Gets or sets the ID of the metascript
    /// </summary>
    public string MetascriptId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the execution was successful
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets the execution status
    /// </summary>
    public MetascriptExecutionStatus Status { get; set; } = MetascriptExecutionStatus.NotExecuted;

    /// <summary>
    /// Gets or sets the execution output
    /// </summary>
    public string? Output { get; set; }

    /// <summary>
    /// Gets or sets the execution error
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution started
    /// </summary>
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the execution completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the execution time in milliseconds
    /// </summary>
    public long ExecutionTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the list of files affected by the execution
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of changes made by the execution
    /// </summary>
    public List<MetascriptChange> Changes { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the execution
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
