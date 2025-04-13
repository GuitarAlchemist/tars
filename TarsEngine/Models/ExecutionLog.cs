namespace TarsEngine.Models;

/// <summary>
/// Represents a log entry in the execution context
/// </summary>
public class ExecutionLog
{
    /// <summary>
    /// Gets or sets the timestamp of the log entry
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the level of the log entry
    /// </summary>
    public LogLevel Level { get; set; } = LogLevel.Information;

    /// <summary>
    /// Gets or sets the message of the log entry
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the source of the log entry
    /// </summary>
    public string Source { get; set; } = string.Empty;
}
