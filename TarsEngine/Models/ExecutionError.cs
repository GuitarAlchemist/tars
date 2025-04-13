namespace TarsEngine.Models;

/// <summary>
/// Represents an error in the execution context
/// </summary>
public class ExecutionError
{
    /// <summary>
    /// Gets or sets the timestamp of the error
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the message of the error
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the source of the error
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the exception that caused the error
    /// </summary>
    public Exception? Exception { get; set; }

    public string? StackTrace { get; set; }
}
