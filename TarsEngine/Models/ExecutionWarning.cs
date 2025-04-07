using System;

namespace TarsEngine.Models;

/// <summary>
/// Represents a warning in the execution context
/// </summary>
public class ExecutionWarning
{
    /// <summary>
    /// Gets or sets the timestamp of the warning
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the message of the warning
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the source of the warning
    /// </summary>
    public string Source { get; set; } = string.Empty;
}
