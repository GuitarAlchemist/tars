using System;

namespace TarsEngine.Models;

/// <summary>
/// Represents an operation in a transaction
/// </summary>
public class Operation
{
    /// <summary>
    /// Gets or sets the operation ID
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the operation type
    /// </summary>
    public OperationType Type { get; set; }

    /// <summary>
    /// Gets or sets the operation target
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the operation details
    /// </summary>
    public string Details { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the operation was performed
    /// </summary>
    public DateTime Timestamp { get; set; }
}
