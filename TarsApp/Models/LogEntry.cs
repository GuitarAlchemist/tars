using System;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsApp.Models;

/// <summary>
/// Represents a log entry for display in the UI
/// </summary>
public class LogEntry
{
    /// <summary>
    /// Gets or sets the log level
    /// </summary>
    public LogLevel LogLevel { get; set; }

    /// <summary>
    /// Gets or sets the log message
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the log entry was created
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the category of the log entry
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the event ID of the log entry
    /// </summary>
    public int EventId { get; set; }

    /// <summary>
    /// Creates a new instance of the LogEntry class
    /// </summary>
    public LogEntry()
    {
    }

    /// <summary>
    /// Creates a new instance of the LogEntry class from an ExecutionLog
    /// </summary>
    /// <param name="log">The execution log to create the log entry from</param>
    public LogEntry(ExecutionLog log)
    {
        LogLevel = (LogLevel)log.LogLevel;
        Message = log.Message;
        Timestamp = log.Timestamp;
        Category = log.Category;
        EventId = log.EventId;
    }
}
