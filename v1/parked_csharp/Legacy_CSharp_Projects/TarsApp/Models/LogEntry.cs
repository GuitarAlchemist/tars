using MsLogLevel = Microsoft.Extensions.Logging.LogLevel;
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
    public MsLogLevel LogLevel { get; set; }

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
        LogLevel = ConvertToMsLogLevel(log.Level);
        Message = log.Message;
        Timestamp = log.Timestamp;
        Category = log.Source; // ExecutionLog uses Source instead of Category
        EventId = 0; // ExecutionLog doesn't have EventId
    }

    /// <summary>
    /// Converts TarsEngine.Models.LogLevel to Microsoft.Extensions.Logging.LogLevel
    /// </summary>
    /// <param name="level">The TarsEngine log level</param>
    /// <returns>The Microsoft log level</returns>
    private static MsLogLevel ConvertToMsLogLevel(TarsEngine.Models.LogLevel level)
    {
        return level switch
        {
            TarsEngine.Models.LogLevel.Trace => MsLogLevel.Trace,
            TarsEngine.Models.LogLevel.Debug => MsLogLevel.Debug,
            TarsEngine.Models.LogLevel.Information => MsLogLevel.Information,
            TarsEngine.Models.LogLevel.Warning => MsLogLevel.Warning,
            TarsEngine.Models.LogLevel.Error => MsLogLevel.Error,
            TarsEngine.Models.LogLevel.Critical => MsLogLevel.Critical,
            _ => MsLogLevel.None
        };
    }
}
