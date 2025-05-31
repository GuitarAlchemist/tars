using TarsEngine.Models;

// Use alias to avoid ambiguity
using MsLogLevel = Microsoft.Extensions.Logging.LogLevel;
using TarsLogLevel = TarsEngine.Models.LogLevel;

namespace TarsApp.Models;

/// <summary>
/// Provides conversion methods between TarsApp and TarsEngine models
/// </summary>
public static class ModelConverters
{
    /// <summary>
    /// Converts a list of TarsApp LogEntry objects to a list of TarsEngine ExecutionLog objects
    /// </summary>
    /// <param name="logEntries">The log entries to convert</param>
    /// <returns>A list of ExecutionLog objects</returns>
    public static List<ExecutionLog> ConvertToExecutionLogs(List<LogEntry> logEntries)
    {
        if (logEntries == null)
            return new List<ExecutionLog>();

        return logEntries.Select(entry => new ExecutionLog
        {
            Timestamp = entry.Timestamp,
            Level = ConvertToTarsLogLevel(entry.LogLevel),
            Message = entry.Message,
            Source = entry.Category ?? "System"
        }).ToList();
    }

    /// <summary>
    /// Converts a list of TarsApp ErrorInfo objects to a list of TarsEngine ExecutionError objects
    /// </summary>
    /// <param name="errors">The errors to convert</param>
    /// <returns>A list of ExecutionError objects</returns>
    public static List<ExecutionError> ConvertToExecutionErrors(List<ErrorInfo> errors)
    {
        if (errors == null)
            return new List<ExecutionError>();

        return errors.Select(error => new ExecutionError
        {
            Timestamp = error.Timestamp,
            Message = error.Message,
            Source = error.Source,
            Exception = null // We can't recreate the actual exception
        }).ToList();
    }

    /// <summary>
    /// Converts a Microsoft LogLevel to a TarsEngine LogLevel
    /// </summary>
    /// <param name="level">The Microsoft LogLevel</param>
    /// <returns>The TarsEngine LogLevel</returns>
    public static TarsLogLevel ConvertToTarsLogLevel(MsLogLevel level)
    {
        return level switch
        {
            MsLogLevel.Trace => TarsLogLevel.Trace,
            MsLogLevel.Debug => TarsLogLevel.Debug,
            MsLogLevel.Information => TarsLogLevel.Information,
            MsLogLevel.Warning => TarsLogLevel.Warning,
            MsLogLevel.Error => TarsLogLevel.Error,
            MsLogLevel.Critical => TarsLogLevel.Critical,
            _ => TarsLogLevel.Information
        };
    }

    /// <summary>
    /// Converts a TarsEngine LogLevel to a Microsoft LogLevel
    /// </summary>
    /// <param name="level">The TarsEngine LogLevel</param>
    /// <returns>The Microsoft LogLevel</returns>
    public static MsLogLevel ConvertToMsLogLevel(TarsLogLevel level)
    {
        return level switch
        {
            TarsLogLevel.Trace => MsLogLevel.Trace,
            TarsLogLevel.Debug => MsLogLevel.Debug,
            TarsLogLevel.Information => MsLogLevel.Information,
            TarsLogLevel.Warning => MsLogLevel.Warning,
            TarsLogLevel.Error => MsLogLevel.Error,
            TarsLogLevel.Critical => MsLogLevel.Critical,
            _ => MsLogLevel.Information
        };
    }

    /// <summary>
    /// Converts a list of TarsEngine ExecutionLog objects to a list of TarsApp LogEntry objects
    /// </summary>
    /// <param name="executionLogs">The execution logs to convert</param>
    /// <returns>A list of LogEntry objects</returns>
    public static List<LogEntry> ConvertToLogEntries(List<ExecutionLog> executionLogs)
    {
        if (executionLogs == null)
            return new List<LogEntry>();

        return executionLogs.Select(log => new LogEntry
        {
            Timestamp = log.Timestamp,
            LogLevel = ConvertToMsLogLevel(log.Level),
            Message = log.Message,
            Category = log.Source ?? "System"
        }).ToList();
    }

    /// <summary>
    /// Converts a list of TarsEngine ExecutionError objects to a list of TarsApp ErrorInfo objects
    /// </summary>
    /// <param name="executionErrors">The execution errors to convert</param>
    /// <returns>A list of ErrorInfo objects</returns>
    public static List<ErrorInfo> ConvertToErrorInfos(List<ExecutionError> executionErrors)
    {
        if (executionErrors == null)
            return new List<ErrorInfo>();

        return executionErrors.Select(error => new ErrorInfo
        {
            Timestamp = error.Timestamp,
            Message = error.Message,
            Source = error.Source ?? string.Empty,
            StackTrace = error.Exception?.StackTrace
        }).ToList();
    }
}