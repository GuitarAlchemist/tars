using System;
using Microsoft.Extensions.Logging;

namespace TarsApp.ViewModels
{
    public class LogEntryViewModel
    {
        public DateTime Timestamp { get; set; }
        public LogLevel LogLevel { get; set; }
        public string Message { get; set; } = string.Empty;
        public string Category { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;

        // Factory method to create from TarsEngine.Models.LogEntry
        public static LogEntryViewModel FromLogEntry(TarsEngine.Models.LogEntry logEntry)
        {
            return new LogEntryViewModel
            {
                Timestamp = logEntry.Timestamp,
                LogLevel = ConvertLogLevel(logEntry.Level),
                Message = logEntry.Message,
                Category = logEntry.Category,
                Source = logEntry.Source
            };
        }

        // Factory method to create from TarsEngine.Models.ExecutionLog
        public static LogEntryViewModel FromExecutionLog(TarsEngine.Models.ExecutionLog executionLog)
        {
            return new LogEntryViewModel
            {
                Timestamp = executionLog.Timestamp,
                LogLevel = ConvertLogLevel(executionLog.Level),
                Message = executionLog.Message,
                Category = string.Empty, // ExecutionLog doesn't have Category
                Source = executionLog.Source
            };
        }

        // Convert TarsEngine.Models.LogLevel to Microsoft.Extensions.Logging.LogLevel
        public static LogLevel ConvertLogLevel(TarsEngine.Models.LogLevel level)
        {
            return level switch
            {
                TarsEngine.Models.LogLevel.Trace => LogLevel.Trace,
                TarsEngine.Models.LogLevel.Debug => LogLevel.Debug,
                TarsEngine.Models.LogLevel.Information => LogLevel.Information,
                TarsEngine.Models.LogLevel.Warning => LogLevel.Warning,
                TarsEngine.Models.LogLevel.Error => LogLevel.Error,
                TarsEngine.Models.LogLevel.Critical => LogLevel.Critical,
                _ => LogLevel.None
            };
        }
    }
}
