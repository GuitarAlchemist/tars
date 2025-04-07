using System;
using System.Collections.Generic;
using System.Linq;
using TarsEngine.Models;

namespace TarsApp.Models
{
    /// <summary>
    /// Extension methods for ExecutionLog
    /// </summary>
    public static class ExecutionLogExtensions
    {
        /// <summary>
        /// Converts a LogEntry to an ExecutionLog
        /// </summary>
        /// <param name="logEntry">The LogEntry to convert</param>
        /// <returns>An ExecutionLog</returns>
        public static ExecutionLog ToExecutionLog(this LogEntry logEntry)
        {
            return new ExecutionLog
            {
                Timestamp = logEntry.Timestamp,
                Level = (LogLevel)logEntry.LogLevel,
                Message = logEntry.Message,
                Source = logEntry.Category ?? "System"
            };
        }

        /// <summary>
        /// Converts a collection of LogEntry objects to a collection of ExecutionLog objects
        /// </summary>
        /// <param name="logEntries">The LogEntry objects to convert</param>
        /// <returns>A collection of ExecutionLog objects</returns>
        public static IEnumerable<ExecutionLog> ToExecutionLogs(this IEnumerable<LogEntry> logEntries)
        {
            return logEntries.Select(logEntry => logEntry.ToExecutionLog());
        }

        /// <summary>
        /// Converts an ExecutionLog to a LogEntry
        /// </summary>
        /// <param name="executionLog">The ExecutionLog to convert</param>
        /// <returns>A LogEntry</returns>
        public static LogEntry ToLogEntry(this ExecutionLog executionLog)
        {
            return new LogEntry
            {
                Timestamp = executionLog.Timestamp,
                LogLevel = (Microsoft.Extensions.Logging.LogLevel)executionLog.Level,
                Message = executionLog.Message,
                Category = executionLog.Source
            };
        }

        /// <summary>
        /// Converts a collection of ExecutionLog objects to a collection of LogEntry objects
        /// </summary>
        /// <param name="executionLogs">The ExecutionLog objects to convert</param>
        /// <returns>A collection of LogEntry objects</returns>
        public static IEnumerable<LogEntry> ToLogEntries(this IEnumerable<ExecutionLog> executionLogs)
        {
            return executionLogs.Select(executionLog => executionLog.ToLogEntry());
        }
    }
}
