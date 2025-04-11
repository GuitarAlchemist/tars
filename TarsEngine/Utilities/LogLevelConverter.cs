using MsLogLevel = Microsoft.Extensions.Logging.LogLevel;
using TarsLogLevel = TarsEngine.Models.LogLevel;

namespace TarsEngine.Utilities
{
    /// <summary>
    /// Provides conversion methods between different LogLevel types
    /// </summary>
    public static class LogLevelConverter
    {
        /// <summary>
        /// Converts from Microsoft.Extensions.Logging.LogLevel to TarsEngine.Models.LogLevel
        /// </summary>
        public static TarsLogLevel ToTarsLogLevel(this MsLogLevel logLevel)
        {
            return logLevel switch
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
        /// Converts from TarsEngine.Models.LogLevel to Microsoft.Extensions.Logging.LogLevel
        /// </summary>
        public static MsLogLevel ToMsLogLevel(this TarsLogLevel logLevel)
        {
            return logLevel switch
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
    }
}
