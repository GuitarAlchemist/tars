using Microsoft.Extensions.Logging;

namespace TarsCliMinimal
{
    /// <summary>
    /// Extension methods for ILogger
    /// </summary>
    public static class LoggerExtensions
    {
        /// <summary>
        /// Gets a logger for a different type from an existing logger
        /// </summary>
        public static ILogger<T> GetLogger<T>(this ILogger logger)
        {
            var loggerFactory = new LoggerFactory();
            return loggerFactory.CreateLogger<T>();
        }
    }
}
