using System.Runtime.CompilerServices;

namespace TarsCli.Services;

public static class LoggingExtensions
{
    public static void LogOperationStarted(this ILogger logger, string operation, 
        [CallerMemberName] string callerName = "")
    {
        logger.LogInformation("Starting {Operation} in {Method}", operation, callerName);
    }

    public static void LogOperationCompleted(this ILogger logger, string operation, 
        [CallerMemberName] string callerName = "")
    {
        logger.LogInformation("Completed {Operation} in {Method}", operation, callerName);
    }

    public static void LogOperationFailed(this ILogger logger, string operation, Exception ex, 
        [CallerMemberName] string callerName = "")
    {
        logger.LogError(ex, "Failed {Operation} in {Method}: {ErrorMessage}", 
            operation, callerName, ex.Message);
    }

    public static void LogFileOperation(this ILogger logger, string operation, string filePath)
    {
        logger.LogDebug("{Operation} file: {FilePath}", operation, filePath);
    }

    public static void LogModelOperation(this ILogger logger, string operation, string model)
    {
        logger.LogDebug("{Operation} model: {Model}", operation, model);
    }

    public static void LogDiagnosticInfo(this ILogger logger, string component, string status)
    {
        logger.LogInformation("Diagnostic: {Component} - {Status}", component, status);
    }
}