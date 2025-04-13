using TarsEngine.Models;

namespace TarsApp.Models;

/// <summary>
/// Represents error information for display in the UI
/// </summary>
public class ErrorInfo
{
    /// <summary>
    /// Gets or sets the error message
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the error type
    /// </summary>
    public string ErrorType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the stack trace
    /// </summary>
    public string StackTrace { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the error occurred
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the source of the error
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Creates a new instance of the ErrorInfo class
    /// </summary>
    public ErrorInfo()
    {
    }

    /// <summary>
    /// Creates a new instance of the ErrorInfo class from an ExecutionError
    /// </summary>
    /// <param name="error">The execution error to create the error info from</param>
    public ErrorInfo(ExecutionError error)
    {
        Message = error.Message;
        ErrorType = error.Exception?.GetType().Name ?? "Unknown";
        StackTrace = error.Exception?.StackTrace ?? string.Empty;
        Timestamp = error.Timestamp;
        Source = error.Source;
    }

    /// <summary>
    /// Creates a new instance of the ErrorInfo class from an exception
    /// </summary>
    /// <param name="exception">The exception to create the error info from</param>
    public ErrorInfo(Exception exception)
    {
        Message = exception.Message;
        ErrorType = exception.GetType().Name;
        StackTrace = exception.StackTrace ?? string.Empty;
        Source = exception.Source ?? string.Empty;
    }
}
