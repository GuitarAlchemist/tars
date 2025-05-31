namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for reporting progress
/// </summary>
public interface IProgressReporter
{
    /// <summary>
    /// Reports progress
    /// </summary>
    /// <param name="message">The progress message</param>
    /// <param name="percentComplete">The percentage complete (0-100)</param>
    void ReportProgress(string message, int percentComplete);

    /// <summary>
    /// Reports a warning
    /// </summary>
    /// <param name="message">The warning message</param>
    /// <param name="exception">The exception, if any</param>
    void ReportWarning(string message, Exception? exception);

    /// <summary>
    /// Reports an error
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="exception">The exception</param>
    void ReportError(string message, Exception exception);
}
