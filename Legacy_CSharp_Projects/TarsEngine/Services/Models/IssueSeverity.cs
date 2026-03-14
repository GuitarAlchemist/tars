namespace TarsEngine.Services.Models;

/// <summary>
/// Represents the severity of an issue
/// </summary>
public enum IssueSeverity
{
    /// <summary>
    /// Information severity level
    /// </summary>
    Info = 0,

    /// <summary>
    /// Suggestion severity level
    /// </summary>
    Suggestion = 1,

    /// <summary>
    /// Warning severity level
    /// </summary>
    Warning = 2,

    /// <summary>
    /// Error severity level
    /// </summary>
    Error = 3,

    /// <summary>
    /// Critical severity level
    /// </summary>
    Critical = 4
}
