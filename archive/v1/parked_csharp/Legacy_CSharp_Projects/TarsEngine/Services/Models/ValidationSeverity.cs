namespace TarsEngine.Services.Models;

/// <summary>
/// Represents the severity of a validation issue
/// </summary>
public enum ValidationSeverity
{
    /// <summary>
    /// Information severity level
    /// </summary>
    Info = 0,

    /// <summary>
    /// Warning severity level
    /// </summary>
    Warning = 1,

    /// <summary>
    /// Error severity level
    /// </summary>
    Error = 2,

    /// <summary>
    /// Critical severity level
    /// </summary>
    Critical = 3
}
