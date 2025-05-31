namespace TarsEngine.Unified;

/// <summary>
/// Represents the severity of an issue (unified version)
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
    /// Trivial severity level
    /// </summary>
    Trivial = 2,

    /// <summary>
    /// Minor severity level
    /// </summary>
    Minor = 3,

    /// <summary>
    /// Warning severity level
    /// </summary>
    Warning = 4,

    /// <summary>
    /// Major severity level
    /// </summary>
    Major = 5,

    /// <summary>
    /// Error severity level
    /// </summary>
    Error = 6,

    /// <summary>
    /// Critical severity level
    /// </summary>
    Critical = 7,

    /// <summary>
    /// Blocker severity level
    /// </summary>
    Blocker = 8
}