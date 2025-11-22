namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Represents the severity of an issue
/// </summary>
public enum IssueSeverity
{
    /// <summary>
    /// Trivial issue
    /// </summary>
    Trivial,

    /// <summary>
    /// Minor issue
    /// </summary>
    Minor,

    /// <summary>
    /// Major issue
    /// </summary>
    Major,

    /// <summary>
    /// Critical issue
    /// </summary>
    Critical,

    /// <summary>
    /// Warning issue
    /// </summary>
    Warning,

    /// <summary>
    /// Error issue
    /// </summary>
    Error
}