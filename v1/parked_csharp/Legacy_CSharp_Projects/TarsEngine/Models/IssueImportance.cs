namespace TarsEngine.Models;

/// <summary>
/// Represents the importance of an issue
/// </summary>
public enum IssueImportance
{
    /// <summary>
    /// Information severity
    /// </summary>
    Information,
    
    /// <summary>
    /// Warning severity
    /// </summary>
    Warning,
    
    /// <summary>
    /// Error severity
    /// </summary>
    Error,
    
    /// <summary>
    /// Critical severity
    /// </summary>
    Critical
}
