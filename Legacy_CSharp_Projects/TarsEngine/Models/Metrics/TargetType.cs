namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a target type for metrics
/// </summary>
public enum TargetType
{
    /// <summary>
    /// Method target
    /// </summary>
    Method,
    
    /// <summary>
    /// Class target
    /// </summary>
    Class,
    
    /// <summary>
    /// File target
    /// </summary>
    File,
    
    /// <summary>
    /// Project target
    /// </summary>
    Project,
    
    /// <summary>
    /// Solution target
    /// </summary>
    Solution,
    
    /// <summary>
    /// Other target
    /// </summary>
    Other
}
