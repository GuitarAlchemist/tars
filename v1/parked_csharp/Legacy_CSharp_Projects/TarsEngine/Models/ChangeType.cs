namespace TarsEngine.Models;

/// <summary>
/// Represents the type of change needed for a component
/// </summary>
public enum ChangeType
{
    /// <summary>
    /// Create a new component
    /// </summary>
    Create,
    
    /// <summary>
    /// Modify an existing component
    /// </summary>
    Modify,
    
    /// <summary>
    /// Delete an existing component
    /// </summary>
    Delete
}
