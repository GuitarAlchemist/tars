namespace TarsEngine.Models;

/// <summary>
/// Represents a component affected by a task
/// </summary>
public class AffectedComponent
{
    /// <summary>
    /// The name of the component
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// The file path of the component
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// The type of change needed for the component
    /// </summary>
    public ChangeType ChangeType { get; set; } = ChangeType.Modify;
    
    /// <summary>
    /// A description of the changes needed for the component
    /// </summary>
    public string Description { get; set; } = string.Empty;
}
