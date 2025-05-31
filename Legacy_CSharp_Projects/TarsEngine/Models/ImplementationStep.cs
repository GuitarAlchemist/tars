namespace TarsEngine.Models;

/// <summary>
/// Represents a step in an implementation plan
/// </summary>
public class ImplementationStep
{
    /// <summary>
    /// The step number
    /// </summary>
    public int StepNumber { get; set; }
    
    /// <summary>
    /// A description of the step
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// The file path affected by the step
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// The type of change needed for the step
    /// </summary>
    public string ChangeType { get; set; } = string.Empty;
    
    /// <summary>
    /// The list of dependencies for the step
    /// </summary>
    public List<string> Dependencies { get; set; } = new();
    
    /// <summary>
    /// The estimated complexity of the step
    /// </summary>
    public TaskComplexity Complexity { get; set; } = TaskComplexity.Medium;
}
