namespace TarsEngine.Models;

/// <summary>
/// Represents a plan for implementing a task
/// </summary>
public class ImplementationPlan
{
    /// <summary>
    /// The task description
    /// </summary>
    public string TaskDescription { get; set; } = string.Empty;
    
    /// <summary>
    /// The list of requirements extracted from the task
    /// </summary>
    public List<string> Requirements { get; set; } = new();
    
    /// <summary>
    /// The list of components affected by the task
    /// </summary>
    public List<AffectedComponent> AffectedComponents { get; set; } = new();
    
    /// <summary>
    /// The list of implementation steps
    /// </summary>
    public List<ImplementationStep> ImplementationSteps { get; set; } = new();
    
    /// <summary>
    /// The estimated complexity of the task
    /// </summary>
    public TaskComplexity Complexity { get; set; } = TaskComplexity.Medium;
    
    /// <summary>
    /// The date and time when the plan was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.Now;
    
    /// <summary>
    /// Convert the implementation plan to a string
    /// </summary>
    /// <returns>A string representation of the implementation plan</returns>
    public override string ToString()
    {
        var result = new System.Text.StringBuilder();
        
        result.AppendLine($"# Implementation Plan for: {TaskDescription}");
        result.AppendLine($"Created: {CreatedAt:yyyy-MM-dd HH:mm:ss}");
        result.AppendLine($"Complexity: {Complexity}");
        result.AppendLine();
        
        result.AppendLine("## Requirements");
        foreach (var requirement in Requirements)
        {
            result.AppendLine($"- {requirement}");
        }
        result.AppendLine();
        
        result.AppendLine("## Affected Components");
        foreach (var component in AffectedComponents)
        {
            result.AppendLine($"- {component.Name} ({component.FilePath}) - {component.ChangeType}");
            result.AppendLine($"  {component.Description}");
        }
        result.AppendLine();
        
        result.AppendLine("## Implementation Steps");
        foreach (var step in ImplementationSteps)
        {
            result.AppendLine($"### Step {step.StepNumber}: {step.Description.Split(Environment.NewLine)[0]}");
            result.AppendLine($"File: {step.FilePath}");
            result.AppendLine($"Change Type: {step.ChangeType}");
            result.AppendLine($"Complexity: {step.Complexity}");
            
            if (step.Dependencies.Count > 0)
            {
                result.AppendLine($"Dependencies: {string.Join(", ", step.Dependencies)}");
            }
            
            // Add the full description (excluding the first line which is already included in the header)
            var descriptionLines = step.Description.Split(Environment.NewLine);
            if (descriptionLines.Length > 1)
            {
                result.AppendLine();
                for (int i = 1; i < descriptionLines.Length; i++)
                {
                    result.AppendLine(descriptionLines[i]);
                }
            }
            
            result.AppendLine();
        }
        
        return result.ToString();
    }
}
