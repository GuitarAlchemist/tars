namespace TarsEngine.Models;

/// <summary>
/// Represents the result of executing an implementation step
/// </summary>
public class StepResult
{
    /// <summary>
    /// The step number
    /// </summary>
    public int StepNumber { get; set; }
    
    /// <summary>
    /// The step description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// The file path affected by the step
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Whether the step was successful
    /// </summary>
    public bool Success { get; set; } = true;
    
    /// <summary>
    /// The error message if the step failed
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
    
    /// <summary>
    /// The start time of the step
    /// </summary>
    public DateTime StartTime { get; set; }
    
    /// <summary>
    /// The end time of the step
    /// </summary>
    public DateTime EndTime { get; set; }
    
    /// <summary>
    /// The duration of the step
    /// </summary>
    public TimeSpan Duration { get; set; }
}
