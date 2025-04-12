namespace TarsEngine.Models;

/// <summary>
/// Represents the result of analyzing a solution
/// </summary>
public class SolutionAnalysisResult
{
    /// <summary>
    /// The name of the solution
    /// </summary>
    public string SolutionName { get; set; } = string.Empty;
    
    /// <summary>
    /// The path to the solution
    /// </summary>
    public string SolutionPath { get; set; } = string.Empty;
    
    /// <summary>
    /// The list of projects in the solution
    /// </summary>
    public List<ProjectAnalysisResult> Projects { get; set; } = new List<ProjectAnalysisResult>();
}
