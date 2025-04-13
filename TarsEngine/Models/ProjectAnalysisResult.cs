namespace TarsEngine.Models;

/// <summary>
/// Represents the result of analyzing a project
/// </summary>
public class ProjectAnalysisResult
{
    /// <summary>
    /// The name of the project
    /// </summary>
    public string ProjectName { get; set; } = string.Empty;
    
    /// <summary>
    /// The path to the project
    /// </summary>
    public string ProjectPath { get; set; } = string.Empty;
    
    /// <summary>
    /// The list of files in the project
    /// </summary>
    public List<string> Files { get; set; } = new();
    
    /// <summary>
    /// The list of references in the project
    /// </summary>
    public List<string> References { get; set; } = new();
    
    /// <summary>
    /// The list of packages in the project
    /// </summary>
    public List<string> Packages { get; set; } = new();
}
