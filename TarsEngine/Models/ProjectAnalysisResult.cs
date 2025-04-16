namespace TarsEngine.Models;

/// <summary>
/// Represents the result of analyzing a project
/// </summary>
public class ProjectAnalysisResult
{
    /// <summary>
    /// Gets or sets the project path
    /// </summary>
    public string ProjectPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the project name
    /// </summary>
    public string ProjectName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the project type
    /// </summary>
    public string ProjectType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the analysis date
    /// </summary>
    public DateTime AnalysisDate { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the file count
    /// </summary>
    public int FileCount { get; set; }

    /// <summary>
    /// Gets or sets the total lines of code
    /// </summary>
    public int TotalLines { get; set; }

    /// <summary>
    /// Gets or sets the languages used in the project
    /// </summary>
    public List<LanguageInfo> Languages { get; set; } = new();

    /// <summary>
    /// Gets or sets the dependencies
    /// </summary>
    public List<string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets the issues found in the project
    /// </summary>
    public List<CodeIssue> Issues { get; set; } = new();

    /// <summary>
    /// Gets or sets the metrics
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Gets or sets the improvement opportunities
    /// </summary>
    public List<ImprovementOpportunity> ImprovementOpportunities { get; set; } = new();

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
