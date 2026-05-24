namespace TarsEngine.Models;

/// <summary>
/// Represents an issue found in code during analysis
/// </summary>
public class CodeIssue
{
    /// <summary>
    /// Gets or sets the ID of the issue
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the type of the issue
    /// </summary>
    public CodeIssueType Type { get; set; }

    /// <summary>
    /// Gets or sets the severity of the issue
    /// </summary>
    public IssueSeverity Severity { get; set; }

    /// <summary>
    /// Gets or sets the title of the issue
    /// </summary>
    public string Title { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the issue
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location of the issue in the code
    /// </summary>
    public CodeLocation Location { get; set; } = new();

    /// <summary>
    /// Gets or sets the code snippet containing the issue
    /// </summary>
    public string CodeSnippet { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggested fix for the issue
    /// </summary>
    public string SuggestedFix { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the impact score of the issue (0.0 to 1.0)
    /// </summary>
    public double ImpactScore { get; set; }

    /// <summary>
    /// Gets or sets the fix difficulty score of the issue (0.0 to 1.0)
    /// </summary>
    public double FixDifficultyScore { get; set; }

    /// <summary>
    /// Gets or sets additional tags for the issue
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the code that caused the issue
    /// </summary>
    public string Code { get; set; } = string.Empty;
}