namespace TarsEngine.Models;

/// <summary>
/// Represents an opportunity for code improvement
/// </summary>
public class ImprovementOpportunity
{
    /// <summary>
    /// Gets or sets the opportunity ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the opportunity description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the opportunity category
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the opportunity priority (0-10)
    /// </summary>
    public int Priority { get; set; }

    /// <summary>
    /// Gets or sets the estimated effort to implement the improvement (in hours)
    /// </summary>
    public double EstimatedEffort { get; set; }

    /// <summary>
    /// Gets or sets the estimated impact of the improvement (0-10)
    /// </summary>
    public int EstimatedImpact { get; set; }

    /// <summary>
    /// Gets or sets the file path where the opportunity was found
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the line number where the opportunity was found
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Gets or sets the code snippet related to the opportunity
    /// </summary>
    public string CodeSnippet { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggested improvement
    /// </summary>
    public string SuggestedImprovement { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the dependencies for this improvement
    /// </summary>
    public List<string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets the tags for this improvement
    /// </summary>
    public List<string> Tags { get; set; } = new();
}
