namespace TarsEngine.Models;

/// <summary>
/// Represents a suggestion for code improvement
/// </summary>
public class ImprovementSuggestion
{
    /// <summary>
    /// Gets or sets the suggestion description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggestion location
    /// </summary>
    public string Location { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggestion recommendation
    /// </summary>
    public string Recommendation { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggestion benefit
    /// </summary>
    public string Benefit { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggestion priority
    /// </summary>
    public int Priority { get; set; }
}
