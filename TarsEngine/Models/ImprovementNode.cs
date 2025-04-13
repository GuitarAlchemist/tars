namespace TarsEngine.Models;

/// <summary>
/// Represents a node in the improvement dependency graph
/// </summary>
public class ImprovementNode
{
    /// <summary>
    /// Gets or sets the ID of the node
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the node
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the category of the node
    /// </summary>
    public ImprovementCategory Category { get; set; } = ImprovementCategory.Other;

    /// <summary>
    /// Gets or sets the priority score of the node
    /// </summary>
    public double PriorityScore { get; set; }

    /// <summary>
    /// Gets or sets the status of the node
    /// </summary>
    public ImprovementStatus Status { get; set; } = ImprovementStatus.Pending;

    /// <summary>
    /// Gets or sets additional metadata about the node
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
