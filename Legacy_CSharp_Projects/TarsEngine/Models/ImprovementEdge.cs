namespace TarsEngine.Models;

/// <summary>
/// Represents an edge in the improvement dependency graph
/// </summary>
public class ImprovementEdge
{
    /// <summary>
    /// Gets or sets the ID of the edge
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ID of the source node
    /// </summary>
    public string SourceId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ID of the target node
    /// </summary>
    public string TargetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of the edge
    /// </summary>
    public ImprovementDependencyType Type { get; set; } = ImprovementDependencyType.Requires;

    /// <summary>
    /// Gets or sets the weight of the edge
    /// </summary>
    public double Weight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets additional metadata about the edge
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
