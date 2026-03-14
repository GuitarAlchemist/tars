namespace TarsEngine.Services.Models;

/// <summary>
/// Represents a knowledge item in the Services.Models namespace
/// </summary>
public class KnowledgeItem
{
    /// <summary>
    /// Gets or sets the unique identifier for the knowledge item
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the content of the knowledge item
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of knowledge
    /// </summary>
    public KnowledgeType Type { get; set; } = KnowledgeType.Unknown;

    /// <summary>
    /// Gets or sets the source of the knowledge
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence level (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the timestamp when the knowledge was created
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the list of related items
    /// </summary>
    public List<string> RelatedItems { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
