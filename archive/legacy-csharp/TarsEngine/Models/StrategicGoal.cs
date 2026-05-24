namespace TarsEngine.Models;

/// <summary>
/// Represents a strategic goal for improvement prioritization
/// </summary>
public class StrategicGoal
{
    /// <summary>
    /// Gets or sets the ID of the goal
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the goal
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the goal
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the category of the goal
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the weight of the goal (0.0 to 1.0)
    /// </summary>
    public double Weight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the timestamp when the goal was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the goal was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the goal expires
    /// </summary>
    public DateTime? ExpiresAt { get; set; }

    /// <summary>
    /// Gets or sets whether the goal is active
    /// </summary>
    public bool IsActive { get; set; } = true;

    /// <summary>
    /// Gets or sets the list of tags associated with the goal
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of keywords associated with the goal
    /// </summary>
    public List<string> Keywords { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the goal
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
