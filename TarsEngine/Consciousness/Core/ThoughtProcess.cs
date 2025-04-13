namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a thought process
/// </summary>
public class ThoughtProcess
{
    /// <summary>
    /// Gets or sets the thought process ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the topic
    /// </summary>
    public string Topic { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the content
    /// </summary>
    public string Content { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the type
    /// </summary>
    public ThoughtType Type { get; set; } = ThoughtType.Analytical;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the attention focus
    /// </summary>
    public string AttentionFocus { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the mental clarity at the time of the thought
    /// </summary>
    public double MentalClarity { get; set; }
    
    /// <summary>
    /// Gets or sets the thought tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the thought context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the related thought IDs
    /// </summary>
    public List<string> RelatedThoughtIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the thought depth (0.0 to 1.0)
    /// </summary>
    public double Depth { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought complexity (0.0 to 1.0)
    /// </summary>
    public double Complexity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought originality (0.0 to 1.0)
    /// </summary>
    public double Originality { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought utility (0.0 to 1.0)
    /// </summary>
    public double Utility { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought emotional impact (0.0 to 1.0)
    /// </summary>
    public double EmotionalImpact { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought outcome
    /// </summary>
    public string Outcome { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the thought duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; } = 0.0;
}
