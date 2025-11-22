namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a self-reflection
/// </summary>
public class SelfReflection
{
    /// <summary>
    /// Gets or sets the reflection ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the reflection timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the reflection topic
    /// </summary>
    public string Topic { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the reflection insight
    /// </summary>
    public string Insight { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the reflection significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; }
    
    /// <summary>
    /// Gets or sets the self-awareness change
    /// </summary>
    public double SelfAwarenessChange { get; set; }
    
    /// <summary>
    /// Gets or sets the related memory IDs
    /// </summary>
    public List<string> RelatedMemoryIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the reflection tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the reflection context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the reflection category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the reflection depth (0.0 to 1.0)
    /// </summary>
    public double Depth { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the reflection quality (0.0 to 1.0)
    /// </summary>
    public double Quality { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the reflection novelty (0.0 to 1.0)
    /// </summary>
    public double Novelty { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the reflection utility (0.0 to 1.0)
    /// </summary>
    public double Utility { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the reflection emotional impact (0.0 to 1.0)
    /// </summary>
    public double EmotionalImpact { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the reflection action items
    /// </summary>
    public List<string> ActionItems { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the reflection follow-up reflections
    /// </summary>
    public List<string> FollowUpReflectionIds { get; set; } = new();
}
