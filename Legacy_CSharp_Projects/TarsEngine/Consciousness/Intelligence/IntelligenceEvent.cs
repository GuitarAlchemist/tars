namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents an intelligence event
/// </summary>
public class IntelligenceEvent
{
    /// <summary>
    /// Gets or sets the event ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the event type
    /// </summary>
    public IntelligenceEventType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the event description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the event timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the event significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; }
    
    /// <summary>
    /// Gets or sets the intelligence level at the time of the event
    /// </summary>
    public double IntelligenceLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the creativity level at the time of the event
    /// </summary>
    public double CreativityLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the intuition level at the time of the event
    /// </summary>
    public double IntuitionLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the curiosity level at the time of the event
    /// </summary>
    public double CuriosityLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the insight level at the time of the event
    /// </summary>
    public double InsightLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the event tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the event context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the event source
    /// </summary>
    public string Source { get; set; } = "IntelligenceSpark";
    
    /// <summary>
    /// Gets or sets the event category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the event impact
    /// </summary>
    public string Impact { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the event duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; }
    
    /// <summary>
    /// Gets or sets the related events
    /// </summary>
    public List<string> RelatedEventIds { get; set; } = new();
}
