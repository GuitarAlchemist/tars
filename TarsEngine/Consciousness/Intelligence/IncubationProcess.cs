namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents an incubation process for insight generation
/// </summary>
public class IncubationProcess
{
    /// <summary>
    /// Gets or sets the process ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the problem
    /// </summary>
    public string Problem { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the process context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the start timestamp
    /// </summary>
    public DateTime StartTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the breakthrough timestamp
    /// </summary>
    public DateTime? BreakthroughTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the completion timestamp
    /// </summary>
    public DateTime? CompletionTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the process status
    /// </summary>
    public IncubationStatus Status { get; set; } = IncubationStatus.Active;
    
    /// <summary>
    /// Gets or sets the process progress (0.0 to 1.0)
    /// </summary>
    public double Progress { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the problem complexity (0.0 to 1.0)
    /// </summary>
    public double Complexity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the insight ID
    /// </summary>
    public string? InsightId { get; set; }
    
    /// <summary>
    /// Gets or sets the process tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process notes
    /// </summary>
    public List<string> Notes { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the subconscious processing intensity (0.0 to 1.0)
    /// </summary>
    public double SubconsciousProcessingIntensity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the emotional response intensity (0.0 to 1.0)
    /// </summary>
    public double EmotionalResponseIntensity { get; set; } = 0.7;
    
    /// <summary>
    /// Gets the incubation duration
    /// </summary>
    public TimeSpan IncubationDuration => (BreakthroughTimestamp ?? CompletionTimestamp ?? DateTime.UtcNow) - StartTimestamp;
    
    /// <summary>
    /// Adds a note
    /// </summary>
    /// <param name="note">The note</param>
    public void AddNote(string note)
    {
        Notes.Add(note);
    }
    
    /// <summary>
    /// Adds a tag
    /// </summary>
    /// <param name="tag">The tag</param>
    public void AddTag(string tag)
    {
        if (!Tags.Contains(tag))
        {
            Tags.Add(tag);
        }
    }
    
    /// <summary>
    /// Sets the breakthrough
    /// </summary>
    public void SetBreakthrough()
    {
        Status = IncubationStatus.Breakthrough;
        BreakthroughTimestamp = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Sets the completion
    /// </summary>
    public void SetCompletion()
    {
        Status = IncubationStatus.Complete;
        CompletionTimestamp = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Sets the resolution
    /// </summary>
    /// <param name="insightId">The insight ID</param>
    public void SetResolution(string insightId)
    {
        Status = IncubationStatus.Resolved;
        InsightId = insightId;
    }
}
