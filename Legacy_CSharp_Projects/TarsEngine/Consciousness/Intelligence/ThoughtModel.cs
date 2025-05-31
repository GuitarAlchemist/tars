namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a thought model
/// </summary>
public class ThoughtModel
{
    /// <summary>
    /// Gets or sets the thought ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the thought content
    /// </summary>
    public string Content { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the thought generation method
    /// </summary>
    public ThoughtGenerationMethod Method { get; set; } = ThoughtGenerationMethod.RandomGeneration;
    
    /// <summary>
    /// Gets or sets the thought significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the thought context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the thought tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the thought source
    /// </summary>
    public string Source { get; set; } = "SpontaneousThought";
    
    /// <summary>
    /// Gets or sets the thought category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the thought impact
    /// </summary>
    public string Impact { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the thought impact level (0.0 to 1.0)
    /// </summary>
    public double ImpactLevel { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the thought follow-up
    /// </summary>
    public string FollowUp { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the related thought IDs
    /// </summary>
    public List<string> RelatedThoughtIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets whether the thought led to an insight
    /// </summary>
    public bool LedToInsight { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the insight ID
    /// </summary>
    public string? InsightId { get; set; }
    
    /// <summary>
    /// Gets or sets the thought originality (0.0 to 1.0)
    /// </summary>
    public double Originality { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the thought coherence (0.0 to 1.0)
    /// </summary>
    public double Coherence { get; set; } = 0.5;
    
    /// <summary>
    /// Records an impact for the thought
    /// </summary>
    /// <param name="impact">The impact</param>
    /// <param name="level">The impact level</param>
    public void RecordImpact(string impact, double level)
    {
        Impact = impact;
        ImpactLevel = level;
    }
    
    /// <summary>
    /// Records a follow-up for the thought
    /// </summary>
    /// <param name="followUp">The follow-up</param>
    public void RecordFollowUp(string followUp)
    {
        FollowUp = followUp;
    }
    
    /// <summary>
    /// Records that the thought led to an insight
    /// </summary>
    /// <param name="insightId">The insight ID</param>
    public void RecordInsight(string insightId)
    {
        LedToInsight = true;
        InsightId = insightId;
    }
    
    /// <summary>
    /// Adds a related thought
    /// </summary>
    /// <param name="thoughtId">The thought ID</param>
    public void AddRelatedThought(string thoughtId)
    {
        if (!RelatedThoughtIds.Contains(thoughtId))
        {
            RelatedThoughtIds.Add(thoughtId);
        }
    }
}
