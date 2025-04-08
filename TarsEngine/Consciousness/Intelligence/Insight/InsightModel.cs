using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence.Insight;

/// <summary>
/// Represents an insight model
/// </summary>
public class InsightModel
{
    /// <summary>
    /// Gets or sets the insight ID
    /// </summary>
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight content
    /// </summary>
    public string Content { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight type
    /// </summary>
    public InsightType Type { get; set; } = InsightType.ConnectionDiscovery;
    
    /// <summary>
    /// Gets or sets the insight significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the insight timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the insight context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight source
    /// </summary>
    public string Source { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight explanation
    /// </summary>
    public string? Explanation { get; set; }
    
    /// <summary>
    /// Gets or sets the related thought IDs
    /// </summary>
    public List<string> RelatedThoughtIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the related question IDs
    /// </summary>
    public List<string> RelatedQuestionIds { get; set; } = new();
}
