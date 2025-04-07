using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an attention focus
/// </summary>
public class AttentionFocus
{
    /// <summary>
    /// Gets or sets the attention focus ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the focus
    /// </summary>
    public string Focus { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intensity (0.0 to 1.0)
    /// </summary>
    public double Intensity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the previous focus
    /// </summary>
    public string PreviousFocus { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the attention tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the attention context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the attention source
    /// </summary>
    public string Source { get; set; } = "Internal";
    
    /// <summary>
    /// Gets or sets the attention priority (0.0 to 1.0)
    /// </summary>
    public double Priority { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the attention category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the attention outcome
    /// </summary>
    public string Outcome { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the attention value (0.0 to 1.0)
    /// </summary>
    public double Value { get; set; } = 0.5;
}
