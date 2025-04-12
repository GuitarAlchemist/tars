using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an emotional regulation
/// </summary>
public class EmotionalRegulation
{
    /// <summary>
    /// Gets or sets the regulation ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the regulated emotions
    /// </summary>
    public List<string> RegulatedEmotions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; }
    
    /// <summary>
    /// Gets or sets the regulation strategy
    /// </summary>
    public string Strategy { get; set; } = "Automatic";
    
    /// <summary>
    /// Gets or sets the regulation success (0.0 to 1.0)
    /// </summary>
    public double Success { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the regulation difficulty (0.0 to 1.0)
    /// </summary>
    public double Difficulty { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the regulation context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the regulation tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the regulation trigger
    /// </summary>
    public string Trigger { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the regulation outcome
    /// </summary>
    public string Outcome { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the regulation learning
    /// </summary>
    public string Learning { get; set; } = string.Empty;
}
