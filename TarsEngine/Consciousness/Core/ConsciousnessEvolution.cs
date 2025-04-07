using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a consciousness evolution
/// </summary>
public class ConsciousnessEvolution
{
    /// <summary>
    /// Gets or sets the evolution ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the evolution type
    /// </summary>
    public string Type { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the previous level
    /// </summary>
    public string PreviousLevel { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the previous depth
    /// </summary>
    public double PreviousDepth { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the evolution tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the evolution context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the evolution trigger
    /// </summary>
    public string Trigger { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the evolution outcome
    /// </summary>
    public string Outcome { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the evolution impact (0.0 to 1.0)
    /// </summary>
    public double Impact { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the evolution duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the evolution learning
    /// </summary>
    public string Learning { get; set; } = string.Empty;
}
