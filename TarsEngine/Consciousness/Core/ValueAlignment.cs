using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an alignment between values
/// </summary>
public class ValueAlignment
{
    /// <summary>
    /// Gets or sets the alignment ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the conflict ID
    /// </summary>
    public string ConflictId { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the values
    /// </summary>
    public List<string> Values { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the resolution
    /// </summary>
    public string Resolution { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the alignment context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the alignment tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the alignment category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the alignment examples
    /// </summary>
    public List<string> Examples { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the alignment principles
    /// </summary>
    public List<string> Principles { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the alignment learning
    /// </summary>
    public string Learning { get; set; } = string.Empty;
}
