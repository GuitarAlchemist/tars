using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a conflict between values
/// </summary>
public class ValueConflict
{
    /// <summary>
    /// Gets or sets the conflict ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the first value
    /// </summary>
    public string Value1 { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the second value
    /// </summary>
    public string Value2 { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the conflict description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the conflict severity (0.0 to 1.0)
    /// </summary>
    public double Severity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the conflict timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the conflict context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the conflict resolution
    /// </summary>
    public string Resolution { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets whether the conflict is resolved
    /// </summary>
    public bool IsResolved => !string.IsNullOrEmpty(Resolution);
    
    /// <summary>
    /// Gets or sets the resolution timestamp
    /// </summary>
    public DateTime? ResolutionTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the conflict tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the conflict category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the conflict examples
    /// </summary>
    public List<string> Examples { get; set; } = new List<string>();
    
    /// <summary>
    /// Resolves the conflict
    /// </summary>
    /// <param name="resolution">The resolution</param>
    public void Resolve(string resolution)
    {
        Resolution = resolution;
        ResolutionTimestamp = DateTime.UtcNow;
    }
}
