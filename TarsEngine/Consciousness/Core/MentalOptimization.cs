using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a mental optimization
/// </summary>
public class MentalOptimization
{
    /// <summary>
    /// Gets or sets the optimization ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the optimization type
    /// </summary>
    public OptimizationType Type { get; set; } = OptimizationType.None;
    
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
    /// Gets or sets the optimization tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the optimization context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the optimization trigger
    /// </summary>
    public string Trigger { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the optimization outcome
    /// </summary>
    public string Outcome { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the optimization impact (0.0 to 1.0)
    /// </summary>
    public double Impact { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the optimization duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the optimization success (0.0 to 1.0)
    /// </summary>
    public double Success { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the optimization learning
    /// </summary>
    public string Learning { get; set; } = string.Empty;
}
