using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a value in the value system
/// </summary>
public class Value
{
    /// <summary>
    /// Gets or sets the value name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the value description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the value importance (0.0 to 1.0)
    /// </summary>
    public double Importance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the value category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the value tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the related values
    /// </summary>
    public List<string> RelatedValues { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the value examples
    /// </summary>
    public List<string> Examples { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the value counter-examples
    /// </summary>
    public List<string> CounterExamples { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the value principles
    /// </summary>
    public List<string> Principles { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the value source
    /// </summary>
    public string Source { get; set; } = "Core";
    
    /// <summary>
    /// Gets or sets the value creation date
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the value last modified date
    /// </summary>
    public DateTime LastModified { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the value stability (0.0 to 1.0)
    /// </summary>
    public double Stability { get; set; } = 0.8;
    
    /// <summary>
    /// Gets or sets the value flexibility (0.0 to 1.0)
    /// </summary>
    public double Flexibility { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the value context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
}
