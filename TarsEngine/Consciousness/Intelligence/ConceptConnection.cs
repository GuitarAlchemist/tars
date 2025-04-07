using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a connection between concepts
/// </summary>
public class ConceptConnection
{
    /// <summary>
    /// Gets or sets the connection ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the first concept
    /// </summary>
    public string Concept1 { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the second concept
    /// </summary>
    public string Concept2 { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the connection strength (0.0 to 1.0)
    /// </summary>
    public double Strength { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the connection type
    /// </summary>
    public ConnectionType Type { get; set; } = ConnectionType.Direct;
    
    /// <summary>
    /// Gets or sets the connection creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the connection attributes
    /// </summary>
    public Dictionary<string, double> Attributes { get; set; } = new Dictionary<string, double>();
    
    /// <summary>
    /// Gets or sets the connection tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the connection description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the connection activation level (0.0 to 1.0)
    /// </summary>
    public double ActivationLevel { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the connection last activation timestamp
    /// </summary>
    public DateTime? LastActivationTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the connection usage count
    /// </summary>
    public int UsageCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the connection insight ID
    /// </summary>
    public string? InsightId { get; set; }
    
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
    /// Sets the attribute
    /// </summary>
    /// <param name="attribute">The attribute</param>
    /// <param name="value">The value</param>
    public void SetAttribute(string attribute, double value)
    {
        Attributes[attribute] = value;
    }
    
    /// <summary>
    /// Activates the connection
    /// </summary>
    /// <param name="activationLevel">The activation level</param>
    public void Activate(double activationLevel)
    {
        ActivationLevel = activationLevel;
        LastActivationTimestamp = DateTime.UtcNow;
        UsageCount++;
    }
    
    /// <summary>
    /// Decays the activation level
    /// </summary>
    /// <param name="decayRate">The decay rate</param>
    public void DecayActivation(double decayRate)
    {
        ActivationLevel = Math.Max(0.0, ActivationLevel - decayRate);
    }
    
    /// <summary>
    /// Sets the insight
    /// </summary>
    /// <param name="insightId">The insight ID</param>
    public void SetInsight(string insightId)
    {
        InsightId = insightId;
    }
}
