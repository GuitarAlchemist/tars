using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an evaluation of an action against values
/// </summary>
public class ValueEvaluation
{
    /// <summary>
    /// Gets or sets the evaluation ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the action
    /// </summary>
    public string Action { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the value alignments (value name to alignment level)
    /// </summary>
    public Dictionary<string, double> ValueAlignments { get; set; } = new Dictionary<string, double>();
    
    /// <summary>
    /// Gets or sets the overall alignment (0.0 to 1.0)
    /// </summary>
    public double OverallAlignment { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets whether the action is aligned with values
    /// </summary>
    public bool IsAligned { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the recommendation
    /// </summary>
    public string Recommendation { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the evaluation tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the evaluation category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the evaluation confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.8;
    
    /// <summary>
    /// Gets or sets the evaluation reasoning
    /// </summary>
    public string Reasoning { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the evaluation alternatives
    /// </summary>
    public List<string> Alternatives { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets the most aligned values
    /// </summary>
    /// <param name="count">The number of values to return</param>
    /// <returns>The most aligned values</returns>
    public List<string> GetMostAlignedValues(int count)
    {
        return ValueAlignments
            .OrderByDescending(v => v.Value)
            .Take(count)
            .Select(v => v.Key)
            .ToList();
    }
    
    /// <summary>
    /// Gets the least aligned values
    /// </summary>
    /// <param name="count">The number of values to return</param>
    /// <returns>The least aligned values</returns>
    public List<string> GetLeastAlignedValues(int count)
    {
        return ValueAlignments
            .OrderBy(v => v.Value)
            .Take(count)
            .Select(v => v.Key)
            .ToList();
    }
}
