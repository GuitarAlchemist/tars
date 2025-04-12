using System;
using System.Collections.Generic;

namespace TarsEngine.ML.Core;

/// <summary>
/// Represents a snapshot of intelligence metrics at a point in time
/// </summary>
public class IntelligenceSnapshot
{
    /// <summary>
    /// Gets or sets the snapshot ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the intelligence score
    /// </summary>
    public double IntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the intelligence ratio compared to human baseline
    /// </summary>
    public double IntelligenceRatio { get; set; }
    
    /// <summary>
    /// Gets or sets the metrics
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the source of the snapshot
    /// </summary>
    public string Source { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the improvement factors
    /// </summary>
    public List<string> ImprovementFactors { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the improvement magnitude
    /// </summary>
    public double ImprovementMagnitude { get; set; }
    
    /// <summary>
    /// Gets or sets the growth rate
    /// </summary>
    public double GrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the strongest dimensions
    /// </summary>
    public List<string> StrongestDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the weakest dimensions
    /// </summary>
    public List<string> WeakestDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the fastest growing dimensions
    /// </summary>
    public List<string> FastestGrowingDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the slowest growing dimensions
    /// </summary>
    public List<string> SlowestGrowingDimensions { get; set; } = new();
}
