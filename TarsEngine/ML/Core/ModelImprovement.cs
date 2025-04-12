using System;
using System.Collections.Generic;

namespace TarsEngine.ML.Core;

/// <summary>
/// Represents an improvement to a machine learning model
/// </summary>
public class ModelImprovement
{
    /// <summary>
    /// Gets or sets the improvement ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the timestamp when the improvement was made
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the improvement description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the improvement type
    /// </summary>
    public ImprovementType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the previous metrics
    /// </summary>
    public Dictionary<string, double> PreviousMetrics { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the new metrics
    /// </summary>
    public Dictionary<string, double> NewMetrics { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the improvement magnitude
    /// </summary>
    public double ImprovementMagnitude { get; set; }
    
    /// <summary>
    /// Gets or sets the changes made
    /// </summary>
    public List<string> Changes { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the training examples added
    /// </summary>
    public int TrainingExamplesAdded { get; set; }
    
    /// <summary>
    /// Gets or sets the hyperparameters changed
    /// </summary>
    public Dictionary<string, string> HyperParametersChanged { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence increase
    /// </summary>
    public double IntelligenceIncrease { get; set; }
    
    /// <summary>
    /// Gets or sets whether the improvement was autonomous
    /// </summary>
    public bool IsAutonomous { get; set; }
    
    /// <summary>
    /// Gets or sets the improvement source
    /// </summary>
    public string Source { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the improvement trigger
    /// </summary>
    public string Trigger { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the improvement reasoning
    /// </summary>
    public string Reasoning { get; set; } = string.Empty;
}
