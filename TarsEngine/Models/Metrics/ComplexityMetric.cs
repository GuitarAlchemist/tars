using System;

namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a complexity metric
/// </summary>
public class ComplexityMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ComplexityMetric"/> class
    /// </summary>
    public ComplexityMetric()
    {
        Category = MetricCategory.Complexity;
    }
    
    /// <summary>
    /// Gets or sets the complexity type
    /// </summary>
    public ComplexityType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the target
    /// </summary>
    public string Target { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the complexity threshold
    /// </summary>
    public double Threshold { get; set; }
    
    /// <summary>
    /// Gets whether the complexity exceeds the threshold
    /// </summary>
    public bool ExceedsThreshold => Value > Threshold;
    
    /// <summary>
    /// Gets or sets the complexity dimension
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the logarithmic value
    /// </summary>
    public double LogValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic threshold
    /// </summary>
    public double LogThreshold { get; set; }
}

/// <summary>
/// Represents a complexity type
/// </summary>
public enum ComplexityType
{
    /// <summary>
    /// Cyclomatic complexity
    /// </summary>
    Cyclomatic,
    
    /// <summary>
    /// Cognitive complexity
    /// </summary>
    Cognitive,
    
    /// <summary>
    /// Halstead complexity
    /// </summary>
    Halstead,
    
    /// <summary>
    /// Maintainability index
    /// </summary>
    Maintainability,
    
    /// <summary>
    /// Structural complexity
    /// </summary>
    Structural,
    
    /// <summary>
    /// Algorithmic complexity
    /// </summary>
    Algorithmic,
    
    /// <summary>
    /// Other complexity
    /// </summary>
    Other
}
