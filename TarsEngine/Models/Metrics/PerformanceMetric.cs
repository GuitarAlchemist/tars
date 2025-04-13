namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a performance metric
/// </summary>
public class PerformanceMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PerformanceMetric"/> class
    /// </summary>
    public PerformanceMetric()
    {
        Category = MetricCategory.Performance;
    }
    
    /// <summary>
    /// Gets or sets the performance dimension
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the performance unit
    /// </summary>
    public string Unit { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the baseline value
    /// </summary>
    public double BaselineValue { get; set; }
    
    /// <summary>
    /// Gets or sets the target value
    /// </summary>
    public double TargetValue { get; set; }
    
    /// <summary>
    /// Gets the performance ratio compared to baseline
    /// </summary>
    public double BaselineRatio => BaselineValue > 0 ? Value / BaselineValue : 0;
    
    /// <summary>
    /// Gets the performance ratio compared to target
    /// </summary>
    public double TargetRatio => TargetValue > 0 ? Value / TargetValue : 0;
    
    /// <summary>
    /// Gets or sets the performance type
    /// </summary>
    public PerformanceType Type { get; set; }
}

/// <summary>
/// Represents a performance type
/// </summary>
public enum PerformanceType
{
    /// <summary>
    /// Speed performance
    /// </summary>
    Speed,
    
    /// <summary>
    /// Accuracy performance
    /// </summary>
    Accuracy,
    
    /// <summary>
    /// Efficiency performance
    /// </summary>
    Efficiency,
    
    /// <summary>
    /// Resource usage performance
    /// </summary>
    ResourceUsage,
    
    /// <summary>
    /// Throughput performance
    /// </summary>
    Throughput,
    
    /// <summary>
    /// Other performance
    /// </summary>
    Other
}
