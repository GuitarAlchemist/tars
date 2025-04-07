using System.Collections.Generic;

namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents the result of a complexity analysis operation
/// </summary>
public class ComplexityAnalysisResult
{
    /// <summary>
    /// Gets or sets the complexity metrics
    /// </summary>
    public List<ComplexityMetric> ComplexityMetrics { get; set; } = new List<ComplexityMetric>();
    
    /// <summary>
    /// Gets or sets the Halstead metrics
    /// </summary>
    public List<HalsteadMetric> HalsteadMetrics { get; set; } = new List<HalsteadMetric>();
    
    /// <summary>
    /// Gets or sets the maintainability metrics
    /// </summary>
    public List<MaintainabilityMetric> MaintainabilityMetrics { get; set; } = new List<MaintainabilityMetric>();
    
    /// <summary>
    /// Gets or sets the readability metrics
    /// </summary>
    public List<ReadabilityMetric> ReadabilityMetrics { get; set; } = new List<ReadabilityMetric>();
}
