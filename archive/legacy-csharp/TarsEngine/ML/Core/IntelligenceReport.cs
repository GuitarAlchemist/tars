namespace TarsEngine.ML.Core;

/// <summary>
/// Represents a comprehensive intelligence report
/// </summary>
public class IntelligenceReport
{
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the current intelligence score
    /// </summary>
    public double CurrentIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the baseline human intelligence score
    /// </summary>
    public double BaselineHumanIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the intelligence ratio compared to human baseline
    /// </summary>
    public double IntelligenceRatio { get; set; }
    
    /// <summary>
    /// Gets or sets the daily growth rate
    /// </summary>
    public double DailyGrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the estimated days to surpass human intelligence
    /// </summary>
    public double EstimatedDaysToSurpassHuman { get; set; }
    
    /// <summary>
    /// Gets or sets whether TARS has surpassed human intelligence
    /// </summary>
    public bool HasSurpassedHuman { get; set; }
    
    /// <summary>
    /// Gets or sets the measurement count
    /// </summary>
    public int MeasurementCount { get; set; }
    
    /// <summary>
    /// Gets or sets the first measurement time
    /// </summary>
    public DateTime FirstMeasurementTime { get; set; }
    
    /// <summary>
    /// Gets or sets the last measurement time
    /// </summary>
    public DateTime LastMeasurementTime { get; set; }
    
    /// <summary>
    /// Gets or sets the current metrics
    /// </summary>
    public Dictionary<string, double> CurrentMetrics { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the comparison with human baseline
    /// </summary>
    public Dictionary<string, ComparisonWithHuman> ComparisonWithHuman { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the strongest dimensions
    /// </summary>
    public List<DimensionStrength> StrongestDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the weakest dimensions
    /// </summary>
    public List<DimensionStrength> WeakestDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the fastest growing dimensions
    /// </summary>
    public List<DimensionGrowth> FastestGrowingDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the slowest growing dimensions
    /// </summary>
    public List<DimensionGrowth> SlowestGrowingDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the improvement recommendations
    /// </summary>
    public List<ImprovementRecommendation> ImprovementRecommendations { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence milestones
    /// </summary>
    public List<IntelligenceMilestone> IntelligenceMilestones { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence projection
    /// </summary>
    public IntelligenceProjection IntelligenceProjection { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the report summary
    /// </summary>
    public string Summary { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the report insights
    /// </summary>
    public List<string> Insights { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the report recommendations
    /// </summary>
    public List<string> Recommendations { get; set; } = new();
}
