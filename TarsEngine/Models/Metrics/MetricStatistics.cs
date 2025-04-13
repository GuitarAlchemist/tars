namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents statistics for a metric
/// </summary>
public class MetricStatistics
{
    /// <summary>
    /// Gets or sets the metric name
    /// </summary>
    public string MetricName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the count
    /// </summary>
    public int Count { get; set; }
    
    /// <summary>
    /// Gets or sets the minimum value
    /// </summary>
    public double MinValue { get; set; }
    
    /// <summary>
    /// Gets or sets the maximum value
    /// </summary>
    public double MaxValue { get; set; }
    
    /// <summary>
    /// Gets or sets the average value
    /// </summary>
    public double AverageValue { get; set; }
    
    /// <summary>
    /// Gets or sets the median value
    /// </summary>
    public double MedianValue { get; set; }
    
    /// <summary>
    /// Gets or sets the standard deviation
    /// </summary>
    public double StandardDeviation { get; set; }
    
    /// <summary>
    /// Gets or sets the first timestamp
    /// </summary>
    public DateTime FirstTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the last timestamp
    /// </summary>
    public DateTime LastTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the growth rate per day
    /// </summary>
    public double GrowthRatePerDay { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic growth rate per day
    /// </summary>
    public double LogGrowthRatePerDay { get; set; }
    
    /// <summary>
    /// Gets or sets the trend
    /// </summary>
    public TrendType Trend { get; set; }
    
    /// <summary>
    /// Gets or sets the confidence level (0.0-1.0)
    /// </summary>
    public double ConfidenceLevel { get; set; } = 0.95;
    
    /// <summary>
    /// Gets or sets the percentiles
    /// </summary>
    public Dictionary<int, double> Percentiles { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the forecast values
    /// </summary>
    public Dictionary<DateTime, double> ForecastValues { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the forecast confidence intervals
    /// </summary>
    public Dictionary<DateTime, (double Lower, double Upper)> ForecastConfidenceIntervals { get; set; } = new();
}

/// <summary>
/// Represents a trend type
/// </summary>
public enum TrendType
{
    /// <summary>
    /// Increasing trend
    /// </summary>
    Increasing,
    
    /// <summary>
    /// Decreasing trend
    /// </summary>
    Decreasing,
    
    /// <summary>
    /// Stable trend
    /// </summary>
    Stable,
    
    /// <summary>
    /// Fluctuating trend
    /// </summary>
    Fluctuating,
    
    /// <summary>
    /// Accelerating trend
    /// </summary>
    Accelerating,
    
    /// <summary>
    /// Decelerating trend
    /// </summary>
    Decelerating,
    
    /// <summary>
    /// Unknown trend
    /// </summary>
    Unknown
}
