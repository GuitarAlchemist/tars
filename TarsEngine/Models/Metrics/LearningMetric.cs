namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a learning metric
/// </summary>
public class LearningMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LearningMetric"/> class
    /// </summary>
    public LearningMetric()
    {
        Category = MetricCategory.Learning;
    }
    
    /// <summary>
    /// Gets or sets the learning dimension
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the learning rate
    /// </summary>
    public double LearningRate { get; set; }
    
    /// <summary>
    /// Gets or sets the previous value
    /// </summary>
    public double PreviousValue { get; set; }
    
    /// <summary>
    /// Gets or sets the time since previous measurement
    /// </summary>
    public TimeSpan TimeSincePrevious { get; set; }
    
    /// <summary>
    /// Gets the improvement ratio
    /// </summary>
    public double ImprovementRatio => PreviousValue > 0 ? Value / PreviousValue : 0;
    
    /// <summary>
    /// Gets the improvement rate per day
    /// </summary>
    public double ImprovementRatePerDay => TimeSincePrevious.TotalDays > 0 ? (Value - PreviousValue) / TimeSincePrevious.TotalDays : 0;
    
    /// <summary>
    /// Gets or sets the learning type
    /// </summary>
    public LearningType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic learning rate
    /// </summary>
    public double LogLearningRate { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic value
    /// </summary>
    public double LogValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic previous value
    /// </summary>
    public double LogPreviousValue { get; set; }
}

/// <summary>
/// Represents a learning type
/// </summary>
public enum LearningType
{
    /// <summary>
    /// Supervised learning
    /// </summary>
    Supervised,
    
    /// <summary>
    /// Unsupervised learning
    /// </summary>
    Unsupervised,
    
    /// <summary>
    /// Reinforcement learning
    /// </summary>
    Reinforcement,
    
    /// <summary>
    /// Transfer learning
    /// </summary>
    Transfer,
    
    /// <summary>
    /// Meta-learning
    /// </summary>
    Meta,
    
    /// <summary>
    /// Other learning
    /// </summary>
    Other
}
