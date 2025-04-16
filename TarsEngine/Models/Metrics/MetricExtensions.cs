namespace TarsEngine.Models.Metrics;

/// <summary>
/// Extension methods for metric classes to provide compatibility with different interfaces
/// </summary>
public static class MetricExtensions
{
    /// <summary>
    /// Gets the description for a metric
    /// </summary>
    public static string GetDescription(this BaseMetric metric)
    {
        return $"{metric.Name} - {metric.Category}";
    }

    /// <summary>
    /// Gets the Type property for a metric
    /// </summary>
    public static MetricType GetType(this BaseMetric metric)
    {
        // Convert the category to a MetricType
        return metric.Category switch
        {
            MetricCategory.Complexity => MetricType.Complexity,
            MetricCategory.Duplication => MetricType.Size,
            MetricCategory.Learning => MetricType.Cohesion,
            MetricCategory.Novelty => MetricType.Inheritance,
            MetricCategory.Performance => MetricType.Performance,
            MetricCategory.Readability => MetricType.Other,
            _ => MetricType.Other
        };
    }

    /// <summary>
    /// Gets the GoodThreshold for a metric
    /// </summary>
    public static double GetGoodThreshold(this BaseMetric metric)
    {
        // Default thresholds based on metric type
        if (metric is ComplexityMetric)
            return 10;
        if (metric is MaintainabilityMetric)
            return 80;
        if (metric is HalsteadMetric)
            return 20;

        return 50; // Default value
    }

    /// <summary>
    /// Gets the AcceptableThreshold for a metric
    /// </summary>
    public static double GetAcceptableThreshold(this BaseMetric metric)
    {
        // Default thresholds based on metric type
        if (metric is ComplexityMetric)
            return 20;
        if (metric is MaintainabilityMetric)
            return 60;
        if (metric is HalsteadMetric)
            return 40;

        return 30; // Default value
    }

    /// <summary>
    /// Gets the PoorThreshold for a metric
    /// </summary>
    public static double GetPoorThreshold(this BaseMetric metric)
    {
        // Default thresholds based on metric type
        if (metric is ComplexityMetric)
            return 30;
        if (metric is MaintainabilityMetric)
            return 40;
        if (metric is HalsteadMetric)
            return 60;

        return 20; // Default value
    }

    /// <summary>
    /// Gets the TargetType for a ComplexityMetric
    /// </summary>
    public static string GetTargetType(this ComplexityMetric metric)
    {
        return metric.Target ?? "Unknown";
    }

    /// <summary>
    /// Gets the ThresholdValue for a ComplexityMetric based on the complexity type
    /// </summary>
    public static double GetThresholdValue(this ComplexityMetric metric, ComplexityType complexityType)
    {
        switch (complexityType)
        {
            case ComplexityType.Cyclomatic:
                return 10;
            case ComplexityType.Cognitive:
                return 15;
            case ComplexityType.Halstead:
                return 20;
            case ComplexityType.Maintainability:
                return 70;
            default:
                return 25;
        }
    }
}