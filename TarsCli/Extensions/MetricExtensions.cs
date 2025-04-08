using TarsEngine.Models.Metrics;

namespace TarsCli.Extensions;

/// <summary>
/// Extension methods for metrics
/// </summary>
public static class MetricExtensions
{
    /// <summary>
    /// Determines if a complexity metric is above its threshold
    /// </summary>
    /// <param name="metric">The complexity metric</param>
    /// <returns>True if the metric is above its threshold</returns>
    public static bool IsAboveThreshold(this ComplexityMetric metric)
    {
        return metric.Value > metric.ThresholdValue && metric.ThresholdValue > 0;
    }

    /// <summary>
    /// Determines if a base metric is above its threshold
    /// </summary>
    /// <param name="metric">The base metric</param>
    /// <returns>True if the metric is above its threshold</returns>
    public static bool IsAboveThreshold(this BaseMetric metric)
    {
        if (metric is ComplexityMetric complexityMetric)
        {
            return complexityMetric.IsAboveThreshold();
        }

        if (metric is HalsteadMetric halsteadMetric)
        {
            return halsteadMetric.IsAboveThreshold;
        }

        if (metric is DuplicationMetric duplicationMetric)
        {
            return duplicationMetric.IsAboveThreshold;
        }

        // Default implementation for other metric types
        return false;
    }

    /// <summary>
    /// Gets the target of a base metric
    /// </summary>
    /// <param name="metric">The base metric</param>
    /// <returns>The target of the metric</returns>
    public static string Target(this BaseMetric metric)
    {
        if (metric is ComplexityMetric complexityMetric)
        {
            return complexityMetric.Target;
        }

        if (metric is HalsteadMetric halsteadMetric)
        {
            return halsteadMetric.Target;
        }

        if (metric is MaintainabilityMetric maintainabilityMetric)
        {
            return maintainabilityMetric.Target;
        }

        if (metric is DuplicationMetric duplicationMetric)
        {
            return duplicationMetric.Target;
        }

        // Default implementation for other metric types
        return string.Empty;
    }

    /// <summary>
    /// Gets the threshold value of a base metric
    /// </summary>
    /// <param name="metric">The base metric</param>
    /// <returns>The threshold value of the metric</returns>
    public static double ThresholdValue(this BaseMetric metric)
    {
        if (metric is ComplexityMetric complexityMetric)
        {
            return complexityMetric.ThresholdValue;
        }

        if (metric is HalsteadMetric halsteadMetric)
        {
            return halsteadMetric.ThresholdValue;
        }

        if (metric is MaintainabilityMetric maintainabilityMetric)
        {
            return maintainabilityMetric.ThresholdValue;
        }

        if (metric is DuplicationMetric duplicationMetric)
        {
            return duplicationMetric.ThresholdValue;
        }

        // Default implementation for other metric types
        return 0;
    }

    /// <summary>
    /// Gets the file path of a base metric
    /// </summary>
    /// <param name="metric">The base metric</param>
    /// <returns>The file path of the metric</returns>
    public static string FilePath(this BaseMetric metric)
    {
        if (metric is ComplexityMetric complexityMetric)
        {
            return complexityMetric.FilePath;
        }

        if (metric is HalsteadMetric halsteadMetric)
        {
            return halsteadMetric.FilePath;
        }

        if (metric is MaintainabilityMetric maintainabilityMetric)
        {
            return maintainabilityMetric.FilePath;
        }

        if (metric is DuplicationMetric duplicationMetric)
        {
            return duplicationMetric.FilePath;
        }

        // Default implementation for other metric types
        return string.Empty;
    }

    /// <summary>
    /// Gets the language of a base metric
    /// </summary>
    /// <param name="metric">The base metric</param>
    /// <returns>The language of the metric</returns>
    public static string Language(this BaseMetric metric)
    {
        if (metric is ComplexityMetric complexityMetric)
        {
            return complexityMetric.Language;
        }

        if (metric is HalsteadMetric halsteadMetric)
        {
            return halsteadMetric.Language;
        }

        if (metric is MaintainabilityMetric maintainabilityMetric)
        {
            return maintainabilityMetric.Language;
        }

        if (metric is DuplicationMetric duplicationMetric)
        {
            return duplicationMetric.Language;
        }

        // Default implementation for other metric types
        return string.Empty;
    }
}
