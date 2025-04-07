using System;
using System.Collections.Generic;

namespace TarsEngine.Models.Metrics;

/// <summary>
/// Base class for all metrics
/// </summary>
public abstract class BaseMetric
{
    /// <summary>
    /// Gets or sets the metric ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the metric name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the metric value
    /// </summary>
    public double Value { get; set; }

    /// <summary>
    /// Gets or sets the metric timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the metric category
    /// </summary>
    public MetricCategory Category { get; set; }

    /// <summary>
    /// Gets or sets the metric source
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the metric tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the metric metadata
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}

/// <summary>
/// Represents a metric category
/// </summary>
public enum MetricCategory
{
    /// <summary>
    /// Performance metric
    /// </summary>
    Performance,

    /// <summary>
    /// Learning metric
    /// </summary>
    Learning,

    /// <summary>
    /// Complexity metric
    /// </summary>
    Complexity,

    /// <summary>
    /// Novelty metric
    /// </summary>
    Novelty,

    /// <summary>
    /// Readability metric
    /// </summary>
    Readability,

    /// <summary>
    /// Duplication metric
    /// </summary>
    Duplication,

    /// <summary>
    /// Other metric
    /// </summary>
    Other
}
