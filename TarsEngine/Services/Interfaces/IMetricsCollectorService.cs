using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Service for collecting and storing metrics related to intelligence progression
/// </summary>
public interface IMetricsCollectorService
{
    /// <summary>
    /// Collects a metric
    /// </summary>
    /// <param name="metric">The metric to collect</param>
    /// <returns>True if the metric was collected successfully</returns>
    Task<bool> CollectMetricAsync(BaseMetric metric);
    
    /// <summary>
    /// Collects multiple metrics
    /// </summary>
    /// <param name="metrics">The metrics to collect</param>
    /// <returns>True if the metrics were collected successfully</returns>
    Task<bool> CollectMetricsAsync(IEnumerable<BaseMetric> metrics);
    
    /// <summary>
    /// Gets metrics by category
    /// </summary>
    /// <param name="category">The metric category</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metrics</returns>
    Task<IEnumerable<BaseMetric>> GetMetricsByCategoryAsync(MetricCategory category, DateTime? startTime = null, DateTime? endTime = null);
    
    /// <summary>
    /// Gets metrics by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metrics</returns>
    Task<IEnumerable<BaseMetric>> GetMetricsByNameAsync(string name, DateTime? startTime = null, DateTime? endTime = null);
    
    /// <summary>
    /// Gets the latest metric value by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <returns>The latest metric value</returns>
    Task<double?> GetLatestMetricValueAsync(string name);
    
    /// <summary>
    /// Gets the metric statistics
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metric statistics</returns>
    Task<MetricStatistics> GetMetricStatisticsAsync(string name, DateTime? startTime = null, DateTime? endTime = null);
    
    /// <summary>
    /// Gets all metric names
    /// </summary>
    /// <returns>All metric names</returns>
    Task<IEnumerable<string>> GetAllMetricNamesAsync();
    
    /// <summary>
    /// Gets all metric categories
    /// </summary>
    /// <returns>All metric categories</returns>
    Task<IEnumerable<MetricCategory>> GetAllMetricCategoriesAsync();
    
    /// <summary>
    /// Clears all metrics
    /// </summary>
    /// <returns>True if the metrics were cleared successfully</returns>
    Task<bool> ClearAllMetricsAsync();
}
