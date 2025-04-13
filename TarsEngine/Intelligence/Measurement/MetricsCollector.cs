using Microsoft.Extensions.Logging;
using TarsEngine.Data;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// Collects and stores metrics related to intelligence progression
/// </summary>
public class MetricsCollector
{
    private readonly ILogger<MetricsCollector> _logger;
    private readonly MetricsRepository _repository;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="MetricsCollector"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="repository">The metrics repository</param>
    public MetricsCollector(ILogger<MetricsCollector> logger, MetricsRepository repository)
    {
        _logger = logger;
        _repository = repository;
    }
    
    /// <summary>
    /// Initializes the metrics collector
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("Initializing Metrics Collector");
        
        // Ensure the repository is ready
        await Task.CompletedTask;
    }
    
    /// <summary>
    /// Collects a metric
    /// </summary>
    /// <param name="metric">The metric to collect</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task CollectMetricAsync(BaseMetric metric)
    {
        _logger.LogInformation("Collecting metric: {MetricName} with value: {MetricValue}", metric.Name, metric.Value);
        
        // Add the metric to the repository
        await _repository.AddMetricAsync(metric);
    }
    
    /// <summary>
    /// Collects multiple metrics
    /// </summary>
    /// <param name="metrics">The metrics to collect</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task CollectMetricsAsync(IEnumerable<BaseMetric> metrics)
    {
        _logger.LogInformation("Collecting multiple metrics");
        
        // Add the metrics to the repository
        await _repository.AddMetricsAsync(metrics);
    }
    
    /// <summary>
    /// Gets metrics by category
    /// </summary>
    /// <param name="category">The metric category</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metrics</returns>
    public async Task<IEnumerable<BaseMetric>> GetMetricsByCategoryAsync(
        MetricCategory category, DateTime? startTime = null, DateTime? endTime = null)
    {
        _logger.LogInformation("Getting metrics by category: {Category}", category);
        
        // Get the metrics from the repository
        return await _repository.GetMetricsByCategoryAsync(category, startTime, endTime);
    }
    
    /// <summary>
    /// Gets metrics by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metrics</returns>
    public async Task<IEnumerable<BaseMetric>> GetMetricsByNameAsync(
        string name, DateTime? startTime = null, DateTime? endTime = null)
    {
        _logger.LogInformation("Getting metrics by name: {MetricName}", name);
        
        // Get the metrics from the repository
        return await _repository.GetMetricsByNameAsync(name, startTime, endTime);
    }
    
    /// <summary>
    /// Gets the latest metric value by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <returns>The latest metric value</returns>
    public async Task<double?> GetLatestMetricValueAsync(string name)
    {
        _logger.LogInformation("Getting latest metric value by name: {MetricName}", name);
        
        // Get the latest metric value from the repository
        return await _repository.GetLatestMetricValueAsync(name);
    }
    
    /// <summary>
    /// Gets the latest metric timestamp by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <returns>The latest metric timestamp</returns>
    public async Task<DateTime?> GetLatestMetricTimestampAsync(string name)
    {
        _logger.LogInformation("Getting latest metric timestamp by name: {MetricName}", name);
        
        // Get the metrics from the repository
        var metrics = await _repository.GetMetricsByNameAsync(name);
        
        // Find the latest metric
        DateTime? latestTimestamp = null;
        foreach (var metric in metrics)
        {
            if (!latestTimestamp.HasValue || metric.Timestamp > latestTimestamp.Value)
            {
                latestTimestamp = metric.Timestamp;
            }
        }
        
        return latestTimestamp;
    }
    
    /// <summary>
    /// Gets metric statistics
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metric statistics</returns>
    public async Task<MetricStatistics> GetMetricStatisticsAsync(
        string name, DateTime? startTime = null, DateTime? endTime = null)
    {
        _logger.LogInformation("Getting metric statistics by name: {MetricName}", name);
        
        // Get the metrics from the repository
        var metrics = await _repository.GetMetricsByNameAsync(name, startTime, endTime);
        
        // Calculate statistics
        return CalculateStatistics(name, metrics);
    }
    
    /// <summary>
    /// Calculates statistics for a set of metrics
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="metrics">The metrics</param>
    /// <returns>The metric statistics</returns>
    private MetricStatistics CalculateStatistics(string name, IEnumerable<BaseMetric> metrics)
    {
        var metricsList = new List<BaseMetric>(metrics);
        
        if (metricsList.Count == 0)
        {
            return new MetricStatistics { MetricName = name };
        }
        
        var values = new List<double>();
        var timestamps = new List<DateTime>();
        
        foreach (var metric in metricsList)
        {
            values.Add(metric.Value);
            timestamps.Add(metric.Timestamp);
        }
        
        values.Sort();
        timestamps.Sort();
        
        var statistics = new MetricStatistics
        {
            MetricName = name,
            Count = metricsList.Count,
            MinValue = values[0],
            MaxValue = values[values.Count - 1],
            AverageValue = CalculateAverage(values),
            MedianValue = CalculateMedian(values),
            StandardDeviation = CalculateStandardDeviation(values),
            FirstTimestamp = timestamps[0],
            LastTimestamp = timestamps[timestamps.Count - 1]
        };
        
        // Calculate growth rate
        if (metricsList.Count >= 2)
        {
            var firstValue = metricsList[0].Value;
            var lastValue = metricsList[metricsList.Count - 1].Value;
            var timeSpan = (statistics.LastTimestamp - statistics.FirstTimestamp).TotalDays;
            
            if (timeSpan > 0)
            {
                statistics.GrowthRatePerDay = (lastValue - firstValue) / timeSpan;
                
                // Calculate logarithmic growth rate
                if (firstValue > 0 && lastValue > 0)
                {
                    statistics.LogGrowthRatePerDay = (Math.Log10(lastValue) - Math.Log10(firstValue)) / timeSpan;
                }
            }
        }
        
        // Determine trend
        statistics.Trend = DetermineTrend(values);
        
        // Calculate percentiles
        statistics.Percentiles = CalculatePercentiles(values);
        
        return statistics;
    }
    
    /// <summary>
    /// Calculates the average of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The average</returns>
    private double CalculateAverage(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }
        
        double sum = 0;
        foreach (var value in values)
        {
            sum += value;
        }
        
        return sum / values.Count;
    }
    
    /// <summary>
    /// Calculates the median of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The median</returns>
    private double CalculateMedian(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }
        
        int middle = values.Count / 2;
        
        if (values.Count % 2 == 0)
        {
            return (values[middle - 1] + values[middle]) / 2;
        }
        else
        {
            return values[middle];
        }
    }
    
    /// <summary>
    /// Calculates the standard deviation of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The standard deviation</returns>
    private double CalculateStandardDeviation(List<double> values)
    {
        if (values.Count <= 1)
        {
            return 0;
        }
        
        double average = CalculateAverage(values);
        double sumOfSquaredDifferences = 0;
        
        foreach (var value in values)
        {
            double difference = value - average;
            sumOfSquaredDifferences += difference * difference;
        }
        
        return Math.Sqrt(sumOfSquaredDifferences / (values.Count - 1));
    }
    
    /// <summary>
    /// Determines the trend of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The trend type</returns>
    private TrendType DetermineTrend(List<double> values)
    {
        if (values.Count < 3)
        {
            return TrendType.Unknown;
        }
        
        int increasing = 0;
        int decreasing = 0;
        
        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] > values[i - 1])
            {
                increasing++;
            }
            else if (values[i] < values[i - 1])
            {
                decreasing++;
            }
        }
        
        double increasingRatio = (double)increasing / (values.Count - 1);
        double decreasingRatio = (double)decreasing / (values.Count - 1);
        
        if (increasingRatio > 0.7)
        {
            return TrendType.Increasing;
        }
        else if (decreasingRatio > 0.7)
        {
            return TrendType.Decreasing;
        }
        else if (increasingRatio > 0.5 && decreasingRatio < 0.3)
        {
            return TrendType.Accelerating;
        }
        else if (decreasingRatio > 0.5 && increasingRatio < 0.3)
        {
            return TrendType.Decelerating;
        }
        else if (Math.Abs(increasingRatio - decreasingRatio) < 0.2)
        {
            return TrendType.Fluctuating;
        }
        else
        {
            return TrendType.Stable;
        }
    }
    
    /// <summary>
    /// Calculates percentiles for a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The percentiles</returns>
    private Dictionary<int, double> CalculatePercentiles(List<double> values)
    {
        var percentiles = new Dictionary<int, double>();
        
        if (values.Count == 0)
        {
            return percentiles;
        }
        
        // Calculate 10th, 25th, 50th, 75th, and 90th percentiles
        percentiles[10] = CalculatePercentile(values, 10);
        percentiles[25] = CalculatePercentile(values, 25);
        percentiles[50] = CalculatePercentile(values, 50);
        percentiles[75] = CalculatePercentile(values, 75);
        percentiles[90] = CalculatePercentile(values, 90);
        
        return percentiles;
    }
    
    /// <summary>
    /// Calculates a percentile for a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <param name="percentile">The percentile</param>
    /// <returns>The percentile value</returns>
    private double CalculatePercentile(List<double> values, int percentile)
    {
        if (values.Count == 0)
        {
            return 0;
        }
        
        double index = (percentile / 100.0) * (values.Count - 1);
        int lowerIndex = (int)Math.Floor(index);
        int upperIndex = (int)Math.Ceiling(index);
        
        if (lowerIndex == upperIndex)
        {
            return values[lowerIndex];
        }
        else
        {
            double weight = index - lowerIndex;
            return values[lowerIndex] * (1 - weight) + values[upperIndex] * weight;
        }
    }
}
