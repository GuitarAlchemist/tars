using Microsoft.Extensions.Logging;
using TarsEngine.Data;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for collecting and storing metrics
/// </summary>
public class MetricsCollectorService : IMetricsCollectorService
{
    private readonly ILogger<MetricsCollectorService> _logger;
    private readonly MetricsRepository _repository;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="MetricsCollectorService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="repository">The repository</param>
    public MetricsCollectorService(ILogger<MetricsCollectorService> logger, MetricsRepository repository)
    {
        _logger = logger;
        _repository = repository;
    }
    
    /// <inheritdoc/>
    public async Task<bool> CollectMetricAsync(BaseMetric metric)
    {
        try
        {
            _logger.LogInformation("Collecting metric: {MetricName} with value: {MetricValue}", metric.Name, metric.Value);
            
            // Add the metric to the repository
            return await _repository.AddMetricAsync(metric);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting metric: {MetricName}", metric.Name);
            return false;
        }
    }
    
    /// <inheritdoc/>
    public async Task<bool> CollectMetricsAsync(IEnumerable<BaseMetric> metrics)
    {
        try
        {
            _logger.LogInformation("Collecting {MetricCount} metrics", metrics.Count());
            
            // Add the metrics to the repository
            return await _repository.AddMetricsAsync(metrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting metrics");
            return false;
        }
    }
    
    /// <inheritdoc/>
    public async Task<IEnumerable<BaseMetric>> GetMetricsByCategoryAsync(MetricCategory category, DateTime? startTime = null, DateTime? endTime = null)
    {
        try
        {
            _logger.LogInformation("Getting metrics by category: {Category}", category);
            
            // Get the metrics from the repository
            return await _repository.GetMetricsByCategoryAsync(category, startTime, endTime);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metrics by category: {Category}", category);
            return [];
        }
    }
    
    /// <inheritdoc/>
    public async Task<IEnumerable<BaseMetric>> GetMetricsByNameAsync(string name, DateTime? startTime = null, DateTime? endTime = null)
    {
        try
        {
            _logger.LogInformation("Getting metrics by name: {MetricName}", name);
            
            // Get the metrics from the repository
            return await _repository.GetMetricsByNameAsync(name, startTime, endTime);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metrics by name: {MetricName}", name);
            return [];
        }
    }
    
    /// <inheritdoc/>
    public async Task<double?> GetLatestMetricValueAsync(string name)
    {
        try
        {
            _logger.LogInformation("Getting latest metric value by name: {MetricName}", name);
            
            // Get the latest metric value from the repository
            return await _repository.GetLatestMetricValueAsync(name);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting latest metric value by name: {MetricName}", name);
            return null;
        }
    }
    
    /// <inheritdoc/>
    public async Task<MetricStatistics> GetMetricStatisticsAsync(string name, DateTime? startTime = null, DateTime? endTime = null)
    {
        try
        {
            _logger.LogInformation("Getting metric statistics by name: {MetricName}", name);
            
            // Get the metrics from the repository
            var metrics = await _repository.GetMetricsByNameAsync(name, startTime, endTime);
            
            // Calculate statistics
            return CalculateStatistics(name, metrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metric statistics by name: {MetricName}", name);
            return new MetricStatistics { MetricName = name };
        }
    }
    
    /// <inheritdoc/>
    public async Task<IEnumerable<string>> GetAllMetricNamesAsync()
    {
        try
        {
            _logger.LogInformation("Getting all metric names");
            
            // Get all metric names from the repository
            return await _repository.GetAllMetricNamesAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all metric names");
            return [];
        }
    }
    
    /// <inheritdoc/>
    public async Task<IEnumerable<MetricCategory>> GetAllMetricCategoriesAsync()
    {
        try
        {
            _logger.LogInformation("Getting all metric categories");
            
            // Get all metric names
            var metricNames = await _repository.GetAllMetricNamesAsync();
            
            // Get a sample metric for each name to determine its category
            var categories = new HashSet<MetricCategory>();
            
            foreach (var name in metricNames)
            {
                var metrics = await _repository.GetMetricsByNameAsync(name, null, null);
                var metric = metrics.FirstOrDefault();
                
                if (metric != null)
                {
                    categories.Add(metric.Category);
                }
            }
            
            return categories;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all metric categories");
            return [];
        }
    }
    
    /// <inheritdoc/>
    public async Task<bool> ClearAllMetricsAsync()
    {
        try
        {
            _logger.LogInformation("Clearing all metrics");
            
            // Clear all metrics from the repository
            return await _repository.ClearAllMetricsAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing all metrics");
            return false;
        }
    }
    
    /// <summary>
    /// Calculates statistics for a set of metrics
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="metrics">The metrics</param>
    /// <returns>The metric statistics</returns>
    private MetricStatistics CalculateStatistics(string name, IEnumerable<BaseMetric> metrics)
    {
        var metricsList = metrics.OrderBy(m => m.Timestamp).ToList();
        
        if (metricsList.Count == 0)
        {
            return new MetricStatistics { MetricName = name };
        }
        
        var values = metricsList.Select(m => m.Value).ToList();
        var timestamps = metricsList.Select(m => m.Timestamp).ToList();
        
        var statistics = new MetricStatistics
        {
            MetricName = name,
            Count = metricsList.Count,
            MinValue = values.Min(),
            MaxValue = values.Max(),
            AverageValue = values.Average(),
            MedianValue = CalculateMedian(values),
            StandardDeviation = CalculateStandardDeviation(values),
            FirstTimestamp = timestamps.First(),
            LastTimestamp = timestamps.Last()
        };
        
        // Calculate growth rate
        if (metricsList.Count >= 2)
        {
            var firstValue = metricsList.First().Value;
            var lastValue = metricsList.Last().Value;
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
        statistics.Trend = DetermineTrend(metricsList);
        
        // Calculate percentiles
        statistics.Percentiles = CalculatePercentiles(values);
        
        // Calculate forecast values
        statistics.ForecastValues = CalculateForecastValues(metricsList, 7); // Forecast for 7 days
        
        // Calculate forecast confidence intervals
        statistics.ForecastConfidenceIntervals = CalculateForecastConfidenceIntervals(metricsList, statistics.ForecastValues, statistics.ConfidenceLevel);
        
        return statistics;
    }
    
    /// <summary>
    /// Calculates the median of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The median</returns>
    private double CalculateMedian(List<double> values)
    {
        var sortedValues = values.OrderBy(v => v).ToList();
        var count = sortedValues.Count;
        
        if (count == 0)
        {
            return 0;
        }
        
        if (count % 2 == 0)
        {
            // Even count, average the middle two values
            return (sortedValues[count / 2 - 1] + sortedValues[count / 2]) / 2;
        }
        else
        {
            // Odd count, return the middle value
            return sortedValues[count / 2];
        }
    }
    
    /// <summary>
    /// Calculates the standard deviation of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The standard deviation</returns>
    private double CalculateStandardDeviation(List<double> values)
    {
        var count = values.Count;
        
        if (count <= 1)
        {
            return 0;
        }
        
        var average = values.Average();
        var sumOfSquares = values.Sum(v => Math.Pow(v - average, 2));
        
        return Math.Sqrt(sumOfSquares / (count - 1));
    }
    
    /// <summary>
    /// Determines the trend of a list of metrics
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The trend type</returns>
    private TrendType DetermineTrend(List<BaseMetric> metrics)
    {
        if (metrics.Count < 3)
        {
            return TrendType.Unknown;
        }
        
        var values = metrics.Select(m => m.Value).ToList();
        var timestamps = metrics.Select(m => m.Timestamp).ToList();
        
        // Calculate first and second derivatives
        var firstDerivatives = new List<double>();
        var secondDerivatives = new List<double>();
        
        for (var i = 1; i < values.Count; i++)
        {
            var timeSpan = (timestamps[i] - timestamps[i - 1]).TotalDays;
            
            if (timeSpan > 0)
            {
                firstDerivatives.Add((values[i] - values[i - 1]) / timeSpan);
            }
        }
        
        for (var i = 1; i < firstDerivatives.Count; i++)
        {
            var timeSpan = (timestamps[i + 1] - timestamps[i]).TotalDays;
            
            if (timeSpan > 0)
            {
                secondDerivatives.Add((firstDerivatives[i] - firstDerivatives[i - 1]) / timeSpan);
            }
        }
        
        // Determine trend based on derivatives
        var avgFirstDerivative = firstDerivatives.Average();
        var avgSecondDerivative = secondDerivatives.Count > 0 ? secondDerivatives.Average() : 0;
        
        if (Math.Abs(avgFirstDerivative) < 0.001)
        {
            return TrendType.Stable;
        }
        else if (avgFirstDerivative > 0)
        {
            if (avgSecondDerivative > 0.001)
            {
                return TrendType.Accelerating;
            }
            else if (avgSecondDerivative < -0.001)
            {
                return TrendType.Decelerating;
            }
            else
            {
                return TrendType.Increasing;
            }
        }
        else
        {
            if (avgSecondDerivative < -0.001)
            {
                return TrendType.Accelerating;
            }
            else if (avgSecondDerivative > 0.001)
            {
                return TrendType.Decelerating;
            }
            else
            {
                return TrendType.Decreasing;
            }
        }
    }
    
    /// <summary>
    /// Calculates percentiles for a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The percentiles</returns>
    private Dictionary<int, double> CalculatePercentiles(List<double> values)
    {
        var sortedValues = values.OrderBy(v => v).ToList();
        var count = sortedValues.Count;
        
        if (count == 0)
        {
            return new Dictionary<int, double>();
        }
        
        var percentiles = new Dictionary<int, double>();
        
        // Calculate 10th, 25th, 50th, 75th, and 90th percentiles
        percentiles[10] = CalculatePercentile(sortedValues, 10);
        percentiles[25] = CalculatePercentile(sortedValues, 25);
        percentiles[50] = CalculatePercentile(sortedValues, 50);
        percentiles[75] = CalculatePercentile(sortedValues, 75);
        percentiles[90] = CalculatePercentile(sortedValues, 90);
        
        return percentiles;
    }
    
    /// <summary>
    /// Calculates a percentile for a list of values
    /// </summary>
    /// <param name="sortedValues">The sorted values</param>
    /// <param name="percentile">The percentile</param>
    /// <returns>The percentile value</returns>
    private double CalculatePercentile(List<double> sortedValues, int percentile)
    {
        var count = sortedValues.Count;
        var index = (percentile / 100.0) * (count - 1);
        var lowerIndex = (int)Math.Floor(index);
        var upperIndex = (int)Math.Ceiling(index);
        
        if (lowerIndex == upperIndex)
        {
            return sortedValues[lowerIndex];
        }
        else
        {
            var weight = index - lowerIndex;
            return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
        }
    }
    
    /// <summary>
    /// Calculates forecast values for a list of metrics
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <param name="days">The number of days to forecast</param>
    /// <returns>The forecast values</returns>
    private Dictionary<DateTime, double> CalculateForecastValues(List<BaseMetric> metrics, int days)
    {
        if (metrics.Count < 2)
        {
            return new Dictionary<DateTime, double>();
        }
        
        var values = metrics.Select(m => m.Value).ToList();
        var timestamps = metrics.Select(m => m.Timestamp).ToList();
        
        // Use linear regression to forecast values
        var xValues = timestamps.Select(t => (t - timestamps[0]).TotalDays).ToList();
        var yValues = values;
        
        var (slope, intercept) = CalculateLinearRegression(xValues, yValues);
        
        var forecast = new Dictionary<DateTime, double>();
        var lastTimestamp = timestamps.Last();
        
        for (var i = 1; i <= days; i++)
        {
            var forecastTimestamp = lastTimestamp.AddDays(i);
            var forecastX = (forecastTimestamp - timestamps[0]).TotalDays;
            var forecastValue = intercept + slope * forecastX;
            
            forecast[forecastTimestamp] = forecastValue;
        }
        
        return forecast;
    }
    
    /// <summary>
    /// Calculates linear regression parameters
    /// </summary>
    /// <param name="xValues">The x values</param>
    /// <param name="yValues">The y values</param>
    /// <returns>The slope and intercept</returns>
    private (double Slope, double Intercept) CalculateLinearRegression(List<double> xValues, List<double> yValues)
    {
        var count = xValues.Count;
        
        if (count < 2)
        {
            return (0, 0);
        }
        
        var sumX = xValues.Sum();
        var sumY = yValues.Sum();
        var sumXY = xValues.Zip(yValues, (x, y) => x * y).Sum();
        var sumXX = xValues.Sum(x => x * x);
        
        var slope = (count * sumXY - sumX * sumY) / (count * sumXX - sumX * sumX);
        var intercept = (sumY - slope * sumX) / count;
        
        return (slope, intercept);
    }
    
    /// <summary>
    /// Calculates forecast confidence intervals
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <param name="forecastValues">The forecast values</param>
    /// <param name="confidenceLevel">The confidence level</param>
    /// <returns>The forecast confidence intervals</returns>
    private Dictionary<DateTime, (double Lower, double Upper)> CalculateForecastConfidenceIntervals(
        List<BaseMetric> metrics, 
        Dictionary<DateTime, double> forecastValues, 
        double confidenceLevel)
    {
        if (metrics.Count < 3 || forecastValues.Count == 0)
        {
            return new Dictionary<DateTime, (double Lower, double Upper)>();
        }
        
        var values = metrics.Select(m => m.Value).ToList();
        var timestamps = metrics.Select(m => m.Timestamp).ToList();
        
        // Calculate standard error of the regression
        var xValues = timestamps.Select(t => (t - timestamps[0]).TotalDays).ToList();
        var yValues = values;
        
        var (slope, intercept) = CalculateLinearRegression(xValues, yValues);
        
        var predictedValues = xValues.Select(x => intercept + slope * x).ToList();
        var residuals = yValues.Zip(predictedValues, (y, p) => y - p).ToList();
        var sumOfSquaredResiduals = residuals.Sum(r => r * r);
        var standardError = Math.Sqrt(sumOfSquaredResiduals / (xValues.Count - 2));
        
        // Calculate t-value for the confidence level
        var tValue = 1.96; // Approximation for 95% confidence level
        
        if (confidenceLevel != 0.95)
        {
            // Adjust t-value for different confidence levels
            if (confidenceLevel == 0.90)
            {
                tValue = 1.645;
            }
            else if (confidenceLevel == 0.99)
            {
                tValue = 2.576;
            }
        }
        
        var intervals = new Dictionary<DateTime, (double Lower, double Upper)>();
        
        foreach (var (timestamp, value) in forecastValues)
        {
            var x = (timestamp - timestamps[0]).TotalDays;
            var xMean = xValues.Average();
            var sumXSquaredDiff = xValues.Sum(xVal => Math.Pow(xVal - xMean, 2));
            
            // Calculate prediction interval
            var standardError2 = standardError * Math.Sqrt(1 + 1.0 / xValues.Count + Math.Pow(x - xMean, 2) / sumXSquaredDiff);
            var margin = tValue * standardError2;
            
            intervals[timestamp] = (value - margin, value + margin);
        }
        
        return intervals;
    }
}
