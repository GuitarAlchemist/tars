using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// Visualizes intelligence progression metrics
/// </summary>
public class ProgressionVisualizer
{
    private readonly ILogger<ProgressionVisualizer> _logger;
    private readonly MetricsCollector _metricsCollector;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ProgressionVisualizer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="metricsCollector">The metrics collector</param>
    public ProgressionVisualizer(ILogger<ProgressionVisualizer> logger, MetricsCollector metricsCollector)
    {
        _logger = logger;
        _metricsCollector = metricsCollector;
    }
    
    /// <summary>
    /// Initializes the progression visualizer
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("Initializing Progression Visualizer");
        
        await Task.CompletedTask;
    }
    
    /// <summary>
    /// Generates visualization data
    /// </summary>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The visualization data</returns>
    public async Task<VisualizationData> GenerateVisualizationDataAsync(DateTime startTime, DateTime endTime)
    {
        _logger.LogInformation("Generating visualization data");
        
        var data = new VisualizationData
        {
            StartTime = startTime,
            EndTime = endTime
        };
        
        // Get all metrics
        var learningMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Learning, startTime, endTime);
        
        var complexityMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Complexity, startTime, endTime);
        
        var noveltyMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Novelty, startTime, endTime);
        
        var performanceMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Performance, startTime, endTime);
        
        var otherMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Other, startTime, endTime);
        
        // Generate time series data
        data.LearningTimeSeries = GenerateLearningTimeSeries(learningMetrics);
        data.ComplexityTimeSeries = GenerateComplexityTimeSeries(complexityMetrics);
        data.NoveltyTimeSeries = GenerateNoveltyTimeSeries(noveltyMetrics);
        data.PerformanceTimeSeries = GeneratePerformanceTimeSeries(performanceMetrics);
        
        // Generate intelligence score time series
        data.IntelligenceScoreTimeSeries = GenerateIntelligenceScoreTimeSeries(otherMetrics);
        
        // Generate radar chart data
        data.RadarChartData = GenerateRadarChartData(
            learningMetrics, complexityMetrics, noveltyMetrics, performanceMetrics);
        
        // Generate histogram data
        data.LearningHistogram = GenerateHistogram(learningMetrics.Select(m => m.Value));
        data.ComplexityHistogram = GenerateHistogram(complexityMetrics.Select(m => m.Value));
        data.NoveltyHistogram = GenerateHistogram(noveltyMetrics.Select(m => m.Value));
        data.PerformanceHistogram = GenerateHistogram(performanceMetrics.Select(m => m.Value));
        
        return data;
    }
    
    /// <summary>
    /// Generates learning time series
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The time series</returns>
    private Dictionary<DateTime, double> GenerateLearningTimeSeries(IEnumerable<BaseMetric> metrics)
    {
        var result = new Dictionary<DateTime, double>();
        
        // Group metrics by day
        var metricsByDay = metrics
            .GroupBy(m => m.Timestamp.Date)
            .OrderBy(g => g.Key);
        
        foreach (var group in metricsByDay)
        {
            // Calculate the average value for the day
            double avgValue = group.Average(m => m.Value);
            result[group.Key] = avgValue;
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates complexity time series
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The time series</returns>
    private Dictionary<DateTime, double> GenerateComplexityTimeSeries(IEnumerable<BaseMetric> metrics)
    {
        var result = new Dictionary<DateTime, double>();
        
        // Group metrics by day
        var metricsByDay = metrics
            .GroupBy(m => m.Timestamp.Date)
            .OrderBy(g => g.Key);
        
        foreach (var group in metricsByDay)
        {
            // Calculate the average value for the day
            double avgValue = group.Average(m => m.Value);
            result[group.Key] = avgValue;
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates novelty time series
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The time series</returns>
    private Dictionary<DateTime, double> GenerateNoveltyTimeSeries(IEnumerable<BaseMetric> metrics)
    {
        var result = new Dictionary<DateTime, double>();
        
        // Group metrics by day
        var metricsByDay = metrics
            .GroupBy(m => m.Timestamp.Date)
            .OrderBy(g => g.Key);
        
        foreach (var group in metricsByDay)
        {
            // Calculate the average value for the day
            double avgValue = group.Average(m => m.Value);
            result[group.Key] = avgValue;
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates performance time series
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The time series</returns>
    private Dictionary<DateTime, double> GeneratePerformanceTimeSeries(IEnumerable<BaseMetric> metrics)
    {
        var result = new Dictionary<DateTime, double>();
        
        // Group metrics by day
        var metricsByDay = metrics
            .GroupBy(m => m.Timestamp.Date)
            .OrderBy(g => g.Key);
        
        foreach (var group in metricsByDay)
        {
            // Calculate the average value for the day
            double avgValue = group.Average(m => m.Value);
            result[group.Key] = avgValue;
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates intelligence score time series
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The time series</returns>
    private Dictionary<DateTime, double> GenerateIntelligenceScoreTimeSeries(IEnumerable<BaseMetric> metrics)
    {
        var result = new Dictionary<DateTime, double>();
        
        // Filter metrics to only include intelligence score metrics
        var intelligenceScoreMetrics = metrics
            .Where(m => m.Name == "Intelligence.OverallScore")
            .OrderBy(m => m.Timestamp);
        
        foreach (var metric in intelligenceScoreMetrics)
        {
            result[metric.Timestamp.Date] = metric.Value;
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates radar chart data
    /// </summary>
    /// <param name="learningMetrics">The learning metrics</param>
    /// <param name="complexityMetrics">The complexity metrics</param>
    /// <param name="noveltyMetrics">The novelty metrics</param>
    /// <param name="performanceMetrics">The performance metrics</param>
    /// <returns>The radar chart data</returns>
    private Dictionary<string, double> GenerateRadarChartData(
        IEnumerable<BaseMetric> learningMetrics,
        IEnumerable<BaseMetric> complexityMetrics,
        IEnumerable<BaseMetric> noveltyMetrics,
        IEnumerable<BaseMetric> performanceMetrics)
    {
        var result = new Dictionary<string, double>();
        
        // Calculate average values for each category
        result["Learning"] = learningMetrics.Any() ? learningMetrics.Average(m => m.Value) : 0;
        result["Complexity"] = complexityMetrics.Any() ? complexityMetrics.Average(m => m.Value) : 0;
        result["Novelty"] = noveltyMetrics.Any() ? noveltyMetrics.Average(m => m.Value) : 0;
        result["Performance"] = performanceMetrics.Any() ? performanceMetrics.Average(m => m.Value) : 0;
        
        // Add subcategories for learning
        var learningMetricsByType = learningMetrics.OfType<LearningMetric>()
            .GroupBy(m => m.Type);
        
        foreach (var group in learningMetricsByType)
        {
            result[$"Learning.{group.Key}"] = group.Average(m => m.Value);
        }
        
        // Add subcategories for complexity
        var complexityMetricsByType = complexityMetrics.OfType<ComplexityMetric>()
            .GroupBy(m => m.Type);
        
        foreach (var group in complexityMetricsByType)
        {
            result[$"Complexity.{group.Key}"] = group.Average(m => m.Value);
        }
        
        // Add subcategories for novelty
        var noveltyMetricsByType = noveltyMetrics.OfType<NoveltyMetric>()
            .GroupBy(m => m.Type);
        
        foreach (var group in noveltyMetricsByType)
        {
            result[$"Novelty.{group.Key}"] = group.Average(m => m.Value);
        }
        
        // Add subcategories for performance
        var performanceMetricsByType = performanceMetrics.OfType<PerformanceMetric>()
            .GroupBy(m => m.Type);
        
        foreach (var group in performanceMetricsByType)
        {
            result[$"Performance.{group.Key}"] = group.Average(m => m.Value);
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates a histogram
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The histogram</returns>
    private Dictionary<string, int> GenerateHistogram(IEnumerable<double> values)
    {
        var result = new Dictionary<string, int>();
        
        if (!values.Any())
        {
            return result;
        }
        
        // Determine the range
        double min = values.Min();
        double max = values.Max();
        
        // Create 10 bins
        int binCount = 10;
        double binWidth = (max - min) / binCount;
        
        // Initialize bins
        for (int i = 0; i < binCount; i++)
        {
            double binStart = min + i * binWidth;
            double binEnd = binStart + binWidth;
            string binLabel = $"{binStart:F2}-{binEnd:F2}";
            result[binLabel] = 0;
        }
        
        // Count values in each bin
        foreach (double value in values)
        {
            int binIndex = (int)Math.Min(binCount - 1, Math.Floor((value - min) / binWidth));
            double binStart = min + binIndex * binWidth;
            double binEnd = binStart + binWidth;
            string binLabel = $"{binStart:F2}-{binEnd:F2}";
            result[binLabel]++;
        }
        
        return result;
    }
}

/// <summary>
/// Represents visualization data
/// </summary>
public class VisualizationData
{
    /// <summary>
    /// Gets or sets the start time
    /// </summary>
    public DateTime StartTime { get; set; }
    
    /// <summary>
    /// Gets or sets the end time
    /// </summary>
    public DateTime EndTime { get; set; }
    
    /// <summary>
    /// Gets or sets the learning time series
    /// </summary>
    public Dictionary<DateTime, double> LearningTimeSeries { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the complexity time series
    /// </summary>
    public Dictionary<DateTime, double> ComplexityTimeSeries { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the novelty time series
    /// </summary>
    public Dictionary<DateTime, double> NoveltyTimeSeries { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the performance time series
    /// </summary>
    public Dictionary<DateTime, double> PerformanceTimeSeries { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence score time series
    /// </summary>
    public Dictionary<DateTime, double> IntelligenceScoreTimeSeries { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the radar chart data
    /// </summary>
    public Dictionary<string, double> RadarChartData { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the learning histogram
    /// </summary>
    public Dictionary<string, int> LearningHistogram { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the complexity histogram
    /// </summary>
    public Dictionary<string, int> ComplexityHistogram { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the novelty histogram
    /// </summary>
    public Dictionary<string, int> NoveltyHistogram { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the performance histogram
    /// </summary>
    public Dictionary<string, int> PerformanceHistogram { get; set; } = new();
}
