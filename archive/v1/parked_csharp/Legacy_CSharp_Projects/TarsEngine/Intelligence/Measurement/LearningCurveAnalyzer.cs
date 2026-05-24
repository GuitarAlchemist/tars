using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Monads;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// Analyzes learning curves and progression patterns
/// </summary>
public class LearningCurveAnalyzer
{
    private readonly ILogger<LearningCurveAnalyzer> _logger;
    private readonly MetricsCollector _metricsCollector;

    // Cache of learning curve analyses
    private readonly Dictionary<string, LearningCurveAnalysis> _learningCurveAnalyses = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="LearningCurveAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="metricsCollector">The metrics collector</param>
    public LearningCurveAnalyzer(ILogger<LearningCurveAnalyzer> logger, MetricsCollector metricsCollector)
    {
        _logger = logger;
        _metricsCollector = metricsCollector;
    }

    /// <summary>
    /// Initializes the learning curve analyzer
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("Initializing Learning Curve Analyzer");

        // Clear the cache
        _learningCurveAnalyses.Clear();

        await Task.CompletedTask;
    }

    /// <summary>
    /// Analyzes a learning event
    /// </summary>
    /// <param name="learningMetric">The learning metric</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task AnalyzeLearningEventAsync(LearningMetric learningMetric)
    {
        _logger.LogInformation("Analyzing learning event: {MetricName}", learningMetric.Name);

        // Get historical metrics for this learning event
        var historicalMetrics = await _metricsCollector.GetMetricsByNameAsync(learningMetric.Name);

        // Create or update the learning curve analysis
        var analysis = await CreateOrUpdateLearningCurveAnalysisAsync(learningMetric.Name, historicalMetrics);

        // Store the analysis in the cache
        _learningCurveAnalyses[learningMetric.Name] = analysis;
    }

    /// <summary>
    /// Gets the learning curve analysis for a specific metric
    /// </summary>
    /// <param name="metricName">The metric name</param>
    /// <returns>The learning curve analysis</returns>
    public async Task<LearningCurveAnalysis> GetLearningCurveAnalysisAsync(string metricName)
    {
        _logger.LogInformation("Getting learning curve analysis for: {MetricName}", metricName);

        // Check if the analysis is in the cache
        if (_learningCurveAnalyses.TryGetValue(metricName, out var cachedAnalysis))
        {
            return cachedAnalysis;
        }

        // Get historical metrics for this learning event
        var historicalMetrics = await _metricsCollector.GetMetricsByNameAsync(metricName);

        // Create the learning curve analysis
        var analysis = await CreateOrUpdateLearningCurveAnalysisAsync(metricName, historicalMetrics);

        // Store the analysis in the cache
        _learningCurveAnalyses[metricName] = analysis;

        return analysis;
    }

    /// <summary>
    /// Gets the learning curve analysis for all metrics
    /// </summary>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The learning curve analysis</returns>
    public async Task<LearningCurveAnalysis> GetLearningCurveAnalysisAsync(DateTime? startTime = null, DateTime? endTime = null)
    {
        _logger.LogInformation("Getting overall learning curve analysis");

        // Get all learning metrics
        var learningMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Learning, startTime, endTime);

        // Group metrics by name
        var metricsByName = new Dictionary<string, List<BaseMetric>>();
        foreach (var metric in learningMetrics)
        {
            if (!metricsByName.TryGetValue(metric.Name, out var metrics))
            {
                metrics = new List<BaseMetric>();
                metricsByName[metric.Name] = metrics;
            }

            metrics.Add(metric);
        }

        // Create analyses for each metric
        var analyses = new List<LearningCurveAnalysis>();
        foreach (var (name, metrics) in metricsByName)
        {
            var analysis = await CreateOrUpdateLearningCurveAnalysisAsync(name, metrics);
            analyses.Add(analysis);
        }

        // Combine analyses into an overall analysis
        return CombineAnalyses(analyses);
    }

    /// <summary>
    /// Creates or updates a learning curve analysis
    /// </summary>
    /// <param name="metricName">The metric name</param>
    /// <param name="metrics">The metrics</param>
    /// <returns>The learning curve analysis</returns>
    private Task<LearningCurveAnalysis> CreateOrUpdateLearningCurveAnalysisAsync(string metricName, IEnumerable<BaseMetric> metrics)
    {
        var metricsList = metrics.OfType<LearningMetric>().OrderBy(m => m.Timestamp).ToList();

        var analysis = new LearningCurveAnalysis
        {
            MetricName = metricName,
            DataPoints = new List<LearningDataPoint>()
        };

        if (metricsList.Count == 0)
        {
            return AsyncMonad.Return(analysis);
        }

        // Extract data points
        foreach (var metric in metricsList)
        {
            analysis.DataPoints.Add(new LearningDataPoint
            {
                Timestamp = metric.Timestamp,
                Value = metric.Value,
                PreviousValue = metric.PreviousValue,
                ImprovementRatio = metric.ImprovementRatio,
                LogValue = metric.LogValue,
                LogPreviousValue = metric.LogPreviousValue,
                LogLearningRate = metric.LogLearningRate
            });
        }

        // Calculate learning curve parameters
        analysis.LearningRate = CalculateLearningRate(metricsList);
        analysis.LogLearningRate = CalculateLogLearningRate(metricsList);
        analysis.PlateauValue = CalculatePlateauValue(metricsList);
        analysis.TimeToPlateauDays = CalculateTimeToPlateauDays(metricsList);
        analysis.CurveType = DetermineCurveType(metricsList);
        analysis.EfficiencyScore = CalculateEfficiencyScore(metricsList);

        // Calculate forecast
        analysis.ForecastValues = CalculateForecastValues(metricsList, 30); // Forecast for 30 days

        return AsyncMonad.Return(analysis);
    }

    /// <summary>
    /// Combines multiple learning curve analyses into an overall analysis
    /// </summary>
    /// <param name="analyses">The analyses to combine</param>
    /// <returns>The combined analysis</returns>
    private LearningCurveAnalysis CombineAnalyses(List<LearningCurveAnalysis> analyses)
    {
        if (analyses.Count == 0)
        {
            return new LearningCurveAnalysis
            {
                MetricName = "Overall",
                DataPoints = new List<LearningDataPoint>()
            };
        }

        // Create a combined analysis
        var combinedAnalysis = new LearningCurveAnalysis
        {
            MetricName = "Overall",
            DataPoints = new List<LearningDataPoint>(),
            LearningRate = analyses.Average(a => a.LearningRate),
            LogLearningRate = analyses.Average(a => a.LogLearningRate),
            PlateauValue = analyses.Average(a => a.PlateauValue),
            TimeToPlateauDays = analyses.Average(a => a.TimeToPlateauDays),
            EfficiencyScore = analyses.Average(a => a.EfficiencyScore),
            CurveType = DetermineDominantCurveType(analyses)
        };

        // Combine forecast values
        combinedAnalysis.ForecastValues = new Dictionary<DateTime, double>();
        foreach (var analysis in analyses)
        {
            foreach (var (date, value) in analysis.ForecastValues)
            {
                if (!combinedAnalysis.ForecastValues.TryGetValue(date, out var existingValue))
                {
                    combinedAnalysis.ForecastValues[date] = value;
                }
                else
                {
                    combinedAnalysis.ForecastValues[date] = (existingValue + value) / 2;
                }
            }
        }

        return combinedAnalysis;
    }

    /// <summary>
    /// Calculates the learning rate
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The learning rate</returns>
    private double CalculateLearningRate(List<LearningMetric> metrics)
    {
        if (metrics.Count < 2)
        {
            return 0;
        }

        // Calculate the average improvement ratio
        double sumImprovementRatio = 0;
        var count = 0;

        for (var i = 1; i < metrics.Count; i++)
        {
            if (metrics[i].ImprovementRatio > 0)
            {
                sumImprovementRatio += metrics[i].ImprovementRatio;
                count++;
            }
        }

        return count > 0 ? sumImprovementRatio / count : 0;
    }

    /// <summary>
    /// Calculates the logarithmic learning rate
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The logarithmic learning rate</returns>
    private double CalculateLogLearningRate(List<LearningMetric> metrics)
    {
        if (metrics.Count < 2)
        {
            return 0;
        }

        // Calculate the average logarithmic learning rate
        double sumLogLearningRate = 0;
        var count = 0;

        for (var i = 1; i < metrics.Count; i++)
        {
            if (metrics[i].LogLearningRate != 0)
            {
                sumLogLearningRate += metrics[i].LogLearningRate;
                count++;
            }
        }

        return count > 0 ? sumLogLearningRate / count : 0;
    }

    /// <summary>
    /// Calculates the plateau value
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The plateau value</returns>
    private double CalculatePlateauValue(List<LearningMetric> metrics)
    {
        if (metrics.Count < 3)
        {
            return metrics.Count > 0 ? metrics[metrics.Count - 1].Value : 0;
        }

        // Check if the learning curve has plateaued
        var hasPlateaued = true;
        var plateauThreshold = 0.05; // 5% improvement threshold

        for (var i = metrics.Count - 3; i < metrics.Count - 1; i++)
        {
            var improvementRatio = metrics[i + 1].Value / metrics[i].Value;
            if (improvementRatio > 1 + plateauThreshold)
            {
                hasPlateaued = false;
                break;
            }
        }

        if (hasPlateaued)
        {
            // Return the average of the last 3 values
            return (metrics[metrics.Count - 3].Value +
                   metrics[metrics.Count - 2].Value +
                   metrics[metrics.Count - 1].Value) / 3;
        }
        else
        {
            // Estimate the plateau value using a power law model
            // For simplicity, return the last value multiplied by a factor
            return metrics[metrics.Count - 1].Value * 1.5;
        }
    }

    /// <summary>
    /// Calculates the time to plateau in days
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The time to plateau in days</returns>
    private double CalculateTimeToPlateauDays(List<LearningMetric> metrics)
    {
        if (metrics.Count < 3)
        {
            return 0;
        }

        // Check if the learning curve has plateaued
        var hasPlateaued = true;
        var plateauThreshold = 0.05; // 5% improvement threshold

        for (var i = metrics.Count - 3; i < metrics.Count - 1; i++)
        {
            var improvementRatio = metrics[i + 1].Value / metrics[i].Value;
            if (improvementRatio > 1 + plateauThreshold)
            {
                hasPlateaued = false;
                break;
            }
        }

        if (hasPlateaued)
        {
            // Return the time from the first measurement to the plateau
            return (metrics[metrics.Count - 3].Timestamp - metrics[0].Timestamp).TotalDays;
        }
        else
        {
            // Estimate the time to plateau using the current learning rate
            var currentValue = metrics[metrics.Count - 1].Value;
            var plateauValue = CalculatePlateauValue(metrics);
            var learningRate = CalculateLearningRate(metrics);

            if (learningRate <= 0 || currentValue >= plateauValue)
            {
                return 0;
            }

            // Simple estimate: how many days at the current learning rate to reach the plateau
            return (plateauValue - currentValue) / (currentValue * (learningRate - 1));
        }
    }

    /// <summary>
    /// Determines the curve type
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The curve type</returns>
    private LearningCurveType DetermineCurveType(List<LearningMetric> metrics)
    {
        if (metrics.Count < 3)
        {
            return LearningCurveType.Unknown;
        }

        // Calculate first and second derivatives
        var firstDerivatives = new List<double>();
        var secondDerivatives = new List<double>();

        for (var i = 1; i < metrics.Count; i++)
        {
            var timeSpan = (metrics[i].Timestamp - metrics[i - 1].Timestamp).TotalDays;
            if (timeSpan > 0)
            {
                firstDerivatives.Add((metrics[i].Value - metrics[i - 1].Value) / timeSpan);
            }
        }

        for (var i = 1; i < firstDerivatives.Count; i++)
        {
            var timeSpan = (metrics[i + 1].Timestamp - metrics[i].Timestamp).TotalDays;
            if (timeSpan > 0)
            {
                secondDerivatives.Add((firstDerivatives[i] - firstDerivatives[i - 1]) / timeSpan);
            }
        }

        // Determine curve type based on derivatives
        if (firstDerivatives.Count == 0)
        {
            return LearningCurveType.Unknown;
        }

        var avgFirstDerivative = firstDerivatives.Average();
        var avgSecondDerivative = secondDerivatives.Count > 0 ? secondDerivatives.Average() : 0;

        if (Math.Abs(avgFirstDerivative) < 0.001)
        {
            return LearningCurveType.Plateau;
        }
        else if (avgFirstDerivative > 0)
        {
            if (avgSecondDerivative > 0.001)
            {
                return LearningCurveType.Exponential;
            }
            else if (avgSecondDerivative < -0.001)
            {
                return LearningCurveType.Logarithmic;
            }
            else
            {
                return LearningCurveType.Linear;
            }
        }
        else
        {
            return LearningCurveType.Declining;
        }
    }

    /// <summary>
    /// Calculates the efficiency score
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The efficiency score</returns>
    private double CalculateEfficiencyScore(List<LearningMetric> metrics)
    {
        if (metrics.Count < 2)
        {
            return 0;
        }

        // Calculate the area under the learning curve
        double area = 0;
        for (var i = 1; i < metrics.Count; i++)
        {
            var timeSpan = (metrics[i].Timestamp - metrics[i - 1].Timestamp).TotalDays;
            var avgValue = (metrics[i].Value + metrics[i - 1].Value) / 2;
            area += timeSpan * avgValue;
        }

        // Calculate the ideal area (immediate jump to the maximum value)
        var totalTimeSpan = (metrics[metrics.Count - 1].Timestamp - metrics[0].Timestamp).TotalDays;
        var maxValue = metrics.Max(m => m.Value);
        var idealArea = totalTimeSpan * maxValue;

        // Calculate efficiency as the ratio of actual to ideal area
        return idealArea > 0 ? area / idealArea : 0;
    }

    /// <summary>
    /// Calculates forecast values
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <param name="days">The number of days to forecast</param>
    /// <returns>The forecast values</returns>
    private Dictionary<DateTime, double> CalculateForecastValues(List<LearningMetric> metrics, int days)
    {
        var forecast = new Dictionary<DateTime, double>();

        if (metrics.Count < 2)
        {
            return forecast;
        }

        // Determine the curve type
        var curveType = DetermineCurveType(metrics);

        // Get the latest value and timestamp
        var latestValue = metrics[metrics.Count - 1].Value;
        var latestTimestamp = metrics[metrics.Count - 1].Timestamp;

        // Calculate learning rate
        var learningRate = CalculateLearningRate(metrics);
        var logLearningRate = CalculateLogLearningRate(metrics);
        var plateauValue = CalculatePlateauValue(metrics);

        // Generate forecast values
        for (var i = 1; i <= days; i++)
        {
            var forecastDate = latestTimestamp.AddDays(i);
            double forecastValue = 0;

            switch (curveType)
            {
                case LearningCurveType.Linear:
                    // Linear growth
                    forecastValue = latestValue * (1 + learningRate * i);
                    break;

                case LearningCurveType.Logarithmic:
                    // Logarithmic growth (approaches plateau)
                    forecastValue = plateauValue - (plateauValue - latestValue) * Math.Exp(-logLearningRate * i);
                    break;

                case LearningCurveType.Exponential:
                    // Exponential growth
                    forecastValue = latestValue * Math.Pow(1 + learningRate, i);
                    break;

                case LearningCurveType.Plateau:
                    // Plateau (constant value)
                    forecastValue = latestValue;
                    break;

                case LearningCurveType.Declining:
                    // Declining (negative growth)
                    forecastValue = latestValue * Math.Pow(1 + learningRate, i);
                    break;

                default:
                    // Unknown (use linear growth)
                    forecastValue = latestValue * (1 + learningRate * i);
                    break;
            }

            forecast[forecastDate] = forecastValue;
        }

        return forecast;
    }

    /// <summary>
    /// Determines the dominant curve type from multiple analyses
    /// </summary>
    /// <param name="analyses">The analyses</param>
    /// <returns>The dominant curve type</returns>
    private LearningCurveType DetermineDominantCurveType(List<LearningCurveAnalysis> analyses)
    {
        if (analyses.Count == 0)
        {
            return LearningCurveType.Unknown;
        }

        // Count the occurrences of each curve type
        var counts = new Dictionary<LearningCurveType, int>();
        foreach (var analysis in analyses)
        {
            if (!counts.TryGetValue(analysis.CurveType, out var count))
            {
                counts[analysis.CurveType] = 1;
            }
            else
            {
                counts[analysis.CurveType] = count + 1;
            }
        }

        // Find the most common curve type
        var dominantType = LearningCurveType.Unknown;
        var maxCount = 0;

        foreach (var (type, count) in counts)
        {
            if (count > maxCount)
            {
                maxCount = count;
                dominantType = type;
            }
        }

        return dominantType;
    }
}

/// <summary>
/// Represents a learning curve analysis
/// </summary>
public class LearningCurveAnalysis
{
    /// <summary>
    /// Gets or sets the metric name
    /// </summary>
    public string MetricName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the data points
    /// </summary>
    public List<LearningDataPoint> DataPoints { get; set; } = new();

    /// <summary>
    /// Gets or sets the learning rate
    /// </summary>
    public double LearningRate { get; set; }

    /// <summary>
    /// Gets or sets the logarithmic learning rate
    /// </summary>
    public double LogLearningRate { get; set; }

    /// <summary>
    /// Gets or sets the plateau value
    /// </summary>
    public double PlateauValue { get; set; }

    /// <summary>
    /// Gets or sets the time to plateau in days
    /// </summary>
    public double TimeToPlateauDays { get; set; }

    /// <summary>
    /// Gets or sets the curve type
    /// </summary>
    public LearningCurveType CurveType { get; set; }

    /// <summary>
    /// Gets or sets the efficiency score
    /// </summary>
    public double EfficiencyScore { get; set; }

    /// <summary>
    /// Gets or sets the forecast values
    /// </summary>
    public Dictionary<DateTime, double> ForecastValues { get; set; } = new();
}

/// <summary>
/// Represents a learning data point
/// </summary>
public class LearningDataPoint
{
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the value
    /// </summary>
    public double Value { get; set; }

    /// <summary>
    /// Gets or sets the previous value
    /// </summary>
    public double PreviousValue { get; set; }

    /// <summary>
    /// Gets or sets the improvement ratio
    /// </summary>
    public double ImprovementRatio { get; set; }

    /// <summary>
    /// Gets or sets the logarithmic value
    /// </summary>
    public double LogValue { get; set; }

    /// <summary>
    /// Gets or sets the logarithmic previous value
    /// </summary>
    public double LogPreviousValue { get; set; }

    /// <summary>
    /// Gets or sets the logarithmic learning rate
    /// </summary>
    public double LogLearningRate { get; set; }
}

/// <summary>
/// Represents a learning curve type
/// </summary>
public enum LearningCurveType
{
    /// <summary>
    /// Unknown curve type
    /// </summary>
    Unknown,

    /// <summary>
    /// Linear curve type
    /// </summary>
    Linear,

    /// <summary>
    /// Logarithmic curve type
    /// </summary>
    Logarithmic,

    /// <summary>
    /// Exponential curve type
    /// </summary>
    Exponential,

    /// <summary>
    /// Plateau curve type
    /// </summary>
    Plateau,

    /// <summary>
    /// Declining curve type
    /// </summary>
    Declining
}
