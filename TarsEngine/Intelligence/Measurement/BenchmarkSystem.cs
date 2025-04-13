using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// System for running benchmarks and establishing baseline measurements
/// </summary>
public class BenchmarkSystem
{
    private readonly ILogger<BenchmarkSystem> _logger;
    private readonly MetricsCollector _metricsCollector;
    private readonly CodeAnalyzerService _codeAnalyzerService;
    private readonly string _baselinePath;

    // Dictionary to store baseline thresholds for different metric types
    private readonly Dictionary<string, Dictionary<string, double>> _baselineThresholds = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="BenchmarkSystem"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="metricsCollector">The metrics collector</param>
    /// <param name="codeAnalyzerService">The code analyzer service</param>
    /// <param name="baselinePath">The path to store baseline data</param>
    public BenchmarkSystem(
        ILogger<BenchmarkSystem> logger,
        MetricsCollector metricsCollector,
        CodeAnalyzerService codeAnalyzerService,
        string baselinePath = "Baselines")
    {
        _logger = logger;
        _metricsCollector = metricsCollector;
        _codeAnalyzerService = codeAnalyzerService;
        _baselinePath = baselinePath;

        // Ensure the baseline directory exists
        Directory.CreateDirectory(_baselinePath);
    }

    /// <summary>
    /// Initializes the benchmark system
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("Initializing Benchmark System");

        // Load existing baseline thresholds if available
        await LoadBaselineThresholdsAsync();
    }

    /// <summary>
    /// Runs benchmarks on the current codebase
    /// </summary>
    /// <param name="codebasePath">The path to the codebase</param>
    /// <returns>The benchmark results</returns>
    public async Task<BenchmarkResults> RunCodebaseBenchmarksAsync(string codebasePath)
    {
        _logger.LogInformation("Running benchmarks on codebase: {CodebasePath}", codebasePath);

        var results = new BenchmarkResults
        {
            CodebasePath = codebasePath,
            StartTime = DateTime.UtcNow
        };

        try
        {
            // Get all C# files in the codebase
            var csharpFiles = Directory.GetFiles(codebasePath, "*.cs", SearchOption.AllDirectories);
            _logger.LogInformation("Found {FileCount} C# files to analyze", csharpFiles.Length);

            // Get all F# files in the codebase
            var fsharpFiles = Directory.GetFiles(codebasePath, "*.fs", SearchOption.AllDirectories);
            _logger.LogInformation("Found {FileCount} F# files to analyze", fsharpFiles.Length);

            // Analyze C# files
            var csharpMetrics = new List<BaseMetric>();
            foreach (var file in csharpFiles)
            {
                try
                {
                    var fileMetrics = await AnalyzeFileAsync(file, "C#");
                    csharpMetrics.AddRange(fileMetrics);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error analyzing file: {FilePath}", file);
                }
            }

            // Analyze F# files
            var fsharpMetrics = new List<BaseMetric>();
            foreach (var file in fsharpFiles)
            {
                try
                {
                    var fileMetrics = await AnalyzeFileAsync(file, "F#");
                    fsharpMetrics.AddRange(fileMetrics);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error analyzing file: {FilePath}", file);
                }
            }

            // Combine all metrics
            var allMetrics = csharpMetrics.Concat(fsharpMetrics).ToList();

            // Calculate statistics for each metric type
            results.ComplexityStatistics = CalculateMetricStatistics(
                allMetrics.OfType<ComplexityMetric>().ToList());

            results.MaintainabilityStatistics = CalculateMetricStatistics(
                allMetrics.OfType<MaintainabilityMetric>().ToList());

            results.HalsteadStatistics = CalculateMetricStatistics(
                allMetrics.OfType<HalsteadMetric>().ToList());

            // Store metrics in the collector
            foreach (var metric in allMetrics)
            {
                await _metricsCollector.CollectMetricAsync(metric);
            }

            // Calculate baseline thresholds
            await CalculateBaselineThresholdsAsync(allMetrics);

            // Save baseline thresholds
            await SaveBaselineThresholdsAsync();

            results.EndTime = DateTime.UtcNow;
            results.TotalMetricsCollected = allMetrics.Count;
            results.Success = true;

            _logger.LogInformation("Benchmark completed successfully. Collected {MetricCount} metrics", allMetrics.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running benchmarks on codebase: {CodebasePath}", codebasePath);
            results.EndTime = DateTime.UtcNow;
            results.Success = false;
            results.ErrorMessage = ex.Message;
        }

        return results;
    }

    /// <summary>
    /// Analyzes a file and collects metrics
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="language">The programming language</param>
    /// <returns>The collected metrics</returns>
    private async Task<List<BaseMetric>> AnalyzeFileAsync(string filePath, string language)
    {
        _logger.LogDebug("Analyzing file: {FilePath}", filePath);

        var metrics = new List<BaseMetric>();

        // Analyze code quality
        var codeAnalysisResult = await _codeAnalyzerService.AnalyzeFileAsync(filePath, new Dictionary<string, string> { { "language", language } });

        // Add all metrics from the analysis result
        if (codeAnalysisResult.Metrics != null)
        {
            // Filter metrics by type
            var complexityMetrics = codeAnalysisResult.Metrics
                .Where(m => m.Type == TarsEngine.Models.MetricType.Complexity)
                .Select(m => new ComplexityMetric
                {
                    Name = m.Name,
                    Value = m.Value,
                    Target = m.Target,
                    Timestamp = codeAnalysisResult.AnalyzedAt
                });
            metrics.AddRange(complexityMetrics);

            var maintainabilityMetrics = codeAnalysisResult.Metrics
                .Where(m => m.Type == TarsEngine.Models.MetricType.Maintainability)
                .Select(m => new MaintainabilityMetric
                {
                    Name = m.Name,
                    Value = m.Value,
                    Target = m.Target,
                    Timestamp = codeAnalysisResult.AnalyzedAt
                });
            metrics.AddRange(maintainabilityMetrics);

            // For Halstead metrics, we'll use metrics with names that start with "Halstead"
            var halsteadMetrics = codeAnalysisResult.Metrics
                .Where(m => m.Name.StartsWith("Halstead", StringComparison.OrdinalIgnoreCase))
                .Select(m => new HalsteadMetric
                {
                    Name = m.Name,
                    Value = m.Value,
                    Target = m.Target,
                    Timestamp = codeAnalysisResult.AnalyzedAt
                });
            metrics.AddRange(halsteadMetrics);
        }

        return metrics;
    }

    /// <summary>
    /// Calculates statistics for a list of metrics
    /// </summary>
    /// <typeparam name="T">The metric type</typeparam>
    /// <param name="metrics">The metrics</param>
    /// <returns>The metric statistics</returns>
    private Dictionary<string, MetricTypeStatistics> CalculateMetricStatistics<T>(List<T> metrics) where T : BaseMetric
    {
        var statistics = new Dictionary<string, MetricTypeStatistics>();

        // Group metrics by target type (Method, Class, File)
        var groupedMetrics = metrics
            .GroupBy(m => GetTargetType(m))
            .ToDictionary(g => g.Key, g => g.ToList());

        foreach (var (targetType, targetMetrics) in groupedMetrics)
        {
            var values = targetMetrics.Select(m => m.Value).ToList();

            if (values.Count == 0)
                continue;

            values.Sort();

            var stats = new MetricTypeStatistics
            {
                Count = values.Count,
                MinValue = values.First(),
                MaxValue = values.Last(),
                AverageValue = values.Average(),
                MedianValue = CalculateMedian(values),
                StandardDeviation = CalculateStandardDeviation(values),
                Percentiles = CalculatePercentiles(values)
            };

            statistics[targetType] = stats;
        }

        return statistics;
    }

    /// <summary>
    /// Gets the target type for a metric
    /// </summary>
    /// <param name="metric">The metric</param>
    /// <returns>The target type</returns>
    private string GetTargetType(BaseMetric metric)
    {
        if (metric is ComplexityMetric complexityMetric)
            return complexityMetric.Target ?? "Unknown";

        if (metric is MaintainabilityMetric maintainabilityMetric)
            return maintainabilityMetric.Target ?? "Unknown";

        if (metric is HalsteadMetric halsteadMetric)
            return halsteadMetric.Target ?? "Unknown";

        return "Unknown";
    }

    /// <summary>
    /// Calculates the median of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The median</returns>
    private double CalculateMedian(List<double> values)
    {
        if (values.Count == 0)
            return 0;

        var sortedValues = values.OrderBy(v => v).ToList();
        var mid = sortedValues.Count / 2;

        if (sortedValues.Count % 2 == 0)
            return (sortedValues[mid - 1] + sortedValues[mid]) / 2;

        return sortedValues[mid];
    }

    /// <summary>
    /// Calculates the standard deviation of a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The standard deviation</returns>
    private double CalculateStandardDeviation(List<double> values)
    {
        if (values.Count <= 1)
            return 0;

        var avg = values.Average();
        var sumOfSquaresOfDifferences = values.Sum(v => Math.Pow(v - avg, 2));
        return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
    }

    /// <summary>
    /// Calculates percentiles for a list of values
    /// </summary>
    /// <param name="values">The values</param>
    /// <returns>The percentiles</returns>
    private Dictionary<int, double> CalculatePercentiles(List<double> values)
    {
        var percentiles = new Dictionary<int, double>();
        var sortedValues = values.OrderBy(v => v).ToList();

        // Calculate 10th, 25th, 50th, 75th, 90th, 95th, and 99th percentiles
        var percentilesOfInterest = new[] { 10, 25, 50, 75, 90, 95, 99 };

        foreach (var p in percentilesOfInterest)
        {
            var index = (int)Math.Ceiling(p / 100.0 * sortedValues.Count) - 1;
            if (index < 0) index = 0;
            if (index >= sortedValues.Count) index = sortedValues.Count - 1;

            percentiles[p] = sortedValues[index];
        }

        return percentiles;
    }

    /// <summary>
    /// Calculates baseline thresholds based on collected metrics
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task CalculateBaselineThresholdsAsync(List<BaseMetric> metrics)
    {
        _logger.LogInformation("Calculating baseline thresholds from {MetricCount} metrics", metrics.Count);

        // Group metrics by type and target
        var complexityMetrics = metrics.OfType<ComplexityMetric>()
            .GroupBy(m => m.Target ?? "Unknown")
            .ToDictionary(g => g.Key, g => g.ToList());

        var maintainabilityMetrics = metrics.OfType<MaintainabilityMetric>()
            .GroupBy(m => m.Target ?? "Unknown")
            .ToDictionary(g => g.Key, g => g.ToList());

        var halsteadMetrics = metrics.OfType<HalsteadMetric>()
            .GroupBy(m => m.Target ?? "Unknown")
            .ToDictionary(g => g.Key, g => g.ToList());

        // Calculate thresholds for complexity metrics
        foreach (var (target, targetMetrics) in complexityMetrics)
        {
            var values = targetMetrics.Select(m => m.Value).OrderBy(v => v).ToList();
            if (values.Count == 0) continue;

            // Use 75th percentile as the threshold
            var threshold = CalculatePercentile(values, 75);

            // Store the threshold
            if (!_baselineThresholds.TryGetValue("Complexity", out var thresholds))
            {
                thresholds = new Dictionary<string, double>();
                _baselineThresholds["Complexity"] = thresholds;
            }

            thresholds[target] = threshold;

            _logger.LogInformation("Set complexity threshold for {Target} to {Threshold}", target, threshold);
        }

        // Calculate thresholds for maintainability metrics
        foreach (var (target, targetMetrics) in maintainabilityMetrics)
        {
            var values = targetMetrics.Select(m => m.Value).OrderBy(v => v).ToList();
            if (values.Count == 0) continue;

            // Use 25th percentile as the threshold (lower is worse for maintainability)
            var threshold = CalculatePercentile(values, 25);

            // Store the threshold
            if (!_baselineThresholds.TryGetValue("Maintainability", out var thresholds))
            {
                thresholds = new Dictionary<string, double>();
                _baselineThresholds["Maintainability"] = thresholds;
            }

            thresholds[target] = threshold;

            _logger.LogInformation("Set maintainability threshold for {Target} to {Threshold}", target, threshold);
        }

        // Calculate thresholds for Halstead metrics
        foreach (var (target, targetMetrics) in halsteadMetrics)
        {
            var values = targetMetrics.Select(m => m.Value).OrderBy(v => v).ToList();
            if (values.Count == 0) continue;

            // Use 75th percentile as the threshold
            var threshold = CalculatePercentile(values, 75);

            // Store the threshold
            if (!_baselineThresholds.TryGetValue("Halstead", out var thresholds))
            {
                thresholds = new Dictionary<string, double>();
                _baselineThresholds["Halstead"] = thresholds;
            }

            thresholds[target] = threshold;

            _logger.LogInformation("Set Halstead threshold for {Target} to {Threshold}", target, threshold);
        }

        await Task.CompletedTask;
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
            return 0;

        var index = (int)Math.Ceiling(percentile / 100.0 * values.Count) - 1;
        if (index < 0) index = 0;
        if (index >= values.Count) index = values.Count - 1;

        return values[index];
    }

    /// <summary>
    /// Saves baseline thresholds to disk
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task SaveBaselineThresholdsAsync()
    {
        var filePath = Path.Combine(_baselinePath, "baseline_thresholds.json");
        _logger.LogInformation("Saving baseline thresholds to {FilePath}", filePath);

        var json = System.Text.Json.JsonSerializer.Serialize(_baselineThresholds, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(filePath, json);
    }

    /// <summary>
    /// Loads baseline thresholds from disk
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task LoadBaselineThresholdsAsync()
    {
        var filePath = Path.Combine(_baselinePath, "baseline_thresholds.json");

        if (!File.Exists(filePath))
        {
            _logger.LogInformation("No baseline thresholds file found at {FilePath}", filePath);
            return;
        }

        _logger.LogInformation("Loading baseline thresholds from {FilePath}", filePath);

        try
        {
            var json = await File.ReadAllTextAsync(filePath);
            var thresholds = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, double>>>(json);

            if (thresholds != null)
            {
                foreach (var (metricType, metricThresholds) in thresholds)
                {
                    _baselineThresholds[metricType] = metricThresholds;
                }
            }

            _logger.LogInformation("Loaded baseline thresholds for {MetricTypeCount} metric types", _baselineThresholds.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading baseline thresholds from {FilePath}", filePath);
        }
    }

    /// <summary>
    /// Gets baseline thresholds for a metric type
    /// </summary>
    /// <param name="metricType">The metric type</param>
    /// <returns>The baseline thresholds</returns>
    public Dictionary<string, double> GetBaselineThresholds(string metricType)
    {
        if (_baselineThresholds.TryGetValue(metricType, out var thresholds))
        {
            return thresholds;
        }

        return new Dictionary<string, double>();
    }

    /// <summary>
    /// Gets a baseline threshold for a specific metric type and target
    /// </summary>
    /// <param name="metricType">The metric type</param>
    /// <param name="target">The target</param>
    /// <returns>The baseline threshold</returns>
    public double GetBaselineThreshold(string metricType, string target)
    {
        if (_baselineThresholds.TryGetValue(metricType, out var thresholds) &&
            thresholds.TryGetValue(target, out var threshold))
        {
            return threshold;
        }

        return 0;
    }

    /// <summary>
    /// Sets a baseline threshold for a specific metric type and target
    /// </summary>
    /// <param name="metricType">The metric type</param>
    /// <param name="target">The target</param>
    /// <param name="threshold">The threshold</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task SetBaselineThresholdAsync(string metricType, string target, double threshold)
    {
        _logger.LogInformation("Setting baseline threshold for {MetricType}.{Target} to {Threshold}", metricType, target, threshold);

        if (!_baselineThresholds.TryGetValue(metricType, out var thresholds))
        {
            thresholds = new Dictionary<string, double>();
            _baselineThresholds[metricType] = thresholds;
        }

        thresholds[target] = threshold;

        // Save the updated thresholds
        await SaveBaselineThresholdsAsync();
    }

    /// <summary>
    /// Compares metrics between TARS-generated code and human-written code
    /// </summary>
    /// <param name="tarsCodePath">The path to TARS-generated code</param>
    /// <param name="humanCodePath">The path to human-written code</param>
    /// <returns>The comparison results</returns>
    public async Task<CodeComparisonResults> CompareMetricsAsync(string tarsCodePath, string humanCodePath)
    {
        _logger.LogInformation("Comparing metrics between TARS code at {TarsPath} and human code at {HumanPath}",
            tarsCodePath, humanCodePath);

        var results = new CodeComparisonResults
        {
            TarsCodePath = tarsCodePath,
            HumanCodePath = humanCodePath,
            StartTime = DateTime.UtcNow
        };

        try
        {
            // Run benchmarks on TARS code
            var tarsResults = await RunCodebaseBenchmarksAsync(tarsCodePath);

            // Run benchmarks on human code
            var humanResults = await RunCodebaseBenchmarksAsync(humanCodePath);

            // Compare complexity metrics
            results.ComplexityComparison = CompareMetricStatistics(
                tarsResults.ComplexityStatistics,
                humanResults.ComplexityStatistics);

            // Compare maintainability metrics
            results.MaintainabilityComparison = CompareMetricStatistics(
                tarsResults.MaintainabilityStatistics,
                humanResults.MaintainabilityStatistics);

            // Compare Halstead metrics
            results.HalsteadComparison = CompareMetricStatistics(
                tarsResults.HalsteadStatistics,
                humanResults.HalsteadStatistics);

            // Calculate quality benchmarks based on human code
            await CalculateQualityBenchmarksAsync(humanResults);

            results.EndTime = DateTime.UtcNow;
            results.Success = true;

            _logger.LogInformation("Comparison completed successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error comparing metrics between TARS and human code");
            results.EndTime = DateTime.UtcNow;
            results.Success = false;
            results.ErrorMessage = ex.Message;
        }

        return results;
    }

    /// <summary>
    /// Compares metric statistics between two sets of metrics
    /// </summary>
    /// <param name="tarsStats">The TARS metrics statistics</param>
    /// <param name="humanStats">The human metrics statistics</param>
    /// <returns>The comparison results</returns>
    private Dictionary<string, MetricComparisonResult> CompareMetricStatistics(
        Dictionary<string, MetricTypeStatistics> tarsStats,
        Dictionary<string, MetricTypeStatistics> humanStats)
    {
        var comparison = new Dictionary<string, MetricComparisonResult>();

        // Get all unique targets from both sets
        var allTargets = tarsStats.Keys.Union(humanStats.Keys).ToList();

        foreach (var target in allTargets)
        {
            var hasTarsStats = tarsStats.TryGetValue(target, out var tarsTargetStats);
            var hasHumanStats = humanStats.TryGetValue(target, out var humanTargetStats);

            if (!hasTarsStats || !hasHumanStats)
                continue;

            var comparisonResult = new MetricComparisonResult
            {
                TarsStats = tarsTargetStats!,
                HumanStats = humanTargetStats!,
                AverageDifference = tarsTargetStats!.AverageValue - humanTargetStats!.AverageValue,
                MedianDifference = tarsTargetStats.MedianValue - humanTargetStats.MedianValue,
                StandardDeviationDifference = tarsTargetStats.StandardDeviation - humanTargetStats.StandardDeviation,
                PercentileDifferences = CalculatePercentileDifferences(tarsTargetStats.Percentiles, humanTargetStats.Percentiles)
            };

            comparison[target] = comparisonResult;
        }

        return comparison;
    }

    /// <summary>
    /// Calculates differences between percentiles
    /// </summary>
    /// <param name="tarsPercentiles">The TARS percentiles</param>
    /// <param name="humanPercentiles">The human percentiles</param>
    /// <returns>The percentile differences</returns>
    private Dictionary<int, double> CalculatePercentileDifferences(
        Dictionary<int, double> tarsPercentiles,
        Dictionary<int, double> humanPercentiles)
    {
        var differences = new Dictionary<int, double>();

        // Get all unique percentiles from both sets
        var allPercentiles = tarsPercentiles.Keys.Union(humanPercentiles.Keys).ToList();

        foreach (var percentile in allPercentiles)
        {
            var hasTarsPercentile = tarsPercentiles.TryGetValue(percentile, out var tarsValue);
            var hasHumanPercentile = humanPercentiles.TryGetValue(percentile, out var humanValue);

            if (!hasTarsPercentile || !hasHumanPercentile)
                continue;

            differences[percentile] = tarsValue - humanValue;
        }

        return differences;
    }

    /// <summary>
    /// Calculates quality benchmarks based on human-written code
    /// </summary>
    /// <param name="humanResults">The benchmark results for human-written code</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task CalculateQualityBenchmarksAsync(BenchmarkResults humanResults)
    {
        _logger.LogInformation("Calculating quality benchmarks from human-written code");

        // Store human code metrics as quality benchmarks
        var benchmarksPath = Path.Combine(_baselinePath, "quality_benchmarks.json");

        var benchmarks = new Dictionary<string, Dictionary<string, MetricTypeStatistics>>
        {
            { "Complexity", humanResults.ComplexityStatistics },
            { "Maintainability", humanResults.MaintainabilityStatistics },
            { "Halstead", humanResults.HalsteadStatistics }
        };

        var json = System.Text.Json.JsonSerializer.Serialize(benchmarks, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(benchmarksPath, json);

        _logger.LogInformation("Quality benchmarks saved to {BenchmarksPath}", benchmarksPath);
    }
}

/// <summary>
/// Represents the results of a benchmark run
/// </summary>
public class BenchmarkResults
{
    /// <summary>
    /// Gets or sets the codebase path
    /// </summary>
    public string CodebasePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time
    /// </summary>
    public DateTime StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time
    /// </summary>
    public DateTime EndTime { get; set; }

    /// <summary>
    /// Gets or sets whether the benchmark was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the error message
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the total number of metrics collected
    /// </summary>
    public int TotalMetricsCollected { get; set; }

    /// <summary>
    /// Gets or sets the complexity statistics
    /// </summary>
    public Dictionary<string, MetricTypeStatistics> ComplexityStatistics { get; set; } = new();

    /// <summary>
    /// Gets or sets the maintainability statistics
    /// </summary>
    public Dictionary<string, MetricTypeStatistics> MaintainabilityStatistics { get; set; } = new();

    /// <summary>
    /// Gets or sets the Halstead statistics
    /// </summary>
    public Dictionary<string, MetricTypeStatistics> HalsteadStatistics { get; set; } = new();

    /// <summary>
    /// Gets the duration of the benchmark
    /// </summary>
    public TimeSpan Duration => EndTime - StartTime;
}

/// <summary>
/// Represents statistics for a metric type
/// </summary>
public class MetricTypeStatistics
{
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
    /// Gets or sets the percentiles
    /// </summary>
    public Dictionary<int, double> Percentiles { get; set; } = new();
}

/// <summary>
/// Represents the results of comparing metrics between TARS-generated code and human-written code
/// </summary>
public class CodeComparisonResults
{
    /// <summary>
    /// Gets or sets the path to the TARS-generated code
    /// </summary>
    public string TarsCodePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the path to the human-written code
    /// </summary>
    public string HumanCodePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time
    /// </summary>
    public DateTime StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time
    /// </summary>
    public DateTime EndTime { get; set; }

    /// <summary>
    /// Gets or sets whether the comparison was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the error message
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the complexity comparison results
    /// </summary>
    public Dictionary<string, MetricComparisonResult> ComplexityComparison { get; set; } = new();

    /// <summary>
    /// Gets or sets the maintainability comparison results
    /// </summary>
    public Dictionary<string, MetricComparisonResult> MaintainabilityComparison { get; set; } = new();

    /// <summary>
    /// Gets or sets the Halstead comparison results
    /// </summary>
    public Dictionary<string, MetricComparisonResult> HalsteadComparison { get; set; } = new();

    /// <summary>
    /// Gets the duration of the comparison
    /// </summary>
    public TimeSpan Duration => EndTime - StartTime;
}

/// <summary>
/// Represents the result of comparing a specific metric between TARS-generated code and human-written code
/// </summary>
public class MetricComparisonResult
{
    /// <summary>
    /// Gets or sets the TARS metrics statistics
    /// </summary>
    public MetricTypeStatistics TarsStats { get; set; } = new();

    /// <summary>
    /// Gets or sets the human metrics statistics
    /// </summary>
    public MetricTypeStatistics HumanStats { get; set; } = new();

    /// <summary>
    /// Gets or sets the difference in average values (TARS - Human)
    /// </summary>
    public double AverageDifference { get; set; }

    /// <summary>
    /// Gets or sets the difference in median values (TARS - Human)
    /// </summary>
    public double MedianDifference { get; set; }

    /// <summary>
    /// Gets or sets the difference in standard deviations (TARS - Human)
    /// </summary>
    public double StandardDeviationDifference { get; set; }

    /// <summary>
    /// Gets or sets the differences in percentiles (TARS - Human)
    /// </summary>
    public Dictionary<int, double> PercentileDifferences { get; set; } = new();

    /// <summary>
    /// Gets the relative difference in average values ((TARS - Human) / Human)
    /// </summary>
    public double RelativeAverageDifference =>
        HumanStats.AverageValue != 0 ? AverageDifference / Math.Abs(HumanStats.AverageValue) : 0;

    /// <summary>
    /// Gets the relative difference in median values ((TARS - Human) / Human)
    /// </summary>
    public double RelativeMedianDifference =>
        HumanStats.MedianValue != 0 ? MedianDifference / Math.Abs(HumanStats.MedianValue) : 0;
}