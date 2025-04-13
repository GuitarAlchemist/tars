using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using UnifiedComplexityType = TarsEngine.Models.Unified.ComplexityTypeUnified;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// Main system for measuring and tracking intelligence progression
/// </summary>
public class IntelligenceProgressionSystem
{
    private readonly ILogger<IntelligenceProgressionSystem> _logger;
    private readonly MetricsCollector _metricsCollector;
    private readonly LearningCurveAnalyzer _learningCurveAnalyzer;
    private readonly ModificationAnalyzer _modificationAnalyzer;
    private readonly ProgressionVisualizer _progressionVisualizer;

    /// <summary>
    /// Initializes a new instance of the <see cref="IntelligenceProgressionSystem"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="metricsCollector">The metrics collector</param>
    /// <param name="learningCurveAnalyzer">The learning curve analyzer</param>
    /// <param name="modificationAnalyzer">The modification analyzer</param>
    /// <param name="progressionVisualizer">The progression visualizer</param>
    public IntelligenceProgressionSystem(
        ILogger<IntelligenceProgressionSystem> logger,
        MetricsCollector metricsCollector,
        LearningCurveAnalyzer learningCurveAnalyzer,
        ModificationAnalyzer modificationAnalyzer,
        ProgressionVisualizer progressionVisualizer)
    {
        _logger = logger;
        _metricsCollector = metricsCollector;
        _learningCurveAnalyzer = learningCurveAnalyzer;
        _modificationAnalyzer = modificationAnalyzer;
        _progressionVisualizer = progressionVisualizer;
    }

    /// <summary>
    /// Initializes the intelligence progression system
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("Initializing Intelligence Progression System");

        await _metricsCollector.InitializeAsync();
        await _learningCurveAnalyzer.InitializeAsync();
        await _modificationAnalyzer.InitializeAsync();
        await _progressionVisualizer.InitializeAsync();

        _logger.LogInformation("Intelligence Progression System initialized");
    }

    /// <summary>
    /// Records a learning event
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="concept">The concept</param>
    /// <param name="proficiency">The proficiency level (0.0-1.0)</param>
    /// <param name="source">The source of the learning</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task RecordLearningEventAsync(string domain, string concept, double proficiency, string source)
    {
        _logger.LogInformation("Recording learning event: {Domain} - {Concept} with proficiency {Proficiency}",
            domain, concept, proficiency);

        var learningMetric = new LearningMetric
        {
            Name = $"Learning.{domain}.{concept}",
            Value = proficiency,
            Dimension = domain,
            Source = source,
            Type = LearningType.Unsupervised,
            Tags = [domain, concept, "learning"]
        };

        // Get previous value if it exists
        var previousValue = await _metricsCollector.GetLatestMetricValueAsync(learningMetric.Name);
        if (previousValue.HasValue)
        {
            learningMetric.PreviousValue = previousValue.Value;
            learningMetric.TimeSincePrevious = DateTime.UtcNow -
                (await _metricsCollector.GetLatestMetricTimestampAsync(learningMetric.Name) ?? DateTime.UtcNow.AddDays(-1));

            // Calculate logarithmic values
            if (previousValue.Value > 0 && proficiency > 0)
            {
                learningMetric.LogPreviousValue = Math.Log10(previousValue.Value);
                learningMetric.LogValue = Math.Log10(proficiency);
                learningMetric.LogLearningRate = (learningMetric.LogValue - learningMetric.LogPreviousValue) /
                    (learningMetric.TimeSincePrevious.TotalDays > 0 ? learningMetric.TimeSincePrevious.TotalDays : 1);
            }
        }

        await _metricsCollector.CollectMetricAsync(learningMetric);
        await _learningCurveAnalyzer.AnalyzeLearningEventAsync(learningMetric);
    }

    /// <summary>
    /// Records a complexity metric
    /// </summary>
    /// <param name="target">The target (e.g., file, module, component)</param>
    /// <param name="complexityType">The complexity type</param>
    /// <param name="value">The complexity value</param>
    /// <param name="threshold">The complexity threshold</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task RecordComplexityMetricAsync(string target, UnifiedComplexityType complexityType, double value, double threshold)
    {
        _logger.LogInformation("Recording complexity metric: {Target} - {ComplexityType} with value {Value}",
            target, complexityType, value);

        var complexityMetric = new ComplexityMetric
        {
            Name = $"Complexity.{complexityType}.{target}",
            Value = value,
            Type = TarsEngine.Services.Adapters.ComplexityTypeConverter.ToModelType(complexityType),
            Target = target,
            Threshold = threshold,
            Dimension = complexityType.ToString(),
            Tags = [target, complexityType.ToString(), "complexity"]
        };

        // Calculate logarithmic values if applicable
        if (value > 0 && threshold > 0)
        {
            complexityMetric.LogValue = Math.Log10(value);
            complexityMetric.LogThreshold = Math.Log10(threshold);
        }

        await _metricsCollector.CollectMetricAsync(complexityMetric);
    }

    /// <summary>
    /// Records a novelty metric
    /// </summary>
    /// <param name="target">The target (e.g., solution, approach, concept)</param>
    /// <param name="noveltyType">The novelty type</param>
    /// <param name="similarityScore">The similarity score (0.0-1.0)</param>
    /// <param name="reference">The reference for comparison</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task RecordNoveltyMetricAsync(string target, NoveltyType noveltyType, double similarityScore, string reference)
    {
        _logger.LogInformation("Recording novelty metric: {Target} - {NoveltyType} with similarity {SimilarityScore}",
            target, noveltyType, similarityScore);

        var noveltyMetric = new NoveltyMetric
        {
            Name = $"Novelty.{noveltyType}.{target}",
            Value = 1.0 - similarityScore, // Novelty is inverse of similarity
            Type = noveltyType,
            SimilarityScore = similarityScore,
            Reference = reference,
            Dimension = noveltyType.ToString(),
            Tags = [target, noveltyType.ToString(), "novelty"]
        };

        await _metricsCollector.CollectMetricAsync(noveltyMetric);
    }

    /// <summary>
    /// Records a performance metric
    /// </summary>
    /// <param name="dimension">The performance dimension</param>
    /// <param name="performanceType">The performance type</param>
    /// <param name="value">The performance value</param>
    /// <param name="baselineValue">The baseline value</param>
    /// <param name="targetValue">The target value</param>
    /// <param name="unit">The unit of measurement</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task RecordPerformanceMetricAsync(
        string dimension,
        PerformanceType performanceType,
        double value,
        double baselineValue,
        double targetValue,
        string unit)
    {
        _logger.LogInformation("Recording performance metric: {Dimension} - {PerformanceType} with value {Value} {Unit}",
            dimension, performanceType, value, unit);

        var performanceMetric = new PerformanceMetric
        {
            Name = $"Performance.{performanceType}.{dimension}",
            Value = value,
            Type = performanceType,
            Dimension = dimension,
            BaselineValue = baselineValue,
            TargetValue = targetValue,
            Unit = unit,
            Tags = [dimension, performanceType.ToString(), "performance"]
        };

        await _metricsCollector.CollectMetricAsync(performanceMetric);
    }

    /// <summary>
    /// Records a code modification
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <param name="reason">The reason for the modification</param>
    /// <param name="improvementType">The improvement type</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task RecordCodeModificationAsync(
        string filePath,
        string originalCode,
        string modifiedCode,
        string reason,
        string improvementType)
    {
        _logger.LogInformation("Recording code modification: {FilePath} - {ImprovementType}",
            filePath, improvementType);

        await _modificationAnalyzer.AnalyzeModificationAsync(filePath, originalCode, modifiedCode, reason, improvementType);
    }

    /// <summary>
    /// Gets the intelligence progression report
    /// </summary>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The intelligence progression report</returns>
    public async Task<IntelligenceProgressionReport> GetProgressionReportAsync(DateTime? startTime = null, DateTime? endTime = null)
    {
        _logger.LogInformation("Generating intelligence progression report");

        var report = new IntelligenceProgressionReport
        {
            StartTime = startTime ?? DateTime.UtcNow.AddDays(-30),
            EndTime = endTime ?? DateTime.UtcNow,
            GeneratedAt = DateTime.UtcNow
        };

        // Get learning metrics
        var learningMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Learning, report.StartTime, report.EndTime);

        // Get complexity metrics
        var complexityMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Complexity, report.StartTime, report.EndTime);

        // Get novelty metrics
        var noveltyMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Novelty, report.StartTime, report.EndTime);

        // Get performance metrics
        var performanceMetrics = await _metricsCollector.GetMetricsByCategoryAsync(
            MetricCategory.Performance, report.StartTime, report.EndTime);

        // Get learning curve analysis
        report.LearningCurveAnalysis = await _learningCurveAnalyzer.GetLearningCurveAnalysisAsync(
            report.StartTime, report.EndTime);

        // Get modification analysis
        report.ModificationAnalysis = await _modificationAnalyzer.GetModificationAnalysisAsync(
            report.StartTime, report.EndTime);

        // Calculate overall intelligence score (logarithmic scale)
        report.OverallIntelligenceScore = await CalculateOverallIntelligenceScoreAsync(
            learningMetrics, complexityMetrics, noveltyMetrics, performanceMetrics);

        // Generate visualization data
        report.VisualizationData = await _progressionVisualizer.GenerateVisualizationDataAsync(
            report.StartTime, report.EndTime);

        return report;
    }

    /// <summary>
    /// Calculates the overall intelligence score
    /// </summary>
    /// <param name="learningMetrics">The learning metrics</param>
    /// <param name="complexityMetrics">The complexity metrics</param>
    /// <param name="noveltyMetrics">The novelty metrics</param>
    /// <param name="performanceMetrics">The performance metrics</param>
    /// <returns>The overall intelligence score</returns>
    private async Task<double> CalculateOverallIntelligenceScoreAsync(
        IEnumerable<BaseMetric> learningMetrics,
        IEnumerable<BaseMetric> complexityMetrics,
        IEnumerable<BaseMetric> noveltyMetrics,
        IEnumerable<BaseMetric> performanceMetrics)
    {
        // Get the latest intelligence score
        var latestScore = await _metricsCollector.GetLatestMetricValueAsync("Intelligence.OverallScore") ?? 1.0;

        // Calculate learning component (weighted average of learning metrics)
        double learningComponent = CalculateLearningComponent(learningMetrics);

        // Calculate complexity component (weighted average of complexity metrics)
        double complexityComponent = CalculateComplexityComponent(complexityMetrics);

        // Calculate novelty component (weighted average of novelty metrics)
        double noveltyComponent = CalculateNoveltyComponent(noveltyMetrics);

        // Calculate performance component (weighted average of performance metrics)
        double performanceComponent = CalculatePerformanceComponent(performanceMetrics);

        // Calculate overall score using logarithmic scale
        // The formula ensures that intelligence growth is logarithmic rather than linear
        double overallScore = latestScore * Math.Pow(10,
            (0.4 * learningComponent +
             0.3 * complexityComponent +
             0.2 * noveltyComponent +
             0.1 * performanceComponent) / 100);

        // Record the overall intelligence score
        var metric = new PerformanceMetric
        {
            Name = "Intelligence.OverallScore",
            Value = overallScore,
            Category = MetricCategory.Other,
            Tags = ["intelligence", "overall"],
            Type = PerformanceType.Other
        };
        await _metricsCollector.CollectMetricAsync(metric);

        return overallScore;
    }

    /// <summary>
    /// Calculates the learning component
    /// </summary>
    /// <param name="learningMetrics">The learning metrics</param>
    /// <returns>The learning component</returns>
    private double CalculateLearningComponent(IEnumerable<BaseMetric> learningMetrics)
    {
        // Implementation would calculate a weighted average of learning metrics
        // For now, return a simple average
        double sum = 0;
        int count = 0;

        foreach (var metric in learningMetrics)
        {
            if (metric is LearningMetric learningMetric)
            {
                sum += learningMetric.ImprovementRatio;
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    /// <summary>
    /// Calculates the complexity component
    /// </summary>
    /// <param name="complexityMetrics">The complexity metrics</param>
    /// <returns>The complexity component</returns>
    private double CalculateComplexityComponent(IEnumerable<BaseMetric> complexityMetrics)
    {
        // Implementation would calculate a weighted average of complexity metrics
        // For now, return a simple average
        double sum = 0;
        int count = 0;

        foreach (var metric in complexityMetrics)
        {
            if (metric is ComplexityMetric complexityMetric)
            {
                // Higher values are better for complexity handling
                sum += complexityMetric.Value / complexityMetric.Threshold;
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    /// <summary>
    /// Calculates the novelty component
    /// </summary>
    /// <param name="noveltyMetrics">The novelty metrics</param>
    /// <returns>The novelty component</returns>
    private double CalculateNoveltyComponent(IEnumerable<BaseMetric> noveltyMetrics)
    {
        // Implementation would calculate a weighted average of novelty metrics
        // For now, return a simple average
        double sum = 0;
        int count = 0;

        foreach (var metric in noveltyMetrics)
        {
            if (metric is NoveltyMetric noveltyMetric)
            {
                sum += noveltyMetric.NoveltyScore;
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    /// <summary>
    /// Calculates the performance component
    /// </summary>
    /// <param name="performanceMetrics">The performance metrics</param>
    /// <returns>The performance component</returns>
    private double CalculatePerformanceComponent(IEnumerable<BaseMetric> performanceMetrics)
    {
        // Implementation would calculate a weighted average of performance metrics
        // For now, return a simple average
        double sum = 0;
        int count = 0;

        foreach (var metric in performanceMetrics)
        {
            if (metric is PerformanceMetric performanceMetric)
            {
                sum += performanceMetric.BaselineRatio;
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }
}
