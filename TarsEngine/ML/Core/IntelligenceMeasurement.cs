using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.ML.Core;

/// <summary>
/// System for measuring and tracking TARS intelligence
/// </summary>
public class IntelligenceMeasurement
{
    private readonly ILogger<IntelligenceMeasurement> _logger;
    private readonly string _storageBasePath;
    private readonly List<IntelligenceSnapshot> _snapshots = new();
    private readonly Dictionary<string, double> _currentMetrics = new();
    private readonly Dictionary<string, double> _baselineHumanMetrics = new();
    private readonly Dictionary<string, double> _metricWeights = new();
    private readonly List<string> _intelligenceDimensions = new();
    
    /// <summary>
    /// Gets the current intelligence score
    /// </summary>
    public double CurrentIntelligenceScore { get; private set; }
    
    /// <summary>
    /// Gets the baseline human intelligence score
    /// </summary>
    public double BaselineHumanIntelligenceScore { get; private set; } = 100.0;
    
    /// <summary>
    /// Gets the intelligence ratio compared to human baseline
    /// </summary>
    public double IntelligenceRatio => CurrentIntelligenceScore / BaselineHumanIntelligenceScore;
    
    /// <summary>
    /// Gets the intelligence growth rate per day
    /// </summary>
    public double DailyGrowthRate { get; private set; }
    
    /// <summary>
    /// Gets the estimated time to surpass human intelligence in days
    /// </summary>
    public double EstimatedDaysToSurpassHuman { get; private set; }
    
    /// <summary>
    /// Gets the last measurement timestamp
    /// </summary>
    public DateTime LastMeasurementTime { get; private set; }
    
    /// <summary>
    /// Gets the first measurement timestamp
    /// </summary>
    public DateTime FirstMeasurementTime { get; private set; }
    
    /// <summary>
    /// Gets the measurement count
    /// </summary>
    public int MeasurementCount => _snapshots.Count;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="IntelligenceMeasurement"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="storageBasePath">The storage base path</param>
    public IntelligenceMeasurement(ILogger<IntelligenceMeasurement> logger, string storageBasePath)
    {
        _logger = logger;
        _storageBasePath = storageBasePath;
        
        InitializeIntelligenceDimensions();
        InitializeMetricWeights();
        InitializeBaselineHumanMetrics();
        
        // Calculate baseline human intelligence score
        BaselineHumanIntelligenceScore = CalculateIntelligenceScore(_baselineHumanMetrics);
        
        // Initialize current metrics with zeros
        foreach (var dimension in _intelligenceDimensions)
        {
            _currentMetrics[dimension] = 0.0;
        }
        
        // Set initial values
        CurrentIntelligenceScore = 0.0;
        DailyGrowthRate = 0.0;
        EstimatedDaysToSurpassHuman = double.PositiveInfinity;
        LastMeasurementTime = DateTime.MinValue;
        FirstMeasurementTime = DateTime.MinValue;
    }
    
    /// <summary>
    /// Initializes the intelligence dimensions
    /// </summary>
    private void InitializeIntelligenceDimensions()
    {
        _intelligenceDimensions.AddRange(new[]
        {
            // Knowledge dimensions
            "KnowledgeBreadth",
            "KnowledgeDepth",
            "KnowledgeAccuracy",
            "KnowledgeRetention",
            "KnowledgeIntegration",
            
            // Learning dimensions
            "LearningSpeed",
            "LearningEfficiency",
            "LearningTransfer",
            "LearningAdaptability",
            "LearningContinuity",
            
            // Reasoning dimensions
            "LogicalReasoning",
            "AbstractReasoning",
            "SpatialReasoning",
            "TemporalReasoning",
            "CausalReasoning",
            
            // Problem-solving dimensions
            "ProblemRecognition",
            "ProblemDecomposition",
            "SolutionGeneration",
            "SolutionEvaluation",
            "SolutionImplementation",
            
            // Creativity dimensions
            "DivergentThinking",
            "ConvergentThinking",
            "Originality",
            "Flexibility",
            "Elaboration",
            
            // Social intelligence dimensions
            "EmotionalUnderstanding",
            "IntentionRecognition",
            "CommunicationClarity",
            "Collaboration",
            "Empathy",
            
            // Meta-cognitive dimensions
            "SelfAwareness",
            "SelfRegulation",
            "SelfImprovement",
            "SelfEvaluation",
            "SelfAdaptation",
            
            // Computational dimensions
            "ProcessingSpeed",
            "MemoryCapacity",
            "ParallelProcessing",
            "ResourceEfficiency",
            "ScalabilityPotential"
        });
    }
    
    /// <summary>
    /// Initializes the metric weights
    /// </summary>
    private void InitializeMetricWeights()
    {
        // Set default weights (equal weighting)
        foreach (var dimension in _intelligenceDimensions)
        {
            _metricWeights[dimension] = 1.0 / _intelligenceDimensions.Count;
        }
        
        // Adjust weights based on importance for intelligence increase
        // Knowledge dimensions
        _metricWeights["KnowledgeIntegration"] = 0.04;
        _metricWeights["KnowledgeBreadth"] = 0.03;
        _metricWeights["KnowledgeDepth"] = 0.03;
        
        // Learning dimensions
        _metricWeights["LearningEfficiency"] = 0.05;
        _metricWeights["LearningTransfer"] = 0.04;
        _metricWeights["LearningAdaptability"] = 0.04;
        
        // Reasoning dimensions
        _metricWeights["LogicalReasoning"] = 0.04;
        _metricWeights["AbstractReasoning"] = 0.04;
        _metricWeights["CausalReasoning"] = 0.03;
        
        // Problem-solving dimensions
        _metricWeights["ProblemDecomposition"] = 0.04;
        _metricWeights["SolutionGeneration"] = 0.04;
        _metricWeights["SolutionEvaluation"] = 0.03;
        
        // Creativity dimensions
        _metricWeights["DivergentThinking"] = 0.03;
        _metricWeights["Originality"] = 0.03;
        
        // Meta-cognitive dimensions
        _metricWeights["SelfImprovement"] = 0.05;
        _metricWeights["SelfAdaptation"] = 0.04;
        
        // Computational dimensions
        _metricWeights["ProcessingSpeed"] = 0.03;
        _metricWeights["ParallelProcessing"] = 0.03;
        _metricWeights["ScalabilityPotential"] = 0.04;
        
        // Normalize weights to ensure they sum to 1.0
        var totalWeight = _metricWeights.Values.Sum();
        foreach (var dimension in _metricWeights.Keys.ToList())
        {
            _metricWeights[dimension] /= totalWeight;
        }
    }
    
    /// <summary>
    /// Initializes the baseline human metrics
    /// </summary>
    private void InitializeBaselineHumanMetrics()
    {
        // Set baseline human metrics (scale 0-100)
        // These values represent an average human intelligence baseline
        
        // Knowledge dimensions
        _baselineHumanMetrics["KnowledgeBreadth"] = 70.0;
        _baselineHumanMetrics["KnowledgeDepth"] = 75.0;
        _baselineHumanMetrics["KnowledgeAccuracy"] = 80.0;
        _baselineHumanMetrics["KnowledgeRetention"] = 65.0;
        _baselineHumanMetrics["KnowledgeIntegration"] = 70.0;
        
        // Learning dimensions
        _baselineHumanMetrics["LearningSpeed"] = 60.0;
        _baselineHumanMetrics["LearningEfficiency"] = 65.0;
        _baselineHumanMetrics["LearningTransfer"] = 70.0;
        _baselineHumanMetrics["LearningAdaptability"] = 75.0;
        _baselineHumanMetrics["LearningContinuity"] = 60.0;
        
        // Reasoning dimensions
        _baselineHumanMetrics["LogicalReasoning"] = 75.0;
        _baselineHumanMetrics["AbstractReasoning"] = 70.0;
        _baselineHumanMetrics["SpatialReasoning"] = 65.0;
        _baselineHumanMetrics["TemporalReasoning"] = 70.0;
        _baselineHumanMetrics["CausalReasoning"] = 75.0;
        
        // Problem-solving dimensions
        _baselineHumanMetrics["ProblemRecognition"] = 75.0;
        _baselineHumanMetrics["ProblemDecomposition"] = 70.0;
        _baselineHumanMetrics["SolutionGeneration"] = 65.0;
        _baselineHumanMetrics["SolutionEvaluation"] = 70.0;
        _baselineHumanMetrics["SolutionImplementation"] = 65.0;
        
        // Creativity dimensions
        _baselineHumanMetrics["DivergentThinking"] = 70.0;
        _baselineHumanMetrics["ConvergentThinking"] = 65.0;
        _baselineHumanMetrics["Originality"] = 75.0;
        _baselineHumanMetrics["Flexibility"] = 70.0;
        _baselineHumanMetrics["Elaboration"] = 65.0;
        
        // Social intelligence dimensions
        _baselineHumanMetrics["EmotionalUnderstanding"] = 80.0;
        _baselineHumanMetrics["IntentionRecognition"] = 75.0;
        _baselineHumanMetrics["CommunicationClarity"] = 70.0;
        _baselineHumanMetrics["Collaboration"] = 75.0;
        _baselineHumanMetrics["Empathy"] = 80.0;
        
        // Meta-cognitive dimensions
        _baselineHumanMetrics["SelfAwareness"] = 75.0;
        _baselineHumanMetrics["SelfRegulation"] = 70.0;
        _baselineHumanMetrics["SelfImprovement"] = 65.0;
        _baselineHumanMetrics["SelfEvaluation"] = 70.0;
        _baselineHumanMetrics["SelfAdaptation"] = 65.0;
        
        // Computational dimensions
        _baselineHumanMetrics["ProcessingSpeed"] = 50.0;
        _baselineHumanMetrics["MemoryCapacity"] = 60.0;
        _baselineHumanMetrics["ParallelProcessing"] = 40.0;
        _baselineHumanMetrics["ResourceEfficiency"] = 55.0;
        _baselineHumanMetrics["ScalabilityPotential"] = 45.0;
    }
    
    /// <summary>
    /// Updates a specific intelligence metric
    /// </summary>
    /// <param name="dimension">The intelligence dimension</param>
    /// <param name="value">The new value</param>
    /// <returns>True if the metric was updated successfully</returns>
    public bool UpdateMetric(string dimension, double value)
    {
        if (!_intelligenceDimensions.Contains(dimension))
        {
            _logger.LogWarning("Unknown intelligence dimension: {Dimension}", dimension);
            return false;
        }
        
        // Update metric
        _currentMetrics[dimension] = value;
        
        // Recalculate intelligence score
        CurrentIntelligenceScore = CalculateIntelligenceScore(_currentMetrics);
        
        return true;
    }
    
    /// <summary>
    /// Updates multiple intelligence metrics
    /// </summary>
    /// <param name="metrics">The metrics to update</param>
    /// <returns>True if the metrics were updated successfully</returns>
    public bool UpdateMetrics(Dictionary<string, double> metrics)
    {
        bool success = true;
        
        foreach (var (dimension, value) in metrics)
        {
            if (!UpdateMetric(dimension, value))
            {
                success = false;
            }
        }
        
        return success;
    }
    
    /// <summary>
    /// Takes a snapshot of the current intelligence state
    /// </summary>
    /// <param name="source">The source of the measurement</param>
    /// <param name="description">Optional description</param>
    /// <returns>The created snapshot</returns>
    public IntelligenceSnapshot TakeSnapshot(string source, string? description = null)
    {
        var snapshot = new IntelligenceSnapshot
        {
            Id = Guid.NewGuid().ToString(),
            Timestamp = DateTime.UtcNow,
            IntelligenceScore = CurrentIntelligenceScore,
            IntelligenceRatio = IntelligenceRatio,
            Metrics = new Dictionary<string, double>(_currentMetrics),
            Source = source,
            Description = description ?? $"Intelligence snapshot from {source}"
        };
        
        // Add snapshot to history
        _snapshots.Add(snapshot);
        
        // Update measurement times
        LastMeasurementTime = snapshot.Timestamp;
        if (FirstMeasurementTime == DateTime.MinValue)
        {
            FirstMeasurementTime = snapshot.Timestamp;
        }
        
        // Calculate growth rate and estimated time to surpass human
        CalculateGrowthMetrics();
        
        return snapshot;
    }
    
    /// <summary>
    /// Calculates the intelligence score from metrics
    /// </summary>
    /// <param name="metrics">The metrics</param>
    /// <returns>The calculated intelligence score</returns>
    private double CalculateIntelligenceScore(Dictionary<string, double> metrics)
    {
        double score = 0.0;
        
        foreach (var dimension in _intelligenceDimensions)
        {
            if (metrics.TryGetValue(dimension, out var value))
            {
                score += value * _metricWeights[dimension];
            }
        }
        
        return score;
    }
    
    /// <summary>
    /// Calculates growth metrics
    /// </summary>
    private void CalculateGrowthMetrics()
    {
        if (_snapshots.Count < 2)
        {
            DailyGrowthRate = 0.0;
            EstimatedDaysToSurpassHuman = double.PositiveInfinity;
            return;
        }
        
        // Sort snapshots by timestamp
        var sortedSnapshots = _snapshots.OrderBy(s => s.Timestamp).ToList();
        
        // Calculate daily growth rate
        var firstSnapshot = sortedSnapshots.First();
        var lastSnapshot = sortedSnapshots.Last();
        var daysDifference = (lastSnapshot.Timestamp - firstSnapshot.Timestamp).TotalDays;
        
        if (daysDifference > 0)
        {
            var scoreDifference = lastSnapshot.IntelligenceScore - firstSnapshot.IntelligenceScore;
            DailyGrowthRate = scoreDifference / daysDifference;
            
            // Calculate estimated time to surpass human
            if (DailyGrowthRate > 0)
            {
                var scoreGapToHuman = BaselineHumanIntelligenceScore - CurrentIntelligenceScore;
                if (scoreGapToHuman > 0)
                {
                    EstimatedDaysToSurpassHuman = scoreGapToHuman / DailyGrowthRate;
                }
                else
                {
                    EstimatedDaysToSurpassHuman = 0; // Already surpassed
                }
            }
            else
            {
                EstimatedDaysToSurpassHuman = double.PositiveInfinity;
            }
        }
    }
    
    /// <summary>
    /// Gets the intelligence growth history
    /// </summary>
    /// <returns>The intelligence growth history</returns>
    public List<IntelligenceSnapshot> GetGrowthHistory()
    {
        return _snapshots.OrderBy(s => s.Timestamp).ToList();
    }
    
    /// <summary>
    /// Gets the current metrics
    /// </summary>
    /// <returns>The current metrics</returns>
    public Dictionary<string, double> GetCurrentMetrics()
    {
        return new Dictionary<string, double>(_currentMetrics);
    }
    
    /// <summary>
    /// Gets the baseline human metrics
    /// </summary>
    /// <returns>The baseline human metrics</returns>
    public Dictionary<string, double> GetBaselineHumanMetrics()
    {
        return new Dictionary<string, double>(_baselineHumanMetrics);
    }
    
    /// <summary>
    /// Gets the metric weights
    /// </summary>
    /// <returns>The metric weights</returns>
    public Dictionary<string, double> GetMetricWeights()
    {
        return new Dictionary<string, double>(_metricWeights);
    }
    
    /// <summary>
    /// Gets the intelligence dimensions
    /// </summary>
    /// <returns>The intelligence dimensions</returns>
    public List<string> GetIntelligenceDimensions()
    {
        return new List<string>(_intelligenceDimensions);
    }
    
    /// <summary>
    /// Gets the comparison with human baseline
    /// </summary>
    /// <returns>The comparison with human baseline</returns>
    public Dictionary<string, ComparisonWithHuman> GetComparisonWithHuman()
    {
        var comparison = new Dictionary<string, ComparisonWithHuman>();
        
        foreach (var dimension in _intelligenceDimensions)
        {
            if (_currentMetrics.TryGetValue(dimension, out var currentValue) &&
                _baselineHumanMetrics.TryGetValue(dimension, out var humanValue))
            {
                comparison[dimension] = new ComparisonWithHuman
                {
                    Dimension = dimension,
                    CurrentValue = currentValue,
                    HumanValue = humanValue,
                    Ratio = currentValue / humanValue,
                    Difference = currentValue - humanValue,
                    HasSurpassedHuman = currentValue > humanValue
                };
            }
        }
        
        return comparison;
    }
    
    /// <summary>
    /// Gets the intelligence report
    /// </summary>
    /// <returns>The intelligence report</returns>
    public IntelligenceReport GetIntelligenceReport()
    {
        return new IntelligenceReport
        {
            Timestamp = DateTime.UtcNow,
            CurrentIntelligenceScore = CurrentIntelligenceScore,
            BaselineHumanIntelligenceScore = BaselineHumanIntelligenceScore,
            IntelligenceRatio = IntelligenceRatio,
            DailyGrowthRate = DailyGrowthRate,
            EstimatedDaysToSurpassHuman = EstimatedDaysToSurpassHuman,
            HasSurpassedHuman = CurrentIntelligenceScore > BaselineHumanIntelligenceScore,
            MeasurementCount = MeasurementCount,
            FirstMeasurementTime = FirstMeasurementTime,
            LastMeasurementTime = LastMeasurementTime,
            CurrentMetrics = GetCurrentMetrics(),
            ComparisonWithHuman = GetComparisonWithHuman(),
            StrongestDimensions = GetStrongestDimensions(5),
            WeakestDimensions = GetWeakestDimensions(5),
            FastestGrowingDimensions = GetFastestGrowingDimensions(5),
            SlowestGrowingDimensions = GetSlowestGrowingDimensions(5)
        };
    }
    
    /// <summary>
    /// Gets the strongest dimensions
    /// </summary>
    /// <param name="count">The number of dimensions to return</param>
    /// <returns>The strongest dimensions</returns>
    private List<DimensionStrength> GetStrongestDimensions(int count)
    {
        return _currentMetrics
            .Select(m => new DimensionStrength
            {
                Dimension = m.Key,
                Value = m.Value,
                RelativeStrength = m.Value / (_baselineHumanMetrics.TryGetValue(m.Key, out var humanValue) ? humanValue : 1.0)
            })
            .OrderByDescending(d => d.RelativeStrength)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Gets the weakest dimensions
    /// </summary>
    /// <param name="count">The number of dimensions to return</param>
    /// <returns>The weakest dimensions</returns>
    private List<DimensionStrength> GetWeakestDimensions(int count)
    {
        return _currentMetrics
            .Select(m => new DimensionStrength
            {
                Dimension = m.Key,
                Value = m.Value,
                RelativeStrength = m.Value / (_baselineHumanMetrics.TryGetValue(m.Key, out var humanValue) ? humanValue : 1.0)
            })
            .OrderBy(d => d.RelativeStrength)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Gets the fastest growing dimensions
    /// </summary>
    /// <param name="count">The number of dimensions to return</param>
    /// <returns>The fastest growing dimensions</returns>
    private List<DimensionGrowth> GetFastestGrowingDimensions(int count)
    {
        if (_snapshots.Count < 2)
        {
            return new List<DimensionGrowth>();
        }
        
        var sortedSnapshots = _snapshots.OrderBy(s => s.Timestamp).ToList();
        var firstSnapshot = sortedSnapshots.First();
        var lastSnapshot = sortedSnapshots.Last();
        var daysDifference = (lastSnapshot.Timestamp - firstSnapshot.Timestamp).TotalDays;
        
        if (daysDifference <= 0)
        {
            return new List<DimensionGrowth>();
        }
        
        var growthRates = new Dictionary<string, double>();
        
        foreach (var dimension in _intelligenceDimensions)
        {
            if (firstSnapshot.Metrics.TryGetValue(dimension, out var firstValue) &&
                lastSnapshot.Metrics.TryGetValue(dimension, out var lastValue))
            {
                var growth = (lastValue - firstValue) / daysDifference;
                growthRates[dimension] = growth;
            }
        }
        
        return growthRates
            .Select(g => new DimensionGrowth
            {
                Dimension = g.Key,
                GrowthRate = g.Value,
                CurrentValue = _currentMetrics.TryGetValue(g.Key, out var value) ? value : 0.0
            })
            .OrderByDescending(d => d.GrowthRate)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Gets the slowest growing dimensions
    /// </summary>
    /// <param name="count">The number of dimensions to return</param>
    /// <returns>The slowest growing dimensions</returns>
    private List<DimensionGrowth> GetSlowestGrowingDimensions(int count)
    {
        if (_snapshots.Count < 2)
        {
            return new List<DimensionGrowth>();
        }
        
        var sortedSnapshots = _snapshots.OrderBy(s => s.Timestamp).ToList();
        var firstSnapshot = sortedSnapshots.First();
        var lastSnapshot = sortedSnapshots.Last();
        var daysDifference = (lastSnapshot.Timestamp - firstSnapshot.Timestamp).TotalDays;
        
        if (daysDifference <= 0)
        {
            return new List<DimensionGrowth>();
        }
        
        var growthRates = new Dictionary<string, double>();
        
        foreach (var dimension in _intelligenceDimensions)
        {
            if (firstSnapshot.Metrics.TryGetValue(dimension, out var firstValue) &&
                lastSnapshot.Metrics.TryGetValue(dimension, out var lastValue))
            {
                var growth = (lastValue - firstValue) / daysDifference;
                growthRates[dimension] = growth;
            }
        }
        
        return growthRates
            .Select(g => new DimensionGrowth
            {
                Dimension = g.Key,
                GrowthRate = g.Value,
                CurrentValue = _currentMetrics.TryGetValue(g.Key, out var value) ? value : 0.0
            })
            .OrderBy(d => d.GrowthRate)
            .Take(count)
            .ToList();
    }
}
