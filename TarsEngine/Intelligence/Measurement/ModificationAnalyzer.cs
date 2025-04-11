using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// Analyzes code modifications and improvements
/// </summary>
public class ModificationAnalyzer
{
    private readonly ILogger<ModificationAnalyzer> _logger;
    private readonly MetricsCollector _metricsCollector;

    // Cache of modification analyses
    private readonly List<CodeModification> _modifications = [];

    /// <summary>
    /// Initializes a new instance of the <see cref="ModificationAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="metricsCollector">The metrics collector</param>
    public ModificationAnalyzer(ILogger<ModificationAnalyzer> logger, MetricsCollector metricsCollector)
    {
        _logger = logger;
        _metricsCollector = metricsCollector;
    }

    /// <summary>
    /// Initializes the modification analyzer
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("Initializing Modification Analyzer");

        // Clear the cache
        _modifications.Clear();

        await Task.CompletedTask;
    }

    /// <summary>
    /// Analyzes a code modification
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <param name="reason">The reason for the modification</param>
    /// <param name="improvementType">The improvement type</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task AnalyzeModificationAsync(
        string filePath,
        string originalCode,
        string modifiedCode,
        string reason,
        string improvementType)
    {
        _logger.LogInformation("Analyzing code modification: {FilePath} - {ImprovementType}",
            filePath, improvementType);

        // Create a modification record
        var modification = new CodeModification
        {
            FilePath = filePath,
            Timestamp = DateTime.UtcNow,
            Reason = reason,
            ImprovementType = improvementType,
            OriginalCode = originalCode,
            ModifiedCode = modifiedCode
        };

        // Calculate metrics
        modification.LinesAdded = CountLinesAdded(originalCode, modifiedCode);
        modification.LinesRemoved = CountLinesRemoved(originalCode, modifiedCode);
        modification.LinesModified = CountLinesModified(originalCode, modifiedCode);
        modification.ComplexityChange = CalculateComplexityChange(originalCode, modifiedCode);
        modification.ReadabilityChange = CalculateReadabilityChange(originalCode, modifiedCode);
        modification.PerformanceImpact = EstimatePerformanceImpact(originalCode, modifiedCode, improvementType);

        // Add to the cache
        _modifications.Add(modification);

        // Record metrics
        await RecordModificationMetricsAsync(modification);
    }

    /// <summary>
    /// Gets the modification analysis
    /// </summary>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The modification analysis</returns>
    public async Task<ModificationAnalysis> GetModificationAnalysisAsync(DateTime? startTime = null, DateTime? endTime = null)
    {
        _logger.LogInformation("Getting modification analysis");

        // Filter modifications by time range
        var filteredModifications = _modifications.AsEnumerable();

        if (startTime.HasValue)
        {
            filteredModifications = filteredModifications.Where(m => m.Timestamp >= startTime.Value);
        }

        if (endTime.HasValue)
        {
            filteredModifications = filteredModifications.Where(m => m.Timestamp <= endTime.Value);
        }

        var modifications = filteredModifications.ToList();

        // Create the analysis
        var analysis = new ModificationAnalysis
        {
            StartTime = startTime ?? (modifications.Count > 0 ? modifications.Min(m => m.Timestamp) : DateTime.UtcNow),
            EndTime = endTime ?? (modifications.Count > 0 ? modifications.Max(m => m.Timestamp) : DateTime.UtcNow),
            TotalModifications = modifications.Count,
            TotalLinesAdded = modifications.Sum(m => m.LinesAdded),
            TotalLinesRemoved = modifications.Sum(m => m.LinesRemoved),
            TotalLinesModified = modifications.Sum(m => m.LinesModified),
            AverageComplexityChange = modifications.Count > 0 ? modifications.Average(m => m.ComplexityChange) : 0,
            AverageReadabilityChange = modifications.Count > 0 ? modifications.Average(m => m.ReadabilityChange) : 0,
            AveragePerformanceImpact = modifications.Count > 0 ? modifications.Average(m => m.PerformanceImpact) : 0,
            ModificationsByType = GetModificationsByType(modifications),
            ModificationsByFile = GetModificationsByFile(modifications),
            ModificationTrend = CalculateModificationTrend(modifications)
        };

        return analysis;
    }

    /// <summary>
    /// Records modification metrics
    /// </summary>
    /// <param name="modification">The modification</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task RecordModificationMetricsAsync(CodeModification modification)
    {
        // Record complexity metric
        await _metricsCollector.CollectMetricAsync(new ComplexityMetric
        {
            Name = $"Modification.Complexity.{modification.FilePath}",
            Value = modification.ComplexityChange,
            Type = (TarsEngine.Models.Metrics.ComplexityType)TarsEngine.Unified.ComplexityType.Cognitive,
            Target = modification.FilePath,
            Threshold = 0,
            Dimension = "Complexity",
            Tags = [modification.FilePath, modification.ImprovementType, "modification", "complexity"]
        });

        // Record readability metric
        await _metricsCollector.CollectMetricAsync(new PerformanceMetric
        {
            Name = $"Modification.Readability.{modification.FilePath}",
            Value = modification.ReadabilityChange,
            Type = PerformanceType.Efficiency,
            Dimension = "Readability",
            BaselineValue = 0,
            TargetValue = 1,
            Unit = "score",
            Tags = [modification.FilePath, modification.ImprovementType, "modification", "readability"]
        });

        // Record performance metric
        await _metricsCollector.CollectMetricAsync(new PerformanceMetric
        {
            Name = $"Modification.Performance.{modification.FilePath}",
            Value = modification.PerformanceImpact,
            Type = PerformanceType.Efficiency,
            Dimension = "Performance",
            BaselineValue = 0,
            TargetValue = 1,
            Unit = "score",
            Tags = [modification.FilePath, modification.ImprovementType, "modification", "performance"]
        });

        // Record novelty metric
        await _metricsCollector.CollectMetricAsync(new NoveltyMetric
        {
            Name = $"Modification.Novelty.{modification.FilePath}",
            Value = CalculateNoveltyScore(modification),
            Type = NoveltyType.Solution,
            SimilarityScore = 0.5, // Default value
            Reference = modification.FilePath,
            Dimension = "Novelty",
            Tags = [modification.FilePath, modification.ImprovementType, "modification", "novelty"]
        });
    }

    /// <summary>
    /// Counts the lines added
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <returns>The number of lines added</returns>
    private int CountLinesAdded(string originalCode, string modifiedCode)
    {
        // Simple implementation: count the difference in line count if the modified code has more lines
        var originalLines = originalCode.Split('\n');
        var modifiedLines = modifiedCode.Split('\n');

        return Math.Max(0, modifiedLines.Length - originalLines.Length);
    }

    /// <summary>
    /// Counts the lines removed
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <returns>The number of lines removed</returns>
    private int CountLinesRemoved(string originalCode, string modifiedCode)
    {
        // Simple implementation: count the difference in line count if the original code has more lines
        var originalLines = originalCode.Split('\n');
        var modifiedLines = modifiedCode.Split('\n');

        return Math.Max(0, originalLines.Length - modifiedLines.Length);
    }

    /// <summary>
    /// Counts the lines modified
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <returns>The number of lines modified</returns>
    private int CountLinesModified(string originalCode, string modifiedCode)
    {
        // Simple implementation: count the number of lines that are different
        var originalLines = originalCode.Split('\n');
        var modifiedLines = modifiedCode.Split('\n');

        int minLength = Math.Min(originalLines.Length, modifiedLines.Length);
        int modifiedCount = 0;

        for (int i = 0; i < minLength; i++)
        {
            if (originalLines[i] != modifiedLines[i])
            {
                modifiedCount++;
            }
        }

        return modifiedCount;
    }

    /// <summary>
    /// Calculates the complexity change
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <returns>The complexity change</returns>
    private double CalculateComplexityChange(string originalCode, string modifiedCode)
    {
        // Simple implementation: estimate complexity based on code length and structure
        double originalComplexity = EstimateComplexity(originalCode);
        double modifiedComplexity = EstimateComplexity(modifiedCode);

        // Return the change (negative is better - less complex)
        return modifiedComplexity - originalComplexity;
    }

    /// <summary>
    /// Estimates the complexity of code
    /// </summary>
    /// <param name="code">The code</param>
    /// <returns>The estimated complexity</returns>
    private double EstimateComplexity(string code)
    {
        // Simple implementation: count control structures and nesting levels
        int controlStructureCount = CountOccurrences(code, "if ") +
                                   CountOccurrences(code, "for ") +
                                   CountOccurrences(code, "while ") +
                                   CountOccurrences(code, "switch ") +
                                   CountOccurrences(code, "catch ");

        int nestingLevel = EstimateNestingLevel(code);

        // Combine factors
        return controlStructureCount * (1 + 0.1 * nestingLevel);
    }

    /// <summary>
    /// Estimates the nesting level of code
    /// </summary>
    /// <param name="code">The code</param>
    /// <returns>The estimated nesting level</returns>
    private int EstimateNestingLevel(string code)
    {
        // Simple implementation: count the maximum number of open braces at any point
        int maxNesting = 0;
        int currentNesting = 0;

        foreach (char c in code)
        {
            if (c == '{')
            {
                currentNesting++;
                maxNesting = Math.Max(maxNesting, currentNesting);
            }
            else if (c == '}')
            {
                currentNesting = Math.Max(0, currentNesting - 1);
            }
        }

        return maxNesting;
    }

    /// <summary>
    /// Calculates the readability change
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <returns>The readability change</returns>
    private double CalculateReadabilityChange(string originalCode, string modifiedCode)
    {
        // Simple implementation: estimate readability based on various factors
        double originalReadability = EstimateReadability(originalCode);
        double modifiedReadability = EstimateReadability(modifiedCode);

        // Return the change (positive is better - more readable)
        return modifiedReadability - originalReadability;
    }

    /// <summary>
    /// Estimates the readability of code
    /// </summary>
    /// <param name="code">The code</param>
    /// <returns>The estimated readability</returns>
    private double EstimateReadability(string code)
    {
        // Simple implementation: consider various readability factors
        double score = 0;

        // Factor 1: Line length (shorter lines are more readable)
        double avgLineLength = code.Split('\n').Where(line => !string.IsNullOrWhiteSpace(line))
            .Average(line => line.Length);
        score -= Math.Max(0, (avgLineLength - 80) / 10); // Penalize lines longer than 80 characters

        // Factor 2: Comment density (more comments are more readable)
        int commentCount = CountOccurrences(code, "//") + CountOccurrences(code, "/*");
        int lineCount = code.Split('\n').Length;
        double commentDensity = (double)commentCount / lineCount;
        score += commentDensity * 5; // Reward comments

        // Factor 3: Variable name length (longer names are generally more descriptive)
        // This is a very simplistic approach
        double avgWordLength = code.Split(new[] { ' ', '\t', '\n', '\r', '(', ')', '{', '}', '[', ']', '.', ',', ';' })
            .Where(word => !string.IsNullOrWhiteSpace(word) && char.IsLetter(word[0]))
            .Average(word => word.Length);
        score += Math.Min(5, avgWordLength) / 2; // Reward longer names, up to a point

        return score;
    }

    /// <summary>
    /// Estimates the performance impact
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="modifiedCode">The modified code</param>
    /// <param name="improvementType">The improvement type</param>
    /// <returns>The estimated performance impact</returns>
    private double EstimatePerformanceImpact(string originalCode, string modifiedCode, string improvementType)
    {
        // Simple implementation: estimate performance impact based on the improvement type
        if (improvementType.Contains("performance", StringComparison.OrdinalIgnoreCase))
        {
            // For performance improvements, estimate based on code changes
            double originalPerformance = EstimatePerformance(originalCode);
            double modifiedPerformance = EstimatePerformance(modifiedCode);

            // Return the change (positive is better - more performant)
            return modifiedPerformance - originalPerformance;
        }
        else
        {
            // For other improvements, assume minimal performance impact
            return 0.1; // Slight positive impact
        }
    }

    /// <summary>
    /// Estimates the performance of code
    /// </summary>
    /// <param name="code">The code</param>
    /// <returns>The estimated performance</returns>
    private double EstimatePerformance(string code)
    {
        // Simple implementation: consider various performance factors
        double score = 0;

        // Factor 1: Loop count (fewer loops are generally more performant)
        int loopCount = CountOccurrences(code, "for ") + CountOccurrences(code, "while ") +
                       CountOccurrences(code, "foreach ");
        score -= loopCount * 0.5; // Penalize loops

        // Factor 2: Allocation count (fewer allocations are generally more performant)
        int allocationCount = CountOccurrences(code, "new ") + CountOccurrences(code, "malloc(") +
                             CountOccurrences(code, "calloc(");
        score -= allocationCount * 0.3; // Penalize allocations

        // Factor 3: Recursion (recursion can be less performant)
        bool hasRecursion = code.Contains("(") && code.Contains(")") &&
                           code.Contains("return") && code.Contains("(");
        if (hasRecursion)
        {
            score -= 1; // Penalize recursion
        }

        return score;
    }

    /// <summary>
    /// Calculates the novelty score
    /// </summary>
    /// <param name="modification">The modification</param>
    /// <returns>The novelty score</returns>
    private double CalculateNoveltyScore(CodeModification modification)
    {
        // Simple implementation: estimate novelty based on the modification
        double score = 0;

        // Factor 1: Code change ratio
        int originalLength = modification.OriginalCode.Length;
        int modifiedLength = modification.ModifiedCode.Length;
        double changeRatio = Math.Abs(modifiedLength - originalLength) / (double)Math.Max(1, originalLength);
        score += changeRatio * 0.5; // More changes suggest more novelty

        // Factor 2: New patterns or structures
        bool hasNewPatterns = modification.ModifiedCode.Contains("class ") && !modification.OriginalCode.Contains("class ") ||
                             modification.ModifiedCode.Contains("interface ") && !modification.OriginalCode.Contains("interface ") ||
                             modification.ModifiedCode.Contains("enum ") && !modification.OriginalCode.Contains("enum ");
        if (hasNewPatterns)
        {
            score += 0.3; // Reward new patterns
        }

        // Factor 3: Improvement type
        if (modification.ImprovementType.Contains("novel", StringComparison.OrdinalIgnoreCase) ||
            modification.ImprovementType.Contains("innovative", StringComparison.OrdinalIgnoreCase))
        {
            score += 0.2; // Reward novel improvements
        }

        return Math.Min(1.0, score); // Cap at 1.0
    }

    /// <summary>
    /// Counts the occurrences of a substring in a string
    /// </summary>
    /// <param name="text">The text</param>
    /// <param name="substring">The substring</param>
    /// <returns>The number of occurrences</returns>
    private int CountOccurrences(string text, string substring)
    {
        int count = 0;
        int index = 0;

        while ((index = text.IndexOf(substring, index, StringComparison.Ordinal)) != -1)
        {
            count++;
            index += substring.Length;
        }

        return count;
    }

    /// <summary>
    /// Gets the modifications by type
    /// </summary>
    /// <param name="modifications">The modifications</param>
    /// <returns>The modifications by type</returns>
    private Dictionary<string, int> GetModificationsByType(List<CodeModification> modifications)
    {
        var result = new Dictionary<string, int>();

        foreach (var modification in modifications)
        {
            if (!result.TryGetValue(modification.ImprovementType, out int count))
            {
                result[modification.ImprovementType] = 1;
            }
            else
            {
                result[modification.ImprovementType] = count + 1;
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the modifications by file
    /// </summary>
    /// <param name="modifications">The modifications</param>
    /// <returns>The modifications by file</returns>
    private Dictionary<string, int> GetModificationsByFile(List<CodeModification> modifications)
    {
        var result = new Dictionary<string, int>();

        foreach (var modification in modifications)
        {
            if (!result.TryGetValue(modification.FilePath, out int count))
            {
                result[modification.FilePath] = 1;
            }
            else
            {
                result[modification.FilePath] = count + 1;
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the modification trend
    /// </summary>
    /// <param name="modifications">The modifications</param>
    /// <returns>The modification trend</returns>
    private ModificationTrend CalculateModificationTrend(List<CodeModification> modifications)
    {
        if (modifications.Count < 3)
        {
            return ModificationTrend.Stable;
        }

        // Group modifications by day
        var modificationsByDay = modifications
            .GroupBy(m => m.Timestamp.Date)
            .OrderBy(g => g.Key)
            .Select(g => new { Date = g.Key, Count = g.Count() })
            .ToList();

        if (modificationsByDay.Count < 3)
        {
            return ModificationTrend.Stable;
        }

        // Calculate the trend
        int increasing = 0;
        int decreasing = 0;

        for (int i = 1; i < modificationsByDay.Count; i++)
        {
            if (modificationsByDay[i].Count > modificationsByDay[i - 1].Count)
            {
                increasing++;
            }
            else if (modificationsByDay[i].Count < modificationsByDay[i - 1].Count)
            {
                decreasing++;
            }
        }

        double increasingRatio = (double)increasing / (modificationsByDay.Count - 1);
        double decreasingRatio = (double)decreasing / (modificationsByDay.Count - 1);

        if (increasingRatio > 0.7)
        {
            return ModificationTrend.Increasing;
        }
        else if (decreasingRatio > 0.7)
        {
            return ModificationTrend.Decreasing;
        }
        else if (increasingRatio > 0.5 && decreasingRatio < 0.3)
        {
            return ModificationTrend.SlightlyIncreasing;
        }
        else if (decreasingRatio > 0.5 && increasingRatio < 0.3)
        {
            return ModificationTrend.SlightlyDecreasing;
        }
        else
        {
            return ModificationTrend.Stable;
        }
    }
}

/// <summary>
/// Represents a code modification
/// </summary>
public class CodeModification
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the reason
    /// </summary>
    public string Reason { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the improvement type
    /// </summary>
    public string ImprovementType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the original code
    /// </summary>
    public string OriginalCode { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the modified code
    /// </summary>
    public string ModifiedCode { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the lines added
    /// </summary>
    public int LinesAdded { get; set; }

    /// <summary>
    /// Gets or sets the lines removed
    /// </summary>
    public int LinesRemoved { get; set; }

    /// <summary>
    /// Gets or sets the lines modified
    /// </summary>
    public int LinesModified { get; set; }

    /// <summary>
    /// Gets or sets the complexity change
    /// </summary>
    public double ComplexityChange { get; set; }

    /// <summary>
    /// Gets or sets the readability change
    /// </summary>
    public double ReadabilityChange { get; set; }

    /// <summary>
    /// Gets or sets the performance impact
    /// </summary>
    public double PerformanceImpact { get; set; }
}

/// <summary>
/// Represents a modification analysis
/// </summary>
public class ModificationAnalysis
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
    /// Gets or sets the total modifications
    /// </summary>
    public int TotalModifications { get; set; }

    /// <summary>
    /// Gets or sets the total lines added
    /// </summary>
    public int TotalLinesAdded { get; set; }

    /// <summary>
    /// Gets or sets the total lines removed
    /// </summary>
    public int TotalLinesRemoved { get; set; }

    /// <summary>
    /// Gets or sets the total lines modified
    /// </summary>
    public int TotalLinesModified { get; set; }

    /// <summary>
    /// Gets or sets the average complexity change
    /// </summary>
    public double AverageComplexityChange { get; set; }

    /// <summary>
    /// Gets or sets the average readability change
    /// </summary>
    public double AverageReadabilityChange { get; set; }

    /// <summary>
    /// Gets or sets the average performance impact
    /// </summary>
    public double AveragePerformanceImpact { get; set; }

    /// <summary>
    /// Gets or sets the modifications by type
    /// </summary>
    public Dictionary<string, int> ModificationsByType { get; set; } = new();

    /// <summary>
    /// Gets or sets the modifications by file
    /// </summary>
    public Dictionary<string, int> ModificationsByFile { get; set; } = new();

    /// <summary>
    /// Gets or sets the modification trend
    /// </summary>
    public ModificationTrend ModificationTrend { get; set; }
}

/// <summary>
/// Represents a modification trend
/// </summary>
public enum ModificationTrend
{
    /// <summary>
    /// Increasing trend
    /// </summary>
    Increasing,

    /// <summary>
    /// Slightly increasing trend
    /// </summary>
    SlightlyIncreasing,

    /// <summary>
    /// Stable trend
    /// </summary>
    Stable,

    /// <summary>
    /// Slightly decreasing trend
    /// </summary>
    SlightlyDecreasing,

    /// <summary>
    /// Decreasing trend
    /// </summary>
    Decreasing
}
