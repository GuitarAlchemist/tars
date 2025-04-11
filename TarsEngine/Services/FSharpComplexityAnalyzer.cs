using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Models.Unified;
using TarsEngine.Utilities;
using TarsEngine.Services.Interfaces;
using TarsEngine.Services.Adapters;
using UnifiedComplexityType = TarsEngine.Models.Unified.ComplexityTypeUnified;
using ModelComplexityType = TarsEngine.Models.Metrics.ComplexityType;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for F# code complexity metrics
/// </summary>
public class FSharpComplexityAnalyzer : ICodeComplexityAnalyzer
{
    private readonly ILogger<FSharpComplexityAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<ModelComplexityType, Dictionary<string, double>>> _thresholds;

    /// <summary>
    /// Initializes a new instance of the <see cref="FSharpComplexityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public FSharpComplexityAnalyzer(ILogger<FSharpComplexityAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<ModelComplexityType, Dictionary<string, double>>>
        {
            ["F#"] = new Dictionary<ModelComplexityType, Dictionary<string, double>>
            {
                [ModelComplexityType.Cyclomatic] = new Dictionary<string, double>
                {
                    ["Function"] = 8,
                    ["Module"] = 15,
                    ["File"] = 40
                },
                [ModelComplexityType.Cognitive] = new Dictionary<string, double>
                {
                    ["Function"] = 12,
                    ["Module"] = 25,
                    ["File"] = 60
                },
                [ModelComplexityType.MaintainabilityIndex] = new Dictionary<string, double>
                {
                    ["Function"] = 20,
                    ["Module"] = 20,
                    ["File"] = 20
                }
            }
        };
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeCyclomaticComplexityAsync(string filePath, string language)
    {
        try
        {
            if (language != "F#")
            {
                _logger.LogWarning("Language {Language} not supported by FSharpComplexityAnalyzer", language);
                return [];
            }

            var sourceCode = await File.ReadAllTextAsync(filePath);

            // For now, we'll use a simplified approach for F# complexity analysis
            // In a real implementation, we would use FSharp.Compiler.Service to parse the code

            var metrics = new List<ComplexityMetric>();
            var fileName = Path.GetFileName(filePath);

            // Estimate complexity based on pattern matching
            var functionCount = CountFunctions(sourceCode);
            var matchCount = CountMatches(sourceCode);
            var ifCount = CountIfs(sourceCode);
            var forCount = CountFors(sourceCode);
            var recursionCount = CountRecursions(sourceCode);

            // Estimate file complexity
            var fileComplexity = matchCount * 2 + ifCount + forCount + recursionCount * 2;

            var fileMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {fileName}",
                Description = $"Estimated cyclomatic complexity for F# file {fileName}",
                Value = fileComplexity,
                Type = TarsEngine.Utilities.ComplexityTypeConverter.ToModelType(UnifiedComplexityType.Cyclomatic),
                FilePath = filePath,
                Language = language,
                Target = fileName,
                TargetType = TargetType.File,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold(language, UnifiedComplexityType.Cyclomatic, "File")
            };

            metrics.Add(fileMetric);

            // In a full implementation, we would analyze individual functions and modules
            // For now, we'll just add a note about the simplified implementation

            _logger.LogInformation("F# complexity analysis is using a simplified approach. For accurate analysis, FSharp.Compiler.Service should be used.");

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing cyclomatic complexity for F# file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeCognitiveComplexityAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("F# cognitive complexity analysis not yet implemented");
        return [];
    }

    /// <inheritdoc/>
    public async Task<List<MaintainabilityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language)
    {
        // Basic implementation for F# maintainability index
        _logger.LogInformation("Using basic maintainability index analysis for F#");

        try
        {
            var metrics = new List<MaintainabilityMetric>();
            var fileName = Path.GetFileName(filePath);

            // Create a simple file-level maintainability metric based on file size
            var fileContent = await File.ReadAllTextAsync(filePath);
            var lines = fileContent.Split('\n');
            var linesOfCode = lines.Count(l => !string.IsNullOrWhiteSpace(l) && !l.TrimStart().StartsWith("//"));
            var commentLines = lines.Count(l => !string.IsNullOrWhiteSpace(l) && l.TrimStart().StartsWith("//"));
            var commentPercentage = linesOfCode > 0 ? (double)commentLines / linesOfCode * 100 : 0;

            // Estimate Halstead volume based on file size
            var halsteadVolume = linesOfCode * 10; // Simple estimation

            // Estimate cyclomatic complexity based on keywords
            var cyclomaticComplexity = fileContent.Split(["if", "match", "for", "while"], StringSplitOptions.None).Length - 1;

            var metric = new MaintainabilityMetric
            {
                Name = $"Maintainability Index - {fileName}",
                Description = $"Estimated maintainability index for F# file {fileName}",
                HalsteadVolume = halsteadVolume,
                CyclomaticComplexity = cyclomaticComplexity,
                LinesOfCode = linesOfCode,
                CommentPercentage = commentPercentage,
                FilePath = filePath,
                Language = language,
                Target = fileName,
                TargetType = TargetType.File,
                Timestamp = DateTime.UtcNow,
                UseMicrosoftFormula = true,
                ThresholdValue = GetThreshold(language, UnifiedComplexityType.MaintainabilityIndex, "File")
            };

            // Calculate maintainability index
            metric.Value = metric.MaintainabilityIndex;

            metrics.Add(metric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing maintainability index for F# file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<HalsteadMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language)
    {
        // Basic implementation for F# Halstead complexity
        _logger.LogInformation("Using basic Halstead complexity analysis for F#");

        try
        {
            var metrics = new List<HalsteadMetric>();
            var fileName = Path.GetFileName(filePath);

            // Create a simple file-level Halstead metric based on file content
            var fileContent = await File.ReadAllTextAsync(filePath);

            // Count distinct operators (simplified for F#)
            var operators = new[] { "+", "-", "*", "/", "=", "<", ">", "|", "&", "^", "!", "~", ".", "<|", "|>", ">>" };
            var distinctOperators = operators.Count(op => fileContent.Contains(op));
            var totalOperators = operators.Sum(op => fileContent.Split(op).Length - 1);

            // Count distinct operands (simplified for F#)
            var words = fileContent.Split([' ', '\n', '\r', '\t', '(', ')', '{', '}', '[', ']', ',', ';', ':', '"'], StringSplitOptions.RemoveEmptyEntries);
            var operands = words.Where(w => !operators.Contains(w) && !new[] { "let", "if", "then", "else", "match", "with", "for", "while", "do", "done", "in", "rec", "fun", "function", "type", "module", "open" }.Contains(w)).ToList();
            var distinctOperands = operands.Distinct().Count();
            var totalOperands = operands.Count;

            // Create metrics for each Halstead type
            foreach (var halsteadType in Enum.GetValues<HalsteadType>())
            {
                var metric = new HalsteadMetric
                {
                    Type = halsteadType,
                    FilePath = filePath,
                    Language = language,
                    Target = fileName,
                    TargetType = TargetType.File,
                    DistinctOperators = distinctOperators,
                    DistinctOperands = distinctOperands,
                    TotalOperators = totalOperators,
                    TotalOperands = totalOperands,
                    Timestamp = DateTime.UtcNow
                };

                // Set name, description, and value based on Halstead type
                switch (halsteadType)
                {
                    case HalsteadType.Vocabulary:
                        metric.Name = $"Halstead Vocabulary - {fileName}";
                        metric.Description = $"Halstead vocabulary (n) for file {fileName}";
                        metric.Value = metric.Vocabulary;
                        break;
                    case HalsteadType.Length:
                        metric.Name = $"Halstead Length - {fileName}";
                        metric.Description = $"Halstead length (N) for file {fileName}";
                        metric.Value = metric.Length;
                        break;
                    case HalsteadType.Volume:
                        metric.Name = $"Halstead Volume - {fileName}";
                        metric.Description = $"Halstead volume (V) for file {fileName}";
                        metric.Value = metric.Volume;
                        metric.ThresholdValue = 10000; // Default threshold for F#
                        break;
                    case HalsteadType.Difficulty:
                        metric.Name = $"Halstead Difficulty - {fileName}";
                        metric.Description = $"Halstead difficulty (D) for file {fileName}";
                        metric.Value = metric.Difficulty;
                        metric.ThresholdValue = 50; // Default threshold for F#
                        break;
                    case HalsteadType.Effort:
                        metric.Name = $"Halstead Effort - {fileName}";
                        metric.Description = $"Halstead effort (E) for file {fileName}";
                        metric.Value = metric.Effort;
                        metric.ThresholdValue = 500000; // Default threshold for F#
                        break;
                    case HalsteadType.TimeRequired:
                        metric.Name = $"Halstead Time Required - {fileName}";
                        metric.Description = $"Halstead time required (T) for file {fileName}";
                        metric.Value = metric.TimeRequired;
                        break;
                    case HalsteadType.DeliveredBugs:
                        metric.Name = $"Halstead Delivered Bugs - {fileName}";
                        metric.Description = $"Halstead delivered bugs (B) for file {fileName}";
                        metric.Value = metric.DeliveredBugs;
                        break;
                }

                metrics.Add(metric);
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing Halstead complexity for F# file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public Task<List<ReadabilityMetric>> AnalyzeReadabilityAsync(string filePath, string language, ReadabilityType readabilityType)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("F# readability analysis not yet implemented");
        return Task.FromResult(new List<ReadabilityMetric>());
    }

    /// <inheritdoc/>
    public async Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeAllComplexityMetricsAsync(string filePath, string language)
    {
        var complexityMetrics = new List<ComplexityMetric>();
        var halsteadMetrics = new List<HalsteadMetric>();
        var maintainabilityMetrics = new List<MaintainabilityMetric>();
        var readabilityMetrics = new List<ReadabilityMetric>();

        // Get cyclomatic complexity metrics
        complexityMetrics.AddRange(await AnalyzeCyclomaticComplexityAsync(filePath, language));

        // Get cognitive complexity metrics
        complexityMetrics.AddRange(await AnalyzeCognitiveComplexityAsync(filePath, language));

        // Get Halstead complexity metrics
        halsteadMetrics.AddRange(await AnalyzeHalsteadComplexityAsync(filePath, language));

        // Get maintainability index metrics
        maintainabilityMetrics.AddRange(await AnalyzeMaintainabilityIndexAsync(filePath, language));

        return (complexityMetrics, halsteadMetrics, maintainabilityMetrics, readabilityMetrics);
    }

    /// <inheritdoc/>
    public async Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeProjectComplexityAsync(string projectPath)
    {
        try
        {
            var complexityMetrics = new List<ComplexityMetric>();
            var halsteadMetrics = new List<HalsteadMetric>();
            var maintainabilityMetrics = new List<MaintainabilityMetric>();
            var readabilityMetrics = new List<ReadabilityMetric>();

            // Analyze F# files
            var fsharpFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories);
            foreach (var file in fsharpFiles)
            {
                var fileMetrics = await AnalyzeAllComplexityMetricsAsync(file, "F#");
                complexityMetrics.AddRange(fileMetrics.ComplexityMetrics);
                halsteadMetrics.AddRange(fileMetrics.HalsteadMetrics);
                maintainabilityMetrics.AddRange(fileMetrics.MaintainabilityMetrics);
                readabilityMetrics.AddRange(fileMetrics.ReadabilityMetrics);
            }

            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);

            // Calculate project-level cyclomatic complexity
            var projectCyclomaticComplexity = complexityMetrics
                .Where(m => m.Type == TarsEngine.Services.Adapters.ComplexityTypeConverter.ToModelType(UnifiedComplexityType.Cyclomatic) && m.TargetType == TargetType.File)
                .Sum(m => m.Value);

            var cyclomaticMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {projectName}",
                Description = $"Estimated cyclomatic complexity for F# project {projectName}",
                Value = projectCyclomaticComplexity,
                Type = TarsEngine.Services.Adapters.ComplexityTypeConverter.ToModelType(UnifiedComplexityType.Cyclomatic),
                FilePath = projectPath,
                Language = "F#",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };

            complexityMetrics.Add(cyclomaticMetric);

            // Calculate project-level Halstead volume
            var projectHalsteadVolume = halsteadMetrics
                .Where(m => m.Type == HalsteadType.Volume && m.TargetType == TargetType.File)
                .Sum(m => m.Value);

            var halsteadMetric = new HalsteadMetric
            {
                Name = $"Halstead Volume - {projectName}",
                Description = $"Halstead volume for F# project {projectName}",
                Value = projectHalsteadVolume,
                Type = HalsteadType.Volume,
                FilePath = projectPath,
                Language = "F#",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };

            halsteadMetrics.Add(halsteadMetric);

            // Calculate project-level maintainability index
            // Use average of file maintainability indices
            var fileMaintenanceIndices = maintainabilityMetrics
                .Where(m => m.TargetType == TargetType.File)
                .ToList();

            if (fileMaintenanceIndices.Any())
            {
                var averageMaintainabilityIndex = fileMaintenanceIndices.Average(m => m.Value);

                var maintainabilityMetric = new MaintainabilityMetric
                {
                    Name = $"Maintainability Index - {projectName}",
                    Description = $"Maintainability index for F# project {projectName}",
                    Value = averageMaintainabilityIndex,
                    HalsteadVolume = projectHalsteadVolume,
                    CyclomaticComplexity = projectCyclomaticComplexity,
                    LinesOfCode = fileMaintenanceIndices.Sum(m => m.LinesOfCode),
                    CommentPercentage = fileMaintenanceIndices.Average(m => m.CommentPercentage),
                    FilePath = projectPath,
                    Language = "F#",
                    Target = projectName,
                    TargetType = TargetType.Project,
                    Timestamp = DateTime.UtcNow,
                    UseMicrosoftFormula = true
                };

                maintainabilityMetrics.Add(maintainabilityMetric);
            }

            return (complexityMetrics, halsteadMetrics, maintainabilityMetrics, readabilityMetrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing F# project complexity for {ProjectPath}", projectPath);
            return ([], [], [], []);
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, UnifiedComplexityType complexityType)
    {
        // Convert to Models.Metrics.ComplexityType
        var modelComplexityType = TarsEngine.Utilities.ComplexityTypeConverter.ToModelType(complexityType);

        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(modelComplexityType, out var typeThresholds))
        {
            return Task.FromResult(typeThresholds);
        }

        return Task.FromResult(new Dictionary<string, double>());
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetHalsteadThresholdsAsync(string language, HalsteadType halsteadType)
    {
        // Default thresholds for F# Halstead metrics
        var thresholds = new Dictionary<string, double>
        {
            ["Function"] = halsteadType switch
            {
                HalsteadType.Volume => 500,
                HalsteadType.Difficulty => 15,
                HalsteadType.Effort => 10000,
                _ => 0
            },
            ["Module"] = halsteadType switch
            {
                HalsteadType.Volume => 4000,
                HalsteadType.Difficulty => 30,
                HalsteadType.Effort => 100000,
                _ => 0
            },
            ["File"] = halsteadType switch
            {
                HalsteadType.Volume => 10000,
                HalsteadType.Difficulty => 50,
                HalsteadType.Effort => 500000,
                _ => 0
            }
        };

        return Task.FromResult(thresholds);
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetMaintainabilityThresholdsAsync(string language)
    {
        // Default thresholds for F# maintainability index
        var thresholds = new Dictionary<string, double>
        {
            ["Function"] = 60,
            ["Module"] = 50,
            ["File"] = 40
        };

        return Task.FromResult(thresholds);
    }

    /// <inheritdoc/>
    public Task<bool> SetComplexityThresholdAsync(string language, UnifiedComplexityType complexityType, string targetType, double threshold)
    {
        try
        {
            // Convert to Models.Metrics.ComplexityType
            var modelComplexityType = TarsEngine.Utilities.ComplexityTypeConverter.ToModelType(complexityType);

            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<ModelComplexityType, Dictionary<string, double>>();
            }

            if (!_thresholds[language].ContainsKey(modelComplexityType))
            {
                _thresholds[language][modelComplexityType] = new Dictionary<string, double>();
            }

            _thresholds[language][modelComplexityType][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting F# complexity threshold for {Language}, {ComplexityType}, {TargetType}",
                language, complexityType, targetType);
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> SetHalsteadThresholdAsync(string language, HalsteadType halsteadType, string targetType, double threshold)
    {
        // F# Halstead thresholds are currently fixed
        _logger.LogInformation("Setting F# Halstead threshold for {HalsteadType}, {TargetType} to {Threshold}",
            halsteadType, targetType, threshold);
        return Task.FromResult(true);
    }

    /// <inheritdoc/>
    public Task<bool> SetMaintainabilityThresholdAsync(string language, string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<ModelComplexityType, Dictionary<string, double>>();
            }

            if (!_thresholds[language].ContainsKey(ModelComplexityType.MaintainabilityIndex))
            {
                _thresholds[language][ModelComplexityType.MaintainabilityIndex] = new Dictionary<string, double>();
            }

            _thresholds[language][ModelComplexityType.MaintainabilityIndex][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting F# maintainability threshold for {Language}, {TargetType}",
                language, targetType);
            return Task.FromResult(false);
        }
    }

    /// <summary>
    /// Counts the number of function definitions in F# code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <returns>Number of functions</returns>
    private int CountFunctions(string sourceCode)
    {
        // Count "let" definitions that are likely functions
        // This is a simplified approach and not 100% accurate
        var lines = sourceCode.Split('\n');
        int count = 0;

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("let ") && trimmedLine.Contains("=") && !trimmedLine.StartsWith("let!"))
            {
                // Check if it's likely a function (has parameters)
                var beforeEquals = trimmedLine.Split('=')[0];
                if (beforeEquals.Contains(" ") && !beforeEquals.Contains("[]") && !beforeEquals.Contains("{}"))
                {
                    count++;
                }
            }
        }

        return count;
    }

    /// <summary>
    /// Counts the number of match expressions in F# code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <returns>Number of match expressions</returns>
    private int CountMatches(string sourceCode)
    {
        // Count "match" expressions
        var lines = sourceCode.Split('\n');
        int count = 0;

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (trimmedLine.Contains("match ") && trimmedLine.Contains("with"))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Counts the number of if expressions in F# code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <returns>Number of if expressions</returns>
    private int CountIfs(string sourceCode)
    {
        // Count "if" expressions
        var lines = sourceCode.Split('\n');
        int count = 0;

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("if ") || trimmedLine.Contains(" if "))
            {
                count++;
            }

            // Count "elif" and "else if"
            if (trimmedLine.StartsWith("elif ") || trimmedLine.StartsWith("else if "))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Counts the number of for loops in F# code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <returns>Number of for loops</returns>
    private int CountFors(string sourceCode)
    {
        // Count "for" loops
        var lines = sourceCode.Split('\n');
        int count = 0;

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("for ") && trimmedLine.Contains(" do"))
            {
                count++;
            }

            // Count "while" loops
            if (trimmedLine.StartsWith("while ") && trimmedLine.Contains(" do"))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Estimates the number of recursive functions in F# code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <returns>Estimated number of recursive functions</returns>
    private int CountRecursions(string sourceCode)
    {
        // This is a very simplified approach to estimate recursion
        // In a real implementation, we would need to analyze the AST

        var lines = sourceCode.Split('\n');
        int count = 0;

        // Count "rec" keyword
        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("let rec "))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Gets the threshold value for a specific language, complexity type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="complexityType">Type of complexity</param>
    /// <param name="targetType">Type of target (function, module, etc.)</param>
    /// <returns>Threshold value</returns>
    private double GetThreshold(string language, UnifiedComplexityType complexityType, string targetType)
    {
        // Convert to Models.Metrics.ComplexityType
        var modelComplexityType = TarsEngine.Utilities.ComplexityTypeConverter.ToModelType(complexityType);

        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(modelComplexityType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }

        // Default thresholds if not configured
        return modelComplexityType switch
        {
            ModelComplexityType.Cyclomatic => targetType switch
            {
                "Function" => 8,
                "Module" => 15,
                "File" => 40,
                _ => 8
            },
            ModelComplexityType.Cognitive => targetType switch
            {
                "Function" => 12,
                "Module" => 25,
                "File" => 60,
                _ => 12
            },
            ModelComplexityType.Halstead => 20,
            _ => 8
        };
    }
}
