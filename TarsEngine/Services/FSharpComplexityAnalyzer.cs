using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for F# code complexity metrics
/// </summary>
public class FSharpComplexityAnalyzer : ICodeComplexityAnalyzer
{
    private readonly ILogger<FSharpComplexityAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<ComplexityType, Dictionary<string, double>>> _thresholds;

    /// <summary>
    /// Initializes a new instance of the <see cref="FSharpComplexityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public FSharpComplexityAnalyzer(ILogger<FSharpComplexityAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<ComplexityType, Dictionary<string, double>>>
        {
            ["F#"] = new Dictionary<ComplexityType, Dictionary<string, double>>
            {
                [ComplexityType.Cyclomatic] = new Dictionary<string, double>
                {
                    ["Function"] = 8,
                    ["Module"] = 15,
                    ["File"] = 40
                },
                [ComplexityType.Cognitive] = new Dictionary<string, double>
                {
                    ["Function"] = 12,
                    ["Module"] = 25,
                    ["File"] = 60
                },
                [ComplexityType.MaintainabilityIndex] = new Dictionary<string, double>
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
                return new List<ComplexityMetric>();
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
                Type = ComplexityType.Cyclomatic,
                FilePath = filePath,
                Language = language,
                Target = fileName,
                TargetType = TargetType.File,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold(language, ComplexityType.Cyclomatic, "File")
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
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeCognitiveComplexityAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("F# cognitive complexity analysis not yet implemented");
        return new List<ComplexityMetric>();
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("F# maintainability index analysis not yet implemented");
        return new List<ComplexityMetric>();
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("F# Halstead complexity analysis not yet implemented");
        return new List<ComplexityMetric>();
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeAllComplexityMetricsAsync(string filePath, string language)
    {
        var metrics = new List<ComplexityMetric>();
        
        metrics.AddRange(await AnalyzeCyclomaticComplexityAsync(filePath, language));
        metrics.AddRange(await AnalyzeCognitiveComplexityAsync(filePath, language));
        metrics.AddRange(await AnalyzeMaintainabilityIndexAsync(filePath, language));
        metrics.AddRange(await AnalyzeHalsteadComplexityAsync(filePath, language));
        
        return metrics;
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeProjectComplexityAsync(string projectPath)
    {
        var metrics = new List<ComplexityMetric>();
        
        try
        {
            var fsharpFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories);
            
            foreach (var file in fsharpFiles)
            {
                metrics.AddRange(await AnalyzeCyclomaticComplexityAsync(file, "F#"));
            }
            
            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);
            
            var projectCyclomaticComplexity = metrics
                .Where(m => m.Type == ComplexityType.Cyclomatic && m.TargetType == TargetType.File)
                .Sum(m => m.Value);
            
            var projectMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {projectName}",
                Description = $"Estimated cyclomatic complexity for F# project {projectName}",
                Value = projectCyclomaticComplexity,
                Type = ComplexityType.Cyclomatic,
                FilePath = projectPath,
                Language = "F#",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };
            
            metrics.Add(projectMetric);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing F# project complexity for {ProjectPath}", projectPath);
        }
        
        return metrics;
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, ComplexityType complexityType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(complexityType, out var typeThresholds))
        {
            return Task.FromResult(typeThresholds);
        }
        
        return Task.FromResult(new Dictionary<string, double>());
    }

    /// <inheritdoc/>
    public Task<bool> SetComplexityThresholdAsync(string language, ComplexityType complexityType, string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<ComplexityType, Dictionary<string, double>>();
            }
            
            if (!_thresholds[language].ContainsKey(complexityType))
            {
                _thresholds[language][complexityType] = new Dictionary<string, double>();
            }
            
            _thresholds[language][complexityType][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting F# complexity threshold for {Language}, {ComplexityType}, {TargetType}", 
                language, complexityType, targetType);
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
    private double GetThreshold(string language, ComplexityType complexityType, string targetType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(complexityType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }
        
        // Default thresholds if not configured
        return complexityType switch
        {
            ComplexityType.Cyclomatic => targetType switch
            {
                "Function" => 8,
                "Module" => 15,
                "File" => 40,
                _ => 8
            },
            ComplexityType.Cognitive => targetType switch
            {
                "Function" => 12,
                "Module" => 25,
                "File" => 60,
                _ => 12
            },
            ComplexityType.MaintainabilityIndex => 20,
            _ => 8
        };
    }
}
