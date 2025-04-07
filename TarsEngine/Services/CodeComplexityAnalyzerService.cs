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
/// Service for analyzing code complexity across multiple languages
/// </summary>
public class CodeComplexityAnalyzerService : ICodeComplexityAnalyzer
{
    private readonly ILogger<CodeComplexityAnalyzerService> _logger;
    private readonly CSharpComplexityAnalyzer _csharpAnalyzer;
    private readonly FSharpComplexityAnalyzer _fsharpAnalyzer;
    private readonly Dictionary<string, ICodeComplexityAnalyzer> _analyzers;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeComplexityAnalyzerService"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="csharpAnalyzer">C# complexity analyzer</param>
    /// <param name="fsharpAnalyzer">F# complexity analyzer</param>
    public CodeComplexityAnalyzerService(
        ILogger<CodeComplexityAnalyzerService> logger,
        CSharpComplexityAnalyzer csharpAnalyzer,
        FSharpComplexityAnalyzer fsharpAnalyzer)
    {
        _logger = logger;
        _csharpAnalyzer = csharpAnalyzer;
        _fsharpAnalyzer = fsharpAnalyzer;
        
        _analyzers = new Dictionary<string, ICodeComplexityAnalyzer>(StringComparer.OrdinalIgnoreCase)
        {
            ["C#"] = _csharpAnalyzer,
            ["CSharp"] = _csharpAnalyzer,
            ["cs"] = _csharpAnalyzer,
            ["F#"] = _fsharpAnalyzer,
            ["FSharp"] = _fsharpAnalyzer,
            ["fs"] = _fsharpAnalyzer
        };
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeCyclomaticComplexityAsync(string filePath, string language)
    {
        try
        {
            if (string.IsNullOrEmpty(language))
            {
                // Try to determine language from file extension
                language = GetLanguageFromFilePath(filePath);
            }
            
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.AnalyzeCyclomaticComplexityAsync(filePath, language);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new List<ComplexityMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing cyclomatic complexity for file {FilePath}", filePath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeCognitiveComplexityAsync(string filePath, string language)
    {
        try
        {
            if (string.IsNullOrEmpty(language))
            {
                // Try to determine language from file extension
                language = GetLanguageFromFilePath(filePath);
            }
            
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.AnalyzeCognitiveComplexityAsync(filePath, language);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new List<ComplexityMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing cognitive complexity for file {FilePath}", filePath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language)
    {
        try
        {
            if (string.IsNullOrEmpty(language))
            {
                // Try to determine language from file extension
                language = GetLanguageFromFilePath(filePath);
            }
            
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.AnalyzeMaintainabilityIndexAsync(filePath, language);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new List<ComplexityMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing maintainability index for file {FilePath}", filePath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language)
    {
        try
        {
            if (string.IsNullOrEmpty(language))
            {
                // Try to determine language from file extension
                language = GetLanguageFromFilePath(filePath);
            }
            
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.AnalyzeHalsteadComplexityAsync(filePath, language);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new List<ComplexityMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing Halstead complexity for file {FilePath}", filePath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeAllComplexityMetricsAsync(string filePath, string language)
    {
        try
        {
            if (string.IsNullOrEmpty(language))
            {
                // Try to determine language from file extension
                language = GetLanguageFromFilePath(filePath);
            }
            
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.AnalyzeAllComplexityMetricsAsync(filePath, language);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new List<ComplexityMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing all complexity metrics for file {FilePath}", filePath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeProjectComplexityAsync(string projectPath)
    {
        try
        {
            var metrics = new List<ComplexityMetric>();
            
            // Analyze C# files
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);
            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await _csharpAnalyzer.AnalyzeCyclomaticComplexityAsync(file, "C#"));
            }
            
            // Analyze F# files
            var fsharpFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories);
            foreach (var file in fsharpFiles)
            {
                metrics.AddRange(await _fsharpAnalyzer.AnalyzeCyclomaticComplexityAsync(file, "F#"));
            }
            
            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);
            
            var projectCyclomaticComplexity = metrics
                .Where(m => m.Type == ComplexityType.Cyclomatic && m.TargetType == TargetType.File)
                .Sum(m => m.Value);
            
            var projectMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {projectName}",
                Description = $"McCabe's cyclomatic complexity for project {projectName}",
                Value = projectCyclomaticComplexity,
                Type = ComplexityType.Cyclomatic,
                FilePath = projectPath,
                Language = "Mixed",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };
            
            metrics.Add(projectMetric);
            
            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project complexity for {ProjectPath}", projectPath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, ComplexityType complexityType)
    {
        try
        {
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.GetComplexityThresholdsAsync(language, complexityType);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new Dictionary<string, double>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting complexity thresholds for {Language}, {ComplexityType}", 
                language, complexityType);
            return new Dictionary<string, double>();
        }
    }

    /// <inheritdoc/>
    public async Task<bool> SetComplexityThresholdAsync(string language, ComplexityType complexityType, string targetType, double threshold)
    {
        try
        {
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.SetComplexityThresholdAsync(language, complexityType, targetType, threshold);
            }
            
            _logger.LogWarning("No analyzer found for language {Language}", language);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting complexity threshold for {Language}, {ComplexityType}, {TargetType}", 
                language, complexityType, targetType);
            return false;
        }
    }
    
    /// <summary>
    /// Gets the programming language from a file path based on its extension
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <returns>Programming language</returns>
    private string GetLanguageFromFilePath(string filePath)
    {
        var extension = Path.GetExtension(filePath)?.ToLowerInvariant();
        
        return extension switch
        {
            ".cs" => "C#",
            ".fs" => "F#",
            _ => "Unknown"
        };
    }
}
