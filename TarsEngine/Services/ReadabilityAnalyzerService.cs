using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code readability
/// </summary>
public class ReadabilityAnalyzerService : IReadabilityAnalyzer
{
    private readonly ILogger<ReadabilityAnalyzerService> _logger;
    private readonly CSharpReadabilityAnalyzer _csharpAnalyzer;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ReadabilityAnalyzerService"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="csharpAnalyzer">C# readability analyzer</param>
    public ReadabilityAnalyzerService(
        ILogger<ReadabilityAnalyzerService> logger,
        CSharpReadabilityAnalyzer csharpAnalyzer)
    {
        _logger = logger;
        _csharpAnalyzer = csharpAnalyzer;
    }
    
    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeIdentifierQualityAsync(string filePath, string language)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(filePath);
            }
            
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.AnalyzeIdentifierQualityAsync(filePath, language),
                "F#" => new List<ReadabilityMetric>(), // F# analyzer not implemented yet
                _ => new List<ReadabilityMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing identifier quality for file {FilePath}", filePath);
            return new List<ReadabilityMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeCommentQualityAsync(string filePath, string language)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(filePath);
            }
            
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.AnalyzeCommentQualityAsync(filePath, language),
                "F#" => new List<ReadabilityMetric>(), // F# analyzer not implemented yet
                _ => new List<ReadabilityMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing comment quality for file {FilePath}", filePath);
            return new List<ReadabilityMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeCodeStructureAsync(string filePath, string language)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(filePath);
            }
            
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.AnalyzeCodeStructureAsync(filePath, language),
                "F#" => new List<ReadabilityMetric>(), // F# analyzer not implemented yet
                _ => new List<ReadabilityMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code structure for file {FilePath}", filePath);
            return new List<ReadabilityMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeOverallReadabilityAsync(string filePath, string language)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(filePath);
            }
            
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.AnalyzeOverallReadabilityAsync(filePath, language),
                "F#" => new List<ReadabilityMetric>(), // F# analyzer not implemented yet
                _ => new List<ReadabilityMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing overall readability for file {FilePath}", filePath);
            return new List<ReadabilityMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeAllReadabilityMetricsAsync(string filePath, string language)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(filePath);
            }
            
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.AnalyzeAllReadabilityMetricsAsync(filePath, language),
                "F#" => new List<ReadabilityMetric>(), // F# analyzer not implemented yet
                _ => new List<ReadabilityMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing all readability metrics for file {FilePath}", filePath);
            return new List<ReadabilityMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeProjectReadabilityAsync(string projectPath)
    {
        try
        {
            var metrics = new List<ReadabilityMetric>();
            
            // Analyze C# files
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);
            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeAllReadabilityMetricsAsync(file, "C#"));
            }
            
            // Analyze F# files
            var fsharpFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories);
            foreach (var file in fsharpFiles)
            {
                metrics.AddRange(await AnalyzeAllReadabilityMetricsAsync(file, "F#"));
            }
            
            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project readability for {ProjectPath}", projectPath);
            return new List<ReadabilityMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<Dictionary<string, double>> GetReadabilityThresholdsAsync(string language, ReadabilityType readabilityType)
    {
        try
        {
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.GetReadabilityThresholdsAsync(language, readabilityType),
                "F#" => new Dictionary<string, double>(), // F# analyzer not implemented yet
                _ => new Dictionary<string, double>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting readability thresholds for {Language}, {ReadabilityType}",
                language, readabilityType);
            return new Dictionary<string, double>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<bool> SetReadabilityThresholdAsync(string language, ReadabilityType readabilityType, string targetType, double threshold)
    {
        try
        {
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.SetReadabilityThresholdAsync(language, readabilityType, targetType, threshold),
                "F#" => false, // F# analyzer not implemented yet
                _ => false
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting readability threshold for {Language}, {ReadabilityType}, {TargetType}",
                language, readabilityType, targetType);
            return false;
        }
    }
    
    /// <summary>
    /// Gets the programming language from a file path
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <returns>Programming language</returns>
    private string GetLanguageFromFilePath(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        
        return extension switch
        {
            ".cs" => "C#",
            ".fs" => "F#",
            _ => string.Empty
        };
    }
}
