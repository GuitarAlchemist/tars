using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code duplication
/// </summary>
public class DuplicationAnalyzerService : IDuplicationAnalyzer
{
    private readonly ILogger<DuplicationAnalyzerService> _logger;
    private readonly CSharpDuplicationAnalyzer _csharpAnalyzer;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="DuplicationAnalyzerService"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="csharpAnalyzer">C# duplication analyzer</param>
    public DuplicationAnalyzerService(
        ILogger<DuplicationAnalyzerService> logger,
        CSharpDuplicationAnalyzer csharpAnalyzer)
    {
        _logger = logger;
        _csharpAnalyzer = csharpAnalyzer;
    }
    
    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeTokenBasedDuplicationAsync(string filePath, string language)
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
                "C#" => await _csharpAnalyzer.AnalyzeTokenBasedDuplicationAsync(filePath, language),
                "F#" => new List<DuplicationMetric>(), // F# analyzer not implemented yet
                _ => new List<DuplicationMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing token-based duplication for file {FilePath}", filePath);
            return new List<DuplicationMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeSemanticDuplicationAsync(string filePath, string language)
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
                "C#" => await _csharpAnalyzer.AnalyzeSemanticDuplicationAsync(filePath, language),
                "F#" => new List<DuplicationMetric>(), // F# analyzer not implemented yet
                _ => new List<DuplicationMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing semantic duplication for file {FilePath}", filePath);
            return new List<DuplicationMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeAllDuplicationMetricsAsync(string filePath, string language)
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
                "C#" => await _csharpAnalyzer.AnalyzeAllDuplicationMetricsAsync(filePath, language),
                "F#" => new List<DuplicationMetric>(), // F# analyzer not implemented yet
                _ => new List<DuplicationMetric>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing all duplication metrics for file {FilePath}", filePath);
            return new List<DuplicationMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeProjectDuplicationAsync(string projectPath)
    {
        try
        {
            var metrics = new List<DuplicationMetric>();
            
            // Analyze C# files
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);
            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeAllDuplicationMetricsAsync(file, "C#"));
            }
            
            // Analyze F# files
            var fsharpFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories);
            foreach (var file in fsharpFiles)
            {
                metrics.AddRange(await AnalyzeAllDuplicationMetricsAsync(file, "F#"));
            }
            
            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project duplication for {ProjectPath}", projectPath);
            return new List<DuplicationMetric>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<Dictionary<string, double>> GetDuplicationThresholdsAsync(string language, DuplicationType duplicationType)
    {
        try
        {
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.GetDuplicationThresholdsAsync(language, duplicationType),
                "F#" => new Dictionary<string, double>(), // F# analyzer not implemented yet
                _ => new Dictionary<string, double>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting duplication thresholds for {Language}, {DuplicationType}",
                language, duplicationType);
            return new Dictionary<string, double>();
        }
    }
    
    /// <inheritdoc/>
    public async Task<bool> SetDuplicationThresholdAsync(string language, DuplicationType duplicationType, string targetType, double threshold)
    {
        try
        {
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.SetDuplicationThresholdAsync(language, duplicationType, targetType, threshold),
                "F#" => false, // F# analyzer not implemented yet
                _ => false
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting duplication threshold for {Language}, {DuplicationType}, {TargetType}",
                language, duplicationType, targetType);
            return false;
        }
    }
    
    /// <inheritdoc/>
    public async Task<bool> VisualizeDuplicationAsync(string path, string language, string outputPath)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(path);
            }
            
            // Use the appropriate analyzer based on language
            return language switch
            {
                "C#" => await _csharpAnalyzer.VisualizeDuplicationAsync(path, language, outputPath),
                "F#" => false, // F# analyzer not implemented yet
                _ => false
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error visualizing duplication for {Path}", path);
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
