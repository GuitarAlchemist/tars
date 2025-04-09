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
    private readonly IReadabilityAnalyzer _readabilityAnalyzer;
    private readonly Dictionary<string, ICodeComplexityAnalyzer> _analyzers;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeComplexityAnalyzerService"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="csharpAnalyzer">C# complexity analyzer</param>
    /// <param name="fsharpAnalyzer">F# complexity analyzer</param>
    /// <param name="readabilityAnalyzer">Readability analyzer</param>
    public CodeComplexityAnalyzerService(
        ILogger<CodeComplexityAnalyzerService> logger,
        CSharpComplexityAnalyzer csharpAnalyzer,
        FSharpComplexityAnalyzer fsharpAnalyzer,
        IReadabilityAnalyzer readabilityAnalyzer)
    {
        _logger = logger;
        _csharpAnalyzer = csharpAnalyzer;
        _fsharpAnalyzer = fsharpAnalyzer;
        _readabilityAnalyzer = readabilityAnalyzer;

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
    public async Task<List<MaintainabilityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language)
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
            return new List<MaintainabilityMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing maintainability index for file {FilePath}", filePath);
            return new List<MaintainabilityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<HalsteadMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language)
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
            return new List<HalsteadMetric>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing Halstead complexity for file {FilePath}", filePath);
            return new List<HalsteadMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeReadabilityAsync(string filePath, string language, ReadabilityType readabilityType)
    {
        try
        {
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                language = GetLanguageFromFilePath(filePath);
            }

            // Use the readability analyzer
            return readabilityType switch
            {
                ReadabilityType.IdentifierQuality => await _readabilityAnalyzer.AnalyzeIdentifierQualityAsync(filePath, language),
                ReadabilityType.CommentQuality => await _readabilityAnalyzer.AnalyzeCommentQualityAsync(filePath, language),
                ReadabilityType.CodeStructure => await _readabilityAnalyzer.AnalyzeCodeStructureAsync(filePath, language),
                ReadabilityType.Overall => await _readabilityAnalyzer.AnalyzeOverallReadabilityAsync(filePath, language),
                _ => await _readabilityAnalyzer.AnalyzeAllReadabilityMetricsAsync(filePath, language)
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing readability for file {FilePath}", filePath);
            return new List<ReadabilityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeAllComplexityMetricsAsync(string filePath, string language)
    {
        try
        {
            if (string.IsNullOrEmpty(language))
            {
                // Try to determine language from file extension
                language = GetLanguageFromFilePath(filePath);
            }

            var complexityMetrics = new List<ComplexityMetric>();
            var halsteadMetrics = new List<HalsteadMetric>();
            var maintainabilityMetrics = new List<MaintainabilityMetric>();
            var readabilityMetrics = new List<ReadabilityMetric>();

            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                var metrics = await analyzer.AnalyzeAllComplexityMetricsAsync(filePath, language);
                complexityMetrics = metrics.ComplexityMetrics;
                halsteadMetrics = metrics.HalsteadMetrics;
                maintainabilityMetrics = metrics.MaintainabilityMetrics;
            }

            // Get readability metrics
            readabilityMetrics = await _readabilityAnalyzer.AnalyzeAllReadabilityMetricsAsync(filePath, language);

            return (complexityMetrics, halsteadMetrics, maintainabilityMetrics, readabilityMetrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing all complexity metrics for file {FilePath}", filePath);
            return (new List<ComplexityMetric>(), new List<HalsteadMetric>(), new List<MaintainabilityMetric>(), new List<ReadabilityMetric>());
        }
    }

    /// <inheritdoc/>
    public async Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics)> AnalyzeProjectComplexityAsync(string projectPath)
    {
        try
        {
            var complexityMetrics = new List<ComplexityMetric>();
            var halsteadMetrics = new List<HalsteadMetric>();
            var maintainabilityMetrics = new List<MaintainabilityMetric>();

            // Analyze C# files
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);
            foreach (var file in csharpFiles)
            {
                var fileMetrics = await AnalyzeAllComplexityMetricsAsync(file, "C#");
                complexityMetrics.AddRange(fileMetrics.ComplexityMetrics);
                halsteadMetrics.AddRange(fileMetrics.HalsteadMetrics);
                maintainabilityMetrics.AddRange(fileMetrics.MaintainabilityMetrics);

            }

            // Analyze F# files
            var fsharpFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories);
            foreach (var file in fsharpFiles)
            {
                var fileMetrics = await AnalyzeAllComplexityMetricsAsync(file, "F#");
                complexityMetrics.AddRange(fileMetrics.ComplexityMetrics);
                halsteadMetrics.AddRange(fileMetrics.HalsteadMetrics);
                maintainabilityMetrics.AddRange(fileMetrics.MaintainabilityMetrics);

            }

            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);

            // Calculate project-level cyclomatic complexity
            var projectCyclomaticComplexity = complexityMetrics
                .Where(m => m.Type == ComplexityType.Cyclomatic && m.TargetType == TargetType.File)
                .Sum(m => m.Value);

            var cyclomaticMetric = new ComplexityMetric
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

            complexityMetrics.Add(cyclomaticMetric);

            // Calculate project-level Halstead volume
            var projectHalsteadVolume = halsteadMetrics
                .Where(m => m.Type == HalsteadType.Volume && m.TargetType == TargetType.File)
                .Sum(m => m.Value);

            var halsteadMetric = new HalsteadMetric
            {
                Name = $"Halstead Volume - {projectName}",
                Description = $"Halstead volume for project {projectName}",
                Value = projectHalsteadVolume,
                Type = HalsteadType.Volume,
                FilePath = projectPath,
                Language = "Mixed",
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
                    Description = $"Maintainability index for project {projectName}",
                    Value = averageMaintainabilityIndex,
                    HalsteadVolume = projectHalsteadVolume,
                    CyclomaticComplexity = projectCyclomaticComplexity,
                    LinesOfCode = fileMaintenanceIndices.Sum(m => m.LinesOfCode),
                    CommentPercentage = fileMaintenanceIndices.Average(m => m.CommentPercentage),
                    FilePath = projectPath,
                    Language = "Mixed",
                    Target = projectName,
                    TargetType = TargetType.Project,
                    Timestamp = DateTime.UtcNow,
                    UseMicrosoftFormula = true
                };

                maintainabilityMetrics.Add(maintainabilityMetric);
            }

            return (complexityMetrics, halsteadMetrics, maintainabilityMetrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project complexity for {ProjectPath}", projectPath);
            return (new List<ComplexityMetric>(), new List<HalsteadMetric>(), new List<MaintainabilityMetric>());
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, TarsEngine.Services.Interfaces.ComplexityType complexityType)
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
    public async Task<Dictionary<string, double>> GetHalsteadThresholdsAsync(string language, HalsteadType halsteadType)
    {
        try
        {
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.GetHalsteadThresholdsAsync(language, halsteadType);
            }

            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new Dictionary<string, double>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting Halstead thresholds for {Language}, {HalsteadType}",
                language, halsteadType);
            return new Dictionary<string, double>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, double>> GetMaintainabilityThresholdsAsync(string language)
    {
        try
        {
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.GetMaintainabilityThresholdsAsync(language);
            }

            _logger.LogWarning("No analyzer found for language {Language}", language);
            return new Dictionary<string, double>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting maintainability thresholds for {Language}", language);
            return new Dictionary<string, double>();
        }
    }

    /// <inheritdoc/>
    public async Task<bool> SetComplexityThresholdAsync(string language, TarsEngine.Services.Interfaces.ComplexityType complexityType, string targetType, double threshold)
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

    /// <inheritdoc/>
    public async Task<bool> SetHalsteadThresholdAsync(string language, HalsteadType halsteadType, string targetType, double threshold)
    {
        try
        {
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.SetHalsteadThresholdAsync(language, halsteadType, targetType, threshold);
            }

            _logger.LogWarning("No analyzer found for language {Language}", language);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting Halstead threshold for {Language}, {HalsteadType}, {TargetType}",
                language, halsteadType, targetType);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> SetMaintainabilityThresholdAsync(string language, string targetType, double threshold)
    {
        try
        {
            if (_analyzers.TryGetValue(language, out var analyzer))
            {
                return await analyzer.SetMaintainabilityThresholdAsync(language, targetType, threshold);
            }

            _logger.LogWarning("No analyzer found for language {Language}", language);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting maintainability threshold for {Language}, {TargetType}",
                language, targetType);
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
