using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Implementation of the readability analyzer
/// </summary>
public class ReadabilityAnalyzer : IReadabilityAnalyzer
{
    private readonly ILogger<ReadabilityAnalyzer> _logger;
    private readonly IFileService _fileService;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReadabilityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="fileService">File service</param>
    public ReadabilityAnalyzer(
        ILogger<ReadabilityAnalyzer> logger,
        IFileService fileService)
    {
        _logger = logger;
        _fileService = fileService;
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeIdentifierQualityAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation($"Analyzing identifier quality for {filePath}");

            var content = await _fileService.ReadFileAsync(filePath);

            var metrics = new List<ReadabilityMetric>();

            // Create a metric for the file
            var fileMetric = new ReadabilityMetric
            {
                Type = ReadabilityType.IdentifierQuality,
                FilePath = filePath,
                Language = language,
                Target = Path.GetFileName(filePath),
                TargetType = TargetType.File,
                Description = $"Identifier quality for {Path.GetFileName(filePath)}",
                AverageIdentifierLength = 5, // Default value
                PoorlyNamedIdentifierCount = 0 // Default value
            };

            // Analyze identifiers in the file
            var identifiers = ExtractIdentifiers(content, language);

            if (identifiers.Any())
            {
                fileMetric.AverageIdentifierLength = identifiers.Average(i => i.Length);
                fileMetric.PoorlyNamedIdentifierCount = identifiers.Count(i => i.Length < 3 || i.Length > 30);
            }

            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing identifier quality for {filePath}");
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeCommentQualityAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation($"Analyzing comment quality for {filePath}");

            var content = await _fileService.ReadFileAsync(filePath);
            var lines = content.Split('\n');

            var metrics = new List<ReadabilityMetric>();

            // Create a metric for the file
            var fileMetric = new ReadabilityMetric
            {
                Type = ReadabilityType.CommentQuality,
                FilePath = filePath,
                Language = language,
                Target = Path.GetFileName(filePath),
                TargetType = TargetType.File,
                Description = $"Comment quality for {Path.GetFileName(filePath)}",
                LinesOfCode = lines.Length,
                CommentPercentage = 0 // Default value
            };

            // Count comment lines
            var commentLines = CountCommentLines(content, language);

            if (lines.Length > 0)
            {
                fileMetric.CommentPercentage = (double)commentLines / lines.Length * 100;
            }

            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing comment quality for {filePath}");
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeCodeStructureAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation($"Analyzing code structure for {filePath}");

            var content = await _fileService.ReadFileAsync(filePath);
            var lines = content.Split('\n');

            var metrics = new List<ReadabilityMetric>();

            // Create a metric for the file
            var fileMetric = new ReadabilityMetric
            {
                Type = ReadabilityType.CodeStructure,
                FilePath = filePath,
                Language = language,
                Target = Path.GetFileName(filePath),
                TargetType = TargetType.File,
                Description = $"Code structure for {Path.GetFileName(filePath)}",
                LinesOfCode = lines.Length,
                MaxNestingDepth = 0, // Default value
                AverageNestingDepth = 0, // Default value
                LongMethodCount = 0, // Default value
                LongLineCount = 0, // Default value
                ComplexExpressionCount = 0, // Default value
                MagicNumberCount = 0 // Default value
            };

            // Analyze code structure
            fileMetric.MaxNestingDepth = CalculateMaxNestingDepth(content, language);
            fileMetric.AverageNestingDepth = CalculateAverageNestingDepth(content, language);
            fileMetric.LongMethodCount = CountLongMethods(content, language);
            fileMetric.LongLineCount = lines.Count(l => l.Length > 100);
            fileMetric.ComplexExpressionCount = CountComplexExpressions(content, language);
            fileMetric.MagicNumberCount = CountMagicNumbers(content, language);

            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing code structure for {filePath}");
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeOverallReadabilityAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation($"Analyzing overall readability for {filePath}");

            var identifierMetrics = await AnalyzeIdentifierQualityAsync(filePath, language);
            var commentMetrics = await AnalyzeCommentQualityAsync(filePath, language);
            var structureMetrics = await AnalyzeCodeStructureAsync(filePath, language);

            var metrics = new List<ReadabilityMetric>();

            // Create a metric for the file
            var fileMetric = new ReadabilityMetric
            {
                Type = ReadabilityType.Overall,
                FilePath = filePath,
                Language = language,
                Target = Path.GetFileName(filePath),
                TargetType = TargetType.File,
                Description = $"Overall readability for {Path.GetFileName(filePath)}"
            };

            // Copy values from other metrics
            if (identifierMetrics.Any())
            {
                var identifierMetric = identifierMetrics.First();
                fileMetric.AverageIdentifierLength = identifierMetric.AverageIdentifierLength;
                fileMetric.PoorlyNamedIdentifierCount = identifierMetric.PoorlyNamedIdentifierCount;
            }

            if (commentMetrics.Any())
            {
                var commentMetric = commentMetrics.First();
                fileMetric.LinesOfCode = commentMetric.LinesOfCode;
                fileMetric.CommentPercentage = commentMetric.CommentPercentage;
            }

            if (structureMetrics.Any())
            {
                var structureMetric = structureMetrics.First();
                fileMetric.MaxNestingDepth = structureMetric.MaxNestingDepth;
                fileMetric.AverageNestingDepth = structureMetric.AverageNestingDepth;
                fileMetric.LongMethodCount = structureMetric.LongMethodCount;
                fileMetric.LongLineCount = structureMetric.LongLineCount;
                fileMetric.ComplexExpressionCount = structureMetric.ComplexExpressionCount;
                fileMetric.MagicNumberCount = structureMetric.MagicNumberCount;
            }

            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing overall readability for {filePath}");
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeAllReadabilityMetricsAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation($"Analyzing all readability metrics for {filePath}");

            var identifierMetrics = await AnalyzeIdentifierQualityAsync(filePath, language);
            var commentMetrics = await AnalyzeCommentQualityAsync(filePath, language);
            var structureMetrics = await AnalyzeCodeStructureAsync(filePath, language);
            var overallMetrics = await AnalyzeOverallReadabilityAsync(filePath, language);

            var metrics = new List<ReadabilityMetric>();
            metrics.AddRange(identifierMetrics);
            metrics.AddRange(commentMetrics);
            metrics.AddRange(structureMetrics);
            metrics.AddRange(overallMetrics);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing all readability metrics for {filePath}");
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeProjectReadabilityAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation($"Analyzing project readability for {projectPath}");

            var metrics = new List<ReadabilityMetric>();

            // Get all source files in the project
            var files = await _fileService.GetFilesAsync(projectPath, "*.cs;*.fs;*.vb;*.js;*.ts;*.py;*.java");

            foreach (var file in files)
            {
                var language = GetLanguageFromExtension(Path.GetExtension(file));
                var fileMetrics = await AnalyzeOverallReadabilityAsync(file, language);
                metrics.AddRange(fileMetrics);
            }

            // Create a metric for the project
            var projectMetric = new ReadabilityMetric
            {
                Type = ReadabilityType.Overall,
                FilePath = projectPath,
                Language = "Multiple",
                Target = Path.GetFileName(projectPath),
                TargetType = TargetType.Project,
                Description = $"Overall readability for {Path.GetFileName(projectPath)}"
            };

            // Calculate average values from file metrics
            if (metrics.Any())
            {
                projectMetric.AverageIdentifierLength = metrics.Average(m => m.AverageIdentifierLength);
                projectMetric.CommentPercentage = metrics.Average(m => m.CommentPercentage);
                projectMetric.AverageNestingDepth = metrics.Average(m => m.AverageNestingDepth);
                projectMetric.MaxNestingDepth = metrics.Max(m => m.MaxNestingDepth);
                projectMetric.LongMethodCount = metrics.Sum(m => m.LongMethodCount);
                projectMetric.LongLineCount = metrics.Sum(m => m.LongLineCount);
                projectMetric.ComplexExpressionCount = metrics.Sum(m => m.ComplexExpressionCount);
                projectMetric.MagicNumberCount = metrics.Sum(m => m.MagicNumberCount);
                projectMetric.PoorlyNamedIdentifierCount = metrics.Sum(m => m.PoorlyNamedIdentifierCount);
                projectMetric.LinesOfCode = metrics.Sum(m => m.LinesOfCode);
            }

            metrics.Add(projectMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing project readability for {projectPath}");
            return [];
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetReadabilityThresholdsAsync(string language, ReadabilityType readabilityType)
    {
        // Default thresholds
        var thresholds = new Dictionary<string, double>
        {
            { "Method", 60 },
            { "Class", 70 },
            { "File", 75 },
            { "Project", 80 }
        };

        return Task.FromResult(thresholds);
    }

    /// <inheritdoc/>
    public Task<bool> SetReadabilityThresholdAsync(string language, ReadabilityType readabilityType, string targetType, double threshold)
    {
        // In a real implementation, this would save the threshold to a configuration file or database
        _logger.LogInformation($"Setting readability threshold for {language}, {readabilityType}, {targetType} to {threshold}");
        return Task.FromResult(true);
    }

    /// <summary>
    /// Extracts identifiers from code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>List of identifiers</returns>
    private List<string> ExtractIdentifiers(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to extract identifiers
        return ["example", "identifier"];
    }

    /// <summary>
    /// Counts comment lines in code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>Number of comment lines</returns>
    private int CountCommentLines(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to count comment lines
        return 0;
    }

    /// <summary>
    /// Calculates maximum nesting depth in code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>Maximum nesting depth</returns>
    private int CalculateMaxNestingDepth(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to calculate nesting depth
        return 0;
    }

    /// <summary>
    /// Calculates average nesting depth in code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>Average nesting depth</returns>
    private double CalculateAverageNestingDepth(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to calculate nesting depth
        return 0;
    }

    /// <summary>
    /// Counts long methods in code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>Number of long methods</returns>
    private int CountLongMethods(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to count long methods
        return 0;
    }

    /// <summary>
    /// Counts complex expressions in code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>Number of complex expressions</returns>
    private int CountComplexExpressions(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to count complex expressions
        return 0;
    }

    /// <summary>
    /// Counts magic numbers in code
    /// </summary>
    /// <param name="content">Code content</param>
    /// <param name="language">Programming language</param>
    /// <returns>Number of magic numbers</returns>
    private int CountMagicNumbers(string content, string language)
    {
        // This is a simplified implementation
        // In a real implementation, this would use a parser to count magic numbers
        return 0;
    }

    /// <summary>
    /// Gets language from file extension
    /// </summary>
    /// <param name="extension">File extension</param>
    /// <returns>Language name</returns>
    private string GetLanguageFromExtension(string extension)
    {
        return extension.ToLowerInvariant() switch
        {
            ".cs" => "C#",
            ".fs" => "F#",
            ".vb" => "VB.NET",
            ".js" => "JavaScript",
            ".ts" => "TypeScript",
            ".py" => "Python",
            ".java" => "Java",
            _ => "Unknown"
        };
    }
}
