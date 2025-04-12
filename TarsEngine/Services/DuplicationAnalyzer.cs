using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;
using System.Text;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for code duplication
/// </summary>
public class DuplicationAnalyzer : IDuplicationAnalyzer
{
    private readonly ILogger<DuplicationAnalyzer> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="DuplicationAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public DuplicationAnalyzer(ILogger<DuplicationAnalyzer> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public Task<List<DuplicationMetric>> AnalyzeTokenBasedDuplicationAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Analyzing token-based duplication for {FilePath}", filePath);

            // This is a demo implementation
            var metrics = new List<DuplicationMetric>
            {
                new()
                {
                    Name = "Token-Based Duplication",
                    Description = "Token-based duplication analysis",
                    Type = DuplicationType.TokenBased,
                    Target = Path.GetFileName(filePath),
                    TargetType = "File",
                    FilePath = filePath,
                    Language = language,
                    TotalLinesOfCode = 100,
                    DuplicatedLinesOfCode = 15,
                    DuplicationPercentage = 15,
                    DuplicatedBlockCount = 2,
                    // DuplicationLevel is calculated from DuplicationPercentage
                    Timestamp = DateTime.UtcNow,
                    DuplicatedBlocks =
                    [
                        new TarsEngine.Models.Metrics.DuplicatedBlock
                        {
                            SourceStartLine = 10,
                            SourceEndLine = 15,
                            TargetStartLine = 50,
                            TargetEndLine = 55,
                            // DuplicatedLines is calculated from SourceStartLine and SourceEndLine
                            SimilarityPercentage = 100,
                            DuplicatedCode =
                                "// Sample duplicated code\nvar x = 10;\nvar y = 20;\nvar z = x + y;\nConsole.WriteLine(z);"
                        },

                        new TarsEngine.Models.Metrics.DuplicatedBlock
                        {
                            SourceStartLine = 25,
                            SourceEndLine = 33,
                            TargetStartLine = 75,
                            TargetEndLine = 83,
                            // DuplicatedLines is calculated from SourceStartLine and SourceEndLine
                            SimilarityPercentage = 95,
                            DuplicatedCode =
                                "// Another sample\nfor (int i = 0; i < 10; i++)\n{\n    Console.WriteLine(i);\n}"
                        }
                    ]
                }
            };

            return Task.FromResult(metrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing token-based duplication for {FilePath}", filePath);
            return Task.FromResult(new List<DuplicationMetric>());
        }
    }

    /// <inheritdoc/>
    public Task<List<DuplicationMetric>> AnalyzeSemanticDuplicationAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Analyzing semantic duplication for {FilePath}", filePath);

            // This is a demo implementation
            var metrics = new List<DuplicationMetric>
            {
                new()
                {
                    Name = "Semantic Duplication",
                    Description = "Semantic duplication analysis",
                    Type = DuplicationType.Semantic,
                    Target = Path.GetFileName(filePath),
                    TargetType = "File",
                    FilePath = filePath,
                    Language = language,
                    TotalLinesOfCode = 100,
                    DuplicatedLinesOfCode = 10,
                    DuplicationPercentage = 10,
                    DuplicatedBlockCount = 1,
                    // DuplicationLevel is calculated from DuplicationPercentage
                    Timestamp = DateTime.UtcNow,
                    DuplicatedBlocks =
                    [
                        new TarsEngine.Models.Metrics.DuplicatedBlock
                        {
                            SourceStartLine = 40,
                            SourceEndLine = 49,
                            TargetStartLine = 60,
                            TargetEndLine = 69,
                            // DuplicatedLines is calculated from SourceStartLine and SourceEndLine
                            SimilarityPercentage = 85,
                            DuplicatedCode =
                                "// Semantically similar code\nvar result = 0;\nfor (int i = 0; i < items.Length; i++)\n{\n    result += items[i];\n}"
                        }
                    ]
                }
            };

            return Task.FromResult(metrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing semantic duplication for {FilePath}", filePath);
            return Task.FromResult(new List<DuplicationMetric>());
        }
    }

    /// <inheritdoc/>
    public Task<List<DuplicationMetric>> AnalyzeAllDuplicationMetricsAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Analyzing all duplication metrics for {FilePath}", filePath);

            // Combine token-based and semantic duplication metrics
            var tokenBasedMetrics = AnalyzeTokenBasedDuplicationAsync(filePath, language).Result;
            var semanticMetrics = AnalyzeSemanticDuplicationAsync(filePath, language).Result;

            var allMetrics = new List<DuplicationMetric>();
            allMetrics.AddRange(tokenBasedMetrics);
            allMetrics.AddRange(semanticMetrics);

            return Task.FromResult(allMetrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing all duplication metrics for {FilePath}", filePath);
            return Task.FromResult(new List<DuplicationMetric>());
        }
    }

    /// <inheritdoc/>
    public Task<List<DuplicationMetric>> AnalyzeProjectDuplicationAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Analyzing project duplication for {ProjectPath}", projectPath);

            // This is a demo implementation
            var metrics = new List<DuplicationMetric>
            {
                new()
                {
                    Name = "Project Duplication",
                    Description = "Project-level duplication analysis",
                    Type = DuplicationType.TokenBased,
                    Target = Path.GetFileName(projectPath),
                    TargetType = "Project",
                    FilePath = projectPath,
                    Language = "C#",
                    TotalLinesOfCode = 1000,
                    DuplicatedLinesOfCode = 150,
                    DuplicationPercentage = 8, // This will result in Moderate DuplicationLevel
                    DuplicatedBlockCount = 12,
                    Timestamp = DateTime.UtcNow
                },
                new()
                {
                    Name = "Project Semantic Duplication",
                    Description = "Project-level semantic duplication analysis",
                    Type = DuplicationType.Semantic,
                    Target = Path.GetFileName(projectPath),
                    TargetType = "Project",
                    FilePath = projectPath,
                    Language = "C#",
                    TotalLinesOfCode = 1000,
                    DuplicatedLinesOfCode = 100,
                    DuplicationPercentage = 15, // This will result in High DuplicationLevel
                    DuplicatedBlockCount = 8,
                    Timestamp = DateTime.UtcNow
                }
            };

            return Task.FromResult(metrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project duplication for {ProjectPath}", projectPath);
            return Task.FromResult(new List<DuplicationMetric>());
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetDuplicationThresholdsAsync(string language, DuplicationType duplicationType)
    {
        try
        {
            _logger.LogInformation("Getting duplication thresholds for {Language}, {DuplicationType}", language, duplicationType);

            // Default thresholds
            var thresholds = new Dictionary<string, double>
            {
                ["Method"] = 5,
                ["Class"] = 10,
                ["File"] = 15,
                ["Project"] = 20
            };

            return Task.FromResult(thresholds);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting duplication thresholds for {Language}, {DuplicationType}", language, duplicationType);
            return Task.FromResult(new Dictionary<string, double>());
        }
    }

    /// <inheritdoc/>
    public Task<bool> SetDuplicationThresholdAsync(string language, DuplicationType duplicationType, string targetType, double threshold)
    {
        try
        {
            _logger.LogInformation("Setting duplication threshold for {Language}, {DuplicationType}, {TargetType} to {Threshold}",
                language, duplicationType, targetType, threshold);

            // In a real implementation, this would store the threshold in a configuration or database
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting duplication threshold for {Language}, {DuplicationType}, {TargetType}",
                language, duplicationType, targetType);
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> VisualizeDuplicationAsync(string path, string language, string outputPath)
    {
        try
        {
            _logger.LogInformation("Visualizing duplication for {Path}, {Language} to {OutputPath}", path, language, outputPath);

            // This is a demo implementation that creates a simple HTML visualization
            var metrics = AnalyzeAllDuplicationMetricsAsync(path, language).Result;

            var sb = new StringBuilder();
            sb.AppendLine("<!DOCTYPE html>");
            sb.AppendLine("<html>");
            sb.AppendLine("<head>");
            sb.AppendLine("  <title>Duplication Visualization</title>");
            sb.AppendLine("  <style>");
            sb.AppendLine("    body { font-family: Arial, sans-serif; margin: 20px; }");
            sb.AppendLine("    h1, h2 { color: #333; }");
            sb.AppendLine("    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }");
            sb.AppendLine("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }");
            sb.AppendLine("    th { background-color: #f2f2f2; }");
            sb.AppendLine("    tr:nth-child(even) { background-color: #f9f9f9; }");
            sb.AppendLine("    .low { color: green; }");
            sb.AppendLine("    .moderate { color: orange; }");
            sb.AppendLine("    .high { color: red; }");
            sb.AppendLine("    .veryhigh { color: darkred; }");
            sb.AppendLine("  </style>");
            sb.AppendLine("</head>");
            sb.AppendLine("<body>");
            sb.AppendLine("  <h1>Duplication Visualization</h1>");

            sb.AppendLine("  <h2>Duplication Metrics</h2>");
            sb.AppendLine("  <table>");
            sb.AppendLine("    <tr>");
            sb.AppendLine("      <th>Type</th>");
            sb.AppendLine("      <th>Target</th>");
            sb.AppendLine("      <th>Total Lines</th>");
            sb.AppendLine("      <th>Duplicated Lines</th>");
            sb.AppendLine("      <th>Duplication %</th>");
            sb.AppendLine("      <th>Level</th>");
            sb.AppendLine("    </tr>");

            foreach (var metric in metrics)
            {
                var levelClass = metric.DuplicationLevel.ToString().ToLowerInvariant();
                sb.AppendLine("    <tr>");
                sb.AppendLine($"      <td>{metric.Type}</td>");
                sb.AppendLine($"      <td>{metric.Target}</td>");
                sb.AppendLine($"      <td>{metric.TotalLinesOfCode}</td>");
                sb.AppendLine($"      <td>{metric.DuplicatedLinesOfCode}</td>");
                sb.AppendLine($"      <td>{metric.DuplicationPercentage:F2}%</td>");
                sb.AppendLine($"      <td class=\"{levelClass}\">{metric.DuplicationLevel}</td>");
                sb.AppendLine("    </tr>");
            }

            sb.AppendLine("  </table>");
            sb.AppendLine("</body>");
            sb.AppendLine("</html>");

            // Write the HTML to the output file
            File.WriteAllText(outputPath, sb.ToString());

            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error visualizing duplication for {Path}, {Language}", path, language);
            return Task.FromResult(false);
        }
    }
}
