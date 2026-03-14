using System.Text;
using DuplicationAnalyzerTests.Models;
using DuplicationAnalyzerTests.Monads;
using DuplicationAnalyzerTests.Services;
using DuplicationAnalyzerTests.Services.Interfaces;
using Microsoft.Extensions.Logging;
using Moq;

namespace DuplicationAnalyzerTests;

/// <summary>
/// Demo command for duplication detection
/// </summary>
public class DuplicationDemoCommand
{
    private readonly IDuplicationAnalyzer _duplicationAnalyzer;

    /// <summary>
    /// Initializes a new instance of the <see cref="DuplicationDemoCommand"/> class
    /// </summary>
    public DuplicationDemoCommand()
    {
        var mockLogger = new Mock<ILogger<CSharpDuplicationAnalyzer>>();
        _duplicationAnalyzer = new CSharpDuplicationAnalyzer(mockLogger.Object);
    }

    /// <summary>
    /// Runs the demo
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="language">Programming language</param>
    /// <param name="type">Duplication type</param>
    /// <param name="output">Output format</param>
    /// <param name="outputPath">Path to save the output file</param>
    /// <returns>Result of the demo</returns>
    public async Task<string> RunAsync(string filePath, string language = "C#", string type = "all", string output = "console", string outputPath = "")
    {
        Console.WriteLine($"Analyzing duplication in {filePath}");

        // Validate the file path
        var pathValidationResult = await Result.TryAsync(() => {
            var isDirectory = Directory.Exists(filePath);
            var isFile = File.Exists(filePath);

            if (!isDirectory && !isFile)
            {
                throw new FileNotFoundException($"Path {filePath} does not exist");
            }

            return Task.FromResult((isFile, isDirectory));
        });

        if (!pathValidationResult.IsSuccess)
        {
            return pathValidationResult.Error;
        }

        // Analyze duplication
        var (isFile, isDirectory) = pathValidationResult.Value;
        var metricsResult = await Result.TryAsync(async () => {
            if (isFile)
            {
                return await AnalyzeFile(filePath, language, type);
            }
            else
            {
                return await _duplicationAnalyzer.AnalyzeProjectDuplicationAsync(filePath);
            }
        });

        if (!metricsResult.IsSuccess)
        {
            return $"Error analyzing duplication: {metricsResult.Error}";
        }

        // Format results
        var formattingResult = Result.Try(() => FormatResults(metricsResult.Value, output));

        if (!formattingResult.IsSuccess)
        {
            return $"Error formatting results: {formattingResult.Error}";
        }

        // Save results if output path is specified
        if (!string.IsNullOrEmpty(outputPath))
        {
            var saveResult = await Result.TryAsync(async () => {
                await File.WriteAllTextAsync(outputPath, formattingResult.Value);
                return $"Results saved to {outputPath}";
            });

            return saveResult.IsSuccess ? saveResult.Value : $"Error saving results: {saveResult.Error}";
        }

        return formattingResult.Value;
    }

    private async Task<List<DuplicationMetric>> AnalyzeFile(string filePath, string language, string type)
    {
        return type.ToLowerInvariant() switch
        {
            "token" => await _duplicationAnalyzer.AnalyzeTokenBasedDuplicationAsync(filePath, language),
            "semantic" => await _duplicationAnalyzer.AnalyzeSemanticDuplicationAsync(filePath, language),
            _ => await _duplicationAnalyzer.AnalyzeAllDuplicationMetricsAsync(filePath, language)
        };
    }

    private string FormatResults(List<DuplicationMetric> metrics, string format)
    {
        return format.ToLowerInvariant() switch
        {
            "json" => FormatAsJson(metrics),
            "csv" => FormatAsCsv(metrics),
            "html" => FormatAsHtml(metrics),
            _ => FormatAsConsole(metrics)
        };
    }

    private string FormatAsConsole(List<DuplicationMetric> metrics)
    {
        var sb = new StringBuilder();

        sb.AppendLine("=== Duplication Analysis Results ===");
        sb.AppendLine();

        // Group metrics by type
        var tokenBasedMetrics = metrics.Where(m => m.Type == DuplicationType.TokenBased).ToList();
        var semanticMetrics = metrics.Where(m => m.Type == DuplicationType.Semantic).ToList();

        if (tokenBasedMetrics.Any())
        {
            sb.AppendLine("=== Token-Based Duplication ===");
            sb.AppendLine();

            foreach (var metric in tokenBasedMetrics.OrderByDescending(m => m.DuplicationPercentage))
            {
                sb.AppendLine($"Target: {metric.Target} ({metric.TargetType})");
                sb.AppendLine($"Duplication: {metric.DuplicationPercentage:F2}% ({metric.DuplicatedLinesOfCode} of {metric.TotalLinesOfCode} lines)");
                sb.AppendLine($"Duplicated Blocks: {metric.DuplicatedBlockCount}");
                sb.AppendLine($"Duplication Level: {metric.DuplicationLevel}");
                sb.AppendLine();

                if (metric.DuplicatedBlocks.Any())
                {
                    sb.AppendLine("Top Duplicated Blocks:");
                    foreach (var block in metric.DuplicatedBlocks.OrderByDescending(b => b.DuplicatedLines).Take(3))
                    {
                        sb.AppendLine($"  - Lines {block.SourceStartLine}-{block.SourceEndLine} duplicated at lines {block.TargetStartLine}-{block.TargetEndLine}");
                        sb.AppendLine($"    Similarity: {block.SimilarityPercentage:F2}%");
                        sb.AppendLine($"    Lines: {block.DuplicatedLines}");
                        sb.AppendLine();
                    }
                }

                sb.AppendLine(new string('-', 50));
            }
        }

        if (semanticMetrics.Any())
        {
            sb.AppendLine("=== Semantic Duplication ===");
            sb.AppendLine();

            foreach (var metric in semanticMetrics.OrderByDescending(m => m.DuplicationPercentage))
            {
                sb.AppendLine($"Target: {metric.Target} ({metric.TargetType})");
                sb.AppendLine($"Duplication: {metric.DuplicationPercentage:F2}% ({metric.DuplicatedLinesOfCode} of {metric.TotalLinesOfCode} lines)");
                sb.AppendLine($"Duplicated Blocks: {metric.DuplicatedBlockCount}");
                sb.AppendLine($"Duplication Level: {metric.DuplicationLevel}");
                sb.AppendLine();

                if (metric.DuplicatedBlocks.Any())
                {
                    sb.AppendLine("Top Semantically Similar Blocks:");
                    foreach (var block in metric.DuplicatedBlocks.OrderByDescending(b => b.SimilarityPercentage).Take(3))
                    {
                        sb.AppendLine($"  - Lines {block.SourceStartLine}-{block.SourceEndLine} similar to lines {block.TargetStartLine}-{block.TargetEndLine}");
                        sb.AppendLine($"    Similarity: {block.SimilarityPercentage:F2}%");
                        sb.AppendLine($"    Lines: {block.DuplicatedLines}");
                        sb.AppendLine();
                    }
                }

                sb.AppendLine(new string('-', 50));
            }
        }

        return sb.ToString();
    }

    private string FormatAsJson(List<DuplicationMetric> metrics)
    {
        return System.Text.Json.JsonSerializer.Serialize(metrics, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    private string FormatAsCsv(List<DuplicationMetric> metrics)
    {
        var sb = new StringBuilder();

        // Header
        sb.AppendLine("Type,Target,TargetType,FilePath,TotalLinesOfCode,DuplicatedLinesOfCode,DuplicationPercentage,DuplicatedBlockCount,DuplicationLevel");

        // Data
        foreach (var metric in metrics)
        {
            sb.AppendLine($"{metric.Type},{metric.Target},{metric.TargetType},{metric.FilePath},{metric.TotalLinesOfCode},{metric.DuplicatedLinesOfCode},{metric.DuplicationPercentage:F2},{metric.DuplicatedBlockCount},{metric.DuplicationLevel}");
        }

        return sb.ToString();
    }

    private string FormatAsHtml(List<DuplicationMetric> metrics)
    {
        var sb = new StringBuilder();

        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html>");
        sb.AppendLine("<head>");
        sb.AppendLine("  <title>Duplication Analysis Results</title>");
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
        sb.AppendLine("    .block { margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; }");
        sb.AppendLine("  </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("  <h1>Duplication Analysis Results</h1>");

        // Group metrics by type
        var tokenBasedMetrics = metrics.Where(m => m.Type == DuplicationType.TokenBased).ToList();
        var semanticMetrics = metrics.Where(m => m.Type == DuplicationType.Semantic).ToList();

        if (tokenBasedMetrics.Any())
        {
            sb.AppendLine("  <h2>Token-Based Duplication</h2>");
            sb.AppendLine("  <table>");
            sb.AppendLine("    <tr>");
            sb.AppendLine("      <th>Target</th>");
            sb.AppendLine("      <th>Target Type</th>");
            sb.AppendLine("      <th>Total Lines</th>");
            sb.AppendLine("      <th>Duplicated Lines</th>");
            sb.AppendLine("      <th>Duplication %</th>");
            sb.AppendLine("      <th>Blocks</th>");
            sb.AppendLine("      <th>Level</th>");
            sb.AppendLine("    </tr>");

            foreach (var metric in tokenBasedMetrics.OrderByDescending(m => m.DuplicationPercentage))
            {
                var levelClass = metric.DuplicationLevel.ToString().ToLowerInvariant();
                sb.AppendLine("    <tr>");
                sb.AppendLine($"      <td>{metric.Target}</td>");
                sb.AppendLine($"      <td>{metric.TargetType}</td>");
                sb.AppendLine($"      <td>{metric.TotalLinesOfCode}</td>");
                sb.AppendLine($"      <td>{metric.DuplicatedLinesOfCode}</td>");
                sb.AppendLine($"      <td>{metric.DuplicationPercentage:F2}%</td>");
                sb.AppendLine($"      <td>{metric.DuplicatedBlockCount}</td>");
                sb.AppendLine($"      <td class=\"{levelClass}\">{metric.DuplicationLevel}</td>");
                sb.AppendLine("    </tr>");
            }

            sb.AppendLine("  </table>");

            sb.AppendLine("  <h3>Top Duplicated Blocks</h3>");

            foreach (var metric in tokenBasedMetrics.OrderByDescending(m => m.DuplicationPercentage).Take(3))
            {
                if (metric.DuplicatedBlocks.Any())
                {
                    sb.AppendLine($"  <h4>{metric.Target} ({metric.TargetType})</h4>");

                    foreach (var block in metric.DuplicatedBlocks.OrderByDescending(b => b.DuplicatedLines).Take(3))
                    {
                        sb.AppendLine("  <div class=\"block\">");
                        sb.AppendLine($"    <p>Lines {block.SourceStartLine}-{block.SourceEndLine} duplicated at lines {block.TargetStartLine}-{block.TargetEndLine}</p>");
                        sb.AppendLine($"    <p>Similarity: {block.SimilarityPercentage:F2}%</p>");
                        sb.AppendLine($"    <p>Lines: {block.DuplicatedLines}</p>");
                        sb.AppendLine("    <pre>");
                        sb.AppendLine(block.DuplicatedCode);
                        sb.AppendLine("    </pre>");
                        sb.AppendLine("  </div>");
                    }
                }
            }
        }

        if (semanticMetrics.Any())
        {
            sb.AppendLine("  <h2>Semantic Duplication</h2>");
            sb.AppendLine("  <table>");
            sb.AppendLine("    <tr>");
            sb.AppendLine("      <th>Target</th>");
            sb.AppendLine("      <th>Target Type</th>");
            sb.AppendLine("      <th>Total Lines</th>");
            sb.AppendLine("      <th>Duplicated Lines</th>");
            sb.AppendLine("      <th>Duplication %</th>");
            sb.AppendLine("      <th>Blocks</th>");
            sb.AppendLine("      <th>Level</th>");
            sb.AppendLine("    </tr>");

            foreach (var metric in semanticMetrics.OrderByDescending(m => m.DuplicationPercentage))
            {
                var levelClass = metric.DuplicationLevel.ToString().ToLowerInvariant();
                sb.AppendLine("    <tr>");
                sb.AppendLine($"      <td>{metric.Target}</td>");
                sb.AppendLine($"      <td>{metric.TargetType}</td>");
                sb.AppendLine($"      <td>{metric.TotalLinesOfCode}</td>");
                sb.AppendLine($"      <td>{metric.DuplicatedLinesOfCode}</td>");
                sb.AppendLine($"      <td>{metric.DuplicationPercentage:F2}%</td>");
                sb.AppendLine($"      <td>{metric.DuplicatedBlockCount}</td>");
                sb.AppendLine($"      <td class=\"{levelClass}\">{metric.DuplicationLevel}</td>");
                sb.AppendLine("    </tr>");
            }

            sb.AppendLine("  </table>");

            sb.AppendLine("  <h3>Top Semantically Similar Blocks</h3>");

            foreach (var metric in semanticMetrics.OrderByDescending(m => m.DuplicationPercentage).Take(3))
            {
                if (metric.DuplicatedBlocks.Any())
                {
                    sb.AppendLine($"  <h4>{metric.Target} ({metric.TargetType})</h4>");

                    foreach (var block in metric.DuplicatedBlocks.OrderByDescending(b => b.SimilarityPercentage).Take(3))
                    {
                        sb.AppendLine("  <div class=\"block\">");
                        sb.AppendLine($"    <p>Lines {block.SourceStartLine}-{block.SourceEndLine} similar to lines {block.TargetStartLine}-{block.TargetEndLine}</p>");
                        sb.AppendLine($"    <p>Similarity: {block.SimilarityPercentage:F2}%</p>");
                        sb.AppendLine($"    <p>Lines: {block.DuplicatedLines}</p>");
                        sb.AppendLine("    <pre>");
                        sb.AppendLine(block.DuplicatedCode);
                        sb.AppendLine("    </pre>");
                        sb.AppendLine("  </div>");
                    }
                }
            }
        }

        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }
}
