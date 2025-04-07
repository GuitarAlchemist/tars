using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Commands;

/// <summary>
/// Command to demonstrate code duplication detection
/// </summary>
public class DuplicationDemoCommand : Command
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DuplicationDemoCommand"/> class
    /// </summary>
    public DuplicationDemoCommand() : base("duplication-demo", "Demonstrates code duplication detection")
    {
        var pathOption = new Option<string>(
            aliases: new[] { "--path", "-p" },
            description: "Path to the file or directory to analyze")
        {
            IsRequired = true
        };
        
        var languageOption = new Option<string>(
            aliases: new[] { "--language", "-l" },
            description: "Programming language (C# or F#)",
            getDefaultValue: () => string.Empty);
        
        var typeOption = new Option<string>(
            aliases: new[] { "--type", "-t" },
            description: "Duplication type (token, semantic, or all)",
            getDefaultValue: () => "all");
        
        var outputOption = new Option<string>(
            aliases: new[] { "--output", "-o" },
            description: "Output format (console, json, csv, or html)",
            getDefaultValue: () => "console");
        
        var outputPathOption = new Option<string>(
            aliases: new[] { "--output-path" },
            description: "Path to save the output file",
            getDefaultValue: () => string.Empty);
        
        AddOption(pathOption);
        AddOption(languageOption);
        AddOption(typeOption);
        AddOption(outputOption);
        AddOption(outputPathOption);
        
        this.SetHandler(HandleCommand);
    }
    
    private async Task HandleCommand(InvocationContext context)
    {
        var path = context.ParseResult.GetValueForOption<string>("--path");
        var language = context.ParseResult.GetValueForOption<string>("--language");
        var type = context.ParseResult.GetValueForOption<string>("--type");
        var output = context.ParseResult.GetValueForOption<string>("--output");
        var outputPath = context.ParseResult.GetValueForOption<string>("--output-path");
        
        var serviceProvider = context.BindingContext.GetService<IServiceProvider>();
        var logger = serviceProvider.GetRequiredService<ILogger<DuplicationDemoCommand>>();
        var duplicationAnalyzer = serviceProvider.GetRequiredService<IDuplicationAnalyzer>();
        
        try
        {
            logger.LogInformation("Analyzing duplication in {Path}", path);
            
            // Determine if path is a file or directory
            var isDirectory = Directory.Exists(path);
            var isFile = File.Exists(path);
            
            if (!isDirectory && !isFile)
            {
                logger.LogError("Path {Path} does not exist", path);
                return;
            }
            
            // Determine language if not specified
            if (string.IsNullOrEmpty(language))
            {
                if (isFile)
                {
                    language = GetLanguageFromFilePath(path);
                }
                else
                {
                    // Default to C# for directories
                    language = "C#";
                }
            }
            
            // Analyze duplication
            List<DuplicationMetric> metrics;
            
            if (isFile)
            {
                metrics = await AnalyzeFile(duplicationAnalyzer, path, language, type);
            }
            else
            {
                metrics = await AnalyzeDirectory(duplicationAnalyzer, path);
            }
            
            // Output results
            await OutputResults(metrics, output, outputPath);
            
            logger.LogInformation("Duplication analysis completed successfully");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error analyzing duplication");
        }
    }
    
    private async Task<List<DuplicationMetric>> AnalyzeFile(IDuplicationAnalyzer analyzer, string filePath, string language, string type)
    {
        return type.ToLowerInvariant() switch
        {
            "token" => await analyzer.AnalyzeTokenBasedDuplicationAsync(filePath, language),
            "semantic" => await analyzer.AnalyzeSemanticDuplicationAsync(filePath, language),
            _ => await analyzer.AnalyzeAllDuplicationMetricsAsync(filePath, language)
        };
    }
    
    private async Task<List<DuplicationMetric>> AnalyzeDirectory(IDuplicationAnalyzer analyzer, string directoryPath)
    {
        return await analyzer.AnalyzeProjectDuplicationAsync(directoryPath);
    }
    
    private async Task OutputResults(List<DuplicationMetric> metrics, string format, string outputPath)
    {
        var output = format.ToLowerInvariant() switch
        {
            "json" => FormatAsJson(metrics),
            "csv" => FormatAsCsv(metrics),
            "html" => FormatAsHtml(metrics),
            _ => FormatAsConsole(metrics)
        };
        
        if (!string.IsNullOrEmpty(outputPath))
        {
            await File.WriteAllTextAsync(outputPath, output);
            Console.WriteLine($"Results saved to {outputPath}");
        }
        else
        {
            Console.WriteLine(output);
        }
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
                        sb.AppendLine(System.Web.HttpUtility.HtmlEncode(block.DuplicatedCode));
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
                        sb.AppendLine(System.Web.HttpUtility.HtmlEncode(block.DuplicatedCode));
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
