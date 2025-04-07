using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Commands;

/// <summary>
/// Command for analyzing code complexity
/// </summary>
public class CodeComplexityCommand : Command
{
    private readonly ILogger<CodeComplexityCommand> _logger;
    private readonly ICodeComplexityAnalyzer _complexityAnalyzer;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="CodeComplexityCommand"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="complexityAnalyzer">Code complexity analyzer</param>
    public CodeComplexityCommand(ILogger<CodeComplexityCommand> logger, ICodeComplexityAnalyzer complexityAnalyzer)
        : base("complexity", "Analyze code complexity")
    {
        _logger = logger;
        _complexityAnalyzer = complexityAnalyzer;
        
        // Add options
        var pathOption = new Option<string>(
            aliases: new[] { "--path", "-p" },
            description: "Path to the file or directory to analyze")
        {
            IsRequired = true
        };
        
        var languageOption = new Option<string>(
            aliases: new[] { "--language", "-l" },
            description: "Programming language (C# or F#)")
        {
            IsRequired = false
        };
        
        var typeOption = new Option<string>(
            aliases: new[] { "--type", "-t" },
            description: "Complexity type (Cyclomatic, Cognitive, Maintainability, Halstead, or All)")
        {
            IsRequired = false
        };
        
        var outputOption = new Option<string>(
            aliases: new[] { "--output", "-o" },
            description: "Output format (Console, Json, or Csv)")
        {
            IsRequired = false
        };
        
        AddOption(pathOption);
        AddOption(languageOption);
        AddOption(typeOption);
        AddOption(outputOption);
        
        this.SetHandler(HandleCommandAsync);
    }
    
    /// <summary>
    /// Handles the command
    /// </summary>
    /// <param name="context">Command context</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task HandleCommandAsync(InvocationContext context)
    {
        try
        {
            var path = context.ParseResult.GetValueForOption<string>("--path");
            var language = context.ParseResult.GetValueForOption<string>("--language");
            var type = context.ParseResult.GetValueForOption<string>("--type") ?? "All";
            var output = context.ParseResult.GetValueForOption<string>("--output") ?? "Console";
            
            if (string.IsNullOrEmpty(path))
            {
                CliSupport.WriteColorLine("Path is required", ConsoleColor.Red);
                return;
            }
            
            if (!File.Exists(path) && !Directory.Exists(path))
            {
                CliSupport.WriteColorLine($"Path not found: {path}", ConsoleColor.Red);
                return;
            }
            
            // If language is not specified, try to determine from file extension
            if (string.IsNullOrEmpty(language) && File.Exists(path))
            {
                var extension = Path.GetExtension(path)?.ToLowerInvariant();
                language = extension switch
                {
                    ".cs" => "C#",
                    ".fs" => "F#",
                    _ => null
                };
            }
            
            // Analyze complexity
            var metrics = await AnalyzeComplexityAsync(path, language, type);
            
            if (metrics.Count == 0)
            {
                CliSupport.WriteColorLine("No complexity metrics found", ConsoleColor.Yellow);
                return;
            }
            
            // Output results
            OutputResults(metrics, output);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code complexity");
            CliSupport.WriteColorLine($"Error analyzing code complexity: {ex.Message}", ConsoleColor.Red);
        }
    }
    
    /// <summary>
    /// Analyzes code complexity
    /// </summary>
    /// <param name="path">Path to the file or directory</param>
    /// <param name="language">Programming language</param>
    /// <param name="type">Complexity type</param>
    /// <returns>Complexity metrics</returns>
    private async Task<List<ComplexityMetric>> AnalyzeComplexityAsync(string path, string language, string type)
    {
        if (File.Exists(path))
        {
            // Analyze a single file
            return await AnalyzeFileComplexityAsync(path, language, type);
        }
        else if (Directory.Exists(path))
        {
            // Analyze a directory
            return await AnalyzeDirectoryComplexityAsync(path, language, type);
        }
        
        return new List<ComplexityMetric>();
    }
    
    /// <summary>
    /// Analyzes code complexity of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <param name="type">Complexity type</param>
    /// <returns>Complexity metrics</returns>
    private async Task<List<ComplexityMetric>> AnalyzeFileComplexityAsync(string filePath, string language, string type)
    {
        return type.ToLowerInvariant() switch
        {
            "cyclomatic" => await _complexityAnalyzer.AnalyzeCyclomaticComplexityAsync(filePath, language),
            "cognitive" => await _complexityAnalyzer.AnalyzeCognitiveComplexityAsync(filePath, language),
            "maintainability" => await _complexityAnalyzer.AnalyzeMaintainabilityIndexAsync(filePath, language),
            "halstead" => await _complexityAnalyzer.AnalyzeHalsteadComplexityAsync(filePath, language),
            _ => await _complexityAnalyzer.AnalyzeAllComplexityMetricsAsync(filePath, language)
        };
    }
    
    /// <summary>
    /// Analyzes code complexity of a directory
    /// </summary>
    /// <param name="directoryPath">Path to the directory</param>
    /// <param name="language">Programming language</param>
    /// <param name="type">Complexity type</param>
    /// <returns>Complexity metrics</returns>
    private async Task<List<ComplexityMetric>> AnalyzeDirectoryComplexityAsync(string directoryPath, string language, string type)
    {
        var metrics = new List<ComplexityMetric>();
        
        // If language is specified, only analyze files of that language
        if (!string.IsNullOrEmpty(language))
        {
            var extension = language.ToLowerInvariant() switch
            {
                "c#" => "*.cs",
                "csharp" => "*.cs",
                "f#" => "*.fs",
                "fsharp" => "*.fs",
                _ => "*.*"
            };
            
            var files = Directory.GetFiles(directoryPath, extension, SearchOption.AllDirectories);
            
            foreach (var file in files)
            {
                metrics.AddRange(await AnalyzeFileComplexityAsync(file, language, type));
            }
        }
        else
        {
            // Analyze all supported files
            var csharpFiles = Directory.GetFiles(directoryPath, "*.cs", SearchOption.AllDirectories);
            var fsharpFiles = Directory.GetFiles(directoryPath, "*.fs", SearchOption.AllDirectories);
            
            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeFileComplexityAsync(file, "C#", type));
            }
            
            foreach (var file in fsharpFiles)
            {
                metrics.AddRange(await AnalyzeFileComplexityAsync(file, "F#", type));
            }
        }
        
        // Add project-level metrics
        metrics.AddRange(await _complexityAnalyzer.AnalyzeProjectComplexityAsync(directoryPath));
        
        return metrics;
    }
    
    /// <summary>
    /// Outputs complexity metrics
    /// </summary>
    /// <param name="metrics">Complexity metrics</param>
    /// <param name="format">Output format</param>
    private void OutputResults(List<ComplexityMetric> metrics, string format)
    {
        switch (format.ToLowerInvariant())
        {
            case "json":
                OutputJson(metrics);
                break;
            case "csv":
                OutputCsv(metrics);
                break;
            default:
                OutputConsole(metrics);
                break;
        }
    }
    
    /// <summary>
    /// Outputs complexity metrics to the console
    /// </summary>
    /// <param name="metrics">Complexity metrics</param>
    private void OutputConsole(List<ComplexityMetric> metrics)
    {
        CliSupport.WriteColorLine("Code Complexity Analysis", ConsoleColor.Cyan);
        Console.WriteLine();
        
        // Group metrics by type
        var groupedMetrics = metrics.GroupBy(m => m.Type);
        
        foreach (var group in groupedMetrics)
        {
            CliSupport.WriteColorLine($"{group.Key} Complexity", ConsoleColor.Yellow);
            Console.WriteLine(new string('-', 80));
            
            // Sort metrics by value (descending)
            var sortedMetrics = group.OrderByDescending(m => m.Value).ToList();
            
            // Calculate max lengths for formatting
            var maxTargetLength = Math.Min(50, sortedMetrics.Max(m => m.Target.Length));
            var maxValueLength = sortedMetrics.Max(m => m.Value.ToString("F2").Length);
            
            // Print header
            Console.WriteLine($"{"Target".PadRight(maxTargetLength)} | {"Value".PadRight(maxValueLength)} | {"Threshold".PadRight(maxValueLength)} | Status");
            Console.WriteLine(new string('-', maxTargetLength + maxValueLength * 2 + 15));
            
            // Print metrics
            foreach (var metric in sortedMetrics)
            {
                var target = metric.Target.Length > maxTargetLength
                    ? metric.Target.Substring(0, maxTargetLength - 3) + "..."
                    : metric.Target.PadRight(maxTargetLength);
                
                var value = metric.Value.ToString("F2").PadRight(maxValueLength);
                var threshold = metric.ThresholdValue.ToString("F2").PadRight(maxValueLength);
                
                var status = metric.IsAboveThreshold
                    ? "EXCEEDS THRESHOLD"
                    : "OK";
                
                var statusColor = metric.IsAboveThreshold
                    ? ConsoleColor.Red
                    : ConsoleColor.Green;
                
                Console.Write($"{target} | {value} | {threshold} | ");
                CliSupport.WriteColorLine(status, statusColor);
            }
            
            Console.WriteLine();
        }
        
        // Print summary
        CliSupport.WriteColorLine("Summary", ConsoleColor.Yellow);
        Console.WriteLine(new string('-', 80));
        
        var totalMetrics = metrics.Count;
        var exceedingThreshold = metrics.Count(m => m.IsAboveThreshold);
        var exceedingPercentage = totalMetrics > 0
            ? (double)exceedingThreshold / totalMetrics * 100
            : 0;
        
        Console.WriteLine($"Total metrics: {totalMetrics}");
        Console.WriteLine($"Metrics exceeding threshold: {exceedingThreshold} ({exceedingPercentage:F2}%)");
        
        Console.WriteLine();
    }
    
    /// <summary>
    /// Outputs complexity metrics as JSON
    /// </summary>
    /// <param name="metrics">Complexity metrics</param>
    private void OutputJson(List<ComplexityMetric> metrics)
    {
        var json = System.Text.Json.JsonSerializer.Serialize(metrics, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        
        Console.WriteLine(json);
    }
    
    /// <summary>
    /// Outputs complexity metrics as CSV
    /// </summary>
    /// <param name="metrics">Complexity metrics</param>
    private void OutputCsv(List<ComplexityMetric> metrics)
    {
        // Print header
        Console.WriteLine("Name,Target,Type,Value,Threshold,IsAboveThreshold,FilePath,Language");
        
        // Print metrics
        foreach (var metric in metrics)
        {
            Console.WriteLine($"\"{metric.Name}\",\"{metric.Target}\",{metric.Type},{metric.Value},{metric.ThresholdValue},{metric.IsAboveThreshold},\"{metric.FilePath}\",\"{metric.Language}\"");
        }
    }
}
