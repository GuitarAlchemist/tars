using System.CommandLine.Invocation;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;
using TarsCli.Extensions;

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
        _pathOption = new Option<string>(
            aliases: ["--path", "-p"],
            description: "Path to the file or directory to analyze")
        {
            IsRequired = true
        };

        _languageOption = new Option<string>(
            aliases: ["--language", "-l"],
            description: "Programming language (C# or F#)")
        {
            IsRequired = false
        };

        _typeOption = new Option<string>(
            aliases: ["--type", "-t"],
            description: "Complexity type (Cyclomatic, Cognitive, Maintainability, Halstead, or All)")
        {
            IsRequired = false
        };

        _halsteadTypeOption = new Option<string>(
            aliases: ["--halstead-type", "-h"],
            description: "Halstead metric type (Vocabulary, Length, Volume, Difficulty, Effort, TimeRequired, DeliveredBugs)")
        {
            IsRequired = false
        };

        _readabilityTypeOption = new Option<string>(
            aliases: ["--readability-type", "-r"],
            description: "Readability metric type (IdentifierQuality, CommentQuality, CodeStructure, Overall)")
        {
            IsRequired = false
        };

        _outputOption = new Option<string>(
            aliases: ["--output", "-o"],
            description: "Output format (Console, Json, or Csv)")
        {
            IsRequired = false
        };

        AddOption(_pathOption);
        AddOption(_languageOption);
        AddOption(_typeOption);
        AddOption(_halsteadTypeOption);
        AddOption(_readabilityTypeOption);
        AddOption(_outputOption);

        this.SetHandler(HandleCommandAsync);
    }

    // Store options as class fields
    private readonly Option<string> _pathOption;
    private readonly Option<string> _languageOption;
    private readonly Option<string> _typeOption;
    private readonly Option<string> _halsteadTypeOption;
    private readonly Option<string> _readabilityTypeOption;
    private readonly Option<string> _outputOption;

    /// <summary>
    /// Handles the command
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>

    private async Task HandleCommandAsync(InvocationContext context)
    {
        try
        {
            var path = context.ParseResult.GetValueForOption(_pathOption);
            var language = context.ParseResult.GetValueForOption(_languageOption);
            var type = context.ParseResult.GetValueForOption(_typeOption) ?? "All";
            var halsteadType = context.ParseResult.GetValueForOption(_halsteadTypeOption);
            var readabilityType = context.ParseResult.GetValueForOption(_readabilityTypeOption);
            var output = context.ParseResult.GetValueForOption(_outputOption) ?? "Console";

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
            var metrics = await AnalyzeComplexityAsync(path, language, type, halsteadType, readabilityType);

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
    /// <param name="halsteadType">Halstead metric type</param>
    /// <param name="readabilityType">Readability metric type</param>
    /// <returns>Complexity metrics</returns>
    private async Task<List<BaseMetric>> AnalyzeComplexityAsync(string path, string language, string type, string halsteadType = null, string readabilityType = null)
    {
        if (File.Exists(path))
        {
            // Analyze a single file
            return await AnalyzeFileComplexityAsync(path, language, type, halsteadType, readabilityType);
        }
        else if (Directory.Exists(path))
        {
            // Analyze a directory
            return await AnalyzeDirectoryComplexityAsync(path, language, type, halsteadType, readabilityType);
        }

        return [];
    }

    /// <summary>
    /// Analyzes code complexity of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <param name="type">Complexity type</param>
    /// <param name="halsteadType">Halstead metric type</param>
    /// <param name="readabilityType">Readability metric type</param>
    /// <returns>Complexity metrics</returns>
    private async Task<List<BaseMetric>> AnalyzeFileComplexityAsync(string filePath, string language, string type, string halsteadType = null, string readabilityType = null)
    {
        var result = new List<BaseMetric>();

        switch (type.ToLowerInvariant())
        {
            case "cyclomatic":
                result.AddRange(await _complexityAnalyzer.AnalyzeCyclomaticComplexityAsync(filePath, language));
                break;
            case "cognitive":
                result.AddRange(await _complexityAnalyzer.AnalyzeCognitiveComplexityAsync(filePath, language));
                break;
            case "maintainability":
                result.AddRange(await _complexityAnalyzer.AnalyzeMaintainabilityIndexAsync(filePath, language));
                break;
            case "halstead":
                var halsteadMetrics = await _complexityAnalyzer.AnalyzeHalsteadComplexityAsync(filePath, language);

                // Filter by Halstead type if specified
                if (!string.IsNullOrEmpty(halsteadType) && Enum.TryParse<HalsteadType>(halsteadType, true, out var parsedHalsteadType))
                {
                    halsteadMetrics = halsteadMetrics.Where(m => m.Type == parsedHalsteadType).ToList();
                }

                result.AddRange(halsteadMetrics);
                break;
            case "readability":
                // Parse readability type if specified
                if (!string.IsNullOrEmpty(readabilityType) && Enum.TryParse<ReadabilityType>(readabilityType, true, out var parsedReadabilityType))
                {
                    result.AddRange(await _complexityAnalyzer.AnalyzeReadabilityAsync(filePath, language, parsedReadabilityType));
                }
                else
                {
                    // If no specific readability type is specified, analyze all readability metrics
                    result.AddRange(await _complexityAnalyzer.AnalyzeReadabilityAsync(filePath, language, ReadabilityType.Overall));
                }
                break;
            default: // "all"
                var allMetrics = await _complexityAnalyzer.AnalyzeAllComplexityMetricsAsync(filePath, language);
                result.AddRange(allMetrics.ComplexityMetrics);

                // Filter Halstead metrics by type if specified
                var filteredHalsteadMetrics = allMetrics.HalsteadMetrics;
                if (!string.IsNullOrEmpty(halsteadType) && Enum.TryParse<HalsteadType>(halsteadType, true, out var parsedType))
                {
                    filteredHalsteadMetrics = filteredHalsteadMetrics.Where(m => m.Type == parsedType).ToList();
                }

                result.AddRange(filteredHalsteadMetrics);
                result.AddRange(allMetrics.MaintainabilityMetrics);

                // Filter readability metrics by type if specified
                var filteredReadabilityMetrics = allMetrics.ReadabilityMetrics;
                if (!string.IsNullOrEmpty(readabilityType) && Enum.TryParse<ReadabilityType>(readabilityType, true, out var parsedReadabilityTypeValue))
                {
                    filteredReadabilityMetrics = filteredReadabilityMetrics.Where(m => m.Type == parsedReadabilityTypeValue).ToList();
                }

                result.AddRange(filteredReadabilityMetrics);
                break;
        }

        return result;
    }

    /// <summary>
    /// Analyzes code complexity of a directory
    /// </summary>
    /// <param name="directoryPath">Path to the directory</param>
    /// <param name="language">Programming language</param>
    /// <param name="type">Complexity type</param>
    /// <param name="halsteadType">Halstead metric type</param>
    /// <param name="readabilityType">Readability metric type</param>
    /// <returns>Complexity metrics</returns>
    private async Task<List<BaseMetric>> AnalyzeDirectoryComplexityAsync(string directoryPath, string language, string type, string halsteadType = null, string readabilityType = null)
    {
        var metrics = new List<BaseMetric>();

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
                metrics.AddRange(await AnalyzeFileComplexityAsync(file, language, type, halsteadType));
            }
        }
        else
        {
            // Analyze all supported files
            var csharpFiles = Directory.GetFiles(directoryPath, "*.cs", SearchOption.AllDirectories);
            var fsharpFiles = Directory.GetFiles(directoryPath, "*.fs", SearchOption.AllDirectories);

            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeFileComplexityAsync(file, "C#", type, halsteadType, readabilityType));
            }

            foreach (var file in fsharpFiles)
            {
                metrics.AddRange(await AnalyzeFileComplexityAsync(file, "F#", type, halsteadType, readabilityType));
            }
        }

        // Add project-level metrics
        var projectMetrics = await _complexityAnalyzer.AnalyzeProjectComplexityAsync(directoryPath);

        // Add appropriate metrics based on type
        switch (type.ToLowerInvariant())
        {
            case "cyclomatic":
            case "cognitive":
                metrics.AddRange(projectMetrics.ComplexityMetrics);
                break;
            case "maintainability":
                metrics.AddRange(projectMetrics.MaintainabilityMetrics);
                break;
            case "halstead":
                var halsteadMetrics = projectMetrics.HalsteadMetrics;

                // Filter by Halstead type if specified
                if (!string.IsNullOrEmpty(halsteadType) && Enum.TryParse<HalsteadType>(halsteadType, true, out var parsedHalsteadType))
                {
                    halsteadMetrics = halsteadMetrics.Where(m => m.Type == parsedHalsteadType).ToList();
                }

                metrics.AddRange(halsteadMetrics);
                break;
            case "readability":
                var readabilityMetrics = projectMetrics.ReadabilityMetrics;

                // Filter by readability type if specified
                if (!string.IsNullOrEmpty(readabilityType) && Enum.TryParse<ReadabilityType>(readabilityType, true, out var parsedReadabilityType))
                {
                    readabilityMetrics = readabilityMetrics.Where(m => m.Type == parsedReadabilityType).ToList();
                }

                metrics.AddRange(readabilityMetrics);
                break;
            default: // "all"
                metrics.AddRange(projectMetrics.ComplexityMetrics);

                // Filter Halstead metrics by type if specified
                var filteredHalsteadMetrics = projectMetrics.HalsteadMetrics;
                if (!string.IsNullOrEmpty(halsteadType) && Enum.TryParse<HalsteadType>(halsteadType, true, out var parsedType))
                {
                    filteredHalsteadMetrics = filteredHalsteadMetrics.Where(m => m.Type == parsedType).ToList();
                }

                metrics.AddRange(filteredHalsteadMetrics);
                metrics.AddRange(projectMetrics.MaintainabilityMetrics);

                // Filter readability metrics by type if specified
                var filteredReadabilityMetrics = projectMetrics.ReadabilityMetrics;
                if (!string.IsNullOrEmpty(readabilityType) && Enum.TryParse<ReadabilityType>(readabilityType, true, out var parsedReadabilityTypeForProject))
                {
                    filteredReadabilityMetrics = filteredReadabilityMetrics.Where(m => m.Type == parsedReadabilityTypeForProject).ToList();
                }

                metrics.AddRange(filteredReadabilityMetrics);
                break;
        }

        return metrics;
    }

    /// <summary>
    /// Outputs complexity metrics
    /// </summary>
    /// <param name="metrics">Complexity metrics</param>
    /// <param name="format">Output format</param>
    private void OutputResults(List<BaseMetric> metrics, string format)
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
    private void OutputConsole(List<BaseMetric> metrics)
    {
        CliSupport.WriteColorLine("Code Complexity Analysis", ConsoleColor.Cyan);
        Console.WriteLine();

        // Group metrics by type
        var complexityMetrics = metrics.OfType<ComplexityMetric>().ToList();
        var halsteadMetrics = metrics.OfType<HalsteadMetric>().ToList();
        var maintainabilityMetrics = metrics.OfType<MaintainabilityMetric>().ToList();

        // Output complexity metrics
        if (complexityMetrics.Any())
        {
            var groupedComplexityMetrics = complexityMetrics.GroupBy(m => m.Type);

            foreach (var group in groupedComplexityMetrics)
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

                    var status = metric.IsAboveThreshold()
                        ? "EXCEEDS THRESHOLD"
                        : "OK";

                    var statusColor = metric.IsAboveThreshold()
                        ? ConsoleColor.Red
                        : ConsoleColor.Green;

                    Console.Write($"{target} | {value} | {threshold} | ");
                    CliSupport.WriteColorLine(status, statusColor);
                }

                Console.WriteLine();
            }
        }

        // Output Halstead metrics
        if (halsteadMetrics.Any())
        {
            var groupedHalsteadMetrics = halsteadMetrics.GroupBy(m => m.Type);

            foreach (var group in groupedHalsteadMetrics)
            {
                CliSupport.WriteColorLine($"Halstead {group.Key}", ConsoleColor.Yellow);
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
                    var threshold = metric.ThresholdValue > 0
                        ? metric.ThresholdValue.ToString("F2").PadRight(maxValueLength)
                        : "N/A".PadRight(maxValueLength);

                    var status = metric.ThresholdValue > 0
                        ? metric.IsAboveThreshold ? "EXCEEDS THRESHOLD" : "OK"
                        : "";

                    var statusColor = metric.IsAboveThreshold ? ConsoleColor.Red : ConsoleColor.Green;

                    Console.Write($"{target} | {value} | {threshold} | ");
                    CliSupport.WriteColorLine(status, statusColor);
                }

                Console.WriteLine();
            }
        }

        // Output Maintainability metrics
        if (maintainabilityMetrics.Any())
        {
            CliSupport.WriteColorLine("Maintainability Index", ConsoleColor.Yellow);
            Console.WriteLine(new string('-', 80));

            // Sort metrics by value (ascending - higher is better for maintainability)
            var sortedMetrics = maintainabilityMetrics.OrderBy(m => m.Value).ToList();

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
                var threshold = metric.ThresholdValue > 0
                    ? metric.ThresholdValue.ToString("F2").PadRight(maxValueLength)
                    : "N/A".PadRight(maxValueLength);

                var status = metric.ThresholdValue > 0
                    ? metric.IsBelowThreshold ? "BELOW THRESHOLD" : "OK"
                    : metric.MaintainabilityLevel.ToString();

                var statusColor = metric.IsBelowThreshold ? ConsoleColor.Red : ConsoleColor.Green;

                Console.Write($"{target} | {value} | {threshold} | ");
                CliSupport.WriteColorLine(status, statusColor);
            }

            Console.WriteLine();
        }

        // Print summary
        CliSupport.WriteColorLine("Summary", ConsoleColor.Yellow);
        Console.WriteLine(new string('-', 80));

        var totalMetrics = metrics.Count;
        var complexityExceedingThreshold = complexityMetrics.Count(m => m.IsAboveThreshold());
        var halsteadExceedingThreshold = halsteadMetrics.Count(m => m.IsAboveThreshold);
        var maintainabilityBelowThreshold = maintainabilityMetrics.Count(m => m.IsBelowThreshold);

        var totalExceeding = complexityExceedingThreshold + halsteadExceedingThreshold + maintainabilityBelowThreshold;
        var exceedingPercentage = totalMetrics > 0
            ? (double)totalExceeding / totalMetrics * 100
            : 0;

        Console.WriteLine($"Total metrics: {totalMetrics}");
        Console.WriteLine($"Complexity metrics exceeding threshold: {complexityExceedingThreshold}");
        Console.WriteLine($"Halstead metrics exceeding threshold: {halsteadExceedingThreshold}");
        Console.WriteLine($"Maintainability metrics below threshold: {maintainabilityBelowThreshold}");
        Console.WriteLine($"Total metrics with issues: {totalExceeding} ({exceedingPercentage:F2}%)");

        Console.WriteLine();
    }

    /// <summary>
    /// Outputs complexity metrics as JSON
    /// </summary>
    /// <param name="metrics">Complexity metrics</param>
    private void OutputJson(List<BaseMetric> metrics)
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
    private void OutputCsv(List<BaseMetric> metrics)
    {
        // Print header
        Console.WriteLine("MetricType,Name,Target,Value,Threshold,HasIssue,FilePath,Language");

        // Print metrics
        foreach (var metric in metrics)
        {
            var metricType = "Unknown";
            var hasIssue = false;

            if (metric is ComplexityMetric complexityMetric)
            {
                metricType = $"Complexity_{complexityMetric.Type}";
                hasIssue = complexityMetric.IsAboveThreshold();
            }
            else if (metric is HalsteadMetric halsteadMetric)
            {
                metricType = $"Halstead_{halsteadMetric.Type}";
                hasIssue = halsteadMetric.IsAboveThreshold;
            }
            else if (metric is MaintainabilityMetric maintainabilityMetric)
            {
                metricType = "Maintainability";
                hasIssue = maintainabilityMetric.IsBelowThreshold;
            }

            Console.WriteLine($"{metricType},\"{metric.Name}\",\"{metric.Target()}\",{metric.Value},{metric.ThresholdValue()},{hasIssue},\"{metric.FilePath()}\",\"{metric.Language()}\"");
        }
    }
}
