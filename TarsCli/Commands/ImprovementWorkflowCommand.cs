using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Commands;

/// <summary>
/// Command for running the end-to-end improvement generation workflow
/// </summary>
public class ImprovementWorkflowCommand : Command
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<ImprovementWorkflowCommand> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImprovementWorkflowCommand"/> class
    /// </summary>
    /// <param name="serviceProvider">The service provider</param>
    /// <param name="logger">The logger</param>
    public ImprovementWorkflowCommand(IServiceProvider serviceProvider, ILogger<ImprovementWorkflowCommand> logger)
        : base("workflow", "Run the end-to-end improvement generation workflow")
    {
        _serviceProvider = serviceProvider;
        _logger = logger;

        // Add options
        var pathOption = new Option<string>(
            ["--path", "-p"],
            "Path to the file or directory to analyze")
        {
            IsRequired = true
        };

        var recursiveOption = new Option<bool>(
            ["--recursive", "-r"],
            () => true,
            "Whether to analyze directories recursively");

        var filePatternOption = new Option<string>(
            ["--pattern", "-f"],
            () => "*.cs;*.fs",
            "File pattern to match (semicolon-separated)");

        var maxImprovementsOption = new Option<int>(
            ["--max-improvements", "-m"],
            () => 10,
            "Maximum number of improvements to generate");

        var executeOption = new Option<bool>(
            ["--execute", "-e"],
            "Whether to execute the improvements");

        var dryRunOption = new Option<bool>(
            ["--dry-run", "-d"],
            () => true,
            "Whether to perform a dry run without making changes");

        var outputOption = new Option<string>(
            ["--output", "-o"],
            "Path to the output file");

        var formatOption = new Option<string>(
            ["--format"],
            () => "json",
            "Output format (json, yaml, or text)");

        var verboseOption = new Option<bool>(
            ["--verbose", "-v"],
            "Show verbose output");

        // Add options to command
        AddOption(pathOption);
        AddOption(recursiveOption);
        AddOption(filePatternOption);
        AddOption(maxImprovementsOption);
        AddOption(executeOption);
        AddOption(dryRunOption);
        AddOption(outputOption);
        AddOption(formatOption);
        AddOption(verboseOption);

        // Set handler
        this.SetHandler(async (InvocationContext context) =>
        {
            var path = context.ParseResult.GetValueForOption(pathOption) ?? "";
            var recursive = context.ParseResult.GetValueForOption(recursiveOption);
            var filePattern = context.ParseResult.GetValueForOption(filePatternOption) ?? "*.cs";
            var maxImprovements = context.ParseResult.GetValueForOption(maxImprovementsOption);
            var execute = context.ParseResult.GetValueForOption(executeOption);
            var dryRun = context.ParseResult.GetValueForOption(dryRunOption);
            var output = context.ParseResult.GetValueForOption(outputOption);
            var format = context.ParseResult.GetValueForOption(formatOption) ?? "json";
            var verbose = context.ParseResult.GetValueForOption(verboseOption);

            await RunWorkflowAsync(path, recursive, filePattern, maxImprovements, execute, dryRun, output, format, verbose);
        });
    }

    private async Task RunWorkflowAsync(
        string path,
        bool recursive,
        string filePattern,
        int maxImprovements,
        bool execute,
        bool dryRun,
        string? output,
        string format,
        bool verbose)
    {
        try
        {
            _logger.LogInformation("Running improvement generation workflow for path: {Path}", path);

            // Create progress reporter
            var progressReporter = new ConsoleProgressReporter(
                _serviceProvider.GetRequiredService<ILogger<ConsoleProgressReporter>>(),
                verbose);

            // Create orchestrator
            var orchestrator = new ImprovementGenerationOrchestrator(
                _serviceProvider.GetRequiredService<ILogger<ImprovementGenerationOrchestrator>>(),
                _serviceProvider.GetRequiredService<ICodeAnalyzerService>(),
                _serviceProvider.GetRequiredService<IPatternMatcherService>(),
                _serviceProvider.GetRequiredService<IMetascriptGeneratorService>(),
                _serviceProvider.GetRequiredService<IImprovementPrioritizerService>(),
                progressReporter);

            // Create options
            var options = new Dictionary<string, string>
            {
                { "Recursive", recursive.ToString() },
                { "FilePattern", filePattern },
                { "MaxImprovements", maxImprovements.ToString() },
                { "ExecuteImprovements", execute.ToString() },
                { "DryRun", dryRun.ToString() },
                { "Verbose", verbose.ToString() }
            };

            // Run workflow
            var improvements = await orchestrator.RunWorkflowAsync(path, options);

            // Output results
            OutputResults(improvements, output, format, verbose);

            _logger.LogInformation("Improvement generation workflow completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running improvement generation workflow");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private void OutputResults<T>(T results, string? outputPath, string format, bool verbose)
    {
        string output;

        // Format results
        switch (format.ToLowerInvariant())
        {
            case "json":
                output = System.Text.Json.JsonSerializer.Serialize(results, new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true
                });
                break;
            case "yaml":
                // Simple YAML serialization (for demo purposes)
                output = "YAML serialization not implemented";
                break;
            case "text":
            default:
                if (results is List<PrioritizedImprovement> improvements)
                {
                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine($"Generated {improvements.Count} improvements:");
                    sb.AppendLine();

                    foreach (var improvement in improvements)
                    {
                        sb.AppendLine($"ID: {improvement.Id}");
                        sb.AppendLine($"Name: {improvement.Name}");
                        sb.AppendLine($"Category: {improvement.Category}");
                        sb.AppendLine($"Priority: {improvement.PriorityScore:F2}");
                        sb.AppendLine($"Status: {improvement.Status}");
                        sb.AppendLine($"Affected Files: {string.Join(", ", improvement.AffectedFiles)}");
                        sb.AppendLine();

                        if (verbose)
                        {
                            sb.AppendLine($"Description: {improvement.Description}");
                            sb.AppendLine($"Impact: {improvement.Impact} ({improvement.ImpactScore:F2})");
                            sb.AppendLine($"Effort: {improvement.Effort} ({improvement.EffortScore:F2})");
                            sb.AppendLine($"Risk: {improvement.Risk} ({improvement.RiskScore:F2})");
                            sb.AppendLine($"Alignment: {improvement.AlignmentScore:F2}");
                            sb.AppendLine($"Tags: {string.Join(", ", improvement.Tags)}");
                            sb.AppendLine($"Dependencies: {string.Join(", ", improvement.Dependencies)}");
                            sb.AppendLine();
                        }
                    }

                    output = sb.ToString();
                }
                else
                {
                    output = results?.ToString() ?? "No results";
                }
                break;
        }

        // Output results
        if (!string.IsNullOrEmpty(outputPath))
        {
            // Write to file
            File.WriteAllText(outputPath, output);
            Console.WriteLine($"Results written to {outputPath}");
        }
        else
        {
            // Write to console
            Console.WriteLine(output);
        }
    }
}
