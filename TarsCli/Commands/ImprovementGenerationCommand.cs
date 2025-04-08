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
/// Command for the Improvement Generation System
/// </summary>
public class ImprovementGenerationCommand : Command
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<ImprovementGenerationCommand> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImprovementGenerationCommand"/> class
    /// </summary>
    /// <param name="serviceProvider">The service provider</param>
    /// <param name="logger">The logger</param>
    public ImprovementGenerationCommand(IServiceProvider serviceProvider, ILogger<ImprovementGenerationCommand> logger)
        : base("improve", "Generate and manage code improvements")
    {
        _serviceProvider = serviceProvider;
        _logger = logger;

        // Add subcommands
        AddCommand(CreateAnalyzeCommand());
        AddCommand(CreateMatchCommand());
        AddCommand(CreateGenerateCommand());
        AddCommand(CreatePrioritizeCommand());
        AddCommand(CreateListCommand());
        AddCommand(CreateExecuteCommand());
        AddCommand(CreateStatusCommand());

        // Add workflow command
        var workflowCommand = new ImprovementWorkflowCommand(serviceProvider, _serviceProvider.GetRequiredService<ILogger<ImprovementWorkflowCommand>>());
        AddCommand(workflowCommand);
    }

    private Command CreateAnalyzeCommand()
    {
        var command = new Command("analyze", "Analyze code for improvement opportunities");

        // Add options
        var pathOption = new Option<string>(
            new[] { "--path", "-p" },
            "Path to the file or directory to analyze")
        {
            IsRequired = true
        };

        var recursiveOption = new Option<bool>(
            new[] { "--recursive", "-r" },
            () => true,
            "Whether to analyze directories recursively");

        var filePatternOption = new Option<string>(
            new[] { "--pattern", "-f" },
            () => "*.cs;*.fs",
            "File pattern to match (semicolon-separated)");

        var outputOption = new Option<string>(
            new[] { "--output", "-o" },
            "Path to the output file");

        var formatOption = new Option<string>(
            new[] { "--format" },
            () => "json",
            "Output format (json, yaml, or text)");

        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(pathOption);
        command.AddOption(recursiveOption);
        command.AddOption(filePatternOption);
        command.AddOption(outputOption);
        command.AddOption(formatOption);
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (string path, bool recursive, string filePattern, string? output, string format, bool verbose) =>
        {
            await AnalyzeCodeAsync(path, recursive, filePattern, output, format, verbose);
        }, pathOption, recursiveOption, filePatternOption, outputOption, formatOption, verboseOption);

        return command;
    }

    private Command CreateMatchCommand()
    {
        var command = new Command("match", "Match code patterns in source code");

        // Add options
        var pathOption = new Option<string>(
            new[] { "--path", "-p" },
            "Path to the file or directory to match patterns in")
        {
            IsRequired = true
        };

        var recursiveOption = new Option<bool>(
            new[] { "--recursive", "-r" },
            () => true,
            "Whether to match patterns in directories recursively");

        var filePatternOption = new Option<string>(
            new[] { "--pattern", "-f" },
            () => "*.cs;*.fs",
            "File pattern to match (semicolon-separated)");

        var languageOption = new Option<string>(
            new[] { "--language", "-l" },
            "Language filter for patterns");

        var outputOption = new Option<string>(
            new[] { "--output", "-o" },
            "Path to the output file");

        var formatOption = new Option<string>(
            new[] { "--format" },
            () => "json",
            "Output format (json, yaml, or text)");

        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(pathOption);
        command.AddOption(recursiveOption);
        command.AddOption(filePatternOption);
        command.AddOption(languageOption);
        command.AddOption(outputOption);
        command.AddOption(formatOption);
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (string path, bool recursive, string filePattern, string? language, string? output, string format, bool verbose) =>
        {
            await MatchPatternsAsync(path, recursive, filePattern, language, output, format, verbose);
        }, pathOption, recursiveOption, filePatternOption, languageOption, outputOption, formatOption, verboseOption);

        return command;
    }

    private Command CreateGenerateCommand()
    {
        var command = new Command("generate", "Generate metascripts from pattern matches");

        // Add options
        var matchIdOption = new Option<string>(
            new[] { "--match-id", "-m" },
            "ID of the pattern match to generate a metascript for");

        var matchFileOption = new Option<string>(
            new[] { "--match-file", "-f" },
            "Path to a file containing pattern matches");

        var outputOption = new Option<string>(
            new[] { "--output", "-o" },
            "Path to the output directory");

        var formatOption = new Option<string>(
            new[] { "--format" },
            () => "json",
            "Output format (json, yaml, or text)");

        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(matchIdOption);
        command.AddOption(matchFileOption);
        command.AddOption(outputOption);
        command.AddOption(formatOption);
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (string? matchId, string? matchFile, string? output, string format, bool verbose) =>
        {
            await GenerateMetascriptsAsync(matchId, matchFile, output, format, verbose);
        }, matchIdOption, matchFileOption, outputOption, formatOption, verboseOption);

        return command;
    }

    private Command CreatePrioritizeCommand()
    {
        var command = new Command("prioritize", "Prioritize improvements");

        // Add options
        var metascriptIdOption = new Option<string>(
            new[] { "--metascript-id", "-m" },
            "ID of the metascript to prioritize");

        var metascriptFileOption = new Option<string>(
            new[] { "--metascript-file", "-f" },
            "Path to a file containing metascripts");

        var outputOption = new Option<string>(
            new[] { "--output", "-o" },
            "Path to the output file");

        var formatOption = new Option<string>(
            new[] { "--format" },
            () => "json",
            "Output format (json, yaml, or text)");

        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(metascriptIdOption);
        command.AddOption(metascriptFileOption);
        command.AddOption(outputOption);
        command.AddOption(formatOption);
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (string? metascriptId, string? metascriptFile, string? output, string format, bool verbose) =>
        {
            await PrioritizeImprovementsAsync(metascriptId, metascriptFile, output, format, verbose);
        }, metascriptIdOption, metascriptFileOption, outputOption, formatOption, verboseOption);

        return command;
    }

    private Command CreateListCommand()
    {
        var command = new Command("list", "List improvements");

        // Add options
        var categoryOption = new Option<string>(
            new[] { "--category", "-c" },
            "Filter by category");

        var statusOption = new Option<string>(
            new[] { "--status", "-s" },
            "Filter by status");

        var tagOption = new Option<string>(
            new[] { "--tag", "-t" },
            "Filter by tag");

        var limitOption = new Option<int>(
            new[] { "--limit", "-l" },
            () => 10,
            "Maximum number of improvements to list");

        var sortByOption = new Option<string>(
            new[] { "--sort-by" },
            () => "priority",
            "Sort by field (priority, impact, effort, risk, alignment, date, name)");

        var outputOption = new Option<string>(
            new[] { "--output", "-o" },
            "Path to the output file");

        var formatOption = new Option<string>(
            new[] { "--format" },
            () => "text",
            "Output format (json, yaml, or text)");

        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(categoryOption);
        command.AddOption(statusOption);
        command.AddOption(tagOption);
        command.AddOption(limitOption);
        command.AddOption(sortByOption);
        command.AddOption(outputOption);
        command.AddOption(formatOption);
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (string? category, string? status, string? tag, int limit, string sortBy, string? output, string format, bool verbose) =>
        {
            await ListImprovementsAsync(category, status, tag, limit, sortBy, output, format, verbose);
        }, categoryOption, statusOption, tagOption, limitOption, sortByOption, outputOption, formatOption, verboseOption);

        return command;
    }

    private Command CreateExecuteCommand()
    {
        var command = new Command("execute", "Execute an improvement");

        // Add options
        var improvementIdOption = new Option<string>(
            new[] { "--id", "-i" },
            "ID of the improvement to execute")
        {
            IsRequired = true
        };

        var dryRunOption = new Option<bool>(
            new[] { "--dry-run", "-d" },
            "Perform a dry run without making changes");

        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(improvementIdOption);
        command.AddOption(dryRunOption);
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (string improvementId, bool dryRun, bool verbose) =>
        {
            await ExecuteImprovementAsync(improvementId, dryRun, verbose);
        }, improvementIdOption, dryRunOption, verboseOption);

        return command;
    }

    private Command CreateStatusCommand()
    {
        var command = new Command("status", "Show the status of the improvement generation system");

        // Add options
        var verboseOption = new Option<bool>(
            new[] { "--verbose", "-v" },
            "Show verbose output");

        // Add options to command
        command.AddOption(verboseOption);

        // Set handler
        command.SetHandler(async (bool verbose) =>
        {
            await ShowStatusAsync(verbose);
        }, verboseOption);

        return command;
    }

    private async Task AnalyzeCodeAsync(string path, bool recursive, string filePattern, string? output, string format, bool verbose)
    {
        try
        {
            _logger.LogInformation("Analyzing code at path: {Path}", path);

            // Get code analyzer service
            var codeAnalyzer = _serviceProvider.GetRequiredService<ICodeAnalyzerService>();

            // Analyze code
            var results = await AnalyzePathAsync(codeAnalyzer, path, recursive, filePattern);

            // Output results
            OutputResults(results, output, format, verbose);

            _logger.LogInformation("Code analysis completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task MatchPatternsAsync(string path, bool recursive, string filePattern, string? language, string? output, string format, bool verbose)
    {
        try
        {
            _logger.LogInformation("Matching patterns in code at path: {Path}", path);

            // Get pattern matcher service
            var patternMatcher = _serviceProvider.GetRequiredService<IPatternMatcherService>();

            // Match patterns
            var results = await MatchPatternsInPathAsync(patternMatcher, path, recursive, filePattern, language);

            // Output results
            OutputResults(results, output, format, verbose);

            _logger.LogInformation("Pattern matching completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error matching patterns");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task GenerateMetascriptsAsync(string? matchId, string? matchFile, string? output, string format, bool verbose)
    {
        try
        {
            _logger.LogInformation("Generating metascripts");

            // Get metascript generator service
            var metascriptGenerator = _serviceProvider.GetRequiredService<IMetascriptGeneratorService>();
            var patternMatcher = _serviceProvider.GetRequiredService<IPatternMatcherService>();

            // Get pattern matches
            var patternMatches = new List<PatternMatch>();

            if (!string.IsNullOrEmpty(matchId))
            {
                // Get pattern match by ID
                var patterns = await patternMatcher.GetPatternsAsync();
                var pattern = patterns.FirstOrDefault(p => p.Id == matchId);
                if (pattern != null)
                {
                    var match = new PatternMatch
                    {
                        PatternId = pattern.Id,
                        PatternName = pattern.Name,
                        MatchedText = pattern.Pattern,
                        Language = pattern.Language,
                        Confidence = 1.0,
                        ExpectedImprovement = pattern.ExpectedImprovement,
                        ImpactScore = pattern.ImpactScore,
                        Tags = pattern.Tags
                    };
                    patternMatches.Add(match);
                }
                else
                {
                    Console.Error.WriteLine($"Pattern match not found: {matchId}");
                    return;
                }
            }
            else if (!string.IsNullOrEmpty(matchFile))
            {
                // Load pattern matches from file
                var json = await File.ReadAllTextAsync(matchFile);
                patternMatches = System.Text.Json.JsonSerializer.Deserialize<List<PatternMatch>>(json) ?? new List<PatternMatch>();
            }
            else
            {
                Console.Error.WriteLine("Either --match-id or --match-file must be specified");
                return;
            }

            // Generate metascripts
            var metascripts = await metascriptGenerator.GenerateMetascriptsAsync(patternMatches);

            // Output results
            OutputResults(metascripts, output, format, verbose);

            _logger.LogInformation("Metascript generation completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating metascripts");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task PrioritizeImprovementsAsync(string? metascriptId, string? metascriptFile, string? output, string format, bool verbose)
    {
        try
        {
            _logger.LogInformation("Prioritizing improvements");

            // Get improvement prioritizer service
            var improvementPrioritizer = _serviceProvider.GetRequiredService<IImprovementPrioritizerService>();
            var metascriptGenerator = _serviceProvider.GetRequiredService<IMetascriptGeneratorService>();

            // Get metascripts
            var metascripts = new List<GeneratedMetascript>();

            if (!string.IsNullOrEmpty(metascriptId))
            {
                // Get metascript by ID
                var metascript = await metascriptGenerator.GetMetascriptAsync(metascriptId);
                if (metascript != null)
                {
                    metascripts.Add(metascript);
                }
                else
                {
                    Console.Error.WriteLine($"Metascript not found: {metascriptId}");
                    return;
                }
            }
            else if (!string.IsNullOrEmpty(metascriptFile))
            {
                // Load metascripts from file
                var json = await File.ReadAllTextAsync(metascriptFile);
                metascripts = System.Text.Json.JsonSerializer.Deserialize<List<GeneratedMetascript>>(json) ?? new List<GeneratedMetascript>();
            }
            else
            {
                Console.Error.WriteLine("Either --metascript-id or --metascript-file must be specified");
                return;
            }

            // Create improvements from metascripts
            var improvements = new List<PrioritizedImprovement>();
            foreach (var metascript in metascripts)
            {
                var improvement = await improvementPrioritizer.CreateImprovementFromMetascriptAsync(metascript);
                improvements.Add(improvement);
            }

            // Prioritize improvements
            improvements = await improvementPrioritizer.PrioritizeImprovementsAsync(improvements);

            // Output results
            OutputResults(improvements, output, format, verbose);

            _logger.LogInformation("Improvement prioritization completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error prioritizing improvements");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task ListImprovementsAsync(string? category, string? status, string? tag, int limit, string sortBy, string? output, string format, bool verbose)
    {
        try
        {
            _logger.LogInformation("Listing improvements");

            // Get improvement prioritizer service
            var improvementPrioritizer = _serviceProvider.GetRequiredService<IImprovementPrioritizerService>();

            // Create options
            var options = new Dictionary<string, string>();
            if (!string.IsNullOrEmpty(category))
            {
                options["Category"] = category;
            }
            if (!string.IsNullOrEmpty(status))
            {
                options["Status"] = status;
            }
            if (!string.IsNullOrEmpty(tag))
            {
                options["Tag"] = tag;
            }
            options["Limit"] = limit.ToString();
            options["SortBy"] = sortBy;

            // Get improvements
            var improvements = await improvementPrioritizer.GetImprovementsAsync(options);

            // Output results
            OutputResults(improvements, output, format, verbose);

            _logger.LogInformation("Listed {ImprovementCount} improvements", improvements.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error listing improvements");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task ExecuteImprovementAsync(string improvementId, bool dryRun, bool verbose)
    {
        try
        {
            _logger.LogInformation("Executing improvement: {ImprovementId}", improvementId);

            // Get improvement prioritizer service
            var improvementPrioritizer = _serviceProvider.GetRequiredService<IImprovementPrioritizerService>();
            var metascriptGenerator = _serviceProvider.GetRequiredService<IMetascriptGeneratorService>();

            // Get improvement
            var improvement = await improvementPrioritizer.GetImprovementAsync(improvementId);
            if (improvement == null)
            {
                Console.Error.WriteLine($"Improvement not found: {improvementId}");
                return;
            }

            // Get metascript
            if (string.IsNullOrEmpty(improvement.MetascriptId))
            {
                Console.Error.WriteLine($"Improvement has no associated metascript: {improvementId}");
                return;
            }

            var metascript = await metascriptGenerator.GetMetascriptAsync(improvement.MetascriptId);
            if (metascript == null)
            {
                Console.Error.WriteLine($"Metascript not found: {improvement.MetascriptId}");
                return;
            }

            // Execute metascript
            var options = new Dictionary<string, string>();
            if (dryRun)
            {
                options["DryRun"] = "true";
            }

            var result = await metascriptGenerator.ExecuteMetascriptAsync(metascript, options);

            // Update improvement status
            if (result.IsSuccessful)
            {
                improvement.Status = dryRun ? ImprovementStatus.Pending : ImprovementStatus.Completed;
                await improvementPrioritizer.UpdateImprovementAsync(improvement);
            }

            // Output result
            Console.WriteLine($"Execution {(result.IsSuccessful ? "succeeded" : "failed")}");
            Console.WriteLine($"Status: {result.Status}");
            if (!string.IsNullOrEmpty(result.Output))
            {
                Console.WriteLine("Output:");
                Console.WriteLine(result.Output);
            }
            if (!string.IsNullOrEmpty(result.Error))
            {
                Console.WriteLine("Error:");
                Console.WriteLine(result.Error);
            }
            if (result.Changes.Count > 0)
            {
                Console.WriteLine("Changes:");
                foreach (var change in result.Changes)
                {
                    Console.WriteLine($"  {change.Type} in {change.FilePath} (lines {change.StartLine}-{change.EndLine})");
                    if (verbose)
                    {
                        Console.WriteLine("  Original:");
                        Console.WriteLine(change.OriginalContent);
                        Console.WriteLine("  New:");
                        Console.WriteLine(change.NewContent);
                    }
                }
            }

            _logger.LogInformation("Improvement execution completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing improvement");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task ShowStatusAsync(bool verbose)
    {
        try
        {
            _logger.LogInformation("Showing improvement generation system status");

            // Get services
            var codeAnalyzer = _serviceProvider.GetRequiredService<ICodeAnalyzerService>();
            var patternMatcher = _serviceProvider.GetRequiredService<IPatternMatcherService>();
            var metascriptGenerator = _serviceProvider.GetRequiredService<IMetascriptGeneratorService>();
            var improvementPrioritizer = _serviceProvider.GetRequiredService<IImprovementPrioritizerService>();

            // Get status information
            var patterns = await patternMatcher.GetPatternsAsync();
            var templates = await metascriptGenerator.GetTemplatesAsync();
            var improvements = await improvementPrioritizer.GetImprovementsAsync();
            var goals = await improvementPrioritizer.GetStrategicGoalsAsync();

            // Output status
            Console.WriteLine("Improvement Generation System Status");
            Console.WriteLine("====================================");
            Console.WriteLine();
            Console.WriteLine($"Patterns: {patterns.Count}");
            Console.WriteLine($"Templates: {templates.Count}");
            Console.WriteLine($"Improvements: {improvements.Count}");
            Console.WriteLine($"Strategic Goals: {goals.Count}");
            Console.WriteLine();

            if (verbose)
            {
                // Show pattern languages
                var languages = await patternMatcher.GetSupportedPatternLanguagesAsync();
                Console.WriteLine("Supported Pattern Languages:");
                foreach (var language in languages)
                {
                    Console.WriteLine($"  {language}");
                }
                Console.WriteLine();

                // Show improvement categories
                var categories = improvements
                    .Select(i => i.Category)
                    .Distinct()
                    .OrderBy(c => c.ToString());
                Console.WriteLine("Improvement Categories:");
                foreach (var category in categories)
                {
                    var count = improvements.Count(i => i.Category == category);
                    Console.WriteLine($"  {category}: {count}");
                }
                Console.WriteLine();

                // Show improvement statuses
                var statuses = improvements
                    .Select(i => i.Status)
                    .Distinct()
                    .OrderBy(s => s.ToString());
                Console.WriteLine("Improvement Statuses:");
                foreach (var status in statuses)
                {
                    var count = improvements.Count(i => i.Status == status);
                    Console.WriteLine($"  {status}: {count}");
                }
                Console.WriteLine();

                // Show next improvements
                var nextImprovements = await improvementPrioritizer.GetNextImprovementsAsync(5);
                Console.WriteLine("Next Improvements:");
                foreach (var improvement in nextImprovements)
                {
                    Console.WriteLine($"  {improvement.Name} (Priority: {improvement.PriorityScore:F2})");
                }
            }

            _logger.LogInformation("Status display completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error showing status");
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task<Dictionary<string, List<TarsEngine.Models.CodeAnalysisResult>>> AnalyzePathAsync(ICodeAnalyzerService codeAnalyzer, string path, bool recursive, string filePattern)
    {
        if (File.Exists(path))
        {
            // Analyze single file
            var result = await codeAnalyzer.AnalyzeFileAsync(path);
            var resultList = new List<TarsEngine.Models.CodeAnalysisResult> { result };
            return new Dictionary<string, List<TarsEngine.Models.CodeAnalysisResult>> { { path, resultList } };
        }
        else if (Directory.Exists(path))
        {
            // Analyze directory
            var results = await codeAnalyzer.AnalyzeDirectoryAsync(path, recursive, filePattern);
            var resultDict = new Dictionary<string, List<TarsEngine.Models.CodeAnalysisResult>>();

            // Create a dictionary with file paths as keys and lists of analysis results as values
            for (int i = 0; i < results.Count; i++)
            {
                var result = results[i];
                var filePath = result.FilePath;
                if (!resultDict.ContainsKey(filePath))
                {
                    resultDict[filePath] = new List<TarsEngine.Models.CodeAnalysisResult>();
                }
                resultDict[filePath].Add(result);
            }

            return resultDict;
        }
        else
        {
            throw new FileNotFoundException($"Path not found: {path}");
        }
    }

    private async Task<Dictionary<string, List<PatternMatch>>> MatchPatternsInPathAsync(IPatternMatcherService patternMatcher, string path, bool recursive, string filePattern, string? language)
    {
        if (File.Exists(path))
        {
            // Match patterns in single file
            var matches = await patternMatcher.FindPatternsInFileAsync(path);
            return new Dictionary<string, List<PatternMatch>> { { path, matches } };
        }
        else if (Directory.Exists(path))
        {
            // Match patterns in directory
            return await patternMatcher.FindPatternsInDirectoryAsync(path, recursive, filePattern);
        }
        else
        {
            throw new FileNotFoundException($"Path not found: {path}");
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
                output = results?.ToString() ?? "No results";
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
