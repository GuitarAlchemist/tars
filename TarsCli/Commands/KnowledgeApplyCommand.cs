using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for applying knowledge to improve files
/// </summary>
public class KnowledgeApplyCommand : Command
{
    private ILogger<KnowledgeApplyCommand> _logger;
    private KnowledgeApplicationService _knowledgeApplicationService;
    private ConsoleService _consoleService;

    /// <summary>
    /// Create a new knowledge-apply command
    /// </summary>
    public KnowledgeApplyCommand() : base("knowledge-apply", "Apply knowledge from the knowledge base to improve files")
    {
        // Add options
        var fileOption = new Option<string>(
            aliases: ["--file", "-f"],
            description: "The file to improve");

        var directoryOption = new Option<string>(
            aliases: ["--directory", "-d"],
            description: "The directory containing files to improve");

        var patternOption = new Option<string>(
            aliases: ["--pattern", "-p"],
            description: "The file pattern to match (e.g., *.cs)",
            getDefaultValue: () => "*.*");

        var recursiveOption = new Option<bool>(
            aliases: ["--recursive", "-r"],
            description: "Search for files recursively",
            getDefaultValue: () => false);

        var modelOption = new Option<string>(
            aliases: ["--model", "-m"],
            description: "Model to use for improvement",
            getDefaultValue: () => "llama3");

        var reportOption = new Option<bool>(
            aliases: ["--report"],
            description: "Generate a knowledge report",
            getDefaultValue: () => false);

        var extractOption = new Option<string>(
            aliases: ["--extract", "-e"],
            description: "Extract knowledge from a file");

        // Add options to command
        AddOption(fileOption);
        AddOption(directoryOption);
        AddOption(patternOption);
        AddOption(recursiveOption);
        AddOption(modelOption);
        AddOption(reportOption);
        AddOption(extractOption);

        // Set handler
        this.SetHandler(async (string file, string directory, string pattern, bool recursive, string model, bool report, string extract) =>
        {
            try
            {
                // Display header
                _consoleService?.WriteHeader("=== TARS Knowledge Application ===");

                // Generate a knowledge report if requested
                if (report)
                {
                    _consoleService?.WriteInfo("Generating knowledge report...");
                    var reportPath = await _knowledgeApplicationService.GenerateKnowledgeReportAsync();

                    if (!string.IsNullOrEmpty(reportPath))
                    {
                        _consoleService?.WriteSuccess("Knowledge report generated successfully");
                        _consoleService?.WriteInfo($"Report saved to: {Path.GetFullPath(reportPath)}");
                    }
                    else
                    {
                        _consoleService?.WriteError("Failed to generate knowledge report");
                    }

                    return;
                }

                // Extract knowledge from a file if requested
                if (!string.IsNullOrEmpty(extract))
                {
                    if (!File.Exists(extract))
                    {
                        _consoleService?.WriteError($"File not found: {extract}");
                        return;
                    }

                    _consoleService?.WriteInfo($"Extracting knowledge from: {Path.GetFullPath(extract)}");
                    var knowledge = await _knowledgeApplicationService.ExtractKnowledgeAsync(extract, model);

                    _consoleService?.WriteSuccess("Knowledge extracted successfully");
                    _consoleService?.WriteInfo($"Title: {knowledge.Title}");
                    _consoleService?.WriteInfo($"Summary: {knowledge.Summary}");
                    _consoleService?.WriteInfo($"Key concepts: {knowledge.KeyConcepts?.Count ?? 0}");
                    _consoleService?.WriteInfo($"Insights: {knowledge.Insights?.Count ?? 0}");

                    return;
                }

                // Apply knowledge to a specific file
                if (!string.IsNullOrEmpty(file))
                {
                    if (!File.Exists(file))
                    {
                        _consoleService?.WriteError($"File not found: {file}");
                        return;
                    }

                    _consoleService?.WriteInfo($"Applying knowledge to: {Path.GetFullPath(file)}");
                    var result = await _knowledgeApplicationService.ApplyKnowledgeToFileAsync(file, model);

                    if (result)
                    {
                        _consoleService?.WriteSuccess("File improved successfully");
                    }
                    else
                    {
                        _consoleService?.WriteInfo("No improvements needed for this file");
                    }

                    return;
                }

                // Apply knowledge to files in a directory
                if (!string.IsNullOrEmpty(directory))
                {
                    if (!Directory.Exists(directory))
                    {
                        _consoleService?.WriteError($"Directory not found: {directory}");
                        return;
                    }

                    _consoleService?.WriteInfo($"Applying knowledge to files in: {Path.GetFullPath(directory)}");
                    _consoleService?.WriteInfo($"Pattern: {pattern}");
                    _consoleService?.WriteInfo($"Recursive: {recursive}");

                    // Get all matching files
                    var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
                    var files = Directory.GetFiles(directory, pattern, searchOption);

                    _consoleService?.WriteInfo($"Found {files.Length} files");

                    // Apply knowledge to each file
                    var improvedCount = 0;
                    foreach (var f in files)
                    {
                        _consoleService?.WriteInfo($"Processing: {Path.GetFileName(f)}");
                        var result = await _knowledgeApplicationService.ApplyKnowledgeToFileAsync(f, model);

                        if (result)
                        {
                            improvedCount++;
                        }
                    }

                    _consoleService?.WriteSuccess($"Improved {improvedCount} out of {files.Length} files");
                    return;
                }

                // If no options were provided, show help
                _consoleService?.WriteInfo("Please specify a file, directory, or other option");
                _consoleService?.WriteInfo("Use --help for more information");
            }
            catch (Exception ex)
            {
                // Handle errors
                _logger?.LogError(ex, "Error applying knowledge");
                _consoleService?.WriteError($"Error applying knowledge: {ex.Message}");
            }
        }, fileOption, directoryOption, patternOption, recursiveOption, modelOption, reportOption, extractOption);
    }

    /// <summary>
    /// Set the required services
    /// </summary>
    public void SetServices(ILogger<KnowledgeApplyCommand> logger, KnowledgeApplicationService knowledgeApplicationService, ConsoleService consoleService)
    {
        _logger = logger;
        _knowledgeApplicationService = knowledgeApplicationService;
        _consoleService = consoleService;
    }
}
