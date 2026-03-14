using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for extracting knowledge from documentation and integrating it with the RetroactionLoop
/// </summary>
public class DocumentationKnowledgeCommand : Command
{
    private readonly ILogger<DocumentationKnowledgeCommand> _logger;
    private readonly DocumentationKnowledgeService _documentationKnowledgeService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="DocumentationKnowledgeCommand"/> class.
    /// </summary>
    public DocumentationKnowledgeCommand(
        ILogger<DocumentationKnowledgeCommand> logger,
        DocumentationKnowledgeService documentationKnowledgeService,
        ConsoleService consoleService)
        : base("doc-extract", "Extract knowledge from documentation and integrate it with the RetroactionLoop")
    {
        _logger = logger;
        _documentationKnowledgeService = documentationKnowledgeService;
        _consoleService = consoleService;

        // Add options
        var extractOption = new Option<bool>(
            ["--extract", "-e"],
            "Extract knowledge from documentation");

        var importOption = new Option<bool>(
            ["--import", "-i"],
            "Import patterns to RetroactionLoop");

        var statsOption = new Option<bool>(
            ["--stats", "-s"],
            "Show knowledge base statistics");

        var maxFilesOption = new Option<int>(
            ["--max-files", "-m"],
            () => 5,
            "Maximum number of files to process");

        var fullOption = new Option<bool>(
            ["--full", "-f"],
            "Run the full knowledge extraction and integration process");

        // Add options to command
        AddOption(extractOption);
        AddOption(importOption);
        AddOption(statsOption);
        AddOption(maxFilesOption);
        AddOption(fullOption);

        // Set the handler
        this.SetHandler(async (extract, import, stats, maxFiles, full) =>
        {
            await HandleCommandAsync(extract, import, stats, maxFiles, full);
        }, extractOption, importOption, statsOption, maxFilesOption, fullOption);
    }

    /// <summary>
    /// Handles the command execution
    /// </summary>
    private async Task HandleCommandAsync(bool extract, bool import, bool stats, int maxFiles, bool full)
    {
        try
        {
            // If no options are specified, show help
            if (!extract && !import && !stats && !full)
            {
                _consoleService.WriteInfo("Please specify an option:");
                _consoleService.WriteInfo("  --extract, -e    Extract knowledge from documentation");
                _consoleService.WriteInfo("  --import, -i     Import patterns to RetroactionLoop");
                _consoleService.WriteInfo("  --stats, -s      Show knowledge base statistics");
                _consoleService.WriteInfo("  --full, -f       Run the full knowledge extraction and integration process");
                _consoleService.WriteInfo("  --max-files, -m  Maximum number of files to process (default: 5)");
                return;
            }

            // Run the full process if requested
            if (full)
            {
                await _documentationKnowledgeService.RunFullKnowledgeIntegrationAsync(maxFiles);
                return;
            }

            // Extract knowledge if requested
            if (extract)
            {
                await _documentationKnowledgeService.ExtractKnowledgeAsync(maxFiles);
            }

            // Import patterns if requested
            if (import)
            {
                await _documentationKnowledgeService.ImportPatternsToRetroactionLoopAsync();
            }

            // Show statistics if requested
            if (stats)
            {
                var statistics = await _documentationKnowledgeService.GetKnowledgeBaseStatisticsAsync();
                _consoleService.WriteHeader("TARS Documentation Knowledge Base Statistics");

                if (!statistics.Exists)
                {
                    _consoleService.WriteInfo("Knowledge base does not exist");
                    return;
                }

                _consoleService.WriteInfo($"Total entries: {statistics.TotalEntries}");
                _consoleService.WriteInfo($"Patterns: {statistics.Patterns}");
                _consoleService.WriteInfo($"Best practices: {statistics.BestPractices}");
                _consoleService.WriteInfo($"Code examples: {statistics.CodeExamples}");
                _consoleService.WriteInfo($"Improvement strategies: {statistics.ImprovementStrategies}");
                _consoleService.WriteInfo($"Architecture insights: {statistics.ArchitectureInsights}");
                _consoleService.WriteInfo($"Last updated: {statistics.LastUpdated}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing documentation knowledge command");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
