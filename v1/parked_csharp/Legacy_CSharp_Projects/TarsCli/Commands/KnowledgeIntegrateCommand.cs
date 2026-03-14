using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for integrating knowledge with other TARS systems
/// </summary>
public class KnowledgeIntegrateCommand : Command
{
    private readonly ILogger<KnowledgeIntegrateCommand> _logger;
    private readonly KnowledgeIntegrationService _knowledgeIntegrationService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Create a new knowledge-integrate command
    /// </summary>
    public KnowledgeIntegrateCommand(
        ILogger<KnowledgeIntegrateCommand> logger,
        KnowledgeIntegrationService knowledgeIntegrationService,
        ConsoleService consoleService)
        : base("knowledge-integrate", "Integrate knowledge with other TARS systems")
    {
        _logger = logger;
        _knowledgeIntegrationService = knowledgeIntegrationService;
        _consoleService = consoleService;

        // Add options
        var explorationOption = new Option<string>(
            aliases: ["--exploration", "-e"],
            description: "The directory containing exploration files to extract knowledge from");

        var targetOption = new Option<string>(
            aliases: ["--target", "-t"],
            description: "The directory to target with improvements");

        var patternOption = new Option<string>(
            aliases: ["--pattern", "-p"],
            description: "The file pattern to match (e.g., *.cs)",
            getDefaultValue: () => "*.cs");

        var modelOption = new Option<string>(
            aliases: ["--model", "-m"],
            description: "Model to use for improvement",
            getDefaultValue: () => "llama3");

        var metascriptOption = new Option<bool>(
            aliases: ["--metascript"],
            description: "Generate a knowledge metascript",
            getDefaultValue: () => false);

        var cycleOption = new Option<bool>(
            aliases: ["--cycle", "-c"],
            description: "Run a complete knowledge improvement cycle",
            getDefaultValue: () => false);

        var retroactionOption = new Option<bool>(
            aliases: ["--retroaction", "-r"],
            description: "Generate a retroaction report",
            getDefaultValue: () => false);

        // Add options to command
        AddOption(explorationOption);
        AddOption(targetOption);
        AddOption(patternOption);
        AddOption(modelOption);
        AddOption(metascriptOption);
        AddOption(cycleOption);
        AddOption(retroactionOption);

        // Set handler
        this.SetHandler(async (string exploration, string target, string pattern, string model, bool metascript, bool cycle, bool retroaction) =>
        {
            try
            {
                // Display header
                _consoleService.WriteHeader("=== TARS Knowledge Integration ===");

                // Generate a knowledge metascript if requested
                if (metascript)
                {
                    if (string.IsNullOrEmpty(target))
                    {
                        _consoleService.WriteError("Target directory is required for generating a metascript");
                        return;
                    }

                    _consoleService.WriteInfo($"Generating knowledge metascript for: {Path.GetFullPath(target)}");
                    _consoleService.WriteInfo($"Pattern: {pattern}");
                    _consoleService.WriteInfo($"Model: {model}");

                    var metascriptPath = await _knowledgeIntegrationService.GenerateKnowledgeMetascriptAsync(target, pattern, model);

                    if (!string.IsNullOrEmpty(metascriptPath))
                    {
                        _consoleService.WriteSuccess("Knowledge metascript generated successfully");
                        _consoleService.WriteInfo($"Metascript saved to: {Path.GetFullPath(metascriptPath)}");
                    }
                    else
                    {
                        _consoleService.WriteError("Failed to generate knowledge metascript");
                    }

                    return;
                }

                // Run a knowledge improvement cycle if requested
                if (cycle)
                {
                    if (string.IsNullOrEmpty(exploration) || string.IsNullOrEmpty(target))
                    {
                        _consoleService.WriteError("Exploration and target directories are required for running a knowledge improvement cycle");
                        return;
                    }

                    _consoleService.WriteInfo($"Running knowledge improvement cycle");
                    _consoleService.WriteInfo($"Exploration directory: {Path.GetFullPath(exploration)}");
                    _consoleService.WriteInfo($"Target directory: {Path.GetFullPath(target)}");
                    _consoleService.WriteInfo($"Pattern: {pattern}");
                    _consoleService.WriteInfo($"Model: {model}");

                    var reportPath = await _knowledgeIntegrationService.RunKnowledgeImprovementCycleAsync(exploration, target, pattern, model);

                    if (!string.IsNullOrEmpty(reportPath))
                    {
                        _consoleService.WriteSuccess("Knowledge improvement cycle completed successfully");
                        _consoleService.WriteInfo($"Report saved to: {Path.GetFullPath(reportPath)}");
                    }
                    else
                    {
                        _consoleService.WriteError("Failed to run knowledge improvement cycle");
                    }

                    return;
                }

                // Generate a retroaction report if requested
                if (retroaction)
                {
                    if (string.IsNullOrEmpty(exploration) || string.IsNullOrEmpty(target))
                    {
                        _consoleService.WriteError("Exploration and target directories are required for generating a retroaction report");
                        return;
                    }

                    _consoleService.WriteInfo($"Generating retroaction report");
                    _consoleService.WriteInfo($"Exploration directory: {Path.GetFullPath(exploration)}");
                    _consoleService.WriteInfo($"Target directory: {Path.GetFullPath(target)}");
                    _consoleService.WriteInfo($"Model: {model}");

                    var reportPath = await _knowledgeIntegrationService.GenerateRetroactionReportAsync(exploration, target, model);

                    if (!string.IsNullOrEmpty(reportPath))
                    {
                        _consoleService.WriteSuccess("Retroaction report generated successfully");
                        _consoleService.WriteInfo($"Report saved to: {Path.GetFullPath(reportPath)}");
                    }
                    else
                    {
                        _consoleService.WriteError("Failed to generate retroaction report");
                    }

                    return;
                }

                // If no options were provided, show help
                _consoleService.WriteInfo("Please specify an operation to perform");
                _consoleService.WriteInfo("Use --help for more information");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in knowledge integration");
                _consoleService.WriteError($"Error: {ex.Message}");
            }
        }, explorationOption, targetOption, patternOption, modelOption, metascriptOption, cycleOption, retroactionOption);
    }
}
