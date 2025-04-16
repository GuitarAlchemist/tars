using System.Diagnostics;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for visualizing the knowledge base
/// </summary>
public class KnowledgeVisualizationCommand : Command
{
    private readonly ILogger<KnowledgeVisualizationCommand> _logger;
    private readonly KnowledgeVisualizationService _knowledgeVisualizationService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeVisualizationCommand"/> class.
    /// </summary>
    public KnowledgeVisualizationCommand(
        ILogger<KnowledgeVisualizationCommand> logger,
        KnowledgeVisualizationService knowledgeVisualizationService,
        ConsoleService consoleService)
        : base("knowledge-viz", "Visualize the knowledge base")
    {
        _logger = logger;
        _knowledgeVisualizationService = knowledgeVisualizationService;
        _consoleService = consoleService;

        // Add options
        var openOption = new Option<bool>(
            ["--open", "-o"],
            "Open the visualization in the default browser");

        // Add options to command
        AddOption(openOption);

        // Set the handler
        this.SetHandler(async (open) =>
        {
            await HandleCommandAsync(open);
        }, openOption);
    }

    /// <summary>
    /// Handles the command execution
    /// </summary>
    private async Task HandleCommandAsync(bool open)
    {
        try
        {
            var success = await _knowledgeVisualizationService.GenerateVisualizationAsync();

            if (success && open)
            {
                var htmlPath = Path.Combine("visualizations", "knowledge_base.html");
                if (File.Exists(htmlPath))
                {
                    _consoleService.WriteInfo($"Opening visualization in browser: {Path.GetFullPath(htmlPath)}");

                    // Open the HTML file in the default browser
                    var processStartInfo = new ProcessStartInfo
                    {
                        FileName = Path.GetFullPath(htmlPath),
                        UseShellExecute = true
                    };
                    Process.Start(processStartInfo);
                }
                else
                {
                    _consoleService.WriteError($"Visualization file not found: {htmlPath}");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing knowledge visualization command");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
