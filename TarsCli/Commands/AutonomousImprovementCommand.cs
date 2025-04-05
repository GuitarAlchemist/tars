using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for autonomous improvement of the codebase
/// </summary>
public class AutonomousImprovementCommand : Command
{
    private readonly ILogger<AutonomousImprovementCommand> _logger;
    private readonly AutonomousImprovementService _autonomousImprovementService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="AutonomousImprovementCommand"/> class.
    /// </summary>
    public AutonomousImprovementCommand(
        ILogger<AutonomousImprovementCommand> logger,
        AutonomousImprovementService autonomousImprovementService,
        ConsoleService consoleService)
        : base("auto-improve-workflow", "Run autonomous improvement workflow")
    {
        _logger = logger;
        _autonomousImprovementService = autonomousImprovementService;
        _consoleService = consoleService;

        // Add options
        var startOption = new Option<bool>(
            ["--start", "-s"],
            "Start the autonomous improvement workflow");

        var stopOption = new Option<bool>(
            ["--stop", "-x"],
            "Stop the autonomous improvement workflow");

        var statusOption = new Option<bool>(
            ["--status", "-t"],
            "Get the status of the autonomous improvement workflow");

        var reportOption = new Option<bool>(
            ["--report", "-r"],
            "Get a detailed report of the autonomous improvement workflow");

        var nameOption = new Option<string>(
            ["--name", "-n"],
            () => "TARS Autonomous Improvement",
            "The name of the workflow");

        var directoriesOption = new Option<List<string>>(
            ["--directories", "-d"],
            "The target directories to improve");

        var maxDurationOption = new Option<int>(
            ["--max-duration", "-m"],
            () => 60,
            "The maximum duration of the workflow in minutes");

        var maxImprovementsOption = new Option<int>(
            ["--max-improvements", "-i"],
            () => 5,
            "The maximum number of improvements to apply");

        // Add options to command
        AddOption(startOption);
        AddOption(stopOption);
        AddOption(statusOption);
        AddOption(reportOption);
        AddOption(nameOption);
        AddOption(directoriesOption);
        AddOption(maxDurationOption);
        AddOption(maxImprovementsOption);

        // Set the handler
        this.SetHandler(async (start, stop, status, report, name, directories, maxDuration, maxImprovements) =>
        {
            await HandleCommandAsync(start, stop, status, report, name, directories, maxDuration, maxImprovements);
        }, startOption, stopOption, statusOption, reportOption, nameOption, directoriesOption, maxDurationOption, maxImprovementsOption);
    }

    /// <summary>
    /// Handles the command execution
    /// </summary>
    private async Task HandleCommandAsync(bool start, bool stop, bool status, bool report, string name, List<string>? directories, int maxDuration, int maxImprovements)
    {
        try
        {
            // Check if multiple actions were specified
            if ((start ? 1 : 0) + (stop ? 1 : 0) + (status ? 1 : 0) > 1)
            {
                _consoleService.WriteError("Please specify only one action: --start, --stop, or --status");
                return;
            }

            // Start the workflow
            if (start)
            {
                // Check if directories were specified
                if (directories == null || directories.Count == 0)
                {
                    _consoleService.WriteError("Please specify at least one target directory with --directories");
                    return;
                }

                // Start the workflow
                bool success = await _autonomousImprovementService.StartWorkflowAsync(name, directories, maxDuration, maxImprovements);
                if (success)
                {
                    _consoleService.WriteSuccess($"Started workflow: {name}");
                    _consoleService.WriteInfo("Use --status to check the status of the workflow");
                    _consoleService.WriteInfo("Use --stop to stop the workflow");
                }
            }
            // Stop the workflow
            else if (stop)
            {
                bool success = _autonomousImprovementService.StopWorkflow();
                if (success)
                {
                    _consoleService.WriteSuccess("Workflow stop requested");
                    _consoleService.WriteInfo("The workflow will stop after the current step completes");
                }
            }
            // Get the status of the workflow
            else if (status)
            {
                var workflowStatus = await _autonomousImprovementService.GetWorkflowStatusAsync();
                _consoleService.WriteHeader("Workflow Status");
                _consoleService.WriteInfo($"Status: {workflowStatus}");
            }
            // Get the report of the workflow
            else if (report)
            {
                var workflowReport = await _autonomousImprovementService.GetWorkflowReportAsync();
                _consoleService.WriteHeader("Workflow Report");
                _consoleService.WriteInfo(workflowReport);
            }
            // Show help
            else
            {
                _consoleService.WriteInfo("Please specify an action:");
                _consoleService.WriteInfo("  --start, -s    Start the autonomous improvement workflow");
                _consoleService.WriteInfo("  --stop, -x     Stop the autonomous improvement workflow");
                _consoleService.WriteInfo("  --status, -t   Get the status of the autonomous improvement workflow");
                _consoleService.WriteInfo("  --report, -r   Get a detailed report of the autonomous improvement workflow");
                _consoleService.WriteInfo("");
                _consoleService.WriteInfo("When starting a workflow, you can specify:");
                _consoleService.WriteInfo("  --name, -n             The name of the workflow (default: TARS Autonomous Improvement)");
                _consoleService.WriteInfo("  --directories, -d      The target directories to improve (required)");
                _consoleService.WriteInfo("  --max-duration, -m     The maximum duration in minutes (default: 60)");
                _consoleService.WriteInfo("  --max-improvements, -i The maximum number of improvements to apply (default: 5)");
                _consoleService.WriteInfo("");
                _consoleService.WriteInfo("Example:");
                _consoleService.WriteInfo("  tars auto-improve-workflow --start --directories docs/Explorations/v1/Chats docs/Explorations/Reflections");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing autonomous improvement command");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
