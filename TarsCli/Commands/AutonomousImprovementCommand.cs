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
        AddOption(nameOption);
        AddOption(directoriesOption);
        AddOption(maxDurationOption);
        AddOption(maxImprovementsOption);

        // Set the handler
        this.SetHandler(async (start, stop, status, name, directories, maxDuration, maxImprovements) =>
        {
            await HandleCommandAsync(start, stop, status, name, directories, maxDuration, maxImprovements);
        }, startOption, stopOption, statusOption, nameOption, directoriesOption, maxDurationOption, maxImprovementsOption);
    }

    /// <summary>
    /// Handles the command execution
    /// </summary>
    private async Task HandleCommandAsync(bool start, bool stop, bool status, string name, List<string> directories, int maxDuration, int maxImprovements)
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
                var workflowState = await _autonomousImprovementService.GetWorkflowStatusAsync();
                if (workflowState != null)
                {
                    _consoleService.WriteHeader("Workflow Status");
                    _consoleService.WriteInfo($"Name: {workflowState.Name}");
                    _consoleService.WriteInfo($"Status: {workflowState.Status}");
                    _consoleService.WriteInfo($"Start Time: {workflowState.StartTime}");
                    
                    if (workflowState.EndTime.HasValue)
                    {
                        _consoleService.WriteInfo($"End Time: {workflowState.EndTime.Value}");
                        _consoleService.WriteInfo($"Duration: {(workflowState.EndTime.Value - workflowState.StartTime).TotalMinutes:F2} minutes");
                    }
                    else
                    {
                        _consoleService.WriteInfo($"Duration: {(DateTime.UtcNow - workflowState.StartTime).TotalMinutes:F2} minutes");
                    }
                    
                    _consoleService.WriteInfo($"Target Directories: {string.Join(", ", workflowState.TargetDirectories)}");
                    _consoleService.WriteInfo($"Maximum Duration: {workflowState.MaxDurationMinutes} minutes");
                    
                    _consoleService.WriteInfo("Steps:");
                    foreach (var step in workflowState.Steps)
                    {
                        string status = step.Status.ToString();
                        string duration = "N/A";
                        
                        if (step.StartTime.HasValue && step.EndTime.HasValue)
                        {
                            duration = $"{(step.EndTime.Value - step.StartTime.Value).TotalMinutes:F2} minutes";
                        }
                        
                        _consoleService.WriteInfo($"  - {step.Name}: {status} ({duration})");
                        
                        if (step.Status == TarsEngine.SelfImprovement.StepStatus.Failed && step.ErrorMessage.HasValue)
                        {
                            _consoleService.WriteError($"    Error: {step.ErrorMessage.Value}");
                        }
                    }
                }
                else
                {
                    _consoleService.WriteWarning("No workflow status found");
                }
            }
            // Show help
            else
            {
                _consoleService.WriteInfo("Please specify an action:");
                _consoleService.WriteInfo("  --start, -s    Start the autonomous improvement workflow");
                _consoleService.WriteInfo("  --stop, -x     Stop the autonomous improvement workflow");
                _consoleService.WriteInfo("  --status, -t   Get the status of the autonomous improvement workflow");
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
