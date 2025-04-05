using System.CommandLine;
using Microsoft.Extensions.Logging;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for autonomous improvement of TARS
/// </summary>
public class AutonomousCommand : Command
{
    private readonly ILogger<AutonomousCommand> _logger;
    private readonly AutonomousImprovementService _autonomousImprovementService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Create a new autonomous command
    /// </summary>
    public AutonomousCommand(
        ILogger<AutonomousCommand> logger,
        AutonomousImprovementService autonomousImprovementService,
        ConsoleService consoleService)
        : base("autonomous", "Autonomous improvement of TARS")
    {
        _logger = logger;
        _autonomousImprovementService = autonomousImprovementService;
        _consoleService = consoleService;

        // Add subcommands
        AddCommand(new AutonomousStartCommand(logger, autonomousImprovementService, consoleService));
        AddCommand(new AutonomousStopCommand(logger, autonomousImprovementService, consoleService));
        AddCommand(new AutonomousStatusCommand(logger, autonomousImprovementService, consoleService));
    }

    /// <summary>
    /// Command for starting autonomous improvement
    /// </summary>
    private class AutonomousStartCommand : Command
    {
        private readonly ILogger<AutonomousCommand> _logger;
        private readonly AutonomousImprovementService _autonomousImprovementService;
        private readonly ConsoleService _consoleService;

        public AutonomousStartCommand(
            ILogger<AutonomousCommand> logger,
            AutonomousImprovementService autonomousImprovementService,
            ConsoleService consoleService)
            : base("start", "Start autonomous improvement")
        {
            _logger = logger;
            _autonomousImprovementService = autonomousImprovementService;
            _consoleService = consoleService;

            // Add options
            var explorationOption = new Option<string[]>(
                aliases: ["--exploration", "-e"],
                description: "The directories containing exploration files to extract knowledge from")
            {
                AllowMultipleArgumentsPerToken = true
            };

            var targetOption = new Option<string[]>(
                aliases: ["--target", "-t"],
                description: "The directories to target with improvements")
            {
                AllowMultipleArgumentsPerToken = true
            };

            var durationOption = new Option<int>(
                aliases: ["--duration", "-d"],
                description: "Duration of the improvement process in minutes",
                getDefaultValue: () => 60);

            var modelOption = new Option<string>(
                aliases: ["--model", "-m"],
                description: "Model to use for improvement",
                getDefaultValue: () => "llama3");

            var autoCommitOption = new Option<bool>(
                aliases: ["--auto-commit", "-c"],
                description: "Whether to automatically commit improvements",
                getDefaultValue: () => false);

            var createPrOption = new Option<bool>(
                aliases: ["--create-pr", "-p"],
                description: "Whether to create a pull request for improvements",
                getDefaultValue: () => false);

            // Add options to command
            AddOption(explorationOption);
            AddOption(targetOption);
            AddOption(durationOption);
            AddOption(modelOption);
            AddOption(autoCommitOption);
            AddOption(createPrOption);

            // Set handler
            this.SetHandler(async (string[] exploration, string[] target, int duration, string model, bool autoCommit, bool createPr) =>
            {
                try
                {
                    // Display header
                    _consoleService.WriteHeader("=== TARS Autonomous Improvement ===");

                    // Validate options
                    if (exploration == null || exploration.Length == 0)
                    {
                        _consoleService.WriteError("At least one exploration directory is required");
                        return;
                    }

                    if (target == null || target.Length == 0)
                    {
                        _consoleService.WriteError("At least one target directory is required");
                        return;
                    }

                    // Start autonomous improvement
                    _consoleService.WriteInfo($"Starting autonomous improvement for {duration} minutes");
                    _consoleService.WriteInfo($"Exploration directories: {string.Join(", ", exploration)}");
                    _consoleService.WriteInfo($"Target directories: {string.Join(", ", target)}");
                    _consoleService.WriteInfo($"Model: {model}");
                    _consoleService.WriteInfo($"Auto-commit: {autoCommit}");
                    _consoleService.WriteInfo($"Create PR: {createPr}");

                    var reportPath = await _autonomousImprovementService.StartAutonomousImprovementAsync(
                        exploration.ToList(),
                        target.ToList(),
                        duration,
                        model,
                        autoCommit,
                        createPr);

                    if (!string.IsNullOrEmpty(reportPath))
                    {
                        _consoleService.WriteSuccess("Autonomous improvement completed successfully");
                        _consoleService.WriteInfo($"Report saved to: {Path.GetFullPath(reportPath)}");
                    }
                    else
                    {
                        _consoleService.WriteError("Autonomous improvement failed");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error starting autonomous improvement");
                    _consoleService.WriteError($"Error: {ex.Message}");
                }
            }, explorationOption, targetOption, durationOption, modelOption, autoCommitOption, createPrOption);
        }
    }

    /// <summary>
    /// Command for stopping autonomous improvement
    /// </summary>
    private class AutonomousStopCommand : Command
    {
        private readonly ILogger<AutonomousCommand> _logger;
        private readonly AutonomousImprovementService _autonomousImprovementService;
        private readonly ConsoleService _consoleService;

        public AutonomousStopCommand(
            ILogger<AutonomousCommand> logger,
            AutonomousImprovementService autonomousImprovementService,
            ConsoleService consoleService)
            : base("stop", "Stop autonomous improvement")
        {
            _logger = logger;
            _autonomousImprovementService = autonomousImprovementService;
            _consoleService = consoleService;

            // Set handler
            this.SetHandler(() =>
            {
                try
                {
                    // Display header
                    _consoleService.WriteHeader("=== TARS Autonomous Improvement ===");

                    // Stop autonomous improvement
                    _consoleService.WriteInfo("Stopping autonomous improvement");
                    _autonomousImprovementService.StopAutonomousImprovement();
                    _consoleService.WriteSuccess("Autonomous improvement stopped");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error stopping autonomous improvement");
                    _consoleService.WriteError($"Error: {ex.Message}");
                }
            });
        }
    }

    /// <summary>
    /// Command for getting autonomous improvement status
    /// </summary>
    private class AutonomousStatusCommand : Command
    {
        private readonly ILogger<AutonomousCommand> _logger;
        private readonly AutonomousImprovementService _autonomousImprovementService;
        private readonly ConsoleService _consoleService;

        public AutonomousStatusCommand(
            ILogger<AutonomousCommand> logger,
            AutonomousImprovementService autonomousImprovementService,
            ConsoleService consoleService)
            : base("status", "Get autonomous improvement status")
        {
            _logger = logger;
            _autonomousImprovementService = autonomousImprovementService;
            _consoleService = consoleService;

            // Set handler
            this.SetHandler(() =>
            {
                try
                {
                    // Display header
                    _consoleService.WriteHeader("=== TARS Autonomous Improvement Status ===");

                    // Get status
                    var status = _autonomousImprovementService.GetStatus();

                    if (status.IsRunning)
                    {
                        _consoleService.WriteInfo("Autonomous improvement is running");
                        _consoleService.WriteInfo($"Start time: {status.StartTime}");
                        _consoleService.WriteInfo($"End time: {status.EndTime}");
                        _consoleService.WriteInfo($"Elapsed time: {status.ElapsedTime.TotalMinutes:F2} minutes");
                        _consoleService.WriteInfo($"Remaining time: {status.RemainingTime.TotalMinutes:F2} minutes");
                    }
                    else
                    {
                        _consoleService.WriteInfo("Autonomous improvement is not running");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error getting autonomous improvement status");
                    _consoleService.WriteError($"Error: {ex.Message}");
                }
            });
        }
    }
}
