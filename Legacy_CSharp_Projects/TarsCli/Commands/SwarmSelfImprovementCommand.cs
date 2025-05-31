using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for managing the swarm self-improvement process
/// </summary>
public class SwarmSelfImprovementCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the SwarmSelfImprovementCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public SwarmSelfImprovementCommand(IServiceProvider serviceProvider) : base("swarm-improve", "Manage swarm self-improvement process")
    {
        _serviceProvider = serviceProvider;

        // Add subcommands
        AddCommand(new StartCommand(_serviceProvider));
        AddCommand(new StopCommand(_serviceProvider));
        AddCommand(new StatusCommand(_serviceProvider));
    }

    /// <summary>
    /// Command to start the swarm self-improvement process
    /// </summary>
    private class StartCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StartCommand(IServiceProvider serviceProvider) : base("start", "Start the swarm self-improvement process")
        {
            _serviceProvider = serviceProvider;

            // Add arguments and options
            var targetOption = new Option<string[]>("--target", "Target directories to improve");
            targetOption.AddAlias("-t");
            targetOption.IsRequired = true;

            var agentCountOption = new Option<int>("--agent-count", () => 3, "Number of agents to create");
            agentCountOption.AddAlias("-a");

            var modelOption = new Option<string>("--model", () => "llama3", "Model to use for improvement");
            modelOption.AddAlias("-m");

            AddOption(targetOption);
            AddOption(agentCountOption);
            AddOption(modelOption);

            this.SetHandler(async (string[] targets, int agentCount, string model) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SwarmSelfImprovementCommand>>();
                var swarmImprovementService = _serviceProvider.GetRequiredService<SwarmSelfImprovementService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Starting swarm self-improvement process with {agentCount} agents");
                    consoleService.WriteInfo($"Target directories: {string.Join(", ", targets)}");
                    consoleService.WriteInfo($"Model: {model}");

                    // Start the improvement process
                    var result = await swarmImprovementService.StartImprovementAsync(targets.ToList(), agentCount, model);

                    if (result)
                    {
                        consoleService.WriteSuccess("Swarm self-improvement process started successfully");
                        consoleService.WriteInfo("Use 'swarm-improve status' to check the status of the process");
                        consoleService.WriteInfo("Use 'swarm-improve stop' to stop the process");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError("Failed to start swarm self-improvement process");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error starting swarm self-improvement process");
                    consoleService.WriteError($"Error starting swarm self-improvement process: {ex.Message}");
                    // Failure
                }
            }, targetOption, agentCountOption, modelOption);
        }
    }

    /// <summary>
    /// Command to stop the swarm self-improvement process
    /// </summary>
    private class StopCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StopCommand(IServiceProvider serviceProvider) : base("stop", "Stop the swarm self-improvement process")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(async () =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SwarmSelfImprovementCommand>>();
                var swarmImprovementService = _serviceProvider.GetRequiredService<SwarmSelfImprovementService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Stopping swarm self-improvement process");

                    // Stop the improvement process
                    var result = await swarmImprovementService.StopImprovementAsync();

                    if (result)
                    {
                        consoleService.WriteSuccess("Swarm self-improvement process stopped successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError("Failed to stop swarm self-improvement process");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error stopping swarm self-improvement process");
                    consoleService.WriteError($"Error stopping swarm self-improvement process: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to get the status of the swarm self-improvement process
    /// </summary>
    private class StatusCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StatusCommand(IServiceProvider serviceProvider) : base("status", "Get the status of the swarm self-improvement process")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(() =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SwarmSelfImprovementCommand>>();
                var swarmImprovementService = _serviceProvider.GetRequiredService<SwarmSelfImprovementService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Getting status of swarm self-improvement process");

                    // Get the status
                    var status = swarmImprovementService.GetStatus();

                    consoleService.WriteInfo("Swarm self-improvement status:");
                    consoleService.WriteInfo(JsonSerializer.Serialize(status, new JsonSerializerOptions { WriteIndented = true }));

                    // Success
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error getting status of swarm self-improvement process");
                    consoleService.WriteError($"Error getting status of swarm self-improvement process: {ex.Message}");
                    // Failure
                }
            });
        }
    }
}