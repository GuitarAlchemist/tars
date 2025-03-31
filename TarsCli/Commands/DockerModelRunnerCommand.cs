using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Constants;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for interacting with Docker Model Runner
/// </summary>
public class DockerModelRunnerCommand : Command
{
    /// <summary>
    /// Create a new Docker Model Runner command
    /// </summary>
    public DockerModelRunnerCommand() : base("docker-model-runner", "Interact with Docker Model Runner")
    {
        AddAlias("dmr");

        // Add subcommands
        AddCommand(new ListModelsCommand());
        AddCommand(new StatusCommand());
    }

    /// <summary>
    /// Command to list available models
    /// </summary>
    private class ListModelsCommand : Command
    {
        public ListModelsCommand() : base("list", "List available models")
        {
            AddAlias("ls");

            this.SetHandler(async (context) =>
            {
                var serviceProvider = context.BindingContext.GetService<IServiceProvider>()
                    ?? throw new InvalidOperationException("Service provider not found");
                var logger = serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var consoleService = serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Fetching available models from Docker Model Runner...");

                    var isAvailable = await dockerModelRunnerService.IsAvailable();
                    if (!isAvailable)
                    {
                        consoleService.WriteError("Docker Model Runner is not available. Make sure it's running and accessible.");
                        context.ExitCode = 1;
                        return;
                    }

                    var models = await dockerModelRunnerService.GetAvailableModels();

                    if (models.Count == 0)
                    {
                        consoleService.WriteWarning("No models found. You may need to pull models first.");
                        context.ExitCode = 0;
                        return;
                    }

                    consoleService.WriteSuccess($"Found {models.Count} models:");

                    // Display models in a table format
                    consoleService.WriteTable(
                        new[] { "ID", "Owner", "Created" },
                        models.Select(m => new[]
                        {
                            m.Id,
                            m.OwnedBy,
                            DateTimeOffset.FromUnixTimeSeconds(m.Created).ToString("yyyy-MM-dd HH:mm:ss")
                        })
                    );

                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error listing models");
                    consoleService.WriteError($"Error listing models: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to check the status of Docker Model Runner
    /// </summary>
    private class StatusCommand : Command
    {
        public StatusCommand() : base("status", "Check the status of Docker Model Runner")
        {
            this.SetHandler(async (context) =>
            {
                var serviceProvider = context.BindingContext.GetService<IServiceProvider>()
                    ?? throw new InvalidOperationException("Service provider not found");
                var logger = serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var gpuService = serviceProvider.GetRequiredService<GpuService>();
                var consoleService = serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Checking Docker Model Runner status...");

                    var isAvailable = await dockerModelRunnerService.IsAvailable();

                    if (isAvailable)
                    {
                        consoleService.WriteSuccess("Docker Model Runner is available");
                        consoleService.WriteInfo($"Base URL: {dockerModelRunnerService.BaseUrl}");
                        consoleService.WriteInfo($"Default model: {dockerModelRunnerService.DefaultModel}");

                        // Check GPU status
                        var isGpuAvailable = gpuService.IsGpuAvailable();
                        if (isGpuAvailable)
                        {
                            consoleService.WriteSuccess("GPU acceleration is available");

                            var gpuInfo = gpuService.GetGpuInfo();
                            foreach (var gpu in gpuInfo.Where(g => gpuService.IsGpuCompatible(g)))
                            {
                                consoleService.WriteInfo($"Compatible GPU: {gpu.Name} with {gpu.MemoryMB}MB memory");
                            }
                        }
                        else
                        {
                            consoleService.WriteWarning("GPU acceleration is not available");
                        }

                        // Get available models
                        var models = await dockerModelRunnerService.GetAvailableModels();
                        consoleService.WriteInfo($"Available models: {models.Count}");

                        context.ExitCode = 0;
                    }
                    else
                    {
                        consoleService.WriteError("Docker Model Runner is not available");
                        consoleService.WriteInfo("Make sure Docker Desktop is running and Docker Model Runner is enabled");
                        consoleService.WriteInfo("You can enable Docker Model Runner in Docker Desktop settings");

                        context.ExitCode = 1;
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error checking Docker Model Runner status");
                    consoleService.WriteError($"Error checking Docker Model Runner status: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }
}
