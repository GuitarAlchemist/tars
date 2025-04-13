using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for managing the self-coding process
/// </summary>
public class SelfCodingCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the SelfCodingCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public SelfCodingCommand(IServiceProvider serviceProvider) : base("self-code", "Manage self-coding process")
    {
        _serviceProvider = serviceProvider;

        // Add subcommands
        AddCommand(new StartCommand(_serviceProvider));
        AddCommand(new StopCommand(_serviceProvider));
        AddCommand(new StatusCommand(_serviceProvider));
        AddCommand(new SetupCommand(_serviceProvider));
        AddCommand(new CleanupCommand(_serviceProvider));
    }

    /// <summary>
    /// Command to start the self-coding process
    /// </summary>
    private class StartCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StartCommand(IServiceProvider serviceProvider) : base("start", "Start the self-coding process")
        {
            _serviceProvider = serviceProvider;

            // Add arguments and options
            var targetOption = new Option<string[]>("--target", "Target directories to improve");
            targetOption.AddAlias("-t");
            targetOption.IsRequired = true;

            var autoApplyOption = new Option<bool>("--auto-apply", () => false, "Automatically apply improvements");
            autoApplyOption.AddAlias("-a");

            AddOption(targetOption);
            AddOption(autoApplyOption);

            this.SetHandler(async (string[] targets, bool autoApply) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SelfCodingCommand>>();
                var selfCodingWorkflow = _serviceProvider.GetRequiredService<SelfCodingWorkflow>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Starting self-coding process for directories: {string.Join(", ", targets)}");
                    consoleService.WriteInfo($"Auto-apply improvements: {autoApply}");

                    // Set the auto-apply configuration
                    var configuration = _serviceProvider.GetRequiredService<Microsoft.Extensions.Configuration.IConfiguration>();
                    var configurationRoot = (Microsoft.Extensions.Configuration.ConfigurationRoot)configuration;
                    configurationRoot.GetSection("Tars:SelfCoding").GetSection("AutoApply").Value = autoApply.ToString();

                    // Start the workflow
                    var result = await selfCodingWorkflow.StartWorkflowAsync(targets.ToList());

                    if (result)
                    {
                        consoleService.WriteSuccess("Self-coding process started successfully");
                        consoleService.WriteInfo("Use 'self-code status' to check the status of the process");
                        consoleService.WriteInfo("Use 'self-code stop' to stop the process");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError("Failed to start self-coding process");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error starting self-coding process");
                    consoleService.WriteError($"Error starting self-coding process: {ex.Message}");
                    // Failure
                }
            }, targetOption, autoApplyOption);
        }
    }

    /// <summary>
    /// Command to stop the self-coding process
    /// </summary>
    private class StopCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StopCommand(IServiceProvider serviceProvider) : base("stop", "Stop the self-coding process")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(async () =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SelfCodingCommand>>();
                var selfCodingWorkflow = _serviceProvider.GetRequiredService<SelfCodingWorkflow>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Stopping self-coding process");

                    // Stop the workflow
                    var result = await selfCodingWorkflow.StopWorkflowAsync();

                    if (result)
                    {
                        consoleService.WriteSuccess("Self-coding process stopped successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError("Failed to stop self-coding process");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error stopping self-coding process");
                    consoleService.WriteError($"Error stopping self-coding process: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to get the status of the self-coding process
    /// </summary>
    private class StatusCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StatusCommand(IServiceProvider serviceProvider) : base("status", "Get the status of the self-coding process")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(() =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SelfCodingCommand>>();
                var selfCodingWorkflow = _serviceProvider.GetRequiredService<SelfCodingWorkflow>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Getting status of self-coding process");

                    // Get the workflow state
                    var state = selfCodingWorkflow.GetWorkflowState();

                    consoleService.WriteInfo("Self-coding status:");
                    consoleService.WriteInfo($"  Status: {state.Status}");
                    consoleService.WriteInfo($"  Current Stage: {state.CurrentStage}");
                    consoleService.WriteInfo($"  Current File: {state.CurrentFile ?? "None"}");
                    consoleService.WriteInfo($"  Start Time: {state.StartTime}");
                    consoleService.WriteInfo($"  End Time: {state.EndTime?.ToString() ?? "N/A"}");
                    consoleService.WriteInfo($"  Files to Process: {state.FilesToProcess.Count}");
                    consoleService.WriteInfo($"  Processed Files: {state.ProcessedFiles.Count}");
                    consoleService.WriteInfo($"  Failed Files: {state.FailedFiles.Count}");

                    if (state.Statistics.Count > 0)
                    {
                        consoleService.WriteInfo("  Statistics:");
                        foreach (var stat in state.Statistics)
                        {
                            consoleService.WriteInfo($"    {stat.Key}: {stat.Value}");
                        }
                    }

                    if (!string.IsNullOrEmpty(state.ErrorMessage))
                    {
                        consoleService.WriteError($"  Error: {state.ErrorMessage}");
                    }

                    // Success
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error getting status of self-coding process");
                    consoleService.WriteError($"Error getting status of self-coding process: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to set up the self-coding environment
    /// </summary>
    private class SetupCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public SetupCommand(IServiceProvider serviceProvider) : base("setup", "Set up the self-coding environment")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(async () =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SelfCodingCommand>>();
                var replicaManager = _serviceProvider.GetRequiredService<TarsReplicaManager>();
                var dockerService = _serviceProvider.GetRequiredService<DockerService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Setting up self-coding environment");

                    // Check if Docker is running
                    if (!await dockerService.IsDockerRunning())
                    {
                        consoleService.WriteError("Docker is not running. Please start Docker first.");
                        // Failure
                        return;
                    }

                    // Create Docker network
                    consoleService.WriteInfo("Creating Docker network for TARS replicas...");

                    try
                    {
                        // Check if network already exists
                        var checkProcess = new System.Diagnostics.Process
                        {
                            StartInfo = new System.Diagnostics.ProcessStartInfo
                            {
                                FileName = "docker",
                                Arguments = "network ls --filter name=tars-network --format {{.Name}}",
                                RedirectStandardOutput = true,
                                RedirectStandardError = true,
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        checkProcess.Start();
                        var output = await checkProcess.StandardOutput.ReadToEndAsync();
                        await checkProcess.WaitForExitAsync();

                        if (!string.IsNullOrWhiteSpace(output) && output.Trim() == "tars-network")
                        {
                            consoleService.WriteInfo("Docker network 'tars-network' already exists");
                        }
                        else
                        {
                            // Create the network
                            var process = new System.Diagnostics.Process
                            {
                                StartInfo = new System.Diagnostics.ProcessStartInfo
                                {
                                    FileName = "docker-compose",
                                    Arguments = "-f docker-compose-network.yml up -d",
                                    RedirectStandardOutput = true,
                                    RedirectStandardError = true,
                                    UseShellExecute = false,
                                    CreateNoWindow = true
                                }
                            };
                            process.Start();
                            var createOutput = await process.StandardOutput.ReadToEndAsync();
                            var createError = await process.StandardError.ReadToEndAsync();
                            await process.WaitForExitAsync();

                            if (process.ExitCode == 0)
                            {
                                consoleService.WriteSuccess("Docker network created successfully");
                            }
                            else
                            {
                                logger.LogError($"Failed to create Docker network: {createError}");
                                consoleService.WriteError($"Failed to create Docker network: {createError}");
                                // Failure
                                return;
                            }
                        }

                        // Create Docker volumes
                        consoleService.WriteInfo("Creating Docker volumes for TARS replicas...");

                        // Ensure volume directories exist
                        Directory.CreateDirectory("docker/volumes/config");
                        Directory.CreateDirectory("docker/volumes/data");
                        Directory.CreateDirectory("docker/volumes/logs");
                        Directory.CreateDirectory("docker/volumes/codebase");

                        // Create the volumes
                        var volumesProcess = new System.Diagnostics.Process
                        {
                            StartInfo = new System.Diagnostics.ProcessStartInfo
                            {
                                FileName = "docker-compose",
                                Arguments = "-f docker-compose-volumes.yml up -d",
                                RedirectStandardOutput = true,
                                RedirectStandardError = true,
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        volumesProcess.Start();
                        var volumesOutput = await volumesProcess.StandardOutput.ReadToEndAsync();
                        var volumesError = await volumesProcess.StandardError.ReadToEndAsync();
                        await volumesProcess.WaitForExitAsync();

                        if (volumesProcess.ExitCode == 0)
                        {
                            consoleService.WriteSuccess("Docker volumes created successfully");
                        }
                        else
                        {
                            logger.LogError($"Failed to create Docker volumes: {volumesError}");
                            consoleService.WriteError($"Failed to create Docker volumes: {volumesError}");
                            // Failure
                            return;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error setting up Docker infrastructure");
                        consoleService.WriteError($"Error setting up Docker infrastructure: {ex.Message}");
                        // Failure
                        return;
                    }

                    // Create the replicas
                    consoleService.WriteInfo("Creating TARS replicas for self-coding...");
                    var result = await replicaManager.CreateSelfCodingReplicasAsync();

                    if (result)
                    {
                        consoleService.WriteSuccess("TARS replicas created successfully");

                        // List the replicas
                        var replicas = replicaManager.GetAllReplicas();
                        consoleService.WriteInfo($"Created {replicas.Count} replicas:");
                        foreach (var replica in replicas)
                        {
                            consoleService.WriteInfo($"  {replica.Name} (ID: {replica.Id}, Role: {replica.Role}, Port: {replica.Port})");
                        }

                        consoleService.WriteInfo("Self-coding environment set up successfully");
                        consoleService.WriteInfo("You can now use 'self-code start' to start the self-coding process");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError("Failed to create TARS replicas");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error setting up self-coding environment");
                    consoleService.WriteError($"Error setting up self-coding environment: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to clean up the self-coding environment
    /// </summary>
    private class CleanupCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public CleanupCommand(IServiceProvider serviceProvider) : base("cleanup", "Clean up the self-coding environment")
        {
            _serviceProvider = serviceProvider;

            // Add options
            var forceOption = new Option<bool>("--force", () => false, "Force cleanup even if replicas are running");
            forceOption.AddAlias("-f");
            AddOption(forceOption);

            this.SetHandler(async (bool force) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<SelfCodingCommand>>();
                var replicaManager = _serviceProvider.GetRequiredService<TarsReplicaManager>();
                var dockerService = _serviceProvider.GetRequiredService<DockerService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Cleaning up self-coding environment");

                    // Check if Docker is running
                    if (!await dockerService.IsDockerRunning())
                    {
                        consoleService.WriteError("Docker is not running. Please start Docker first.");
                        return;
                    }

                    // Check if any replicas are running
                    var replicas = replicaManager.GetAllReplicas();
                    var runningReplicas = replicas.Where(r => r.Status == "running").ToList();

                    if (runningReplicas.Count > 0 && !force)
                    {
                        consoleService.WriteError($"There are {runningReplicas.Count} replicas still running. Use --force to clean up anyway.");
                        return;
                    }

                    // Stop and remove all replicas
                    if (replicas.Count > 0)
                    {
                        consoleService.WriteInfo($"Stopping and removing {replicas.Count} replicas...");
                        foreach (var replica in replicas)
                        {
                            await replicaManager.RemoveReplicaAsync(replica.Id);
                            consoleService.WriteInfo($"  Removed replica {replica.Name} (ID: {replica.Id})");
                        }
                        consoleService.WriteSuccess("All replicas removed");
                    }
                    else
                    {
                        consoleService.WriteInfo("No replicas found");
                    }

                    // Remove Docker volumes
                    consoleService.WriteInfo("Removing Docker volumes...");
                    var volumesProcess = new System.Diagnostics.Process
                    {
                        StartInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "docker-compose",
                            Arguments = "-f docker-compose-volumes.yml down -v",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }
                    };
                    volumesProcess.Start();
                    var volumesOutput = await volumesProcess.StandardOutput.ReadToEndAsync();
                    var volumesError = await volumesProcess.StandardError.ReadToEndAsync();
                    await volumesProcess.WaitForExitAsync();

                    if (volumesProcess.ExitCode == 0)
                    {
                        consoleService.WriteSuccess("Docker volumes removed successfully");
                    }
                    else
                    {
                        logger.LogError($"Failed to remove Docker volumes: {volumesError}");
                        consoleService.WriteError($"Failed to remove Docker volumes: {volumesError}");
                    }

                    // Remove Docker network
                    consoleService.WriteInfo("Removing Docker network...");
                    var networkProcess = new System.Diagnostics.Process
                    {
                        StartInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "docker-compose",
                            Arguments = "-f docker-compose-network.yml down",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }
                    };
                    networkProcess.Start();
                    var networkOutput = await networkProcess.StandardOutput.ReadToEndAsync();
                    var networkError = await networkProcess.StandardError.ReadToEndAsync();
                    await networkProcess.WaitForExitAsync();

                    if (networkProcess.ExitCode == 0)
                    {
                        consoleService.WriteSuccess("Docker network removed successfully");
                    }
                    else
                    {
                        logger.LogError($"Failed to remove Docker network: {networkError}");
                        consoleService.WriteError($"Failed to remove Docker network: {networkError}");
                    }

                    // Clean up volume directories
                    try
                    {
                        if (System.IO.Directory.Exists("docker/volumes/config"))
                        {
                            System.IO.Directory.Delete("docker/volumes/config", true);
                        }
                        if (System.IO.Directory.Exists("docker/volumes/data"))
                        {
                            System.IO.Directory.Delete("docker/volumes/data", true);
                        }
                        if (System.IO.Directory.Exists("docker/volumes/logs"))
                        {
                            System.IO.Directory.Delete("docker/volumes/logs", true);
                        }
                        if (System.IO.Directory.Exists("docker/volumes/codebase"))
                        {
                            System.IO.Directory.Delete("docker/volumes/codebase", true);
                        }
                        consoleService.WriteSuccess("Volume directories cleaned up");
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error cleaning up volume directories");
                        consoleService.WriteError($"Error cleaning up volume directories: {ex.Message}");
                    }

                    consoleService.WriteSuccess("Self-coding environment cleaned up successfully");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error cleaning up self-coding environment");
                    consoleService.WriteError($"Error cleaning up self-coding environment: {ex.Message}");
                }
            }, forceOption);
        }
    }
}