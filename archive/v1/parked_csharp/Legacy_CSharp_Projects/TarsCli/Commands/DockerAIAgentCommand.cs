using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Services;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for interacting with Docker AI Agent
    /// </summary>
    public class DockerAIAgentCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Constructor for DockerAIAgentCommand
        /// </summary>
        /// <param name="serviceProvider">Service provider</param>
        public DockerAIAgentCommand(IServiceProvider serviceProvider) : base("docker-ai-agent", "Interact with Docker AI Agent")
        {
            _serviceProvider = serviceProvider;
            AddAlias("dai");

            // Add subcommands
            AddCommand(new StartCommand(_serviceProvider));
            AddCommand(new StopCommand(_serviceProvider));
            AddCommand(new StatusCommand(_serviceProvider));
            AddCommand(new RunModelCommand(_serviceProvider));
            AddCommand(new GenerateCommand(_serviceProvider));
            AddCommand(new ShellCommand(_serviceProvider));
            AddCommand(new ListModelsCommand(_serviceProvider));
            AddCommand(new BridgeCommand(_serviceProvider));
        }

        /// <summary>
        /// Command to start Docker AI Agent
        /// </summary>
        private class StartCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public StartCommand(IServiceProvider serviceProvider) : base("start", "Start Docker AI Agent")
            {
                _serviceProvider = serviceProvider;

                this.SetHandler(async (InvocationContext context) =>
                {
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo("Starting Docker AI Agent...");

                        var result = await dockerAIAgentService.StartDockerAIAgentAsync();
                        if (result)
                        {
                            consoleService.WriteSuccess("Docker AI Agent started successfully");
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteError("Failed to start Docker AI Agent");
                            context.ExitCode = 1;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error starting Docker AI Agent");
                        consoleService.WriteError($"Error starting Docker AI Agent: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to stop Docker AI Agent
        /// </summary>
        private class StopCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public StopCommand(IServiceProvider serviceProvider) : base("stop", "Stop Docker AI Agent")
            {
                _serviceProvider = serviceProvider;

                this.SetHandler(async (InvocationContext context) =>
                {
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo("Stopping Docker AI Agent...");

                        var result = await dockerAIAgentService.StopDockerAIAgentAsync();
                        if (result)
                        {
                            consoleService.WriteSuccess("Docker AI Agent stopped successfully");
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteError("Failed to stop Docker AI Agent");
                            context.ExitCode = 1;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error stopping Docker AI Agent");
                        consoleService.WriteError($"Error stopping Docker AI Agent: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to check Docker AI Agent status
        /// </summary>
        private class StatusCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public StatusCommand(IServiceProvider serviceProvider) : base("status", "Check Docker AI Agent status")
            {
                _serviceProvider = serviceProvider;

                this.SetHandler(async (InvocationContext context) =>
                {
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo("Checking Docker AI Agent status...");

                        var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                        if (isAvailable)
                        {
                            consoleService.WriteSuccess("Docker AI Agent is running");
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteWarning("Docker AI Agent is not running");
                            context.ExitCode = 0;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error checking Docker AI Agent status");
                        consoleService.WriteError($"Error checking Docker AI Agent status: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to run a model using Docker AI Agent
        /// </summary>
        private class RunModelCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public RunModelCommand(IServiceProvider serviceProvider) : base("run-model", "Run a model using Docker AI Agent")
            {
                _serviceProvider = serviceProvider;

                var modelArgument = new Argument<string>("model", "Name of the model to run");
                AddArgument(modelArgument);

                this.SetHandler(async (InvocationContext context) =>
                {
                    var model = context.ParseResult.GetValueForArgument(modelArgument);
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo($"Running model {model} using Docker AI Agent...");

                        var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                        if (!isAvailable)
                        {
                            consoleService.WriteError("Docker AI Agent is not running. Start it first with 'tarscli docker-ai-agent start'");
                            context.ExitCode = 1;
                            return;
                        }

                        var result = await dockerAIAgentService.RunModelAsync(model);
                        if (result)
                        {
                            consoleService.WriteSuccess($"Model {model} started successfully");
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteError($"Failed to run model {model}");
                            context.ExitCode = 1;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, $"Error running model {model}");
                        consoleService.WriteError($"Error running model {model}: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to generate text using Docker AI Agent
        /// </summary>
        private class GenerateCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public GenerateCommand(IServiceProvider serviceProvider) : base("generate", "Generate text using Docker AI Agent")
            {
                _serviceProvider = serviceProvider;

                var promptArgument = new Argument<string>("prompt", "Prompt to generate text from");
                AddArgument(promptArgument);

                var modelOption = new Option<string>("--model", "Name of the model to use");
                AddOption(modelOption);

                this.SetHandler(async (InvocationContext context) =>
                {
                    var prompt = context.ParseResult.GetValueForArgument(promptArgument);
                    var model = context.ParseResult.GetValueForOption(modelOption) ?? "";
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo($"Generating text using Docker AI Agent...");

                        var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                        if (!isAvailable)
                        {
                            consoleService.WriteError("Docker AI Agent is not running. Start it first with 'tarscli docker-ai-agent start'");
                            context.ExitCode = 1;
                            return;
                        }

                        var result = await dockerAIAgentService.GenerateTextAsync(prompt, model);
                        if (!string.IsNullOrEmpty(result))
                        {
                            consoleService.WriteSuccess("Generated text:");
                            Console.WriteLine(result);
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteError("Failed to generate text");
                            context.ExitCode = 1;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error generating text");
                        consoleService.WriteError($"Error generating text: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to execute a shell command using Docker AI Agent
        /// </summary>
        private class ShellCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public ShellCommand(IServiceProvider serviceProvider) : base("shell", "Execute a shell command using Docker AI Agent")
            {
                _serviceProvider = serviceProvider;

                var commandArgument = new Argument<string>("command", "Command to execute");
                AddArgument(commandArgument);

                this.SetHandler(async (InvocationContext context) =>
                {
                    var command = context.ParseResult.GetValueForArgument(commandArgument);
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo($"Executing shell command using Docker AI Agent: {command}");

                        var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                        if (!isAvailable)
                        {
                            consoleService.WriteError("Docker AI Agent is not running. Start it first with 'tarscli docker-ai-agent start'");
                            context.ExitCode = 1;
                            return;
                        }

                        var result = await dockerAIAgentService.ExecuteShellCommandAsync(command);
                        if (!string.IsNullOrEmpty(result))
                        {
                            consoleService.WriteSuccess("Command output:");
                            Console.WriteLine(result);
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteError("Failed to execute command");
                            context.ExitCode = 1;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, $"Error executing shell command: {command}");
                        consoleService.WriteError($"Error executing shell command: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to list available models from Docker AI Agent
        /// </summary>
        private class ListModelsCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public ListModelsCommand(IServiceProvider serviceProvider) : base("list-models", "List available models from Docker AI Agent")
            {
                _serviceProvider = serviceProvider;
                AddAlias("models");

                this.SetHandler(async (InvocationContext context) =>
                {
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo("Getting available models from Docker AI Agent...");

                        var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                        if (!isAvailable)
                        {
                            consoleService.WriteError("Docker AI Agent is not running. Start it first with 'tarscli docker-ai-agent start'");
                            context.ExitCode = 1;
                            return;
                        }

                        var models = await dockerAIAgentService.GetAvailableModelsAsync();
                        if (models.Count > 0)
                        {
                            consoleService.WriteSuccess("Available models:");
                            foreach (var model in models)
                            {
                                Console.WriteLine($"- {model}");
                            }
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteWarning("No models available");
                            context.ExitCode = 0;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error getting available models");
                        consoleService.WriteError($"Error getting available models: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }

        /// <summary>
        /// Command to bridge Docker AI Agent with MCP
        /// </summary>
        private class BridgeCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            public BridgeCommand(IServiceProvider serviceProvider) : base("bridge", "Bridge Docker AI Agent with MCP")
            {
                _serviceProvider = serviceProvider;

                var mcpUrlOption = new Option<string>("--mcp-url", "MCP URL");
                mcpUrlOption.SetDefaultValue("http://localhost:8999/");
                AddOption(mcpUrlOption);

                this.SetHandler(async (InvocationContext context) =>
                {
                    var mcpUrl = context.ParseResult.GetValueForOption(mcpUrlOption);
                    var logger = _serviceProvider.GetRequiredService<ILogger<DockerAIAgentCommand>>();
                    var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteInfo($"Bridging Docker AI Agent with MCP at {mcpUrl}...");

                        var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                        if (!isAvailable)
                        {
                            consoleService.WriteError("Docker AI Agent is not running. Start it first with 'tarscli docker-ai-agent start'");
                            context.ExitCode = 1;
                            return;
                        }

                        var result = await dockerAIAgentService.BridgeWithMcpAsync(mcpUrl);
                        if (result)
                        {
                            consoleService.WriteSuccess($"Docker AI Agent bridged with MCP at {mcpUrl}");
                            context.ExitCode = 0;
                        }
                        else
                        {
                            consoleService.WriteError($"Failed to bridge Docker AI Agent with MCP at {mcpUrl}");
                            context.ExitCode = 1;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, $"Error bridging Docker AI Agent with MCP at {mcpUrl}");
                        consoleService.WriteError($"Error bridging Docker AI Agent with MCP: {ex.Message}");
                        context.ExitCode = 1;
                    }
                });
            }
        }
    }
}
