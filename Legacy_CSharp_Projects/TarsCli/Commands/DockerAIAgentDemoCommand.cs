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
    /// Command for demonstrating Docker AI Agent integration with TARS
    /// </summary>
    public class DockerAIAgentDemoCommand : Command
    {
        private readonly ILogger<DockerAIAgentDemoCommand>? _logger;
        private readonly IServiceProvider? _serviceProvider;

        /// <summary>
        /// Constructor for DockerAIAgentDemoCommand
        /// </summary>
        /// <param name="logger">Logger</param>
        /// <param name="serviceProvider">Service provider</param>
        public DockerAIAgentDemoCommand(ILogger<DockerAIAgentDemoCommand>? logger = null, IServiceProvider? serviceProvider = null)
            : base("docker-ai-agent", "Demonstrate Docker AI Agent integration with TARS")
        {
            _logger = logger;
            _serviceProvider = serviceProvider;

            this.SetHandler(async (InvocationContext context) =>
            {
                if (_serviceProvider == null)
                {
                    Console.WriteLine("Service provider is not available. Cannot run demo.");
                    context.ExitCode = 1;
                    return;
                }

                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                var dockerAIAgentService = _serviceProvider.GetRequiredService<DockerAIAgentService>();

                try
                {
                    // Introduction
                    consoleService.WriteHeader("DOCKER AI AGENT INTEGRATION DEMO");
                    Console.WriteLine("This demo showcases the integration between TARS and Docker AI Agent.");
                    Console.WriteLine("Docker AI Agent provides access to local LLMs and Docker capabilities.");
                    Console.WriteLine();

                    // Step 1: Start Docker AI Agent
                    consoleService.WriteSubHeader("Step 1: Starting Docker AI Agent");
                    Console.WriteLine("Starting Docker AI Agent...");
                    var startResult = await dockerAIAgentService.StartDockerAIAgentAsync();
                    if (!startResult)
                    {
                        consoleService.WriteError("Failed to start Docker AI Agent. Demo cannot continue.");
                        context.ExitCode = 1;
                        return;
                    }
                    consoleService.WriteSuccess("Docker AI Agent started successfully");
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Step 2: Check Docker AI Agent status
                    consoleService.WriteSubHeader("Step 2: Checking Docker AI Agent Status");
                    Console.WriteLine("Checking if Docker AI Agent is available...");
                    var isAvailable = await dockerAIAgentService.IsAvailableAsync();
                    if (!isAvailable)
                    {
                        consoleService.WriteError("Docker AI Agent is not available. Demo cannot continue.");
                        context.ExitCode = 1;
                        return;
                    }
                    consoleService.WriteSuccess("Docker AI Agent is available");
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Step 3: List available models
                    consoleService.WriteSubHeader("Step 3: Listing Available Models");
                    Console.WriteLine("Getting available models from Docker AI Agent...");
                    var models = await dockerAIAgentService.GetAvailableModelsAsync();
                    if (models.Count > 0)
                    {
                        consoleService.WriteSuccess("Available models:");
                        foreach (var model in models)
                        {
                            Console.WriteLine($"- {model}");
                        }
                    }
                    else
                    {
                        consoleService.WriteWarning("No models available");
                    }
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Step 4: Generate text
                    consoleService.WriteSubHeader("Step 4: Generating Text");
                    Console.WriteLine("Generating text using Docker AI Agent...");
                    var prompt = "Write a short poem about artificial intelligence and creativity";
                    Console.WriteLine($"Prompt: {prompt}");
                    var generatedText = await dockerAIAgentService.GenerateTextAsync(prompt);
                    if (!string.IsNullOrEmpty(generatedText))
                    {
                        consoleService.WriteSuccess("Generated text:");
                        Console.WriteLine(generatedText);
                    }
                    else
                    {
                        consoleService.WriteWarning("Failed to generate text");
                    }
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Step 5: Execute shell command
                    consoleService.WriteSubHeader("Step 5: Executing Shell Command");
                    Console.WriteLine("Executing shell command using Docker AI Agent...");
                    var command = "docker ps";
                    Console.WriteLine($"Command: {command}");
                    var commandOutput = await dockerAIAgentService.ExecuteShellCommandAsync(command);
                    if (!string.IsNullOrEmpty(commandOutput))
                    {
                        consoleService.WriteSuccess("Command output:");
                        Console.WriteLine(commandOutput);
                    }
                    else
                    {
                        consoleService.WriteWarning("Failed to execute command");
                    }
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Step 6: Bridge with MCP
                    consoleService.WriteSubHeader("Step 6: Bridging with MCP");
                    Console.WriteLine("Bridging Docker AI Agent with TARS MCP...");
                    var mcpUrl = "http://localhost:8999/";
                    var bridgeResult = await dockerAIAgentService.BridgeWithMcpAsync(mcpUrl);
                    if (bridgeResult)
                    {
                        consoleService.WriteSuccess($"Docker AI Agent bridged with MCP at {mcpUrl}");
                    }
                    else
                    {
                        consoleService.WriteWarning($"Failed to bridge Docker AI Agent with MCP at {mcpUrl}");
                    }
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Step 7: Stop Docker AI Agent
                    consoleService.WriteSubHeader("Step 7: Stopping Docker AI Agent");
                    Console.WriteLine("Stopping Docker AI Agent...");
                    var stopResult = await dockerAIAgentService.StopDockerAIAgentAsync();
                    if (stopResult)
                    {
                        consoleService.WriteSuccess("Docker AI Agent stopped successfully");
                    }
                    else
                    {
                        consoleService.WriteWarning("Failed to stop Docker AI Agent");
                    }
                    Console.WriteLine();
                    await Task.Delay(2000);

                    // Conclusion
                    consoleService.WriteHeader("DEMO COMPLETE");
                    Console.WriteLine("This concludes the demonstration of Docker AI Agent integration with TARS.");
                    Console.WriteLine("The Docker AI Agent provides access to local LLMs and Docker capabilities,");
                    Console.WriteLine("enabling TARS to leverage Docker's AI features for autonomous self-improvement.");
                    Console.WriteLine();
                    Console.WriteLine("For more information, see the Docker AI Agent documentation:");
                    Console.WriteLine("docs/Docker-AI-Agent.md");
                    Console.WriteLine();

                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error running Docker AI Agent demo");
                    consoleService.WriteError($"Error running Docker AI Agent demo: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }
}
