using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for managing the TARS MCP swarm
/// </summary>
public class McpSwarmCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the McpSwarmCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public McpSwarmCommand(IServiceProvider serviceProvider) : base("mcp-swarm", "Manage TARS MCP swarm")
    {
        _serviceProvider = serviceProvider;

        // Add subcommands
        AddCommand(new CreateAgentCommand(_serviceProvider));
        AddCommand(new StartAgentCommand(_serviceProvider));
        AddCommand(new StopAgentCommand(_serviceProvider));
        AddCommand(new RemoveAgentCommand(_serviceProvider));
        AddCommand(new ListAgentsCommand(_serviceProvider));
        AddCommand(new StartAllCommand(_serviceProvider));
        AddCommand(new StopAllCommand(_serviceProvider));
        AddCommand(new SendRequestCommand(_serviceProvider));
    }

    /// <summary>
    /// Command to create a new agent in the swarm
    /// </summary>
    private class CreateAgentCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public CreateAgentCommand(IServiceProvider serviceProvider) : base("create", "Create a new agent in the swarm")
        {
            _serviceProvider = serviceProvider;

            // Add arguments and options
            var nameArgument = new Argument<string>("name", "Name of the agent");
            var roleArgument = new Argument<string>("role", "Role of the agent");
            var capabilitiesOption = new Option<string[]>("--capabilities", "Capabilities of the agent");
            capabilitiesOption.AddAlias("-c");
            var metadataOption = new Option<string[]>("--metadata", "Additional metadata for the agent (key=value pairs)");
            metadataOption.AddAlias("-m");

            AddArgument(nameArgument);
            AddArgument(roleArgument);
            AddOption(capabilitiesOption);
            AddOption(metadataOption);

            this.SetHandler(async (string name, string role, string[] capabilities, string[] metadata) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Creating agent '{name}' with role '{role}'...");

                    // Parse capabilities
                    var capabilitiesList = capabilities?.ToList() ?? new List<string>();

                    // Parse metadata
                    var metadataDict = new Dictionary<string, string>();
                    if (metadata != null)
                    {
                        foreach (var item in metadata)
                        {
                            var parts = item.Split('=', 2);
                            if (parts.Length == 2)
                            {
                                metadataDict[parts[0]] = parts[1];
                            }
                        }
                    }

                    // Create the agent
                    var agent = await swarmService.CreateAgentAsync(name, role, capabilitiesList, metadataDict);

                    consoleService.WriteSuccess($"Agent created successfully:");
                    consoleService.WriteInfo($"  ID: {agent.Id}");
                    consoleService.WriteInfo($"  Name: {agent.Name}");
                    consoleService.WriteInfo($"  Role: {agent.Role}");
                    consoleService.WriteInfo($"  Port: {agent.Port}");
                    consoleService.WriteInfo($"  Status: {agent.Status}");
                    consoleService.WriteInfo($"  Container: {agent.ContainerName}");

                    // Success
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error creating agent");
                    consoleService.WriteError($"Error creating agent: {ex.Message}");
                    // Failure
                }
            }, nameArgument, roleArgument, capabilitiesOption, metadataOption);
        }
    }

    /// <summary>
    /// Command to start an agent in the swarm
    /// </summary>
    private class StartAgentCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StartAgentCommand(IServiceProvider serviceProvider) : base("start", "Start an agent in the swarm")
        {
            _serviceProvider = serviceProvider;

            // Add arguments
            var idArgument = new Argument<string>("id", "ID of the agent to start");

            AddArgument(idArgument);

            this.SetHandler(async (string id) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Starting agent with ID '{id}'...");

                    // Start the agent
                    var result = await swarmService.StartAgentAsync(id);

                    if (result)
                    {
                        consoleService.WriteSuccess($"Agent started successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError($"Failed to start agent");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error starting agent");
                    consoleService.WriteError($"Error starting agent: {ex.Message}");
                    // Failure
                }
            }, idArgument);
        }
    }

    /// <summary>
    /// Command to stop an agent in the swarm
    /// </summary>
    private class StopAgentCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StopAgentCommand(IServiceProvider serviceProvider) : base("stop", "Stop an agent in the swarm")
        {
            _serviceProvider = serviceProvider;

            // Add arguments
            var idArgument = new Argument<string>("id", "ID of the agent to stop");

            AddArgument(idArgument);

            this.SetHandler(async (string id) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Stopping agent with ID '{id}'...");

                    // Stop the agent
                    var result = await swarmService.StopAgentAsync(id);

                    if (result)
                    {
                        consoleService.WriteSuccess($"Agent stopped successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError($"Failed to stop agent");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error stopping agent");
                    consoleService.WriteError($"Error stopping agent: {ex.Message}");
                    // Failure
                }
            }, idArgument);
        }
    }

    /// <summary>
    /// Command to remove an agent from the swarm
    /// </summary>
    private class RemoveAgentCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public RemoveAgentCommand(IServiceProvider serviceProvider) : base("remove", "Remove an agent from the swarm")
        {
            _serviceProvider = serviceProvider;

            // Add arguments
            var idArgument = new Argument<string>("id", "ID of the agent to remove");

            AddArgument(idArgument);

            this.SetHandler(async (string id) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Removing agent with ID '{id}'...");

                    // Remove the agent
                    var result = await swarmService.RemoveAgentAsync(id);

                    if (result)
                    {
                        consoleService.WriteSuccess($"Agent removed successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteError($"Failed to remove agent");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error removing agent");
                    consoleService.WriteError($"Error removing agent: {ex.Message}");
                    // Failure
                }
            }, idArgument);
        }
    }

    /// <summary>
    /// Command to list all agents in the swarm
    /// </summary>
    private class ListAgentsCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public ListAgentsCommand(IServiceProvider serviceProvider) : base("list", "List all agents in the swarm")
        {
            _serviceProvider = serviceProvider;
            AddAlias("ls");

            this.SetHandler(async () =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Updating agent statuses...");
                    await swarmService.UpdateAgentStatusesAsync();

                    var agents = swarmService.GetAllAgents();

                    if (agents.Count == 0)
                    {
                        consoleService.WriteInfo("No agents found in the swarm");
                    }
                    else
                    {
                        consoleService.WriteInfo($"Found {agents.Count} agents in the swarm:");
                        foreach (var agent in agents)
                        {
                            consoleService.WriteInfo($"  ID: {agent.Id}");
                            consoleService.WriteInfo($"  Name: {agent.Name}");
                            consoleService.WriteInfo($"  Role: {agent.Role}");
                            consoleService.WriteInfo($"  Port: {agent.Port}");
                            consoleService.WriteInfo($"  Status: {agent.Status}");
                            consoleService.WriteInfo($"  Container: {agent.ContainerName}");
                            consoleService.WriteInfo($"  Created: {agent.CreatedAt}");
                            consoleService.WriteInfo($"  Last Active: {agent.LastActiveAt}");
                            consoleService.WriteInfo($"  Capabilities: {string.Join(", ", agent.Capabilities)}");
                            consoleService.WriteInfo("  ---");
                        }
                    }

                    // Success
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error listing agents");
                    consoleService.WriteError($"Error listing agents: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to start all agents in the swarm
    /// </summary>
    private class StartAllCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StartAllCommand(IServiceProvider serviceProvider) : base("start-all", "Start all agents in the swarm")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(async () =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Starting all agents in the swarm...");

                    // Start all agents
                    var result = await swarmService.StartAllAgentsAsync();

                    if (result)
                    {
                        consoleService.WriteSuccess("All agents started successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteWarning("Some agents failed to start. Check the logs for details.");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error starting all agents");
                    consoleService.WriteError($"Error starting all agents: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to stop all agents in the swarm
    /// </summary>
    private class StopAllCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StopAllCommand(IServiceProvider serviceProvider) : base("stop-all", "Stop all agents in the swarm")
        {
            _serviceProvider = serviceProvider;

            this.SetHandler(async () =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Stopping all agents in the swarm...");

                    // Stop all agents
                    var result = await swarmService.StopAllAgentsAsync();

                    if (result)
                    {
                        consoleService.WriteSuccess("All agents stopped successfully");
                        // Success
                    }
                    else
                    {
                        consoleService.WriteWarning("Some agents failed to stop. Check the logs for details.");
                        // Failure
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error stopping all agents");
                    consoleService.WriteError($"Error stopping all agents: {ex.Message}");
                    // Failure
                }
            });
        }
    }

    /// <summary>
    /// Command to send a request to an agent
    /// </summary>
    private class SendRequestCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public SendRequestCommand(IServiceProvider serviceProvider) : base("send", "Send a request to an agent")
        {
            _serviceProvider = serviceProvider;

            // Add arguments
            var idArgument = new Argument<string>("id", "ID of the agent to send the request to");
            var actionArgument = new Argument<string>("action", "Action to perform");
            var operationArgument = new Argument<string>("operation", "Operation to perform");
            var contentOption = new Option<string>("--content", "Content to send");
            contentOption.AddAlias("-c");
            var fileOption = new Option<string>("--file", "File containing the content to send");
            fileOption.AddAlias("-f");

            AddArgument(idArgument);
            AddArgument(actionArgument);
            AddArgument(operationArgument);
            AddOption(contentOption);
            AddOption(fileOption);

            this.SetHandler(async (string id, string action, string operation, string content, string file) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmCommand>>();
                var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    // Get the content from the file if specified
                    if (!string.IsNullOrEmpty(file))
                    {
                        if (!File.Exists(file))
                        {
                            consoleService.WriteError($"File not found: {file}");
                            // Failure
                            return;
                        }

                        content = File.ReadAllText(file);
                    }

                    // Create the request
                    var requestObj = new
                    {
                        action = action,
                        operation = operation,
                        content = content
                    };

                    var requestJson = JsonSerializer.Serialize(requestObj);
                    var request = JsonSerializer.Deserialize<JsonElement>(requestJson);

                    consoleService.WriteInfo($"Sending request to agent with ID '{id}'...");

                    // Send the request
                    var response = await swarmService.SendRequestToAgentAsync(id, request);

                    consoleService.WriteSuccess("Request sent successfully");
                    consoleService.WriteInfo("Response:");
                    consoleService.WriteInfo(JsonSerializer.Serialize(response, new JsonSerializerOptions { WriteIndented = true }));

                    // Success
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error sending request to agent");
                    consoleService.WriteError($"Error sending request to agent: {ex.Message}");
                    // Failure
                }
            }, idArgument, actionArgument, operationArgument, contentOption, fileOption);
        }
    }
}