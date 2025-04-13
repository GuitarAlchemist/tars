using System.Text.Json;
using TarsEngine.A2A;
using TarsCli.Services;
using Task = System.Threading.Tasks.Task;

namespace TarsCli.Commands;

/// <summary>
/// Commands for interacting with A2A protocol
/// </summary>
public class A2ACommands
{
    private readonly ILogger<A2ACommands> _logger;
    private readonly TarsA2AService _tarsA2AService;

    /// <summary>
    /// Initializes a new instance of the A2ACommands class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="tarsA2AService">TARS A2A service instance</param>
    public A2ACommands(ILogger<A2ACommands> logger, TarsA2AService tarsA2AService)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _tarsA2AService = tarsA2AService ?? throw new ArgumentNullException(nameof(tarsA2AService));
    }

    /// <summary>
    /// Gets the A2A command
    /// </summary>
    public Command GetCommand()
    {
        var command = new Command("a2a", "A2A protocol commands");

        // Start command
        var startCommand = new Command("start", "Start the A2A server");
        startCommand.SetHandler(HandleStartCommand);
        command.AddCommand(startCommand);

        // Stop command
        var stopCommand = new Command("stop", "Stop the A2A server");
        stopCommand.SetHandler(HandleStopCommand);
        command.AddCommand(stopCommand);

        // Send command
        var sendCommand = new Command("send", "Send a task to an A2A agent");
        var agentUrlOption = new Option<string>("--agent-url", "The URL of the agent") { IsRequired = true };
        var messageOption = new Option<string>("--message", "The message to send") { IsRequired = true };
        var skillIdOption = new Option<string>("--skill-id", "The skill ID to use");
        var outputFileOption = new Option<string>("--output", "The output file to write the response to");
        sendCommand.AddOption(agentUrlOption);
        sendCommand.AddOption(messageOption);
        sendCommand.AddOption(skillIdOption);
        sendCommand.AddOption(outputFileOption);
        sendCommand.SetHandler(HandleSendCommand, agentUrlOption, messageOption, skillIdOption, outputFileOption);
        command.AddCommand(sendCommand);

        // Get command
        var getCommand = new Command("get", "Get a task from an A2A agent");
        var taskIdOption = new Option<string>("--task-id", "The task ID to get") { IsRequired = true };
        getCommand.AddOption(agentUrlOption);
        getCommand.AddOption(taskIdOption);
        getCommand.AddOption(outputFileOption);
        getCommand.SetHandler(HandleGetCommand, agentUrlOption, taskIdOption, outputFileOption);
        command.AddCommand(getCommand);

        // Cancel command
        var cancelCommand = new Command("cancel", "Cancel a task on an A2A agent");
        cancelCommand.AddOption(agentUrlOption);
        cancelCommand.AddOption(taskIdOption);
        cancelCommand.SetHandler(HandleCancelCommand, agentUrlOption, taskIdOption);
        command.AddCommand(cancelCommand);

        // Get agent card command
        var getAgentCardCommand = new Command("get-agent-card", "Get the agent card from an A2A agent");
        getAgentCardCommand.AddOption(agentUrlOption);
        getAgentCardCommand.AddOption(outputFileOption);
        getAgentCardCommand.SetHandler(HandleGetAgentCardCommand, agentUrlOption, outputFileOption);
        command.AddCommand(getAgentCardCommand);

        return command;
    }

    /// <summary>
    /// Handles the start command
    /// </summary>
    private async Task HandleStartCommand()
    {
        try
        {
            await _tarsA2AService.StartAsync();
            Console.WriteLine("TARS A2A server started");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting TARS A2A server");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Handles the stop command
    /// </summary>
    private async Task HandleStopCommand()
    {
        try
        {
            await _tarsA2AService.StopAsync();
            Console.WriteLine("TARS A2A server stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping TARS A2A server");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Handles the send command
    /// </summary>
    private async Task HandleSendCommand(string agentUrl, string message, string skillId, string outputFile)
    {
        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);
            Console.WriteLine($"Connected to agent: {agentCard.Name}");

            // Create A2A client
            var client = new A2AClient(agentCard);

            // Create message
            var a2aMessage = new Message
            {
                Role = "user",
                Parts = new List<Part>
                {
                    new TextPart
                    {
                        Text = message
                    }
                }
            };

            // Add skill ID to metadata if provided
            if (!string.IsNullOrEmpty(skillId))
            {
                a2aMessage.Metadata = new Dictionary<string, object>
                {
                    { "skillId", skillId }
                };
            }

            // Send task with streaming if supported
            if (agentCard.Capabilities.Streaming)
            {
                Console.WriteLine("Sending task with streaming...");
                var task = await client.SendTaskStreamingAsync(
                    a2aMessage,
                    onUpdate: t => 
                    {
                        Console.WriteLine($"Task status: {t.Status}");
                        if (t.Messages.Count > 1)
                        {
                            var lastMessage = t.Messages[t.Messages.Count - 1];
                            if (lastMessage.Parts.Count > 0 && lastMessage.Parts[0] is TextPart textPart)
                            {
                                Console.WriteLine($"Response: {textPart.Text}");
                            }
                        }
                    });

                // Write output to file if specified
                if (!string.IsNullOrEmpty(outputFile))
                {
                    WriteTaskToFile(task, outputFile);
                }
            }
            else
            {
                Console.WriteLine("Sending task...");
                var task = await client.SendTaskAsync(a2aMessage);
                Console.WriteLine($"Task ID: {task.TaskId}");
                Console.WriteLine($"Task status: {task.Status}");

                if (task.Messages.Count > 1)
                {
                    var lastMessage = task.Messages[task.Messages.Count - 1];
                    if (lastMessage.Parts.Count > 0 && lastMessage.Parts[0] is TextPart textPart)
                    {
                        Console.WriteLine($"Response: {textPart.Text}");
                    }
                }

                // Write output to file if specified
                if (!string.IsNullOrEmpty(outputFile))
                {
                    WriteTaskToFile(task, outputFile);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending task to A2A agent");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Handles the get command
    /// </summary>
    private async Task HandleGetCommand(string agentUrl, string taskId, string outputFile)
    {
        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);
            Console.WriteLine($"Connected to agent: {agentCard.Name}");

            // Create A2A client
            var client = new A2AClient(agentCard);

            // Get task
            Console.WriteLine($"Getting task {taskId}...");
            var task = await client.GetTaskAsync(taskId);
            Console.WriteLine($"Task status: {task.Status}");

            if (task.Messages.Count > 1)
            {
                var lastMessage = task.Messages[task.Messages.Count - 1];
                if (lastMessage.Parts.Count > 0 && lastMessage.Parts[0] is TextPart textPart)
                {
                    Console.WriteLine($"Response: {textPart.Text}");
                }
            }

            // Write output to file if specified
            if (!string.IsNullOrEmpty(outputFile))
            {
                WriteTaskToFile(task, outputFile);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting task from A2A agent");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Handles the cancel command
    /// </summary>
    private async Task HandleCancelCommand(string agentUrl, string taskId)
    {
        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);
            Console.WriteLine($"Connected to agent: {agentCard.Name}");

            // Create A2A client
            var client = new A2AClient(agentCard);

            // Cancel task
            Console.WriteLine($"Canceling task {taskId}...");
            var task = await client.CancelTaskAsync(taskId);
            Console.WriteLine($"Task status: {task.Status}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error canceling task on A2A agent");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Handles the get agent card command
    /// </summary>
    private async Task HandleGetAgentCardCommand(string agentUrl, string outputFile)
    {
        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);
                
            // Display agent card information
            Console.WriteLine($"Agent Name: {agentCard.Name}");
            Console.WriteLine($"Description: {agentCard.Description}");
            Console.WriteLine($"URL: {agentCard.Url}");
            Console.WriteLine($"Provider: {agentCard.Provider.Organization}");
            Console.WriteLine($"Version: {agentCard.Version}");
            Console.WriteLine($"Documentation: {agentCard.DocumentationUrl}");
                
            Console.WriteLine("\nCapabilities:");
            Console.WriteLine($"  Streaming: {agentCard.Capabilities.Streaming}");
            Console.WriteLine($"  Push Notifications: {agentCard.Capabilities.PushNotifications}");
            Console.WriteLine($"  State Transition History: {agentCard.Capabilities.StateTransitionHistory}");
                
            Console.WriteLine("\nSkills:");
            foreach (var skill in agentCard.Skills)
            {
                Console.WriteLine($"  ID: {skill.Id}");
                Console.WriteLine($"  Name: {skill.Name}");
                Console.WriteLine($"  Description: {skill.Description}");
                Console.WriteLine($"  Tags: {string.Join(", ", skill.Tags)}");
                Console.WriteLine();
            }
                
            // Write output to file if specified
            if (!string.IsNullOrEmpty(outputFile))
            {
                var options = new JsonSerializerOptions
                {
                    WriteIndented = true
                };
                var json = JsonSerializer.Serialize(agentCard, options);
                File.WriteAllText(outputFile, json);
                Console.WriteLine($"Agent card written to {outputFile}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting agent card");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Writes a task to a file
    /// </summary>
    private void WriteTaskToFile(TarsEngine.A2A.Task task, string outputFile)
    {
        try
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };
            var json = JsonSerializer.Serialize(task, options);
            File.WriteAllText(outputFile, json);
            Console.WriteLine($"Task written to {outputFile}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error writing task to file");
            Console.WriteLine($"Error writing to file: {ex.Message}");
        }
    }
}