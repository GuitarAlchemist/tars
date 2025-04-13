using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;
using A2AClient = TarsEngine.A2A.A2AClient;
using A2ACardResolver = TarsEngine.A2A.AgentCardResolver;
using A2AMessage = TarsEngine.A2A.Message;
using A2APart = TarsEngine.A2A.Part;
using A2ATextPart = TarsEngine.A2A.TextPart;

namespace TarsCli.Commands;

/// <summary>
/// Command for demonstrating A2A protocol capabilities
/// </summary>
public class A2ADemoCommand : Command
{
    private readonly ILogger<A2ADemoCommand>? _logger;
    private readonly IServiceProvider? _serviceProvider;
    private readonly ConsoleService? _consoleService;
    private readonly TarsA2AService? _tarsA2AService;

    /// <summary>
    /// Initializes a new instance of the <see cref="A2ADemoCommand"/> class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="serviceProvider">Service provider</param>
    public A2ADemoCommand(ILogger<A2ADemoCommand>? logger = null, IServiceProvider? serviceProvider = null)
        : base("a2a-demo", "Demonstrate A2A protocol capabilities")
    {
        _logger = logger;
        _serviceProvider = serviceProvider;

        if (_serviceProvider != null)
        {
            _consoleService = _serviceProvider.GetService<ConsoleService>();
            _tarsA2AService = _serviceProvider.GetService<TarsA2AService>();
        }

        // Add handler
        this.SetHandler(HandleCommand);
    }

    /// <summary>
    /// Handles the command
    /// </summary>
    private async Task HandleCommand()
    {
        try
        {
            if (_consoleService == null || _tarsA2AService == null)
            {
                Console.WriteLine("Error: Required services not available.");
                return;
            }

            // Display header
            _consoleService.WriteHeader("A2A PROTOCOL DEMONSTRATION");
            _consoleService.WriteInfo("This demo showcases the A2A (Agent-to-Agent) protocol capabilities of TARS.");
            _consoleService.WriteInfo("The A2A protocol enables interoperability between different AI agents.");
            Console.WriteLine();

            // Start the A2A server
            _consoleService.WriteSubHeader("Starting the A2A Server");
            await _tarsA2AService.StartAsync();
            _consoleService.WriteSuccess("A2A server started successfully");
            Console.WriteLine();
            await Task.Delay(1000);

            // Create a client
            var cardResolver = new A2ACardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync("http://localhost:8998/");
            var client = new A2AClient(agentCard);

            // Display agent card information
            _consoleService.WriteSubHeader("TARS Agent Card");
            _consoleService.WriteInfo($"Agent Name: {agentCard.Name}");
            _consoleService.WriteInfo($"Description: {agentCard.Description}");
            _consoleService.WriteInfo($"URL: {agentCard.Url}");
            _consoleService.WriteInfo($"Provider: {agentCard.Provider.Organization}");

            _consoleService.WriteInfo("\nCapabilities:");
            _consoleService.WriteInfo($"  Streaming: {agentCard.Capabilities.Streaming}");
            _consoleService.WriteInfo($"  Push Notifications: {agentCard.Capabilities.PushNotifications}");
            _consoleService.WriteInfo($"  State Transition History: {agentCard.Capabilities.StateTransitionHistory}");

            _consoleService.WriteInfo("\nSkills:");
            foreach (var skill in agentCard.Skills)
            {
                _consoleService.WriteInfo($"  ID: {skill.Id}");
                _consoleService.WriteInfo($"  Name: {skill.Name}");
                _consoleService.WriteInfo($"  Description: {skill.Description}");
                Console.WriteLine();
            }

            await Task.Delay(1000);

            // Send a code generation task
            _consoleService.WriteSubHeader("Code Generation Task");
            _consoleService.WriteInfo("Sending a code generation task to the A2A server...");

            var message = new A2AMessage
            {
                Role = "user",
                Parts = new System.Collections.Generic.List<A2APart>
                {
                    new A2ATextPart
                    {
                        Text = "Generate a C# class for a Logger with methods for logging different levels (Info, Warning, Error)"
                    }
                },
                Metadata = new System.Collections.Generic.Dictionary<string, object>
                {
                    { "skillId", "code_generation" }
                }
            };

            var task = await client.SendTaskAsync(message);
            _consoleService.WriteSuccess($"Task ID: {task.TaskId}");
            _consoleService.WriteSuccess($"Task Status: {task.Status}");

            if (task.Messages.Count > 1)
            {
                var responseMessage = task.Messages[1];
                if (responseMessage.Parts.Count > 0 && responseMessage.Parts[0] is A2ATextPart textPart)
                {
                    _consoleService.WriteInfo("\nResponse:");
                    Console.WriteLine(textPart.Text);
                }
            }

            Console.WriteLine();
            await Task.Delay(1000);

            // Send a code analysis task
            _consoleService.WriteSubHeader("Code Analysis Task");
            _consoleService.WriteInfo("Sending a code analysis task to the A2A server...");

            var codeToAnalyze = "public void ProcessData(string data) { var result = data.Split(','); Console.WriteLine(result[0]); }";

            message = new A2AMessage
            {
                Role = "user",
                Parts = new System.Collections.Generic.List<A2APart>
                {
                    new A2ATextPart
                    {
                        Text = $"Analyze this code for potential issues: {codeToAnalyze}"
                    }
                },
                Metadata = new System.Collections.Generic.Dictionary<string, object>
                {
                    { "skillId", "code_analysis" }
                }
            };

            task = await client.SendTaskAsync(message);
            _consoleService.WriteSuccess($"Task ID: {task.TaskId}");
            _consoleService.WriteSuccess($"Task Status: {task.Status}");

            if (task.Messages.Count > 1)
            {
                var responseMessage = task.Messages[1];
                if (responseMessage.Parts.Count > 0 && responseMessage.Parts[0] is A2ATextPart textPart)
                {
                    _consoleService.WriteInfo("\nResponse:");
                    Console.WriteLine(textPart.Text);
                }
            }

            Console.WriteLine();
            await Task.Delay(1000);

            // Use A2A through MCP
            _consoleService.WriteSubHeader("A2A through MCP");
            _consoleService.WriteInfo("Using A2A through the MCP bridge...");

            // This would normally be done through the MCP service, but we'll simulate it here
            _consoleService.WriteInfo("MCP Request:");
            _consoleService.WriteInfo("{");
            _consoleService.WriteInfo("  \"action\": \"a2a\",");
            _consoleService.WriteInfo("  \"operation\": \"send_task\",");
            _consoleService.WriteInfo("  \"agent_url\": \"http://localhost:8998/\",");
            _consoleService.WriteInfo("  \"content\": \"Generate a simple utility class for string operations\",");
            _consoleService.WriteInfo("  \"skill_id\": \"code_generation\"");
            _consoleService.WriteInfo("}");

            message = new A2AMessage
            {
                Role = "user",
                Parts = new System.Collections.Generic.List<A2APart>
                {
                    new A2ATextPart
                    {
                        Text = "Generate a simple utility class for string operations"
                    }
                },
                Metadata = new System.Collections.Generic.Dictionary<string, object>
                {
                    { "skillId", "code_generation" }
                }
            };

            task = await client.SendTaskAsync(message);

            _consoleService.WriteInfo("\nMCP Response:");
            _consoleService.WriteInfo("{");
            _consoleService.WriteInfo("  \"success\": true,");
            _consoleService.WriteInfo($"  \"task_id\": \"{task.TaskId}\",");
            _consoleService.WriteInfo($"  \"status\": \"{task.Status}\",");

            if (task.Messages.Count > 1)
            {
                var responseMessage = task.Messages[1];
                if (responseMessage.Parts.Count > 0 && responseMessage.Parts[0] is A2ATextPart textPart)
                {
                    _consoleService.WriteInfo("  \"result\": \"" + textPart.Text.Replace("\n", "\\n").Replace("\"", "\\\"").Substring(0, Math.Min(50, textPart.Text.Length)) + "...\",");
                }
            }

            _consoleService.WriteInfo("  \"artifacts\": []");
            _consoleService.WriteInfo("}");

            Console.WriteLine();
            await Task.Delay(1000);

            // Stop the A2A server
            _consoleService.WriteSubHeader("Stopping the A2A Server");
            await _tarsA2AService.StopAsync();
            _consoleService.WriteSuccess("A2A server stopped successfully");
            Console.WriteLine();

            // Conclusion
            _consoleService.WriteHeader("DEMO COMPLETE");
            _consoleService.WriteInfo("This concludes the demonstration of TARS A2A protocol capabilities.");
            _consoleService.WriteInfo("The A2A protocol enables TARS to communicate with other A2A-compatible agents");
            _consoleService.WriteInfo("and expose its capabilities through a standardized interface.");
            Console.WriteLine();
            _consoleService.WriteInfo("For more information, see the A2A protocol documentation:");
            _consoleService.WriteInfo("docs/A2A-Protocol.md");
            Console.WriteLine();
        }
        catch (System.Net.HttpListenerException ex) when (ex.ErrorCode == 995)
        {
            // This is expected during shutdown, no need to log an error
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error running A2A demo");
            Console.WriteLine($"Error: {ex.Message}");

            // Try to stop the A2A server if it's running
            try
            {
                if (_tarsA2AService != null)
                {
                    await _tarsA2AService.StopAsync();
                }
            }
            catch
            {
                // Ignore errors when stopping the server
            }
        }
    }
}