using System.CommandLine.Invocation;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for demonstrating the MCP swarm capabilities
/// </summary>
public class McpSwarmDemoCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the McpSwarmDemoCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public McpSwarmDemoCommand(IServiceProvider serviceProvider) : base("mcp-swarm-demo", "Demonstrate MCP swarm capabilities")
    {
        _serviceProvider = serviceProvider;

        this.SetHandler(async (InvocationContext context) =>
        {
            var logger = _serviceProvider.GetRequiredService<ILogger<McpSwarmDemoCommand>>();
            var swarmService = _serviceProvider.GetRequiredService<TarsMcpSwarmService>();
            var dockerService = _serviceProvider.GetRequiredService<DockerService>();
            var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

            try
            {
                // Display demo header
                consoleService.WriteHeader("MCP SWARM DEMONSTRATION");
                consoleService.WriteInfo("This demo showcases the MCP swarm capabilities of TARS.");
                consoleService.WriteInfo("The MCP swarm enables TARS to deploy and manage multiple MCP agents in Docker containers.");
                consoleService.WriteInfo("These agents can work together to perform complex tasks, such as self-improvement.");
                Console.WriteLine();

                // Check if Docker is running
                consoleService.WriteSubHeader("Checking Docker");
                var isDockerRunning = await dockerService.IsDockerRunning();
                if (!isDockerRunning)
                {
                    consoleService.WriteError("Docker is not running. Please start Docker first.");
                    context.ExitCode = 1;
                    return;
                }
                consoleService.WriteSuccess("Docker is running");
                Console.WriteLine();

                // Create Docker network
                consoleService.WriteSubHeader("Creating Docker Network");
                consoleService.WriteInfo("Creating Docker network for MCP agents...");
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
                await process.WaitForExitAsync();
                if (process.ExitCode == 0)
                {
                    consoleService.WriteSuccess("Docker network created successfully");
                }
                else
                {
                    consoleService.WriteError("Failed to create Docker network");
                    context.ExitCode = 1;
                    return;
                }
                Console.WriteLine();

                // Create agents
                consoleService.WriteSubHeader("Creating MCP Agents");
                consoleService.WriteInfo("Creating code analyzer agent...");
                var analyzerAgent = await swarmService.CreateAgentAsync(
                    "CodeAnalyzer",
                    "code_analyzer",
                    ["analyze_code", "detect_issues", "suggest_improvements"]);
                consoleService.WriteSuccess($"Code analyzer agent created with ID: {analyzerAgent.Id}");

                consoleService.WriteInfo("Creating code generator agent...");
                var generatorAgent = await swarmService.CreateAgentAsync(
                    "CodeGenerator",
                    "code_generator",
                    ["generate_code", "refactor_code", "optimize_code"]);
                consoleService.WriteSuccess($"Code generator agent created with ID: {generatorAgent.Id}");

                consoleService.WriteInfo("Creating test generator agent...");
                var testAgent = await swarmService.CreateAgentAsync(
                    "TestGenerator",
                    "test_generator",
                    ["generate_tests", "run_tests", "analyze_test_results"]);
                consoleService.WriteSuccess($"Test generator agent created with ID: {testAgent.Id}");
                Console.WriteLine();

                // Wait for agents to start
                consoleService.WriteSubHeader("Waiting for Agents to Start");
                consoleService.WriteInfo("Waiting for agents to start...");
                await Task.Delay(5000);
                await swarmService.UpdateAgentStatusesAsync();
                var agents = swarmService.GetAllAgents();
                foreach (var agent in agents)
                {
                    consoleService.WriteInfo($"Agent {agent.Name} (ID: {agent.Id}) status: {agent.Status}");
                }
                Console.WriteLine();

                // Simulate sending requests to agents
                consoleService.WriteSubHeader("Simulating Agent Requests");
                consoleService.WriteInfo("Simulating a code analysis request...");
                consoleService.WriteInfo("Request: Analyze TarsCli/Program.cs for potential improvements");
                await Task.Delay(2000);
                consoleService.WriteSuccess("Analysis completed (simulated)");
                consoleService.WriteInfo("Results: Found 3 potential improvements (simulated)");
                Console.WriteLine();

                consoleService.WriteInfo("Simulating a code generation request...");
                consoleService.WriteInfo("Request: Generate a unit test for TarsCli/Services/TarsMcpSwarmService.cs");
                await Task.Delay(2000);
                consoleService.WriteSuccess("Code generation completed (simulated)");
                consoleService.WriteInfo("Results: Generated 2 unit tests (simulated)");
                Console.WriteLine();

                // Stop and remove agents
                consoleService.WriteSubHeader("Cleaning Up");
                consoleService.WriteInfo("Stopping and removing agents...");
                foreach (var agent in agents)
                {
                    await swarmService.RemoveAgentAsync(agent.Id);
                    consoleService.WriteInfo($"Agent {agent.Name} (ID: {agent.Id}) removed");
                }
                Console.WriteLine();

                // Remove Docker network
                consoleService.WriteInfo("Removing Docker network...");
                process = new System.Diagnostics.Process
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
                process.Start();
                await process.WaitForExitAsync();
                if (process.ExitCode == 0)
                {
                    consoleService.WriteSuccess("Docker network removed successfully");
                }
                else
                {
                    consoleService.WriteWarning("Failed to remove Docker network");
                }
                Console.WriteLine();

                // Display demo footer
                consoleService.WriteHeader("DEMO COMPLETE");
                consoleService.WriteInfo("This concludes the MCP swarm demonstration.");
                consoleService.WriteInfo("You can use the following commands to manage the MCP swarm:");
                consoleService.WriteInfo("  mcp-swarm create <name> <role> - Create a new agent");
                consoleService.WriteInfo("  mcp-swarm start <id> - Start an agent");
                consoleService.WriteInfo("  mcp-swarm stop <id> - Stop an agent");
                consoleService.WriteInfo("  mcp-swarm remove <id> - Remove an agent");
                consoleService.WriteInfo("  mcp-swarm list - List all agents");
                consoleService.WriteInfo("  mcp-swarm send <id> <action> <operation> - Send a request to an agent");
                Console.WriteLine();
                consoleService.WriteInfo("You can also use the swarm-improve command to start a self-improvement process:");
                consoleService.WriteInfo("  swarm-improve start --target <directories> --agent-count <count> --model <model>");
                Console.WriteLine();

                context.ExitCode = 0;
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error running MCP swarm demo");
                consoleService.WriteError($"Error running MCP swarm demo: {ex.Message}");
                context.ExitCode = 1;
            }
        });
    }
}