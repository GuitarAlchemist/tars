using System;
using System.CommandLine;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for demonstrating how to use Augment Agent from TARS through VS Code
/// </summary>
public class AugmentVSCodeDemoCommand : Command
{
    private readonly ILogger<AugmentVSCodeDemoCommand> _logger;
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="AugmentVSCodeDemoCommand"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="serviceProvider">The service provider</param>
    public AugmentVSCodeDemoCommand(
        ILogger<AugmentVSCodeDemoCommand> logger,
        IServiceProvider serviceProvider)
        : base("augment-vscode-demo", "Demonstrate how to use Augment Agent from TARS through VS Code")
    {
        _logger = logger;
        _serviceProvider = serviceProvider;

        // Add options
        var taskOption = new Option<string>(
            aliases: ["--task", "-t"],
            description: "The task to perform with Augment Agent",
            getDefaultValue: () => "Analyze the codebase and suggest improvements");
        AddOption(taskOption);

        // Set handler
        this.SetHandler(async (string task) =>
        {
            await RunDemoAsync(task);
        }, taskOption);
    }

    /// <summary>
    /// Run the demo
    /// </summary>
    /// <param name="task">The task to perform with Augment Agent</param>
    private async Task RunDemoAsync(string task)
    {
        try
        {
            _logger.LogInformation($"Running Augment-VSCode demo with task: {task}");
            
            CliSupport.WriteHeader("TARS-Augment-VSCode Collaboration Demo");
            Console.WriteLine("This demo shows how to use Augment Agent from TARS through VS Code");
            Console.WriteLine();
            
            // Step 1: Start the MCP server
            CliSupport.WriteSubHeader("Step 1: Starting TARS MCP server");
            
            var mcpService = _serviceProvider.GetRequiredService<TarsMcpService>();
            await mcpService.StartAsync();
            
            Console.WriteLine("MCP server started successfully");
            Console.WriteLine();
            
            // Step 2: Enable collaboration
            CliSupport.WriteSubHeader("Step 2: Enabling collaboration");
            
            var collaborationService = _serviceProvider.GetRequiredService<CollaborationService>();
            await collaborationService.InitiateCollaborationAsync();
            
            Console.WriteLine("Collaboration enabled successfully");
            Console.WriteLine();
            
            // Step 3: Open VS Code and guide the user
            CliSupport.WriteSubHeader("Step 3: Setting up VS Code");
            Console.WriteLine("Opening VS Code...");
            
            var vsCodeProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "code",
                    UseShellExecute = true
                }
            };
            
            vsCodeProcess.Start();
            
            Console.WriteLine("Please follow these steps in VS Code:");
            Console.WriteLine("1. Open VS Code Settings (Ctrl+,)");
            Console.WriteLine("2. Search for 'chat.agent.enabled'");
            Console.WriteLine("3. Check the box to enable it");
            Console.WriteLine("4. Open the Chat view (Ctrl+Alt+I)");
            Console.WriteLine("5. Select 'Agent' mode from the dropdown");
            Console.WriteLine();
            
            // Step 4: Wait for user to set up VS Code
            Console.WriteLine("Press Enter when you have completed the VS Code setup...");
            Console.ReadLine();
            
            // Step 5: Show how to use Augment Agent
            CliSupport.WriteSubHeader("Step 4: Using Augment Agent through VS Code");
            Console.WriteLine("In the VS Code Chat view, you can now type:");
            Console.WriteLine($"\n> {task}\n");
            Console.WriteLine("VS Code Agent Mode will use the TARS MCP server to execute the task,");
            Console.WriteLine("collaborating with Augment Agent to provide enhanced capabilities.");
            Console.WriteLine();
            
            Console.WriteLine("You can also use TARS-specific commands in VS Code Agent Mode:");
            Console.WriteLine("- #vscode_agent execute_metascript: Execute a TARS metascript");
            Console.WriteLine("- #vscode_agent analyze_codebase: Analyze the codebase structure and quality");
            Console.WriteLine("- #vscode_agent generate_metascript: Generate a TARS metascript for a specific task");
            Console.WriteLine();
            
            CliSupport.WriteColorLine("Demo completed successfully!", ConsoleColor.Green);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running Augment-VSCode demo");
            CliSupport.WriteColorLine($"Demo failed: {ex.Message}", ConsoleColor.Red);
        }
    }
}
