using Microsoft.Extensions.Logging;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;

namespace TarsCli.Commands;

/// <summary>
/// Command for controlling VS Code
/// </summary>
public class VSCodeControlCommand : Command
{
    private readonly ILogger<VSCodeControlCommand> _logger;
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="VSCodeControlCommand"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="serviceProvider">The service provider</param>
    public VSCodeControlCommand(
        ILogger<VSCodeControlCommand> logger,
        IServiceProvider serviceProvider)
        : base("vscode", "Control VS Code programmatically")
    {
        _logger = logger;
        _serviceProvider = serviceProvider;

        // Add subcommands
        AddCommand(CreateOpenCommand());
        AddCommand(CreateCommandCommand());
        AddCommand(CreateTypeCommand());
        AddCommand(CreateClickCommand());
        AddCommand(CreateDemoCommand());
        AddCommand(CreateAugmentCommand());
    }

    private Command CreateOpenCommand()
    {
        var command = new Command("open", "Open a file in VS Code");
        var filePathArgument = new Argument<string>("file-path", "The path to the file to open");
        command.AddArgument(filePathArgument);

        command.SetHandler(async (string filePath) =>
        {
            _logger.LogInformation($"Opening file in VS Code: {filePath}");

            try
            {
                // Use the VS Code CLI to open the file
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "code",
                        Arguments = $"\"{filePath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                await process.WaitForExitAsync();

                Console.WriteLine(process.ExitCode == 0 ? "File opened successfully" : "Failed to open file");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error opening file in VS Code: {filePath}");
                Console.WriteLine($"Failed to open file: {ex.Message}");
            }
        }, filePathArgument);

        return command;
    }

    private Command CreateCommandCommand()
    {
        var command = new Command("command", "Execute a VS Code command");
        var commandArgument = new Argument<string>("command", "The command to execute");
        command.AddArgument(commandArgument);

        command.SetHandler(async (string cmd) =>
        {
            _logger.LogInformation($"Executing VS Code command: {cmd}");

            try
            {
                // Use PowerShell to execute the VS Code command via keyboard shortcuts
                await ExecuteVSCodeCommandAsync(cmd);
                Console.WriteLine("Command executed successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error executing VS Code command: {cmd}");
                Console.WriteLine($"Failed to execute command: {ex.Message}");
            }
        }, commandArgument);

        return command;
    }

    private async Task ExecuteVSCodeCommandAsync(string command)
    {
        // Open the command palette
        await SendKeysAsync("^+p");

        // Wait for the command palette to open
        await Task.Delay(500);

        // Type the command
        await SendKeysAsync(command);

        // Wait for the command to be found
        await Task.Delay(500);

        // Press Enter to execute the command
        await SendKeysAsync("{ENTER}");
    }

    private async Task SendKeysAsync(string keys)
    {
        // Escape single quotes in the keys string
        var escapedKeys = keys.Replace("'", "''");

        // Create the PowerShell script
        var script = $@"
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.SendKeys]::SendWait('{escapedKeys}')
        ";

        // Execute the PowerShell script
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-Command \"{script}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };

        process.Start();
        await process.WaitForExitAsync();
    }

    private Command CreateTypeCommand()
    {
        var command = new Command("type", "Type text in VS Code");
        var textArgument = new Argument<string>("text", "The text to type");
        command.AddArgument(textArgument);

        command.SetHandler(async (string text) =>
        {
            _logger.LogInformation($"Typing text in VS Code: {text}");

            try
            {
                // Use PowerShell to type the text
                await SendKeysAsync(text);
                Console.WriteLine("Text typed successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error typing text in VS Code");
                Console.WriteLine($"Failed to type text: {ex.Message}");
            }
        }, textArgument);

        return command;
    }

    private Command CreateClickCommand()
    {
        var command = new Command("click", "Click at a specific position in VS Code");
        var xArgument = new Argument<int>("x", "The x-coordinate");
        var yArgument = new Argument<int>("y", "The y-coordinate");
        command.AddArgument(xArgument);
        command.AddArgument(yArgument);

        command.SetHandler(async (int x, int y) =>
        {
            _logger.LogInformation($"Clicking at position ({x}, {y}) in VS Code");

            try
            {
                // Use PowerShell to move the mouse and click
                await MoveMouseAsync(x, y);
                await Task.Delay(100); // Wait for the move to complete
                await ClickMouseAsync();
                Console.WriteLine("Click performed successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error clicking at position ({x}, {y}) in VS Code");
                Console.WriteLine($"Failed to click: {ex.Message}");
            }
        }, xArgument, yArgument);

        return command;
    }

    private async Task MoveMouseAsync(int x, int y)
    {
        // Create the PowerShell script
        var script = $@"
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({x}, {y})
        ";

        // Execute the PowerShell script
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-Command \"{script}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };

        process.Start();
        await process.WaitForExitAsync();
    }

    private async Task ClickMouseAsync(string button = "left")
    {
        // Create the PowerShell script
        var script = $@"
            Add-Type -AssemblyName System.Windows.Forms
            $signature = @""
                [DllImport(""user32.dll"", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
                public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint cButtons, uint dwExtraInfo);
            ""@
            $type = Add-Type -MemberDefinition $signature -Name ""MouseEvents"" -Namespace ""Win32"" -PassThru

            # Constants for mouse events
            $MOUSEEVENTF_LEFTDOWN = 0x02
            $MOUSEEVENTF_LEFTUP = 0x04
            $MOUSEEVENTF_RIGHTDOWN = 0x08
            $MOUSEEVENTF_RIGHTUP = 0x10
            $MOUSEEVENTF_MIDDLEDOWN = 0x20
            $MOUSEEVENTF_MIDDLEUP = 0x40

            # Perform the click based on the button
            switch (""{button.ToLower()}"") {{
                ""left"" {{
                    $type::mouse_event($MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    Start-Sleep -Milliseconds 10
                    $type::mouse_event($MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                }}
                ""right"" {{
                    $type::mouse_event($MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    Start-Sleep -Milliseconds 10
                    $type::mouse_event($MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                }}
                ""middle"" {{
                    $type::mouse_event($MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                    Start-Sleep -Milliseconds 10
                    $type::mouse_event($MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                }}
            }}
        ";

        // Execute the PowerShell script
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-Command \"{script}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };

        process.Start();
        await process.WaitForExitAsync();
    }

    private Command CreateDemoCommand()
    {
        var command = new Command("demo", "Run a demo of VS Code control");

        command.SetHandler(async () =>
        {
            _logger.LogInformation("Running VS Code control demo");

            Console.WriteLine("VS Code Control Demo");
            Console.WriteLine("===================");

            // Give the user time to switch to VS Code
            Console.WriteLine("Please switch to VS Code within 5 seconds...");
            await Task.Delay(5000);

            try
            {
                // Open a new file using keyboard shortcut Ctrl+N
                Console.WriteLine("Creating a new file...");
                await SendKeysAsync("^n");
                await Task.Delay(1000);

                // Type some text
                Console.WriteLine("Typing some text...");
                await SendKeysAsync("// This is a demo of TARS controlling VS Code\n\n");
                await Task.Delay(500);

                await SendKeysAsync("using System;\n\n");
                await Task.Delay(500);

                await SendKeysAsync("public class TarsDemo\n{\n    public static void Main(string[] args)\n    {\n        Console.WriteLine(\"Hello from TARS!\");\n    }\n}");
                await Task.Delay(1000);

                // Format the document using keyboard shortcut Alt+Shift+F
                Console.WriteLine("Formatting the document...");
                await SendKeysAsync("%+f");
                await Task.Delay(1000);

                // Save the file using keyboard shortcut Ctrl+S
                Console.WriteLine("Saving the file...");
                await SendKeysAsync("^s");
                await Task.Delay(1000);

                Console.WriteLine("Demo completed successfully!");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running VS Code control demo");
                Console.WriteLine($"Demo failed: {ex.Message}");
            }
        });

        return command;
    }

    private Command CreateAugmentCommand()
    {
        var command = new Command("augment", "Use Augment Agent through VS Code");
        var taskArgument = new Argument<string>("task", "The task to perform with Augment Agent");
        command.AddArgument(taskArgument);

        command.SetHandler(async (string task) =>
        {
            _logger.LogInformation($"Using Augment Agent through VS Code for task: {task}");

            Console.WriteLine("TARS-Augment-VSCode Collaboration");
            Console.WriteLine("================================");

            try
            {
                // Step 1: Start the MCP server if not already running
                Console.WriteLine("Step 1: Starting TARS MCP server...");
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "tarscli",
                        Arguments = "mcp start",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                await Task.Delay(2000); // Wait for the MCP server to start

                // Step 2: Enable collaboration
                Console.WriteLine("Step 2: Enabling collaboration...");
                var collaborateProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "tarscli",
                        Arguments = "mcp collaborate start",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                collaborateProcess.Start();
                await collaborateProcess.WaitForExitAsync();

                // Step 3: Open VS Code and enable Agent Mode
                Console.WriteLine("Step 3: Opening VS Code and enabling Agent Mode...");
                Console.WriteLine("Please follow these steps in VS Code:");
                Console.WriteLine("1. Open VS Code Settings (Ctrl+,)");
                Console.WriteLine("2. Search for 'chat.agent.enabled'");
                Console.WriteLine("3. Check the box to enable it");
                Console.WriteLine("4. Open the Chat view (Ctrl+Alt+I)");
                Console.WriteLine("5. Select 'Agent' mode from the dropdown");

                // Step 4: Open VS Code
                var vsCodeProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "code",
                        UseShellExecute = true
                    }
                };

                vsCodeProcess.Start();

                // Step 5: Wait for user to set up VS Code
                Console.WriteLine("\nPress Enter when you have completed the VS Code setup...");
                Console.ReadLine();

                // Step 6: Show how to use Augment Agent
                Console.WriteLine("\nStep 4: Using Augment Agent through VS Code");
                Console.WriteLine("In the VS Code Chat view, you can now type:");
                Console.WriteLine($"\n> {task}\n");
                Console.WriteLine("VS Code Agent Mode will use the TARS MCP server to execute the task,");
                Console.WriteLine("collaborating with Augment Agent to provide enhanced capabilities.");

                Console.WriteLine("\nYou can also use TARS-specific commands in VS Code Agent Mode:");
                Console.WriteLine("- #vscode_agent execute_metascript: Execute a TARS metascript");
                Console.WriteLine("- #vscode_agent analyze_codebase: Analyze the codebase structure and quality");
                Console.WriteLine("- #vscode_agent generate_metascript: Generate a TARS metascript for a specific task");

                Console.WriteLine("\nCollaboration setup completed successfully!");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error setting up TARS-Augment-VSCode collaboration");
                Console.WriteLine($"Collaboration setup failed: {ex.Message}");
            }
        }, taskArgument);

        return command;
    }
}
