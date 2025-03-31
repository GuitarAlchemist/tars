using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.Text.Json;
using TarsCli.Mcp;

namespace TarsCli.Services;

public class EnhancedMcpService
{
    private readonly ILogger<EnhancedMcpService> _logger;
    private readonly IConfiguration _configuration;
    private readonly McpController _mcpController;
    private readonly bool _autoExecuteCommands;
    private readonly bool _autoCodeGeneration;

    public EnhancedMcpService(
        ILogger<EnhancedMcpService> logger,
        IConfiguration configuration,
        McpController mcpController)
    {
        _logger = logger;
        _configuration = configuration;
        _mcpController = mcpController;

        // Get auto-execution settings from configuration
        _autoExecuteCommands = _configuration.GetValue<bool>("Tars:Mcp:AutoExecuteCommands", false);
        _autoCodeGeneration = _configuration.GetValue<bool>("Tars:Mcp:AutoCodeGeneration", false);

        _logger.LogInformation($"Enhanced MCP Service initialized. Auto-execute: {_autoExecuteCommands}, Auto-code: {_autoCodeGeneration}");
    }

    /// <summary>
    /// Executes a terminal command without asking for permission if auto-execute is enabled
    /// </summary>
    public async Task<string> ExecuteTerminalCommand(string command, string? workingDirectory = null)
    {
        _logger.LogInformation($"Executing terminal command: {command}");

        if (!_autoExecuteCommands)
        {
            _logger.LogWarning("Auto-execute is disabled. Command will not be executed.");
            return $"Auto-execute is disabled. Command not executed: {command}";
        }

        try
        {
            var processStartInfo = new ProcessStartInfo
            {
                FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "cmd.exe" : "/bin/bash",
                Arguments = Environment.OSVersion.Platform == PlatformID.Win32NT ? $"/c {command}" : $"-c \"{command}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            if (!string.IsNullOrEmpty(workingDirectory))
            {
                processStartInfo.WorkingDirectory = workingDirectory;
            }

            var process = Process.Start(processStartInfo);

            if (process == null)
            {
                return "Failed to start process.";
            }

            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();

            await process.WaitForExitAsync();

            if (!string.IsNullOrEmpty(error))
            {
                _logger.LogWarning($"Command execution produced errors: {error}");
                return $"Command executed with errors:\nOutput: {output}\nErrors: {error}";
            }

            _logger.LogInformation($"Command executed successfully: {command}");
            return $"Command executed successfully:\n{output}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing terminal command: {command}");
            return $"Error executing command: {ex.Message}";
        }
    }

    /// <summary>
    /// Generates and saves code to a file without asking for permission if auto-code is enabled
    /// </summary>
    public async Task<string> GenerateAndSaveCode(string filePath, string codeContent)
    {
        _logger.LogInformation($"Generating code for file: {filePath}");

        if (!_autoCodeGeneration)
        {
            _logger.LogWarning("Auto-code generation is disabled. Code will not be saved.");
            return $"Auto-code generation is disabled. Code not saved to: {filePath}";
        }

        try
        {
            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Save the code to the file
            await File.WriteAllTextAsync(filePath, codeContent);

            _logger.LogInformation($"Code generated and saved to: {filePath}");
            return $"Code generated and saved to: {filePath}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating code for file: {filePath}");
            return $"Error generating code: {ex.Message}";
        }
    }

    /// <summary>
    /// Runs a code generation and execution workflow
    /// </summary>
    public async Task<string> RunCodeGenerationWorkflow(string taskDescription, string outputDirectory)
    {
        _logger.LogInformation($"Running code generation workflow for task: {taskDescription}");

        try
        {
            // Create output directory if it doesn't exist
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            // Step 1: Generate code based on task description
            var codeFilePath = Path.Combine(outputDirectory, "generated_code.cs");
            var codeContent = await GenerateCodeFromDescription(taskDescription);

            // Step 2: Save the generated code
            await GenerateAndSaveCode(codeFilePath, codeContent);

            // Step 3: Compile the code if it's a C# file
            if (codeFilePath.EndsWith(".cs"))
            {
                var compileResult = await ExecuteTerminalCommand($"dotnet build {codeFilePath}", outputDirectory);

                if (compileResult.Contains("error"))
                {
                    return $"Code generation completed but compilation failed:\n{compileResult}";
                }
            }

            // Step 4: Run the compiled code
            var runResult = await ExecuteTerminalCommand($"dotnet run --project {codeFilePath}", outputDirectory);

            return $"Code generation workflow completed successfully:\n{runResult}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running code generation workflow for task: {taskDescription}");
            return $"Error running code generation workflow: {ex.Message}";
        }
    }

    /// <summary>
    /// Generates code from a task description using an AI model
    /// </summary>
    private async Task<string> GenerateCodeFromDescription(string taskDescription)
    {
        _logger.LogInformation($"Generating code from description: {taskDescription}");

        // This is a placeholder for actual AI code generation
        // In a real implementation, this would call an AI model to generate code

        var codeTemplate = @"
using System;

namespace GeneratedCode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine(""Generated code for task: {0}"");
            Console.WriteLine(""This is a placeholder implementation."");

            // TODO: Implement the actual task

            Console.WriteLine(""Task completed successfully."");
        }
    }
}";

        return string.Format(codeTemplate, taskDescription);
    }

    /// <summary>
    /// Executes an MCP command through the MCP controller
    /// </summary>
    public async Task<string> ExecuteMcpCommand(string commandType, string target = "")
    {
        _logger.LogInformation($"Executing MCP command: {commandType} with target: {target}");

        try
        {
            return await _mcpController.ExecuteCommand(commandType, target);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing MCP command: {commandType}");
            return $"Error executing MCP command: {ex.Message}";
        }
    }

    /// <summary>
    /// Configures an MCP server for integration with Augment Code
    /// </summary>
    public async Task<string> ConfigureMcpServer(string serverName, string command, string[] args)
    {
        _logger.LogInformation($"Configuring MCP server: {serverName}");

        try
        {
            // Create the MCP server configuration
            var mcpServerConfig = new
            {
                name = serverName,
                command = command,
                args = args
            };

            // Get the current settings file path
            var settingsPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "Code", "User", "settings.json");

            // Check if the settings file exists
            if (!File.Exists(settingsPath))
            {
                _logger.LogWarning($"Settings file not found: {settingsPath}");
                return $"Settings file not found: {settingsPath}";
            }

            // Read the current settings
            var settingsJson = await File.ReadAllTextAsync(settingsPath);
            var settings = JsonDocument.Parse(settingsJson);

            // Create a new settings object with the MCP server configuration
            var newSettingsObject = new Dictionary<string, object>();

            // Copy existing settings
            foreach (var property in settings.RootElement.EnumerateObject())
            {
                if (property.Name == "augment.advanced")
                {
                    // Update the augment.advanced settings
                    var advancedSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(property.Value.GetRawText());

                    if (advancedSettings.ContainsKey("mcpServers"))
                    {
                        // Update existing mcpServers array
                        var mcpServers = JsonSerializer.Deserialize<List<object>>(advancedSettings["mcpServers"].ToString());
                        mcpServers.Add(mcpServerConfig);
                        advancedSettings["mcpServers"] = mcpServers;
                    }
                    else
                    {
                        // Create new mcpServers array
                        advancedSettings["mcpServers"] = new List<object> { mcpServerConfig };
                    }

                    newSettingsObject[property.Name] = advancedSettings;
                }
                else
                {
                    // Copy other settings as-is
                    newSettingsObject[property.Name] = JsonSerializer.Deserialize<object>(property.Value.GetRawText());
                }
            }

            // If augment.advanced doesn't exist, create it
            if (!newSettingsObject.ContainsKey("augment.advanced"))
            {
                newSettingsObject["augment.advanced"] = new Dictionary<string, object>
                {
                    ["mcpServers"] = new List<object> { mcpServerConfig }
                };
            }

            // Serialize the updated settings
            var newSettingsJson = JsonSerializer.Serialize(newSettingsObject, new JsonSerializerOptions { WriteIndented = true });

            // Save the updated settings
            await File.WriteAllTextAsync(settingsPath, newSettingsJson);

            _logger.LogInformation($"MCP server configured: {serverName}");
            return $"MCP server configured: {serverName}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error configuring MCP server: {serverName}");
            return $"Error configuring MCP server: {ex.Message}";
        }
    }
}
