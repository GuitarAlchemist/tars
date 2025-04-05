using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Mcp;

public class McpController
{
    private readonly ILogger<McpController> _logger;
    private readonly IConfiguration? _configuration;
    private readonly bool _autoExecuteEnabled;
    private readonly TarsEngine.Services.Interfaces.IKnowledgeTransferService? _knowledgeTransferService;

    public McpController(ILogger<McpController> logger, IConfiguration? configuration = null, TarsEngine.Services.Interfaces.IKnowledgeTransferService? knowledgeTransferService = null)
    {
        _logger = logger;
        _configuration = configuration;
        _knowledgeTransferService = knowledgeTransferService;

        // Check if auto-execute is enabled in configuration
        var autoExecuteValue = _configuration?["Tars:Mcp:AutoExecuteEnabled"];
        _autoExecuteEnabled = autoExecuteValue != null && (autoExecuteValue.Equals("true", StringComparison.OrdinalIgnoreCase) || autoExecuteValue.Equals("1"));

        _logger.LogInformation($"MCP Controller initialized. Auto-execute: {_autoExecuteEnabled}");
    }

    public async Task<string> ExecuteCommand(string commandType, string target = "")
    {
        _logger.LogInformation($"Executing MCP command: {commandType} with target: {target}");

        try
        {
            return commandType switch
            {
                "run" => RunApplication(target),
                "processes" => ListProcesses(),
                "status" => GetSystemStatus(),
                "execute" => await ExecuteTerminalCommand(target),
                "code" => await GenerateCode(target),
                "augment" => ConfigureAugmentMcp(target),
                "extract-knowledge" => await ExtractKnowledgeAsync(target),
                "share-knowledge" => await ShareKnowledgeAsync(target),
                "retrieve-knowledge" => await RetrieveKnowledgeAsync(target),
                "apply-knowledge" => await ApplyKnowledgeAsync(target),
                // Add more commands as needed
                _ => $"Unknown command: {commandType}"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing MCP command {commandType}");
            return $"Error: {ex.Message}";
        }
    }

    private string RunApplication(string appPath)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = appPath,
                UseShellExecute = true
            });
            return $"Started application: {appPath}";
        }
        catch (Exception ex)
        {
            return $"Failed to start application: {ex.Message}";
        }
    }

    private string ListProcesses()
    {
        var processes = Process.GetProcesses();
        var result = new System.Text.StringBuilder();

        foreach (var process in processes)
        {
            result.AppendLine($"{process.Id}: {process.ProcessName}");
        }

        return result.ToString();
    }

    private string GetSystemStatus()
    {
        // Basic system info - expand as needed
        return $"System: {Environment.OSVersion}\n" +
               $"Machine: {Environment.MachineName}\n" +
               $"Processors: {Environment.ProcessorCount}\n" +
               $"Memory: {GC.GetTotalMemory(false) / (1024 * 1024)} MB";
    }

    private async Task<string> ExecuteTerminalCommand(string command)
    {
        if (!_autoExecuteEnabled)
        {
            _logger.LogWarning($"Auto-execute is disabled. Command not executed: {command}");
            return $"Auto-execute is disabled. Command not executed: {command}";
        }

        _logger.LogInformation($"Executing terminal command: {command}");

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

            return $"Command executed successfully:\n{output}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing terminal command: {command}");
            return $"Error executing command: {ex.Message}";
        }
    }

    private async Task<string> GenerateCode(string codeSpec)
    {
        if (!_autoExecuteEnabled)
        {
            _logger.LogWarning($"Auto-execute is disabled. Code generation not performed.");
            return $"Auto-execute is disabled. Code generation not performed.";
        }

        try
        {
            string filePath;
            string content;

            // Check if the input is JSON or direct parameters
            if (codeSpec.TrimStart().StartsWith("{"))
            {
                // Parse the code specification JSON
                var codeSpecObj = JsonSerializer.Deserialize<JsonElement>(codeSpec);

                if (!codeSpecObj.TryGetProperty("filePath", out var filePathElement) ||
                    !codeSpecObj.TryGetProperty("content", out var contentElement))
                {
                    return "Invalid code specification. Must include 'filePath' and 'content'.";
                }

                filePath = filePathElement.GetString();
                content = contentElement.GetString();
            }
            else
            {
                // Parse direct parameters (filePath:::content format)
                var parts = codeSpec.Split([":::"], 2, StringSplitOptions.None);
                if (parts.Length != 2)
                {
                    return "Invalid code specification format. Use filePath:::content format.";
                }

                filePath = parts[0];
                content = parts[1];

                // Handle triple-quoted strings by removing the triple quotes
                if (content.StartsWith("\"\"\"") && content.EndsWith("\"\"\""))
                {
                    content = content.Substring(3, content.Length - 6);
                }
                // Also check for the -triple-quoted syntax
                else if (content == "-triple-quoted")
                {
                    // The next part should be the actual content
                    if (parts.Length > 2)
                    {
                        content = parts[2];

                        // If the content is wrapped in triple quotes, remove them
                        if (content.StartsWith("\"\"\"") && content.EndsWith("\"\"\""))
                        {
                            content = content.Substring(3, content.Length - 6);
                        }
                    }
                    else
                    {
                        return "Missing content after -triple-quoted flag.";
                    }
                }
                // Handle the case where the content itself is a triple-quoted string
                else if (parts.Length == 2 && content.Contains("\n"))
                {
                    // This is likely a multi-line string that was triple-quoted
                    // No need to do anything special, just use it as-is
                }
            }

            if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(content))
            {
                return "Invalid code specification. 'filePath' and 'content' cannot be empty.";
            }

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Write the content to the file
            await File.WriteAllTextAsync(filePath, content);

            return $"Code generated and saved to: {filePath}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating code");
            return $"Error generating code: {ex.Message}";
        }
    }

    private async Task<string> ExtractKnowledgeAsync(string text)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from text of length {Length}", text.Length);

            if (_knowledgeTransferService == null)
            {
                return "Error: Knowledge transfer service is not available";
            }

            var knowledgeItems = await _knowledgeTransferService.ExtractKnowledgeAsync(text);
            var itemsList = knowledgeItems.ToList();

            _logger.LogInformation("Extracted {Count} knowledge items", itemsList.Count);

            return System.Text.Json.JsonSerializer.Serialize(itemsList, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge");
            return $"Error extracting knowledge: {ex.Message}";
        }
    }

    private async Task<string> ShareKnowledgeAsync(string target)
    {
        try
        {
            _logger.LogInformation("Sharing knowledge with {Target}", target);

            if (_knowledgeTransferService == null)
            {
                return "Error: Knowledge transfer service is not available";
            }

            // Parse the target as JSON
            var targetData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(target);
            if (targetData == null)
            {
                return "Error: Invalid target data";
            }

            // Get the knowledge items
            if (!targetData.TryGetValue("items", out var itemsObj) || itemsObj == null)
            {
                return "Error: No knowledge items specified";
            }

            // Get the target system
            if (!targetData.TryGetValue("target", out var targetSystemObj) || targetSystemObj == null)
            {
                return "Error: No target system specified";
            }

            // Get the options
            Dictionary<string, string>? options = null;
            if (targetData.TryGetValue("options", out var optionsObj) && optionsObj != null)
            {
                options = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(optionsObj.ToString() ?? "{}");
            }

            // Convert the items to KnowledgeItems
            var itemsJson = System.Text.Json.JsonSerializer.Serialize(itemsObj);
            var knowledgeItems = System.Text.Json.JsonSerializer.Deserialize<List<TarsEngine.Services.Interfaces.KnowledgeItem>>(itemsJson);
            if (knowledgeItems == null)
            {
                return "Error: Invalid knowledge items";
            }

            // Share the knowledge
            var result = await _knowledgeTransferService.ShareKnowledgeAsync(knowledgeItems, targetSystemObj.ToString() ?? "", options);

            return System.Text.Json.JsonSerializer.Serialize(result, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sharing knowledge");
            return $"Error sharing knowledge: {ex.Message}";
        }
    }

    private async Task<string> RetrieveKnowledgeAsync(string target)
    {
        try
        {
            _logger.LogInformation("Retrieving knowledge with query: {Query}", target);

            if (_knowledgeTransferService == null)
            {
                return "Error: Knowledge transfer service is not available";
            }

            // Parse the target as JSON
            var targetData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(target);
            if (targetData == null)
            {
                return "Error: Invalid target data";
            }

            // Get the query
            if (!targetData.TryGetValue("query", out var queryObj) || queryObj == null)
            {
                return "Error: No query specified";
            }

            // Get the source
            if (!targetData.TryGetValue("source", out var sourceObj) || sourceObj == null)
            {
                return "Error: No source specified";
            }

            // Get the options
            Dictionary<string, string>? options = null;
            if (targetData.TryGetValue("options", out var optionsObj) && optionsObj != null)
            {
                options = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(optionsObj.ToString() ?? "{}");
            }

            // Retrieve the knowledge
            var knowledgeItems = await _knowledgeTransferService.RetrieveKnowledgeAsync(queryObj.ToString() ?? "", sourceObj.ToString() ?? "", options);
            var itemsList = knowledgeItems.ToList();

            _logger.LogInformation("Retrieved {Count} knowledge items", itemsList.Count);

            return System.Text.Json.JsonSerializer.Serialize(itemsList, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving knowledge");
            return $"Error retrieving knowledge: {ex.Message}";
        }
    }

    private async Task<string> ApplyKnowledgeAsync(string target)
    {
        try
        {
            _logger.LogInformation("Applying knowledge to {Target}", target);

            if (_knowledgeTransferService == null)
            {
                return "Error: Knowledge transfer service is not available";
            }

            // Parse the target as JSON
            var targetData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(target);
            if (targetData == null)
            {
                return "Error: Invalid target data";
            }

            // Get the knowledge items
            if (!targetData.TryGetValue("items", out var itemsObj) || itemsObj == null)
            {
                return "Error: No knowledge items specified";
            }

            // Get the target system
            if (!targetData.TryGetValue("target", out var targetSystemObj) || targetSystemObj == null)
            {
                return "Error: No target system specified";
            }

            // Get the options
            Dictionary<string, string>? options = null;
            if (targetData.TryGetValue("options", out var optionsObj) && optionsObj != null)
            {
                options = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(optionsObj.ToString() ?? "{}");
            }

            // Convert the items to KnowledgeItems
            var itemsJson = System.Text.Json.JsonSerializer.Serialize(itemsObj);
            var knowledgeItems = System.Text.Json.JsonSerializer.Deserialize<List<TarsEngine.Services.Interfaces.KnowledgeItem>>(itemsJson);
            if (knowledgeItems == null)
            {
                return "Error: Invalid knowledge items";
            }

            // Apply the knowledge
            var result = await _knowledgeTransferService.ApplyKnowledgeAsync(knowledgeItems, targetSystemObj.ToString() ?? "", options);

            return System.Text.Json.JsonSerializer.Serialize(result, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying knowledge");
            return $"Error applying knowledge: {ex.Message}";
        }
    }

    private string ConfigureAugmentMcp(string configSpec)
    {
        try
        {
            // Parse the configuration specification JSON
            var configSpecObj = JsonSerializer.Deserialize<JsonElement>(configSpec);

            if (!configSpecObj.TryGetProperty("serverName", out var serverNameElement) ||
                !configSpecObj.TryGetProperty("command", out var commandElement) ||
                !configSpecObj.TryGetProperty("args", out var argsElement))
            {
                return "Invalid configuration specification. Must include 'serverName', 'command', and 'args'.";
            }

            var serverName = serverNameElement.GetString();
            var command = commandElement.GetString();
            var args = argsElement.EnumerateArray().Select(a => a.GetString()).ToArray();

            if (string.IsNullOrEmpty(serverName) || string.IsNullOrEmpty(command))
            {
                return "Invalid configuration specification. 'serverName' and 'command' cannot be empty.";
            }

            // Get the VS Code settings file path
            var settingsPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "Code", "User", "settings.json");

            // Check if the settings file exists
            if (!File.Exists(settingsPath))
            {
                return $"VS Code settings file not found: {settingsPath}";
            }

            // Read the current settings
            var settingsJson = File.ReadAllText(settingsPath);
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

                        // Create the new MCP server configuration
                        var mcpServerConfig = new Dictionary<string, object>
                        {
                            { "name", serverName },
                            { "command", command },
                            { "args", args }
                        };

                        mcpServers.Add(mcpServerConfig);
                        advancedSettings["mcpServers"] = mcpServers;
                    }
                    else
                    {
                        // Create new mcpServers array
                        var mcpServerConfig = new Dictionary<string, object>
                        {
                            { "name", serverName },
                            { "command", command },
                            { "args", args }
                        };

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
                var mcpServerConfig = new Dictionary<string, object>
                {
                    { "name", serverName },
                    { "command", command },
                    { "args", args }
                };

                newSettingsObject["augment.advanced"] = new Dictionary<string, object>
                {
                    ["mcpServers"] = new List<object> { mcpServerConfig }
                };
            }

            // Serialize the updated settings
            var newSettingsJson = JsonSerializer.Serialize(newSettingsObject, new JsonSerializerOptions { WriteIndented = true });

            // Save the updated settings
            File.WriteAllText(settingsPath, newSettingsJson);

            return $"Augment MCP server '{serverName}' configured successfully.";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error configuring Augment MCP");
            return $"Error configuring Augment MCP: {ex.Message}";
        }
    }
}