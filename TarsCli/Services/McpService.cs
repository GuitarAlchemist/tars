using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.Http.Json;
using System.Text;

namespace TarsCli.Services;

/// <summary>
/// Implementation of the Model Context Protocol (MCP) service
/// Based on Anthropic's MCP specification: https://www.anthropic.com/news/model-context-protocol
/// </summary>
public class McpService
{
    private readonly ILogger<McpService> _logger;
    private readonly IConfiguration _configuration;
    private readonly HttpClient _httpClient;
    private readonly string _serviceUrl;
    private readonly int _servicePort;
    private HttpListener? _listener;
    private bool _isRunning = false;
    private readonly Dictionary<string, Func<JsonElement, Task<JsonElement>>> _handlers = new();
    private readonly ConversationLoggingService? _conversationLoggingService;

    public McpService(
        ILogger<McpService> logger,
        IConfiguration configuration,
        ConversationLoggingService? conversationLoggingService = null)
    {
        _logger = logger;
        _configuration = configuration;
        _httpClient = new HttpClient();
        _conversationLoggingService = conversationLoggingService;

        // Get configuration values
        _servicePort = _configuration.GetValue<int>("Tars:Mcp:Port", 8999);
        _serviceUrl = $"http://localhost:{_servicePort}/";

        // Register default handlers
        RegisterDefaultHandlers();

        _logger.LogInformation($"MCP Service initialized with URL: {_serviceUrl}");
    }

    /// <summary>
    /// Register the default MCP handlers
    /// </summary>
    private void RegisterDefaultHandlers()
    {
        // Register the execute handler
        RegisterHandler("execute", async (request) =>
        {
            var command = request.GetProperty("command").GetString();
            _logger.LogInformation($"Executing command: {command}");

            var result = await ExecuteCommand(command);
            return JsonSerializer.SerializeToElement(new { success = true, output = result });
        });

        // Register the code handler
        RegisterHandler("code", async (request) =>
        {
            var filePath = request.GetProperty("filePath").GetString();
            var content = request.GetProperty("content").GetString();
            _logger.LogInformation($"Generating code for file: {filePath}");

            var result = await GenerateCode(filePath, content);
            return JsonSerializer.SerializeToElement(new { success = true, message = result });
        });

        // Register the status handler
        RegisterHandler("status", async (request) =>
        {
            _logger.LogInformation("Getting system status");

            var status = GetSystemStatus();
            return JsonSerializer.SerializeToElement(new { success = true, status = status });
        });

        // Register the tars handler for TARS-specific operations
        RegisterHandler("tars", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing TARS operation: {operation}");

            var result = await ExecuteTarsOperation(operation, request);
            return result;
        });
    }

    /// <summary>
    /// Register a handler for a specific MCP action
    /// </summary>
    public void RegisterHandler(string action, Func<JsonElement, Task<JsonElement>> handler)
    {
        _handlers[action] = handler;
        _logger.LogInformation($"Registered handler for action: {action}");
    }

    /// <summary>
    /// Start the MCP service
    /// </summary>
    public async Task StartAsync()
    {
        if (_isRunning)
        {
            _logger.LogWarning("MCP Service is already running");
            return;
        }

        try
        {
            _listener = new HttpListener();
            _listener.Prefixes.Add(_serviceUrl);
            _listener.Start();

            _isRunning = true;
            _logger.LogInformation($"MCP Service started on {_serviceUrl}");

            // Start listening for requests
            await Task.Run(async () =>
            {
                while (_isRunning)
                {
                    try
                    {
                        var context = await _listener.GetContextAsync();
                        _ = ProcessRequestAsync(context);
                    }
                    catch (Exception ex)
                    {
                        if (_isRunning)
                        {
                            _logger.LogError(ex, "Error processing MCP request");
                        }
                    }
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting MCP Service");
            _isRunning = false;
            throw;
        }
    }

    /// <summary>
    /// Stop the MCP service
    /// </summary>
    public void Stop()
    {
        if (!_isRunning)
        {
            return;
        }

        _isRunning = false;
        _listener?.Stop();
        _listener = null;

        _logger.LogInformation("MCP Service stopped");
    }

    /// <summary>
    /// Process an incoming MCP request
    /// </summary>
    private async Task ProcessRequestAsync(HttpListenerContext context)
    {
        try
        {
            var request = context.Request;
            var response = context.Response;

            // Only accept POST requests
            if (request.HttpMethod != "POST")
            {
                response.StatusCode = 405; // Method Not Allowed
                response.Close();
                return;
            }

            // Read the request body
            string requestBody;
            using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            {
                requestBody = await reader.ReadToEndAsync();
            }

            _logger.LogInformation($"Received MCP request: {requestBody}");

            // Parse the request
            var requestJson = JsonSerializer.Deserialize<JsonElement>(requestBody);

            // Get the action
            if (!requestJson.TryGetProperty("action", out var actionElement))
            {
                await SendErrorResponse(response, "Missing 'action' property");
                return;
            }

            var action = actionElement.GetString();

            // Execute the handler for the action
            if (_handlers.TryGetValue(action, out var handler))
            {
                var result = await handler(requestJson);

                // Log the conversation if logging service is available
                if (_conversationLoggingService != null)
                {
                    // Try to determine the source from the request headers
                    string source = "unknown";
                    if (request.Headers["User-Agent"] != null && request.Headers["User-Agent"].Contains("Augment"))
                    {
                        source = "augment";
                    }
                    else if (request.Headers["X-Source"] != null)
                    {
                        source = request.Headers["X-Source"];
                    }

                    // Log the conversation
                    await _conversationLoggingService.LogConversationAsync(source, action, requestJson, result);
                }

                // Send the response
                response.ContentType = "application/json";
                response.StatusCode = 200;

                var responseBytes = Encoding.UTF8.GetBytes(result.ToString());
                response.ContentLength64 = responseBytes.Length;
                await response.OutputStream.WriteAsync(responseBytes);
            }
            else
            {
                await SendErrorResponse(response, $"Unknown action: {action}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing MCP request");
            await SendErrorResponse(context.Response, $"Error: {ex.Message}");
        }
        finally
        {
            context.Response.Close();
        }
    }

    /// <summary>
    /// Send an error response
    /// </summary>
    private async Task SendErrorResponse(HttpListenerResponse response, string message)
    {
        response.ContentType = "application/json";
        response.StatusCode = 400;

        var errorResponse = JsonSerializer.Serialize(new { success = false, error = message });
        var responseBytes = Encoding.UTF8.GetBytes(errorResponse);
        response.ContentLength64 = responseBytes.Length;
        await response.OutputStream.WriteAsync(responseBytes);
    }

    /// <summary>
    /// Execute a terminal command
    /// </summary>
    public async Task<string> ExecuteCommand(string command)
    {
        try
        {
            var autoExecuteEnabled = _configuration.GetValue<bool>("Tars:Mcp:AutoExecuteEnabled", false);
            if (!autoExecuteEnabled)
            {
                _logger.LogWarning($"Auto-execute is disabled. Command not executed: {command}");
                return "Auto-execute is disabled. Command not executed.";
            }

            var processStartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "cmd.exe" : "/bin/bash",
                Arguments = Environment.OSVersion.Platform == PlatformID.Win32NT ? $"/c {command}" : $"-c \"{command}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = System.Diagnostics.Process.Start(processStartInfo);
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();

            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogWarning($"Command exited with code {process.ExitCode}: {error}");
                return $"Command exited with code {process.ExitCode}: {error}";
            }

            return output;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing command: {command}");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Generate and save code to a file
    /// </summary>
    public async Task<string> GenerateCode(string filePath, string content)
    {
        try
        {
            var autoCodeEnabled = _configuration.GetValue<bool>("Tars:Mcp:AutoCodeEnabled", false);
            if (!autoCodeEnabled)
            {
                _logger.LogWarning($"Auto-code generation is disabled. Code not saved to: {filePath}");
                return "Auto-code generation is disabled. Code not saved.";
            }

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Save the code to the file
            await File.WriteAllTextAsync(filePath, content);

            _logger.LogInformation($"Code generated and saved to: {filePath}");
            return $"Code generated and saved to: {filePath}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating code for file: {filePath}");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Get system status information
    /// </summary>
    private Dictionary<string, object> GetSystemStatus()
    {
        return new Dictionary<string, object>
        {
            ["system"] = Environment.OSVersion.ToString(),
            ["machine"] = Environment.MachineName,
            ["processors"] = Environment.ProcessorCount,
            ["memory"] = GC.GetTotalMemory(false) / (1024 * 1024),
            ["dotnet"] = Environment.Version.ToString(),
            ["mcp_version"] = "1.0.0"
        };
    }

    /// <summary>
    /// Execute a TARS-specific operation
    /// </summary>
    private async Task<JsonElement> ExecuteTarsOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "version":
                    var version = typeof(McpService).Assembly.GetName().Version.ToString();
                    return JsonSerializer.SerializeToElement(new { success = true, version = version });

                case "capabilities":
                    var capabilities = new Dictionary<string, object>
                    {
                        ["execute"] = "Execute terminal commands",
                        ["code"] = "Generate and save code",
                        ["status"] = "Get system status",
                        ["tars"] = "Execute TARS-specific operations"
                    };
                    return JsonSerializer.SerializeToElement(new { success = true, capabilities = capabilities });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown TARS operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing TARS operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Send an MCP request to another MCP service
    /// </summary>
    public async Task<JsonElement> SendRequestAsync(string url, string action, object data)
    {
        try
        {
            var request = new Dictionary<string, object>
            {
                ["action"] = action,
            };

            // Add all properties from data
            foreach (var property in data.GetType().GetProperties())
            {
                request[property.Name] = property.GetValue(data);
            }

            // Add a custom header to identify the source
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = JsonContent.Create(request)
            };
            httpRequest.Headers.Add("X-Source", "tars");

            var response = await _httpClient.SendAsync(httpRequest);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            var responseJson = JsonSerializer.Deserialize<JsonElement>(responseContent);

            // Log the conversation if logging service is available
            if (_conversationLoggingService != null)
            {
                await _conversationLoggingService.LogConversationAsync("tars", action, request, responseJson);
            }

            return responseJson;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error sending MCP request to {url}");
            throw;
        }
    }
}