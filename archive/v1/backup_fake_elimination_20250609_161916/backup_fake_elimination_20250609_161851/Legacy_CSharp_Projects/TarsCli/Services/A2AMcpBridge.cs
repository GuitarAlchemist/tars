using System.Text;
using System.Text.Json;
using TarsEngine.A2A;
using A2ATask = TarsEngine.A2A.Task;
using A2ATaskStatus = TarsEngine.A2A.TaskStatus;

namespace TarsCli.Services;

/// <summary>
/// Bridge between A2A protocol and MCP protocol
/// </summary>
public class A2AMcpBridge
{
    private readonly ILogger<A2AMcpBridge> _logger;
    private readonly McpService _mcpService;
    private readonly A2AServer _a2aServer;
    private readonly Dictionary<string, string> _taskToMcpSessionMap = new();

    /// <summary>
    /// Initializes a new instance of the A2AMcpBridge class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="mcpService">MCP service instance</param>
    /// <param name="a2aServer">A2A server instance</param>
    public A2AMcpBridge(ILogger<A2AMcpBridge> logger, McpService mcpService, A2AServer a2aServer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _mcpService = mcpService ?? throw new ArgumentNullException(nameof(mcpService));
        _a2aServer = a2aServer ?? throw new ArgumentNullException(nameof(a2aServer));
    }

    /// <summary>
    /// Initializes the bridge
    /// </summary>
    public void Initialize()
    {
        // Register A2A task handlers for each skill
        // Create a temporary agent card for testing
        var agentCard = new AgentCard
        {
            Name = "TARS Agent",
            Description = "TARS Agent with A2A protocol support",
            Url = "http://localhost:8998/",
            Skills =
            [
                new() { Id = "code_generation", Name = "Code Generation" },
                new() { Id = "code_analysis", Name = "Code Analysis" },
                new() { Id = "metascript_execution", Name = "Metascript Execution" },
                new() { Id = "knowledge_extraction", Name = "Knowledge Extraction" },
                new() { Id = "self_improvement", Name = "Self Improvement" }
            ]
        };

        foreach (var skill in agentCard.Skills)
        {
            _a2aServer.RegisterTaskHandler(skill.Id, HandleA2ATaskAsync);
        }

        // Register MCP handler for A2A
        // Register handler for A2A operations
        _mcpService.RegisterHandler("a2a", HandleMcpRequestAsync);

        _logger.LogInformation("A2A-MCP bridge initialized");
    }

    /// <summary>
    /// Handles an A2A task
    /// </summary>
    private async Task<A2ATask> HandleA2ATaskAsync(Message message, CancellationToken cancellationToken)
    {
        try
        {
            // Extract skill ID from message metadata
            string skillId = null;
            if (message.Metadata != null && message.Metadata.TryGetValue("skillId", out var skillIdObj))
            {
                skillId = skillIdObj.ToString();
            }

            if (string.IsNullOrEmpty(skillId))
            {
                throw new ArgumentException("Missing skillId in message metadata");
            }

            // Create a new MCP session or use an existing one
            string sessionId;
            if (message.Metadata != null && message.Metadata.TryGetValue("taskId", out var taskIdObj))
            {
                var taskId = taskIdObj.ToString();
                if (_taskToMcpSessionMap.TryGetValue(taskId, out var existingSessionId))
                {
                    sessionId = existingSessionId;
                }
                else
                {
                    sessionId = Guid.NewGuid().ToString();
                    _taskToMcpSessionMap[taskId] = sessionId;
                }
            }
            else
            {
                sessionId = Guid.NewGuid().ToString();
            }

            // Convert A2A message to MCP request
            var mcpRequest = new JsonElement();
            using (var jsonDoc = JsonDocument.Parse("{}"))
            {
                var root = jsonDoc.RootElement.Clone();
                var memoryStream = new MemoryStream();
                var writer = new Utf8JsonWriter(memoryStream);
                writer.WriteStartObject();

                writer.WriteString("action", skillId);
                writer.WriteString("operation", "process");
                writer.WriteString("sessionId", sessionId);

                // Extract text content from message parts
                var textContent = new List<string>();
                foreach (var part in message.Parts)
                {
                    if (part is TextPart textPart)
                    {
                        textContent.Add(textPart.Text);
                    }
                }

                writer.WriteString("content", string.Join("\n", textContent));

                // Add any additional metadata
                if (message.Metadata != null)
                {
                    writer.WriteStartObject("metadata");
                    foreach (var kvp in message.Metadata)
                    {
                        if (kvp.Key != "skillId" && kvp.Key != "taskId")
                        {
                            writer.WritePropertyName(kvp.Key);
                            JsonSerializer.Serialize(writer, kvp.Value);
                        }
                    }
                    writer.WriteEndObject();
                }

                writer.WriteEndObject();
                writer.Flush();
                var jsonString = Encoding.UTF8.GetString(memoryStream.ToArray());
                mcpRequest = JsonSerializer.Deserialize<JsonElement>(jsonString);
            }

            // Send request to MCP service
            var mcpResponse = await HandleMcpRequestInternalAsync(mcpRequest);

            // Convert MCP response to A2A task
            var task = new A2ATask
            {
                TaskId = message.Metadata != null && message.Metadata.TryGetValue("taskId", out var tid) ? tid.ToString() : Guid.NewGuid().ToString(),
                Status = A2ATaskStatus.Completed,
                Messages =
                [
                    message, // Include the original message
                    new()
                    {
                        Role = "agent",
                        Parts =
                        [
                            new TextPart
                            {
                                Text = mcpResponse.GetProperty("result").ToString()
                            }
                        ]
                    }
                ],
                Artifacts = []
            };

            // Check if there are any artifacts in the MCP response
            if (mcpResponse.TryGetProperty("artifacts", out var artifacts))
            {
                foreach (var artifact in artifacts.EnumerateArray())
                {
                    var name = artifact.GetProperty("name").GetString();
                    var content = artifact.GetProperty("content").GetString();
                    var mimeType = artifact.TryGetProperty("mimeType", out var mt) ? mt.GetString() : "text/plain";

                    task.Artifacts.Add(new Artifact
                    {
                        Name = name,
                        Parts =
                        [
                            new FilePart
                            {
                                File = new FileContent
                                {
                                    Name = name,
                                    MimeType = mimeType,
                                    Bytes = Convert.ToBase64String(Encoding.UTF8.GetBytes(content))
                                }
                            }
                        ]
                    });
                }
            }

            return task;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling A2A task");

            // Return a failed task
            return new A2ATask
            {
                TaskId = message.Metadata != null && message.Metadata.TryGetValue("taskId", out var tid) ? tid.ToString() : Guid.NewGuid().ToString(),
                Status = A2ATaskStatus.Failed,
                Messages =
                [
                    message, // Include the original message
                    new()
                    {
                        Role = "agent",
                        Parts =
                        [
                            new TextPart
                            {
                                Text = $"Error: {ex.Message}"
                            }
                        ]
                    }
                ]
            };
        }
    }

    /// <summary>
    /// Internal method to handle MCP requests
    /// </summary>
    private async Task<JsonElement> HandleMcpRequestInternalAsync(JsonElement request)
    {
        // For now, just simulate a response
        return JsonSerializer.SerializeToElement(new
        {
            success = true,
            result = "This is a simulated response from the MCP service."
        });
    }

    /// <summary>
    /// Handles an MCP request for A2A
    /// </summary>
    private async Task<JsonElement> HandleMcpRequestAsync(JsonElement request)
    {
        try
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Handling MCP request for A2A operation: {operation}");

            switch (operation)
            {
                case "send_task":
                    return await HandleMcpSendTaskAsync(request);

                case "get_task":
                    return await HandleMcpGetTaskAsync(request);

                case "cancel_task":
                    return await HandleMcpCancelTaskAsync(request);

                case "get_agent_card":
                    return await HandleMcpGetAgentCardAsync(request);

                case "get_capabilities":
                    return JsonSerializer.SerializeToElement(new
                    {
                        success = true,
                        capabilities = new[]
                        {
                            new { name = "send_task", description = "Send a task to an A2A agent" },
                            new { name = "get_task", description = "Get the status of a task" },
                            new { name = "cancel_task", description = "Cancel a task" },
                            new { name = "get_agent_card", description = "Get the agent card for an A2A agent" }
                        }
                    });

                default:
                    return JsonSerializer.SerializeToElement(new
                    {
                        success = false,
                        error = $"Unknown A2A operation: {operation}"
                    });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling MCP request for A2A");
            return JsonSerializer.SerializeToElement(new
            {
                success = false,
                error = ex.Message
            });
        }
    }

    /// <summary>
    /// Handles an MCP request to send a task
    /// </summary>
    private async Task<JsonElement> HandleMcpSendTaskAsync(JsonElement request)
    {
        var agentUrl = request.GetProperty("agent_url").GetString();
        var content = request.GetProperty("content").GetString();
        var skillId = request.TryGetProperty("skill_id", out var sid) ? sid.GetString() : null;

        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);

            // Create A2A client
            var client = new A2AClient(agentCard);

            // Create message
            var message = new Message
            {
                Role = "user",
                Parts =
                [
                    new TextPart
                    {
                        Text = content
                    }
                ]
            };

            // Add skill ID to metadata if provided
            if (!string.IsNullOrEmpty(skillId))
            {
                message.Metadata = new Dictionary<string, object>
                {
                    { "skillId", skillId }
                };
            }

            // Send task
            var task = await client.SendTaskAsync(message);

            // Convert A2A task to MCP response
            return JsonSerializer.SerializeToElement(new
            {
                success = true,
                task_id = task.TaskId,
                status = task.Status.ToString(),
                result = task.Messages.Count > 1 ? ((TextPart)task.Messages[1].Parts[0]).Text : null,
                artifacts = task.Artifacts.Count > 0 ? task.Artifacts : null
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error sending task to A2A agent at {agentUrl}");
            return JsonSerializer.SerializeToElement(new
            {
                success = false,
                error = ex.Message
            });
        }
    }

    /// <summary>
    /// Handles an MCP request to get a task
    /// </summary>
    private async Task<JsonElement> HandleMcpGetTaskAsync(JsonElement request)
    {
        var agentUrl = request.GetProperty("agent_url").GetString();
        var taskId = request.GetProperty("task_id").GetString();

        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);

            // Create A2A client
            var client = new A2AClient(agentCard);

            // Get task
            var task = await client.GetTaskAsync(taskId);

            // Convert A2A task to MCP response
            return JsonSerializer.SerializeToElement(new
            {
                success = true,
                task_id = task.TaskId,
                status = task.Status.ToString(),
                result = task.Messages.Count > 1 ? ((TextPart)task.Messages[1].Parts[0]).Text : null,
                artifacts = task.Artifacts.Count > 0 ? task.Artifacts : null
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting task from A2A agent at {agentUrl}");
            return JsonSerializer.SerializeToElement(new
            {
                success = false,
                error = ex.Message
            });
        }
    }

    /// <summary>
    /// Handles an MCP request to cancel a task
    /// </summary>
    private async Task<JsonElement> HandleMcpCancelTaskAsync(JsonElement request)
    {
        var agentUrl = request.GetProperty("agent_url").GetString();
        var taskId = request.GetProperty("task_id").GetString();

        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);

            // Create A2A client
            var client = new A2AClient(agentCard);

            // Cancel task
            var task = await client.CancelTaskAsync(taskId);

            // Convert A2A task to MCP response
            return JsonSerializer.SerializeToElement(new
            {
                success = true,
                task_id = task.TaskId,
                status = task.Status.ToString()
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error canceling task from A2A agent at {agentUrl}");
            return JsonSerializer.SerializeToElement(new
            {
                success = false,
                error = ex.Message
            });
        }
    }

    /// <summary>
    /// Handles an MCP request to get an agent card
    /// </summary>
    private async Task<JsonElement> HandleMcpGetAgentCardAsync(JsonElement request)
    {
        var agentUrl = request.GetProperty("agent_url").GetString();

        try
        {
            // Resolve the agent card
            var cardResolver = new AgentCardResolver();
            var agentCard = await cardResolver.ResolveAgentCardAsync(agentUrl);

            // Convert agent card to MCP response
            return JsonSerializer.SerializeToElement(new
            {
                success = true,
                agent_card = agentCard
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting agent card from {agentUrl}");
            return JsonSerializer.SerializeToElement(new
            {
                success = false,
                error = ex.Message
            });
        }
    }
}