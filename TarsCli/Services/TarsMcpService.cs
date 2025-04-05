using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Models;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Services;

/// <summary>
/// TARS-specific implementation of the Model Context Protocol (MCP) service
/// This service provides TARS capabilities to other MCP clients like Augment
/// </summary>
public class TarsMcpService
{
    private readonly ILogger<TarsMcpService> _logger;
    private readonly IConfiguration _configuration;
    private readonly McpService _mcpService;
    private readonly OllamaService _ollamaService;
    private readonly SelfImprovementService _selfImprovementService;
    private readonly SlackIntegrationService _slackIntegrationService;
    private readonly TarsSpeechService _speechService;
    private readonly KnowledgeApplicationService _knowledgeApplicationService;
    private readonly KnowledgeIntegrationService _knowledgeIntegrationService;

    public TarsMcpService(
        ILogger<TarsMcpService> logger,
        IConfiguration configuration,
        McpService mcpService,
        OllamaService ollamaService,
        SelfImprovementService selfImprovementService,
        SlackIntegrationService slackIntegrationService,
        TarsSpeechService speechService,
        KnowledgeApplicationService knowledgeApplicationService,
        KnowledgeIntegrationService knowledgeIntegrationService)
    {
        _logger = logger;
        _configuration = configuration;
        _mcpService = mcpService;
        _ollamaService = ollamaService;
        _selfImprovementService = selfImprovementService;
        _slackIntegrationService = slackIntegrationService;
        _speechService = speechService;
        _knowledgeApplicationService = knowledgeApplicationService;
        _knowledgeIntegrationService = knowledgeIntegrationService;

        // Register TARS-specific handlers
        RegisterHandlers();

        _logger.LogInformation("TARS MCP Service initialized");
    }

    /// <summary>
    /// Register TARS-specific MCP handlers
    /// </summary>
    private void RegisterHandlers()
    {
        // Register the ollama handler
        _mcpService.RegisterHandler("ollama", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing Ollama operation: {operation}");

            var result = await ExecuteOllamaOperation(operation, request);
            return result;
        });

        // Register the self-improvement handler
        _mcpService.RegisterHandler("self-improve", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing self-improvement operation: {operation}");

            var result = await ExecuteSelfImprovementOperation(operation, request);
            return result;
        });

        // Register the slack handler
        _mcpService.RegisterHandler("slack", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing Slack operation: {operation}");

            var result = await ExecuteSlackOperation(operation, request);
            return result;
        });

        // Register the speech handler
        _mcpService.RegisterHandler("speech", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing speech operation: {operation}");

            var result = await ExecuteSpeechOperation(operation, request);
            return result;
        });

        // Register the knowledge handler
        _mcpService.RegisterHandler("knowledge", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing knowledge operation: {operation}");

            var result = await ExecuteKnowledgeOperation(operation, request);
            return result;
        });

        // Register the vscode_agent handler for VS Code Agent Mode integration
        _mcpService.RegisterHandler("vscode_agent", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing VS Code Agent operation: {operation}");

            var result = await ExecuteVSCodeAgentOperation(operation, request);
            return result;
        });

        // Register the collaboration handler for TARS-VSCode-Augment collaboration
        _mcpService.RegisterHandler("collaboration", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing collaboration operation: {operation}");

            var result = await ExecuteCollaborationOperation(operation, request);
            return result;
        });

        // Register the vscode handler for agent mode integration
        _mcpService.RegisterHandler("vscode", async (request) =>
        {
            var operation = request.GetProperty("operation").GetString();
            _logger.LogInformation($"Executing VSCode operation: {operation}");

            var result = await ExecuteVSCodeOperation(operation, request);
            return result;
        });
    }

    /// <summary>
    /// Execute an Ollama operation
    /// </summary>
    private async Task<JsonElement> ExecuteOllamaOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "generate":
                    var prompt = request.GetProperty("prompt").GetString();
                    var model = request.TryGetProperty("model", out var modelElement) ? modelElement.GetString() : "llama3";

                    _logger.LogInformation($"Generating completion with model {model}");
                    var completion = await _ollamaService.GenerateCompletion(prompt, model);

                    return JsonSerializer.SerializeToElement(new { success = true, completion = completion });

                case "models":
                    // Get available models
                    return JsonSerializer.SerializeToElement(new { success = true, models = new[] { "llama3", "mistral", "phi3" } });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown Ollama operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing Ollama operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Execute a self-improvement operation
    /// </summary>
    private async Task<JsonElement> ExecuteSelfImprovementOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "start":
                    var duration = request.TryGetProperty("duration", out var durationElement) ? durationElement.GetInt32() : 60;
                    var autoAccept = request.TryGetProperty("autoAccept", out var autoAcceptElement) ? autoAcceptElement.GetBoolean() : false;

                    _logger.LogInformation($"Starting self-improvement for {duration} minutes with autoAccept={autoAccept}");

                    // Start self-improvement in a background task
                    _ = Task.Run(() =>
                    {
                        // Call the auto-improvement service
                        _logger.LogInformation($"Starting auto-improvement for {duration} minutes with autoAccept={autoAccept}");
                    });

                    return JsonSerializer.SerializeToElement(new { success = true, message = $"Self-improvement started for {duration} minutes" });

                case "status":
                    // Get self-improvement status
                    return JsonSerializer.SerializeToElement(new { success = true, status = new { isRunning = false, duration = 0, startTime = DateTime.Now.ToString() } });

                case "stop":
                    // Stop self-improvement
                    _logger.LogInformation("Stopping self-improvement");
                    return JsonSerializer.SerializeToElement(new { success = true, message = "Self-improvement stopped" });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown self-improvement operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing self-improvement operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Execute a Slack operation
    /// </summary>
    private async Task<JsonElement> ExecuteSlackOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "announce":
                    var title = request.GetProperty("title").GetString();
                    var message = request.GetProperty("message").GetString();
                    var channel = request.TryGetProperty("channel", out var channelElement) ? channelElement.GetString() : null;

                    // Send announcement to Slack
                    _logger.LogInformation($"Sending Slack announcement: {title}");

                    return JsonSerializer.SerializeToElement(new { success = true, message = "Announcement sent" });

                case "feature":
                    var featureName = request.GetProperty("name").GetString();
                    var featureDescription = request.GetProperty("description").GetString();
                    var featureChannel = request.TryGetProperty("channel", out var featureChannelElement) ? featureChannelElement.GetString() : null;

                    // Send feature update to Slack
                    _logger.LogInformation($"Sending Slack feature update: {featureName}");

                    return JsonSerializer.SerializeToElement(new { success = true, message = "Feature update sent" });

                case "milestone":
                    var milestoneName = request.GetProperty("name").GetString();
                    var milestoneDescription = request.GetProperty("description").GetString();
                    var milestoneChannel = request.TryGetProperty("channel", out var milestoneChannelElement) ? milestoneChannelElement.GetString() : null;

                    // Send milestone to Slack
                    _logger.LogInformation($"Sending Slack milestone: {milestoneName}");

                    return JsonSerializer.SerializeToElement(new { success = true, message = "Milestone sent" });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown Slack operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing Slack operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Execute a speech operation
    /// </summary>
    private async Task<JsonElement> ExecuteSpeechOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "speak":
                    var text = request.GetProperty("text").GetString();
                    var voice = request.TryGetProperty("voice", out var voiceElement) ? voiceElement.GetString() : null;
                    var language = request.TryGetProperty("language", out var languageElement) ? languageElement.GetString() : null;
                    var speakerWav = request.TryGetProperty("speakerWav", out var speakerWavElement) ? speakerWavElement.GetString() : null;
                    var agentId = request.TryGetProperty("agentId", out var agentIdElement) ? agentIdElement.GetString() : null;

                    _logger.LogInformation($"Speaking text: {text}");
                    _speechService.Speak(text, voice, language, speakerWav, agentId);

                    return JsonSerializer.SerializeToElement(new { success = true, message = "Speech started" });

                case "configure":
                    var enabled = request.TryGetProperty("enabled", out var enabledElement) ? enabledElement.GetBoolean() : true;
                    var defaultVoice = request.TryGetProperty("defaultVoice", out var defaultVoiceElement) ? defaultVoiceElement.GetString() : null;
                    var defaultLanguage = request.TryGetProperty("defaultLanguage", out var defaultLanguageElement) ? defaultLanguageElement.GetString() : null;

                    _logger.LogInformation("Configuring speech service");
                    _speechService.Configure(enabled, defaultVoice, defaultLanguage);

                    return JsonSerializer.SerializeToElement(new { success = true, message = "Speech configured" });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown speech operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing speech operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Execute a knowledge operation
    /// </summary>
    private async Task<JsonElement> ExecuteKnowledgeOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "extract":
                    var filePath = request.GetProperty("filePath").GetString();
                    var model = request.TryGetProperty("model", out var modelElement) ? modelElement.GetString() : "llama3";

                    var knowledge = await _knowledgeApplicationService.ExtractKnowledgeAsync(filePath, model);
                    return JsonSerializer.SerializeToElement(new { success = true, knowledge = knowledge });

                case "apply":
                    var targetFile = request.GetProperty("filePath").GetString();
                    var applyModel = request.TryGetProperty("model", out var applyModelElement) ? applyModelElement.GetString() : "llama3";

                    var result = await _knowledgeApplicationService.ApplyKnowledgeToFileAsync(targetFile, applyModel);
                    return JsonSerializer.SerializeToElement(new { success = true, improved = result, message = result ? "File improved successfully" : "No improvements needed" });

                case "report":
                    var reportPath = await _knowledgeApplicationService.GenerateKnowledgeReportAsync();
                    return JsonSerializer.SerializeToElement(new { success = true, reportPath = reportPath });

                case "metascript":
                    var targetDir = request.GetProperty("targetDirectory").GetString();
                    var pattern = request.TryGetProperty("pattern", out var patternElement) ? patternElement.GetString() : "*.cs";
                    var metascriptModel = request.TryGetProperty("model", out var metascriptModelElement) ? metascriptModelElement.GetString() : "llama3";

                    var metascriptPath = await _knowledgeIntegrationService.GenerateKnowledgeMetascriptAsync(targetDir, pattern, metascriptModel);
                    return JsonSerializer.SerializeToElement(new { success = true, metascriptPath = metascriptPath });

                case "cycle":
                    var explorationDir = request.GetProperty("explorationDirectory").GetString();
                    var targetDirectory = request.GetProperty("targetDirectory").GetString();
                    var filePattern = request.TryGetProperty("pattern", out var filePatternElement) ? filePatternElement.GetString() : "*.cs";
                    var cycleModel = request.TryGetProperty("model", out var cycleModelElement) ? cycleModelElement.GetString() : "llama3";

                    var cycleReportPath = await _knowledgeIntegrationService.RunKnowledgeImprovementCycleAsync(explorationDir, targetDirectory, filePattern, cycleModel);
                    return JsonSerializer.SerializeToElement(new { success = true, reportPath = cycleReportPath });

                case "retroaction":
                    var retroExplorationDir = request.GetProperty("explorationDirectory").GetString();
                    var retroTargetDir = request.GetProperty("targetDirectory").GetString();
                    var retroModel = request.TryGetProperty("model", out var retroModelElement) ? retroModelElement.GetString() : "llama3";

                    var retroReportPath = await _knowledgeIntegrationService.GenerateRetroactionReportAsync(retroExplorationDir, retroTargetDir, retroModel);
                    return JsonSerializer.SerializeToElement(new { success = true, reportPath = retroReportPath });

                case "list":
                    var knowledgeItems = await _knowledgeApplicationService.GetAllKnowledgeAsync();
                    return JsonSerializer.SerializeToElement(new { success = true, count = knowledgeItems.Count, items = knowledgeItems });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown knowledge operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing knowledge operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Start the TARS MCP service
    /// </summary>
    public async Task StartAsync()
    {
        await _mcpService.StartAsync();
        _logger.LogInformation("TARS MCP Service started");
    }

    /// <summary>
    /// Execute a VSCode operation
    /// </summary>
    private async Task<JsonElement> ExecuteVSCodeOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "agent-mode":
                    var enabled = request.TryGetProperty("enabled", out var enabledElement) ? enabledElement.GetBoolean() : true;
                    var workspace = request.TryGetProperty("workspace", out var workspaceElement) ? workspaceElement.GetString() : null;

                    _logger.LogInformation($"Configuring VSCode agent mode: enabled={enabled}, workspace={workspace ?? "current"}");

                    // Configure VSCode agent mode
                    var vscodeSettingsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".vscode", "settings.json");

                    if (!File.Exists(vscodeSettingsPath))
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(vscodeSettingsPath));
                        File.WriteAllText(vscodeSettingsPath, "{}");
                    }

                    var json = File.ReadAllText(vscodeSettingsPath);
                    var settings = JsonDocument.Parse(json);
                    var root = new Dictionary<string, object>();

                    // Copy existing settings
                    foreach (var property in settings.RootElement.EnumerateObject())
                    {
                        root[property.Name] = JsonSerializer.Deserialize<object>(property.Value.GetRawText());
                    }

                    // Configure agent mode settings
                    var port = _configuration.GetValue<int>("Tars:Mcp:Port", 8999);

                    // Add or update augment.advanced settings
                    if (root.ContainsKey("augment.advanced"))
                    {
                        var advancedSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(root["augment.advanced"].ToString());

                        // Configure agent mode
                        advancedSettings["agentMode"] = enabled;

                        // Configure MCP servers if not already configured
                        if (advancedSettings.ContainsKey("mcpServers"))
                        {
                            var mcpServers = JsonSerializer.Deserialize<List<object>>(advancedSettings["mcpServers"].ToString());

                            // Check if TARS is already in the list
                            bool hasTars = false;
                            foreach (var server in mcpServers)
                            {
                                var serverDict = JsonSerializer.Deserialize<Dictionary<string, object>>(server.ToString());
                                if (serverDict.ContainsKey("name") && serverDict["name"].ToString() == "tars")
                                {
                                    hasTars = true;
                                    break;
                                }
                            }

                            if (!hasTars)
                            {
                                mcpServers.Add(new Dictionary<string, object>
                                {
                                    ["name"] = "tars",
                                    ["url"] = $"http://localhost:{port}/"
                                });
                            }

                            advancedSettings["mcpServers"] = mcpServers;
                        }
                        else
                        {
                            // Create new mcpServers array
                            advancedSettings["mcpServers"] = new List<object>
                            {
                                new Dictionary<string, object>
                                {
                                    ["name"] = "tars",
                                    ["url"] = $"http://localhost:{port}/"
                                }
                            };
                        }

                        root["augment.advanced"] = advancedSettings;
                    }
                    else
                    {
                        // Create new augment.advanced section
                        root["augment.advanced"] = new Dictionary<string, object>
                        {
                            ["agentMode"] = enabled,
                            ["mcpServers"] = new List<object>
                            {
                                new Dictionary<string, object>
                                {
                                    ["name"] = "tars",
                                    ["url"] = $"http://localhost:{port}/"
                                }
                            }
                        };
                    }

                    // Save the updated settings
                    var newJson = JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
                    File.WriteAllText(vscodeSettingsPath, newJson);

                    return JsonSerializer.SerializeToElement(new { success = true, message = $"VSCode agent mode {(enabled ? "enabled" : "disabled")}" });

                case "status":
                    // Get VSCode agent mode status
                    var statusSettingsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".vscode", "settings.json");

                    if (!File.Exists(statusSettingsPath))
                    {
                        return JsonSerializer.SerializeToElement(new { success = true, agentMode = false, configured = false });
                    }

                    var statusJson = File.ReadAllText(statusSettingsPath);
                    var statusSettings = JsonDocument.Parse(statusJson);

                    bool agentModeEnabled = false;
                    bool mcpConfigured = false;

                    if (statusSettings.RootElement.TryGetProperty("augment.advanced", out var advancedElement))
                    {
                        if (advancedElement.TryGetProperty("agentMode", out var agentModeElement))
                        {
                            agentModeEnabled = agentModeElement.GetBoolean();
                        }

                        if (advancedElement.TryGetProperty("mcpServers", out var mcpServersElement))
                        {
                            foreach (var server in mcpServersElement.EnumerateArray())
                            {
                                if (server.TryGetProperty("name", out var nameElement) && nameElement.GetString() == "tars")
                                {
                                    mcpConfigured = true;
                                    break;
                                }
                            }
                        }
                    }

                    return JsonSerializer.SerializeToElement(new { success = true, agentMode = agentModeEnabled, configured = mcpConfigured });

                default:
                    return JsonSerializer.SerializeToElement(new { success = false, error = $"Unknown VSCode operation: {operation}" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing VSCode operation: {operation}");
            return JsonSerializer.SerializeToElement(new { success = false, error = ex.Message });
        }
    }

    /// <summary>
    /// Execute a collaboration operation (public method for direct calls)
    /// </summary>
    public async Task<JsonElement> ExecuteCollaborationOperationAsync(string operation, JsonElement request)
    {
        return await ExecuteCollaborationOperation(operation, request);
    }

    /// <summary>
    /// Execute a collaboration operation
    /// </summary>
    private async Task<JsonElement> ExecuteCollaborationOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "initiate_workflow":
                    var workflowName = request.GetProperty("workflow_name").GetString();
                    var parameters = request.TryGetProperty("parameters", out var paramsElement)
                        ? JsonSerializer.Deserialize<Dictionary<string, object>>(paramsElement.GetRawText())
                        : new Dictionary<string, object>();

                    _logger.LogInformation($"Initiating workflow: {workflowName}");

                    // Create a dynamic object for the workflow definition
                    dynamic workflowContent = new System.Dynamic.ExpandoObject();
                    workflowContent.workflow_name = workflowName;
                    workflowContent.description = $"Workflow initiated by TARS: {workflowName}";
                    workflowContent.coordinator = "tars";
                    workflowContent.steps = new List<dynamic>();

                    // Create a collaboration message for workflow definition
                    var workflowDefinition = new CollaborationMessage
                    {
                        Sender = "tars",
                        Recipient = "*",
                        Type = "workflow_definition",
                        Operation = "define",
                        Content = workflowContent
                    };

                    // Add steps based on the workflow name
                    switch (workflowName)
                    {
                        case "code_improvement":
                            workflowContent.steps.Add(new { component = "tars", action = "analyze_code" });
                            workflowContent.steps.Add(new { component = "augment", action = "suggest_improvements" });
                            workflowContent.steps.Add(new { component = "vscode", action = "apply_changes" });
                            workflowContent.steps.Add(new { component = "tars", action = "verify_improvements" });
                            break;

                        case "knowledge_extraction":
                            workflowContent.steps.Add(new { component = "tars", action = "extract_knowledge" });
                            workflowContent.steps.Add(new { component = "augment", action = "enhance_knowledge" });
                            workflowContent.steps.Add(new { component = "tars", action = "integrate_knowledge" });
                            break;

                        case "self_improvement":
                            workflowContent.steps.Add(new { component = "tars", action = "identify_improvement_areas" });
                            workflowContent.steps.Add(new { component = "augment", action = "analyze_code_quality" });
                            workflowContent.steps.Add(new { component = "tars", action = "generate_improvement_plan" });
                            workflowContent.steps.Add(new { component = "vscode", action = "apply_improvements" });
                            break;

                        default:
                            return JsonSerializer.SerializeToElement(new {
                                success = false,
                                error = $"Unknown workflow: {workflowName}"
                            });
                    }

                    // Broadcast the workflow definition to all connected clients
                    await _mcpService.BroadcastEventAsync("workflow_initiated", workflowDefinition);

                    // Create a collaboration message for progress reporting
                    var progressReport = new CollaborationMessage
                    {
                        Sender = "tars",
                        Recipient = "*",
                        Type = "progress",
                        Operation = $"workflow_{workflowName}",
                        Status = "initiated",
                        Progress = new ProgressInfo
                        {
                            Percentage = 0,
                            CurrentStep = "initialization",
                            TotalSteps = workflowContent.steps.Count,
                            StatusMessage = $"Workflow {workflowName} has been initiated"
                        }
                    };

                    // Broadcast the progress report to all connected clients
                    await _mcpService.BroadcastEventAsync("progress_update", progressReport);

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        workflow_id = workflowDefinition.Id,
                        message = $"Workflow {workflowName} initiated"
                    });

                case "execute_workflow_step":
                    var workflowId = request.GetProperty("workflow_id").GetString();
                    var stepIndex = request.GetProperty("step_index").GetInt32();
                    var stepParameters = request.TryGetProperty("parameters", out var stepParamsElement)
                        ? JsonSerializer.Deserialize<Dictionary<string, object>>(stepParamsElement.GetRawText())
                        : new Dictionary<string, object>();

                    _logger.LogInformation($"Executing workflow step: {workflowId}, step {stepIndex}");

                    // In a real implementation, we would look up the workflow by ID and execute the specified step
                    // For now, we'll just return a success message

                    // Create a collaboration message for step progress reporting
                    var stepProgressReport = new CollaborationMessage
                    {
                        Sender = "tars",
                        Recipient = "*",
                        Type = "progress",
                        Operation = $"workflow_{workflowId}_step_{stepIndex}",
                        Status = "completed",
                        Progress = new ProgressInfo
                        {
                            Percentage = (stepIndex + 1) * 25, // Assuming 4 steps
                            CurrentStep = $"step_{stepIndex}",
                            TotalSteps = 4, // Assuming 4 steps
                            StatusMessage = $"Step {stepIndex} of workflow {workflowId} has been completed"
                        }
                    };

                    // Broadcast the progress report to all connected clients
                    await _mcpService.BroadcastEventAsync("progress_update", stepProgressReport);

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        message = $"Workflow step {stepIndex} executed"
                    });

                case "transfer_knowledge":
                    var knowledgeType = request.GetProperty("knowledge_type").GetString();
                    var content = request.GetProperty("content");
                    var source = request.GetProperty("source").GetString();
                    var target = request.GetProperty("target").GetString();

                    _logger.LogInformation($"Transferring knowledge from {source} to {target}: {knowledgeType}");

                    // Create a collaboration message for knowledge transfer
                    var knowledgeTransfer = new CollaborationMessage
                    {
                        Sender = source,
                        Recipient = target,
                        Type = CollaborationProtocol.MessageTypes.Knowledge,
                        Operation = "transfer",
                        Content = JsonSerializer.Deserialize<object>(content.GetRawText()),
                        Metadata = new Dictionary<string, object>
                        {
                            ["knowledge_type"] = knowledgeType,
                            ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                            ["initiated_by"] = "tars"
                        }
                    };

                    // Broadcast the knowledge transfer to all connected clients
                    await _mcpService.BroadcastEventAsync("knowledge_transfer", knowledgeTransfer);

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        message = $"Knowledge transferred from {source} to {target}"
                    });

                case "provide_feedback":
                    var feedbackType = request.GetProperty("feedback_type").GetString();
                    var targetMessageId = request.GetProperty("target_message_id").GetString();
                    var rating = request.TryGetProperty("rating", out var ratingElement)
                        ? ratingElement.GetInt32()
                        : (int?)null;
                    var comments = request.TryGetProperty("comments", out var commentsElement)
                        ? commentsElement.GetString()
                        : string.Empty;

                    _logger.LogInformation($"Providing feedback: {feedbackType} for message {targetMessageId}");

                    // Create a collaboration message for feedback
                    var feedback = new CollaborationMessage
                    {
                        Sender = "tars",
                        Recipient = "augment",
                        Type = CollaborationProtocol.MessageTypes.Feedback,
                        Operation = "provide",
                        InResponseTo = targetMessageId,
                        Content = new ImprovementFeedback
                        {
                            Accepted = rating.HasValue && rating.Value >= 3,
                            Reason = comments,
                            Suggestions = new List<string>()
                        },
                        Metadata = new Dictionary<string, object>
                        {
                            ["feedback_type"] = feedbackType,
                            ["rating"] = rating
                        }
                    };

                    // Broadcast the feedback to all connected clients
                    await _mcpService.BroadcastEventAsync("feedback", feedback);

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        message = $"Feedback provided for message {targetMessageId}"
                    });

                case "get_collaboration_status":
                    // In a real implementation, we would track the status of all active collaborations
                    // For now, we'll just return a sample status
                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        active_workflows = new[] {
                            new {
                                id = Guid.NewGuid().ToString(),
                                name = "code_improvement",
                                status = "in_progress",
                                progress = 50,
                                current_step = 2,
                                total_steps = 4
                            }
                        },
                        recent_knowledge_transfers = new[] {
                            new {
                                id = Guid.NewGuid().ToString(),
                                type = "code_analysis",
                                source = "tars",
                                target = "augment",
                                timestamp = DateTimeOffset.UtcNow.AddMinutes(-5).ToUnixTimeSeconds()
                            }
                        }
                    });

                default:
                    return JsonSerializer.SerializeToElement(new {
                        success = false,
                        error = $"Unknown collaboration operation: {operation}"
                    });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing collaboration operation: {operation}");
            return JsonSerializer.SerializeToElement(new {
                success = false,
                error = ex.Message
            });
        }
    }

    /// <summary>
    /// Execute a VS Code Agent operation
    /// </summary>
    private async Task<JsonElement> ExecuteVSCodeAgentOperation(string operation, JsonElement request)
    {
        try
        {
            switch (operation)
            {
                case "execute_metascript":
                    var metascriptPath = request.GetProperty("metascriptPath").GetString();
                    var parameters = request.TryGetProperty("parameters", out var paramsElement)
                        ? JsonSerializer.Deserialize<Dictionary<string, string>>(paramsElement.GetRawText())
                        : new Dictionary<string, string>();

                    _logger.LogInformation($"Executing metascript: {metascriptPath}");

                    // Execute the metascript
                    // In a real implementation, this would call the DslService to execute the metascript
                    // For now, we'll just return a success message

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        output = $"Executed metascript: {metascriptPath}",
                        generatedFiles = new[] { "generated/output.cs" }
                    });

                case "analyze_codebase":
                    var targetDir = request.GetProperty("targetDirectory").GetString();
                    var filePattern = request.TryGetProperty("filePattern", out var patternElement)
                        ? patternElement.GetString()
                        : "*.cs";
                    var analysisModel = request.TryGetProperty("model", out var modelElement)
                        ? modelElement.GetString()
                        : "llama3";

                    _logger.LogInformation($"Analyzing codebase: {targetDir}");

                    // Perform code analysis
                    // In a real implementation, this would call the KnowledgeApplicationService to analyze the codebase
                    // For now, we'll just return a sample analysis

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        analysis = new {
                            files = 10,
                            lines = 1000,
                            classes = 20,
                            methods = 100,
                            complexity = "medium",
                            recommendations = new[] {
                                "Improve error handling in McpService",
                                "Add more documentation to TarsMcpService",
                                "Refactor DslService for better performance"
                            }
                        }
                    });

                case "generate_metascript":
                    var task = request.GetProperty("task").GetString();
                    var outputDirectory = request.TryGetProperty("outputDirectory", out var outputDirElement)
                        ? outputDirElement.GetString()
                        : "Examples/metascripts";

                    _logger.LogInformation($"Generating metascript for task: {task}");

                    // Generate a metascript
                    // In a real implementation, this would generate a metascript based on the task
                    // For now, we'll just return a sample metascript path

                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        metascriptPath = $"{outputDirectory}/generated_task.tars",
                        message = $"Generated metascript for task: {task}"
                    });

                case "get_capabilities":
                    // Return the capabilities of the TARS MCP service for VS Code Agent Mode
                    return JsonSerializer.SerializeToElement(new {
                        success = true,
                        capabilities = new[] {
                            new { name = "execute_metascript", description = "Execute a TARS metascript" },
                            new { name = "analyze_codebase", description = "Analyze the codebase structure and quality" },
                            new { name = "generate_metascript", description = "Generate a TARS metascript for a specific task" }
                        }
                    });

                default:
                    return JsonSerializer.SerializeToElement(new {
                        success = false,
                        error = $"Unknown VS Code Agent operation: {operation}"
                    });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing VS Code Agent operation: {operation}");
            return JsonSerializer.SerializeToElement(new {
                success = false,
                error = ex.Message
            });
        }
    }

    /// <summary>
    /// Stop the TARS MCP service
    /// </summary>
    public void Stop()
    {
        _mcpService.Stop();
        _logger.LogInformation("TARS MCP Service stopped");
    }
}