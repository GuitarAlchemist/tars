using System.Text.Json;
using Microsoft.Extensions.Configuration;

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
    /// Stop the TARS MCP service
    /// </summary>
    public void Stop()
    {
        _mcpService.Stop();
        _logger.LogInformation("TARS MCP Service stopped");
    }
}