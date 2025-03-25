using System.Diagnostics;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

public class OllamaSetupService
{
    private readonly ILogger<OllamaSetupService> _logger;
    private readonly OllamaService _ollamaService;
    private readonly string _defaultModel;

    public OllamaSetupService(
        ILogger<OllamaSetupService> logger,
        OllamaService ollamaService,
        IConfiguration configuration)
    {
        _logger = logger;
        _ollamaService = ollamaService;
        _defaultModel = configuration["Ollama:DefaultModel"] ?? "codellama:13b-code";
    }

    public async Task<bool> CheckOllamaSetupAsync()
    {
        try
        {
            // Check if Ollama is installed and running
            if (!IsOllamaRunning())
            {
                _logger.LogWarning("Ollama service is not running. Please start Ollama.");
                return false;
            }

            // Check if the default model is available
            if (!await IsModelAvailable(_defaultModel))
            {
                _logger.LogWarning($"Default model '{_defaultModel}' is not available. Please pull it using 'ollama pull {_defaultModel}'");
                return false;
            }

            _logger.LogInformation("Ollama setup verified successfully.");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking Ollama setup");
            return false;
        }
    }

    private bool IsOllamaRunning()
    {
        try
        {
            // Simple ping to Ollama API
            var result = _ollamaService.GenerateCompletion("test", _defaultModel).Result;
            return !string.IsNullOrEmpty(result) && !result.StartsWith("Error");
        }
        catch
        {
            return false;
        }
    }

    private async Task<bool> IsModelAvailable(string model)
    {
        try
        {
            // Try to generate a simple completion with the model
            var result = await _ollamaService.GenerateCompletion("test", model);
            return !string.IsNullOrEmpty(result) && !result.StartsWith("Error");
        }
        catch
        {
            return false;
        }
    }
}