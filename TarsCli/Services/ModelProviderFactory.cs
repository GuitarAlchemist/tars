using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Constants;
using TarsCli.Models;

namespace TarsCli.Services;

/// <summary>
/// Factory for creating model providers
/// </summary>
public class ModelProviderFactory
{
    private readonly ILogger<ModelProviderFactory> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;
    private readonly DockerModelRunnerService _dockerModelRunnerService;
    private readonly ModelProvider _defaultProvider;

    public ModelProviderFactory(
        ILogger<ModelProviderFactory> logger,
        IConfiguration configuration,
        OllamaService ollamaService,
        DockerModelRunnerService dockerModelRunnerService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;
        _dockerModelRunnerService = dockerModelRunnerService;

        // Get default provider from configuration
        var defaultProviderStr = configuration[ConfigurationKeys.ModelProvider.Default] ?? "Ollama";
        var providerOption = ModelProviderExtensions.FromString(defaultProviderStr);
        _defaultProvider = providerOption ?? ModelProvider.Ollama;

        _logger.LogInformation($"Default model provider: {_defaultProvider.ToString()}");
    }

    /// <summary>
    /// Get the default model provider
    /// </summary>
    public ModelProvider DefaultProvider => _defaultProvider;

    /// <summary>
    /// Check if a provider is available
    /// </summary>
    /// <param name="provider">The provider to check</param>
    /// <returns>True if the provider is available, false otherwise</returns>
    public async Task<bool> IsProviderAvailable(ModelProvider provider)
    {
        try
        {
            return provider switch
            {
                ModelProvider.Ollama => await _ollamaService.IsModelAvailable(_ollamaService.DefaultModel),
                ModelProvider.DockerModelRunner => await _dockerModelRunnerService.IsAvailable(),
                _ => throw new ArgumentException($"Unsupported model provider: {provider}")
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error checking if provider {provider.ToString()} is available");
            return false;
        }
    }

    /// <summary>
    /// Generate a completion using the specified provider
    /// </summary>
    /// <param name="prompt">The prompt to generate a completion for</param>
    /// <param name="model">The model to use</param>
    /// <param name="provider">The provider to use (optional, defaults to the default provider)</param>
    /// <returns>The generated completion</returns>
    public async Task<string> GenerateCompletion(string prompt, string? model = null, ModelProvider? provider = null)
    {
        var selectedProvider = provider ?? _defaultProvider;
        
        // Use default model if not specified
        var selectedModel = model ?? GetDefaultModelForProvider(selectedProvider);

        _logger.LogInformation($"Generating completion using {selectedProvider.ToString()} with model {selectedModel}");

        return selectedProvider switch
        {
            ModelProvider.Ollama => await _ollamaService.GenerateCompletion(prompt, selectedModel),
            ModelProvider.DockerModelRunner => await _dockerModelRunnerService.GenerateCompletion(prompt, selectedModel),
            _ => throw new ArgumentException($"Unsupported model provider: {selectedProvider}")
        };
    }

    /// <summary>
    /// Get available models from the specified provider
    /// </summary>
    /// <param name="provider">The provider to get models from (optional, defaults to the default provider)</param>
    /// <returns>List of available models</returns>
    public async Task<List<string>> GetAvailableModelNames(ModelProvider? provider = null)
    {
        var selectedProvider = provider ?? _defaultProvider;

        _logger.LogInformation($"Getting available models from {selectedProvider.ToString()}");

        return selectedProvider switch
        {
            ModelProvider.Ollama => await _ollamaService.GetAvailableModels(),
            ModelProvider.DockerModelRunner => (await _dockerModelRunnerService.GetAvailableModels()).Select(m => m.Id).ToList(),
            _ => throw new ArgumentException($"Unsupported model provider: {selectedProvider}")
        };
    }

    /// <summary>
    /// Check if a model is available from the specified provider
    /// </summary>
    /// <param name="model">The model to check</param>
    /// <param name="provider">The provider to check (optional, defaults to the default provider)</param>
    /// <returns>True if the model is available, false otherwise</returns>
    public async Task<bool> IsModelAvailable(string model, ModelProvider? provider = null)
    {
        var selectedProvider = provider ?? _defaultProvider;

        _logger.LogInformation($"Checking if model {model} is available from {selectedProvider.ToString()}");

        return selectedProvider switch
        {
            ModelProvider.Ollama => await _ollamaService.IsModelAvailable(model),
            ModelProvider.DockerModelRunner => await _dockerModelRunnerService.IsModelAvailable(model),
            _ => throw new ArgumentException($"Unsupported model provider: {selectedProvider}")
        };
    }

    /// <summary>
    /// Generate a chat completion using the specified provider
    /// </summary>
    /// <param name="messages">The chat messages</param>
    /// <param name="model">The model to use</param>
    /// <param name="provider">The provider to use (optional, defaults to the default provider)</param>
    /// <returns>The generated chat completion</returns>
    public async Task<string> GenerateChatCompletion(List<ChatMessage> messages, string? model = null, ModelProvider? provider = null)
    {
        var selectedProvider = provider ?? _defaultProvider;
        
        // Use default model if not specified
        var selectedModel = model ?? GetDefaultModelForProvider(selectedProvider);

        _logger.LogInformation($"Generating chat completion using {selectedProvider.ToString()} with model {selectedModel}");

        if (selectedProvider == ModelProvider.DockerModelRunner)
        {
            return await _dockerModelRunnerService.GenerateChatCompletion(messages, selectedModel);
        }
        else
        {
            // For Ollama, convert chat messages to a prompt
            var prompt = string.Join("\n\n", messages.Select(m => $"{m.Role}: {m.Content}"));
            return await _ollamaService.GenerateCompletion(prompt, selectedModel);
        }
    }

    /// <summary>
    /// Get the default model for a provider
    /// </summary>
    /// <param name="provider">The provider</param>
    /// <returns>The default model for the provider</returns>
    private string GetDefaultModelForProvider(ModelProvider provider)
    {
        return provider.GetDefaultModel();
    }
}
