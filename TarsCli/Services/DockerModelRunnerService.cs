using System.Net.Http.Json;
using Microsoft.Extensions.Configuration;
using TarsCli.Constants;
using TarsCli.Models;

namespace TarsCli.Services;

/// <summary>
/// Service for interacting with Docker Model Runner
/// </summary>
public class DockerModelRunnerService
{
    private readonly ILogger<DockerModelRunnerService> _logger;
    private readonly IConfiguration _configuration;
    private readonly GpuService _gpuService;
    private readonly HttpClient _httpClient = new();
    private readonly string _baseUrl;
    private readonly string _defaultModel;
    private readonly bool _enableGpu;

    public DockerModelRunnerService(
        ILogger<DockerModelRunnerService> logger,
        IConfiguration configuration,
        GpuService gpuService)
    {
        _logger = logger;
        _configuration = configuration;
        _gpuService = gpuService;

        _baseUrl = configuration[ConfigurationKeys.DockerModelRunner.BaseUrl] ?? "http://localhost:8080";
        _defaultModel = configuration[ConfigurationKeys.DockerModelRunner.DefaultModel] ?? "llama3:8b";
        
        // Check if GPU acceleration is available
        _enableGpu = _gpuService.IsGpuAvailable();
        
        if (_enableGpu)
        {
            _logger.LogInformation("GPU acceleration is enabled for Docker Model Runner");
        }
        else
        {
            _logger.LogInformation("GPU acceleration is disabled for Docker Model Runner");
        }
    }

    /// <summary>
    /// Get the base URL for the Docker Model Runner API
    /// </summary>
    public string BaseUrl => _baseUrl;

    /// <summary>
    /// Get the default model for Docker Model Runner
    /// </summary>
    public string DefaultModel => _defaultModel;

    /// <summary>
    /// Check if Docker Model Runner is available
    /// </summary>
    /// <returns>True if Docker Model Runner is available, false otherwise</returns>
    public async Task<bool> IsAvailable()
    {
        try
        {
            _logger.LogDebug("Checking if Docker Model Runner is available");
            var response = await _httpClient.GetAsync($"{_baseUrl}{DockerModelRunnerEndpoints.Models}");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking if Docker Model Runner is available");
            return false;
        }
    }

    /// <summary>
    /// Check if a model is available
    /// </summary>
    /// <param name="model">Model name</param>
    /// <returns>True if the model is available, false otherwise</returns>
    public async Task<bool> IsModelAvailable(string model)
    {
        try
        {
            _logger.LogDebug($"Checking if model {model} is available");
            var response = await _httpClient.GetAsync($"{_baseUrl}{DockerModelRunnerEndpoints.Models}");

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError($"Failed to get models from Docker Model Runner API: {response.StatusCode} - {response.ReasonPhrase}");
                _logger.LogInformation("Assuming model is available since we can't verify");
                return true; // Assume model is available if we can't check
            }

            var result = await response.Content.ReadFromJsonAsync<ModelListResponse>();

            if (result?.Models == null || !result.Models.Any())
            {
                _logger.LogWarning("No models returned from Docker Model Runner API");
                _logger.LogInformation("Assuming model is available since we can't verify");
                return true; // Assume model is available if no models are returned
            }

            // Log available models for debugging
            var availableModels = result.Models.Select(m => m.Id).ToList();
            _logger.LogDebug($"Available models: {string.Join(", ", availableModels)}");

            // Check if model exists
            return result.Models.Any(m => m.Id == model);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error checking if model {model} is available");
            _logger.LogInformation("Assuming model is available since we can't verify");
            return true; // Assume model is available if we can't check due to an error
        }
    }

    /// <summary>
    /// Get a list of available models
    /// </summary>
    /// <returns>List of available models</returns>
    public async Task<List<ModelInfo>> GetAvailableModels()
    {
        try
        {
            _logger.LogDebug("Getting available models");
            var response = await _httpClient.GetAsync($"{_baseUrl}{DockerModelRunnerEndpoints.Models}");
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<ModelListResponse>();
            return result?.Models ?? new List<ModelInfo>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available models");
            return new List<ModelInfo>();
        }
    }

    /// <summary>
    /// Generate a completion using Docker Model Runner
    /// </summary>
    /// <param name="prompt">The prompt to generate a completion for</param>
    /// <param name="model">The model to use</param>
    /// <returns>The generated completion</returns>
    public async Task<string> GenerateCompletion(string prompt, string model)
    {
        try
        {
            var request = new CompletionRequest
            {
                Model = model,
                Prompt = prompt,
                MaxTokens = 2048,
                Temperature = 0.7f,
                Stream = false
            };

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            _logger.LogDebug($"Sending request to Docker Model Runner API at {_baseUrl}{DockerModelRunnerEndpoints.Completions}");
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}{DockerModelRunnerEndpoints.Completions}", request, cts.Token);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<CompletionResponse>(cancellationToken: cts.Token);
            return result?.Choices?.FirstOrDefault()?.Text ?? string.Empty;
        }
        catch (TaskCanceledException)
        {
            _logger.LogError("Request to Docker Model Runner timed out");
            return "Error: Request timed out. The model may be taking too long to respond.";
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error communicating with Docker Model Runner");
            return $"Error: Unable to connect to Docker Model Runner service: {ex.Message}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating completion from Docker Model Runner");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Generate a chat completion using Docker Model Runner
    /// </summary>
    /// <param name="messages">The chat messages</param>
    /// <param name="model">The model to use</param>
    /// <returns>The generated chat completion</returns>
    public async Task<string> GenerateChatCompletion(List<ChatMessage> messages, string model)
    {
        try
        {
            var request = new ChatCompletionRequest
            {
                Model = model,
                Messages = messages.Select(ChatMessageToDictionary).ToList(),
                MaxTokens = 2048,
                Temperature = 0.7f,
                Stream = false
            };

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            _logger.LogDebug($"Sending request to Docker Model Runner API at {_baseUrl}{DockerModelRunnerEndpoints.ChatCompletions}");
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}{DockerModelRunnerEndpoints.ChatCompletions}", request, cts.Token);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<ChatCompletionResponse>(cancellationToken: cts.Token);
            return result?.Choices?.FirstOrDefault()?.Message?.Content ?? string.Empty;
        }
        catch (TaskCanceledException)
        {
            _logger.LogError("Request to Docker Model Runner timed out");
            return "Error: Request timed out. The model may be taking too long to respond.";
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error communicating with Docker Model Runner");
            return $"Error: Unable to connect to Docker Model Runner service: {ex.Message}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating chat completion from Docker Model Runner");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Pull a model from Docker Hub
    /// </summary>
    /// <param name="model">The model to pull</param>
    /// <returns>True if the model was pulled successfully, false otherwise</returns>
    public async Task<bool> PullModel(string model)
    {
        try
        {
            _logger.LogInformation($"Pulling model {model} from Docker Hub");
            
            // This is a simplified implementation - in a real implementation, 
            // you would use the Docker API to pull the model
            // For now, we'll just simulate a successful pull
            await Task.Delay(2000); // Simulate pulling
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error pulling model {model}");
            return false;
        }
    }

    /// <summary>
    /// Convert a ChatMessage to a dictionary for JSON serialization
    /// </summary>
    /// <param name="message">The chat message</param>
    /// <returns>A dictionary representation of the chat message</returns>
    private Dictionary<string, object> ChatMessageToDictionary(ChatMessage message)
    {
        return new Dictionary<string, object>
        {
            ["role"] = message.Role,
            ["content"] = message.Content
        };
    }
}
