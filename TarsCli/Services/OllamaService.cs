using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

public class OllamaService
{
    private readonly ILogger<OllamaService> _logger;
    private readonly IConfiguration _configuration;
    private readonly GpuService _gpuService;
    private readonly HttpClient _httpClient = new();
    private readonly string _baseUrl;
    private readonly string _defaultModel;
    private readonly bool _enableGpu;
    private readonly Dictionary<string, object> _gpuParameters;

    public OllamaService(
        ILogger<OllamaService> logger,
        IConfiguration configuration,
        GpuService gpuService)
    {
        _logger = logger;
        _configuration = configuration;
        _gpuService = gpuService;

        _baseUrl = configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
        _defaultModel = configuration["Ollama:DefaultModel"] ?? "codellama:13b-code";

        // Check if GPU acceleration is available
        _enableGpu = _gpuService.IsGpuAvailable();
        _gpuParameters = _enableGpu ? _gpuService.GetOllamaGpuParameters() : new Dictionary<string, object>();

        if (_enableGpu)
        {
            _logger.LogInformation("GPU acceleration is enabled for Ollama");
        }
        else
        {
            _logger.LogInformation("GPU acceleration is disabled for Ollama");
        }
    }

    // Add a public property to expose the base URL
    public string BaseUrl => _baseUrl;

    /// <summary>
    /// Get the default model for Ollama
    /// </summary>
    public string DefaultModel => _defaultModel;

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
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError($"Failed to get models from Ollama API: {response.StatusCode} - {response.ReasonPhrase}");
                _logger.LogInformation("Assuming model is available since we can't verify");
                return true; // Assume model is available if we can't check
            }

            var result = await response.Content.ReadFromJsonAsync<OllamaTagsResponse>();

            if (result?.Models == null || !result.Models.Any())
            {
                _logger.LogWarning("No models returned from Ollama API");
                _logger.LogInformation("Assuming model is available since we can't verify");
                return true; // Assume model is available if no models are returned
            }

            // Log available models for debugging
            var availableModels = result.Models.Select(m => m.Name).ToList();
            _logger.LogDebug($"Available models: {string.Join(", ", availableModels)}");

            // Check if model exists or if a model with the same prefix exists
            var exactMatch = result.Models.Any(m => m.Name == model);
            var prefixMatch = result.Models.Any(m => m.Name.StartsWith(model + ":") || model.StartsWith(m.Name + ":"));

            return exactMatch || prefixMatch;
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
    public async Task<List<string>> GetAvailableModels()
    {
        try
        {
            _logger.LogDebug("Getting available models");
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaTagsResponse>();
            return result?.Models?.Select(m => m.Name).ToList() ?? new List<string>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available models");
            return new List<string>();
        }
    }

    public async Task<string> GenerateCompletion(string prompt, string model)
    {
        try
        {
            var request = new OllamaRequest
            {
                Model = model,
                Prompt = prompt,
                Stream = false,
                // Add reasonable timeout
                Options = new OllamaOptions { TimeoutMs = 120000 }
            };

            // Add GPU parameters if GPU acceleration is enabled
            if (_enableGpu && _gpuParameters.Count > 0)
            {
                _logger.LogDebug("Adding GPU parameters to Ollama request");

                // Add GPU parameters to options
                foreach (var param in _gpuParameters)
                {
                    // Use reflection to set the property
                    var property = typeof(OllamaOptions).GetProperty(param.Key, System.Reflection.BindingFlags.IgnoreCase | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                    if (property != null)
                    {
                        property.SetValue(request.Options, param.Value);
                    }
                }
            }

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            _logger.LogDebug($"Sending request to Ollama API at {_baseUrl}/api/generate");
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/generate", request, cts.Token);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaResponse>(cancellationToken: cts.Token);
            return result?.Response ?? string.Empty;
        }
        catch (TaskCanceledException)
        {
            _logger.LogError("Request to Ollama timed out");
            return "Error: Request timed out. The model may be taking too long to respond.";
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error communicating with Ollama");
            return $"Error: Unable to connect to Ollama service: {ex.Message}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating completion from Ollama");
            return $"Error: {ex.Message}";
        }
    }

    private class OllamaRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("prompt")]
        public string Prompt { get; set; } = string.Empty;

        [JsonPropertyName("stream")]
        public bool Stream { get; set; }

        [JsonPropertyName("options")]
        public OllamaOptions Options { get; set; } = new();
    }

    private class OllamaOptions
    {
        [JsonPropertyName("timeout_ms")]
        public int TimeoutMs { get; set; }

        [JsonPropertyName("use_gpu")]
        public bool UseGpu { get; set; }

        [JsonPropertyName("gpu_layers")]
        public int GpuLayers { get; set; }
    }

    private class OllamaResponse
    {
        [JsonPropertyName("response")]
        public string Response { get; set; } = string.Empty;
    }

    private class OllamaTagsResponse
    {
        [JsonPropertyName("models")]
        public List<OllamaModel>? Models { get; set; }
    }

    private class OllamaModel
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("size")]
        public long Size { get; set; }

        [JsonPropertyName("modified_at")]
        public string? ModifiedAt { get; set; }
    }
}