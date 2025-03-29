using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

public class OllamaService(ILogger<OllamaService> logger, IConfiguration configuration)
{
    private readonly HttpClient _httpClient = new();
    private readonly string _baseUrl = configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
    private readonly string _defaultModel = configuration["Ollama:DefaultModel"] ?? "codellama:13b-code";

    // Add a public property to expose the base URL
    public string BaseUrl => _baseUrl;

    /// <summary>
    /// Check if a model is available
    /// </summary>
    /// <param name="model">Model name</param>
    /// <returns>True if the model is available, false otherwise</returns>
    public async Task<bool> IsModelAvailable(string model)
    {
        try
        {
            logger.LogDebug($"Checking if model {model} is available");
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaTagsResponse>();
            return result?.Models?.Any(m => m.Name == model) ?? false;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error checking if model {model} is available");
            return false;
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
            logger.LogDebug("Getting available models");
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaTagsResponse>();
            return result?.Models?.Select(m => m.Name).ToList() ?? new List<string>();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error getting available models");
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

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            logger.LogDebug($"Sending request to Ollama API at {_baseUrl}/api/generate");
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/generate", request, cts.Token);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaResponse>(cancellationToken: cts.Token);
            return result?.Response ?? string.Empty;
        }
        catch (TaskCanceledException)
        {
            logger.LogError("Request to Ollama timed out");
            return "Error: Request timed out. The model may be taking too long to respond.";
        }
        catch (HttpRequestException ex)
        {
            logger.LogError(ex, "HTTP error communicating with Ollama");
            return $"Error: Unable to connect to Ollama service: {ex.Message}";
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error generating completion from Ollama");
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