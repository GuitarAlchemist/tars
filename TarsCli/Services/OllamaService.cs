using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;

namespace TarsCli.Services;

public class OllamaService
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<OllamaService> _logger;
    private readonly string _baseUrl;
    private readonly string _defaultModel;

    // Add a public property to expose the base URL
    public string BaseUrl => _baseUrl;

    public OllamaService(ILogger<OllamaService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _httpClient = new HttpClient();
        _baseUrl = configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
        _defaultModel = configuration["Ollama:DefaultModel"] ?? "codellama:13b-code";
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
        public OllamaOptions Options { get; set; }
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
}