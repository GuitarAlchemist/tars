using System.Diagnostics;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

public class OllamaSetupService
{
    private readonly ILogger<OllamaSetupService> _logger;
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly string[] _requiredModels;

    public OllamaSetupService(ILogger<OllamaSetupService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _httpClient = new HttpClient();
        _baseUrl = configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
        _requiredModels = (configuration["Ollama:RequiredModels"] ?? "llama3.2,all-minilm")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    public async Task<bool> CheckOllamaSetupAsync()
    {
        if (!await IsOllamaRunningAsync())
        {
            _logger.LogWarning("Ollama service is not running. Attempting to start it...");
            if (!StartOllamaService())
            {
                _logger.LogError("Failed to start Ollama service. Please install and start Ollama manually.");
                return false;
            }
        }

        var missingModels = await GetMissingModelsAsync();
        if (missingModels.Any())
        {
            _logger.LogWarning($"Missing required models: {string.Join(", ", missingModels)}");
            _logger.LogInformation("Attempting to install missing models...");
            
            foreach (var model in missingModels)
            {
                if (!await InstallModelAsync(model))
                {
                    _logger.LogError($"Failed to install model: {model}");
                    return false;
                }
            }
        }

        return true;
    }

    public async Task<bool> IsOllamaRunningAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<List<string>> GetMissingModelsAsync()
    {
        var missingModels = new List<string>(_requiredModels);
        
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var tagsResponse = JsonSerializer.Deserialize<TagsResponse>(content);
                
                if (tagsResponse?.Models != null)
                {
                    var installedModels = tagsResponse.Models.Select(m => m.Name).ToList();
                    missingModels.RemoveAll(m => installedModels.Contains(m));
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking installed models");
        }
        
        return missingModels;
    }

    public async Task<bool> InstallModelAsync(string modelName)
    {
        _logger.LogInformation($"Installing model: {modelName}");
        
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ollama",
                    Arguments = $"pull {modelName}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            
            process.Start();
            await process.WaitForExitAsync();
            
            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error installing model {modelName}");
            return false;
        }
    }

    private bool StartOllamaService()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ollama",
                    Arguments = "serve",
                    UseShellExecute = true,
                    CreateNoWindow = false
                }
            };
            
            process.Start();
            
            // Give it a moment to start
            Thread.Sleep(2000);
            
            return IsOllamaRunningAsync().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting Ollama service");
            return false;
        }
    }

    private class TagsResponse
    {
        [JsonPropertyName("models")]
        public List<ModelInfo> Models { get; set; } = new();
    }

    private class ModelInfo
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;
    }
}