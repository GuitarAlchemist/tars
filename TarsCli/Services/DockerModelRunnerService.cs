using System.Net.Http.Json;
using System.Text;
using Microsoft.Extensions.Configuration;
using TarsCli.Constants;
using TarsCli.Models;
using ModelInfoType = TarsCli.Models.ModelInfo;

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

    /// <summary>
    /// Gets the base URL for the Docker Model Runner API
    /// </summary>
    public string BaseUrl => _baseUrl;

    /// <summary>
    /// Gets the default model for the Docker Model Runner
    /// </summary>
    public string DefaultModel => _defaultModel;

    /// <summary>
    /// Gets the HTTP client for the Docker Model Runner API
    /// </summary>
    public HttpClient HttpClient => _httpClient;

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
    /// Check if Docker Model Runner is available
    /// </summary>
    /// <returns>True if Docker Model Runner is available, false otherwise</returns>
    public async Task<bool> IsAvailable()
    {
        try
        {
            _logger.LogDebug("Checking if Docker Model Runner is available");
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking if Docker Model Runner is available");
            return false;
        }
    }

    /// <summary>
    /// Get a list of available models
    /// </summary>
    /// <returns>List of available models</returns>
    public async Task<List<ModelInfoType>> GetAvailableModels()
    {
        try
        {
            _logger.LogDebug("Getting available models");
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<OllamaTagsResponse>();
            return result?.Models?.Select(m => new ModelInfoType { Id = m.Name }).ToList() ?? [];
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available models");
            return [];
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
            var request = new OllamaCompletionRequest
            {
                Model = model,
                Prompt = prompt,
                Options = GetModelOptions(model)
            };

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            _logger.LogDebug($"Sending request to Docker Model Runner API at {_baseUrl}/api/generate");

            // First check if the model exists
            try
            {
                var modelResponse = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
                modelResponse.EnsureSuccessStatusCode();

                var modelResult = await modelResponse.Content.ReadFromJsonAsync<OllamaTagsResponse>();
                if (modelResult?.Models == null || !modelResult.Models.Any(m => m.Name == model))
                {
                    _logger.LogWarning($"Model {model} not found in Docker Model Runner. Pulling model...");

                    // Pull the model
                    var pullRequest = new { name = model };
                    var pullResponse = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/pull", pullRequest, cts.Token);
                    pullResponse.EnsureSuccessStatusCode();

                    _logger.LogInformation($"Model {model} pulled successfully");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error checking/pulling model {model}");
            }

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/generate", request, cts.Token);
            response.EnsureSuccessStatusCode();

            // Read the response as a string first
            var responseString = await response.Content.ReadAsStringAsync(cts.Token);
            _logger.LogDebug($"Raw response: {responseString}");

            try
            {
                // The response might be a stream of JSON objects
                var lines = responseString.Split(['\n'], StringSplitOptions.RemoveEmptyEntries);
                var fullResponse = new StringBuilder();

                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    try
                    {
                        var jsonObject = System.Text.Json.JsonDocument.Parse(line);
                        if (jsonObject.RootElement.TryGetProperty("response", out var responseElement))
                        {
                            fullResponse.Append(responseElement.GetString());
                        }
                    }
                    catch (Exception lineEx)
                    {
                        _logger.LogWarning(lineEx, $"Error parsing line: {line}");
                    }
                }

                if (fullResponse.Length > 0)
                {
                    return fullResponse.ToString();
                }

                // If we couldn't parse the streaming response, try parsing the whole thing
                var result = System.Text.Json.JsonSerializer.Deserialize<OllamaCompletionResponse>(responseString);
                return result?.Response ?? string.Empty;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing response from Docker Model Runner");

                // Try to extract the response from the string
                if (responseString.Contains("\"response\":"))
                {
                    var startIndex = responseString.IndexOf("\"response\":") + 11;
                    var endIndex = responseString.IndexOf("\",", startIndex);
                    if (endIndex > startIndex)
                    {
                        return responseString.Substring(startIndex, endIndex - startIndex);
                    }
                }

                return "Error parsing response: " + ex.Message;
            }
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
    /// <param name="progressCallback">Optional callback for streaming progress</param>
    /// <returns>The generated chat completion</returns>
    public async Task<string> GenerateChatCompletion(List<ChatMessage> messages, string model, Action<string>? progressCallback = null)
    {
        try
        {
            // Convert messages to Ollama format
            var formattedMessages = messages.Select(m => new OllamaMessage
            {
                Role = m.Role,
                Content = m.Content
            }).ToList();

            var request = new OllamaChatRequest
            {
                Model = model,
                Messages = formattedMessages,
                Options = GetModelOptions(model)
            };

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            _logger.LogDebug($"Sending request to Docker Model Runner API at {_baseUrl}/api/chat");

            // First check if the model exists
            try
            {
                var modelResponse = await _httpClient.GetAsync($"{_baseUrl}/api/tags");
                modelResponse.EnsureSuccessStatusCode();

                var modelResult = await modelResponse.Content.ReadFromJsonAsync<OllamaTagsResponse>();
                if (modelResult?.Models == null || !modelResult.Models.Any(m => m.Name == model))
                {
                    _logger.LogWarning($"Model {model} not found in Docker Model Runner. Pulling model...");

                    // Pull the model
                    var pullRequest = new { name = model };
                    var pullResponse = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/pull", pullRequest, cts.Token);
                    pullResponse.EnsureSuccessStatusCode();

                    _logger.LogInformation($"Model {model} pulled successfully");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error checking/pulling model {model}");
            }

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/chat", request, cts.Token);
            response.EnsureSuccessStatusCode();

            // If we have a progress callback, stream the response
            if (progressCallback != null)
            {
                var fullResponse = new StringBuilder();
                using var stream = await response.Content.ReadAsStreamAsync(cts.Token);
                using var reader = new StreamReader(stream);
                
                string? line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    
                    try
                    {
                        var jsonObject = System.Text.Json.JsonDocument.Parse(line);
                        string? content = null;
                        
                        if (jsonObject.RootElement.TryGetProperty("message", out var messageElement) &&
                            messageElement.TryGetProperty("content", out var contentElement))
                        {
                            content = contentElement.GetString();
                        }
                        else if (jsonObject.RootElement.TryGetProperty("response", out var responseElement))
                        {
                            content = responseElement.GetString();
                        }
                        
                        if (!string.IsNullOrEmpty(content))
                        {
                            fullResponse.Append(content);
                            progressCallback(content);
                        }
                    }
                    catch (Exception lineEx)
                    {
                        _logger.LogWarning(lineEx, $"Error parsing line: {line}");
                    }
                }
                
                return fullResponse.ToString();
            }
            else
            {
                // Read the response as a string first
                var responseString = await response.Content.ReadAsStringAsync(cts.Token);
                _logger.LogDebug($"Raw response: {responseString}");

                try
                {
                    // The response might be a stream of JSON objects
                    var lines = responseString.Split(['\n'], StringSplitOptions.RemoveEmptyEntries);
                    var fullResponse = new StringBuilder();

                    foreach (var line in lines)
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;

                        try
                        {
                            var jsonObject = System.Text.Json.JsonDocument.Parse(line);
                            if (jsonObject.RootElement.TryGetProperty("message", out var messageElement) &&
                                messageElement.TryGetProperty("content", out var contentElement))
                            {
                                fullResponse.Append(contentElement.GetString());
                            }
                            else if (jsonObject.RootElement.TryGetProperty("response", out var responseElement))
                            {
                                fullResponse.Append(responseElement.GetString());
                            }
                        }
                        catch (Exception lineEx)
                        {
                            _logger.LogWarning(lineEx, $"Error parsing line: {line}");
                        }
                    }

                    if (fullResponse.Length > 0)
                    {
                        return fullResponse.ToString();
                    }

                    // If we couldn't parse the streaming response, try parsing the whole thing
                    var result = System.Text.Json.JsonSerializer.Deserialize<OllamaCompletionResponse>(responseString);
                    return result?.Message?.Content ?? result?.Response ?? string.Empty;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error parsing response from Docker Model Runner");

                    // Try to extract the response from the string
                    if (responseString.Contains("\"content\":"))
                    {
                        var startIndex = responseString.IndexOf("\"content\":") + 10;
                        var endIndex = responseString.IndexOf("\",", startIndex);
                        if (endIndex > startIndex)
                        {
                            return responseString.Substring(startIndex, endIndex - startIndex);
                        }
                    }
                    else if (responseString.Contains("\"response\":"))
                    {
                        var startIndex = responseString.IndexOf("\"response\":") + 11;
                        var endIndex = responseString.IndexOf("\",", startIndex);
                        if (endIndex > startIndex)
                        {
                            return responseString.Substring(startIndex, endIndex - startIndex);
                        }
                    }

                    return "Error parsing response: " + ex.Message;
                }
            }
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
    /// Check if a model is available in Docker Model Runner
    /// </summary>
    /// <param name="model">The model to check</param>
    /// <returns>True if the model is available, false otherwise</returns>
    public async Task<bool> IsModelAvailable(string model)
    {
        try
        {
            _logger.LogDebug($"Checking if model {model} is available");
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags");

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError($"Failed to get models from Docker Model Runner API: {response.StatusCode} - {response.ReasonPhrase}");
                _logger.LogInformation("Assuming model is available since we can't verify");
                return true; // Assume model is available if we can't check
            }

            var result = await response.Content.ReadFromJsonAsync<OllamaTagsResponse>();

            if (result?.Models == null || !result.Models.Any())
            {
                _logger.LogWarning("No models returned from Docker Model Runner API");
                _logger.LogInformation("Assuming model is available since we can't verify");
                return true; // Assume model is available if no models are returned
            }

            // Log available models for debugging
            var availableModels = result.Models.Select(m => m.Name).ToList();
            _logger.LogDebug($"Available models: {string.Join(", ", availableModels)}");

            // Check if model exists
            return result.Models.Any(m => m.Name == model);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error checking if model {model} is available");
            _logger.LogInformation("Assuming model is available since we can't verify");
            return true; // Assume model is available if we can't check due to an error
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

            // Create a pull request
            var pullRequest = new { name = model };
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(10));
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/pull", pullRequest, cts.Token);
            response.EnsureSuccessStatusCode();

            _logger.LogInformation($"Model {model} pulled successfully");
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

    /// <summary>
    /// Get model-specific options for a given model
    /// </summary>
    /// <param name="model">The model name</param>
    /// <returns>Options for the model</returns>
    private OllamaOptions GetModelOptions(string model)
    {
        // Default options
        var options = new OllamaOptions
        {
            Temperature = 0.7f,
            NumPredict = 2048,
            Stream = true
        };
        
        // Adjust options based on model
        if (model.StartsWith("llama3:70b") || model.StartsWith("llama3:405b"))
        {
            // Larger Llama 3 models
            options.Temperature = 0.6f;
            options.NumPredict = 4096;
        }
        else if (model.StartsWith("llama3:8b"))
        {
            // Smaller Llama 3 models
            options.Temperature = 0.7f;
            options.NumPredict = 2048;
        }
        else if (model.StartsWith("claude-3-opus"))
        {
            // Claude 3 Opus
            options.Temperature = 0.5f;
            options.NumPredict = 4096;
        }
        else if (model.StartsWith("claude-3-sonnet"))
        {
            // Claude 3 Sonnet
            options.Temperature = 0.6f;
            options.NumPredict = 4096;
        }
        else if (model.StartsWith("claude-3-haiku"))
        {
            // Claude 3 Haiku
            options.Temperature = 0.7f;
            options.NumPredict = 2048;
        }
        else if (model.StartsWith("gpt-4o") || model.StartsWith("gpt-4-turbo"))
        {
            // GPT-4o and GPT-4 Turbo
            options.Temperature = 0.6f;
            options.NumPredict = 4096;
        }
        else if (model.StartsWith("gemini-1.5-pro") || model.StartsWith("gemini-1.5-ultra"))
        {
            // Gemini 1.5 Pro/Ultra
            options.Temperature = 0.6f;
            options.NumPredict = 8192; // Supports very long context
        }
        
        return options;
    }

    /// <summary>
    /// Get recommended models for different use cases
    /// </summary>
    /// <returns>A dictionary of use cases and their recommended models</returns>
    public Dictionary<string, List<string>> GetRecommendedModels()
    {
        return new Dictionary<string, List<string>>
        {
            ["General Purpose"] =
            [
                ModelProviders.OpenAI.GPT4o,
                ModelProviders.Anthropic.Claude3Opus,
                ModelProviders.Meta.Llama3_70B_Instruct
            ],
            ["Code Generation"] =
            [
                ModelProviders.OpenAI.GPT4o,
                ModelProviders.Anthropic.Claude3Opus,
                ModelProviders.Meta.Llama3_70B_Instruct
            ],
            ["Long Context"] =
            [
                ModelProviders.Google.Gemini15Pro,
                ModelProviders.Anthropic.Claude3Opus,
                ModelProviders.OpenAI.GPT4o
            ],
            ["Fast Response"] =
            [
                ModelProviders.Anthropic.Claude3Haiku,
                ModelProviders.Meta.Llama3_8B_Instruct,
                ModelProviders.Google.Gemini15Flash
            ],
            ["Local Deployment"] =
            [
                ModelProviders.Meta.Llama3_8B_Instruct,
                ModelProviders.Meta.Llama3_70B_Instruct
            ]
        };
    }
    
    /// <summary>
    /// Get information about a specific model
    /// </summary>
    /// <param name="model">The model name</param>
    /// <returns>Information about the model</returns>
    public ModelInfo GetModelInfo(string model)
    {
        // Set provider and description based on model name
        if (model.StartsWith("llama3:"))
        {
            var description = "Meta's Llama 3 model";
            var contextLength = 4096;
            
            if (model.Contains("70b"))
            {
                description += " (70B parameters)";
                contextLength = 8192;
            }
            else if (model.Contains("8b"))
            {
                description += " (8B parameters)";
                contextLength = 4096;
            }
            else if (model.Contains("405b"))
            {
                description += " (405B parameters)";
                contextLength = 128000;
            }
            
            if (model.Contains("instruct"))
            {
                description += " - Instruction tuned";
            }
            
            return new ModelInfo
            {
                Id = model,
                Provider = ModelProviders.Meta.Name,
                Description = description,
                ContextLength = contextLength
            };
        }
        else if (model.StartsWith("claude-3"))
        {
            var description = "Claude 3";
            var contextLength = 200000;
            
            if (model.Contains("opus"))
            {
                description = "Claude 3 Opus - Anthropic's most powerful model";
            }
            else if (model.Contains("sonnet"))
            {
                description = "Claude 3 Sonnet - Balanced performance and speed";
            }
            else if (model.Contains("haiku"))
            {
                description = "Claude 3 Haiku - Fast and efficient model";
            }
            
            return new ModelInfo
            {
                Id = model,
                Provider = ModelProviders.Anthropic.Name,
                Description = description,
                ContextLength = contextLength
            };
        }
        else if (model.StartsWith("gpt-4"))
        {
            var description = "GPT-4";
            var contextLength = 128000;
            
            if (model.Contains("o"))
            {
                description = "GPT-4o - OpenAI's most capable multimodal model";
            }
            else if (model.Contains("turbo"))
            {
                description = "GPT-4 Turbo - Powerful and efficient model";
            }
            
            return new ModelInfo
            {
                Id = model,
                Provider = ModelProviders.OpenAI.Name,
                Description = description,
                ContextLength = contextLength
            };
        }
        else if (model.StartsWith("gemini-1.5"))
        {
            var description = "Gemini 1.5";
            var contextLength = 1000000;
            
            if (model.Contains("pro"))
            {
                description = "Gemini 1.5 Pro - Google's advanced multimodal model";
            }
            else if (model.Contains("flash"))
            {
                description = "Gemini 1.5 Flash - Fast and efficient model";
            }
            
            return new ModelInfo
            {
                Id = model,
                Provider = ModelProviders.Google.Name,
                Description = description,
                ContextLength = contextLength
            };
        }
        
        // Default model info
        return new ModelInfo
        {
            Id = model,
            Provider = "Unknown",
            Description = "Unknown model",
            ContextLength = 4096
        };
    }
}
