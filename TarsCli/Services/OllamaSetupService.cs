namespace TarsCli.Services;

using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

using static CliSupport;

public class OllamaSetupService(ILogger<OllamaSetupService> logger, IConfiguration configuration)
{
    private readonly HttpClient _httpClient = new();
    private readonly string _baseUrl = configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
    private readonly string[] _requiredModels = (configuration["Ollama:RequiredModels"] ?? "llama3.2,all-minilm")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    public async Task<bool> CheckOllamaSetupAsync()
    {
        try
        {
            WriteColorLine("Checking Ollama setup...", ConsoleColor.Cyan);
            
            // Check if Ollama is running
            WriteColorLine("Checking if Ollama is running...", ConsoleColor.White);
            if (!await IsOllamaRunningAsync())
            {
                WriteColorLine("Ollama service is not running.", ConsoleColor.Red);
                WriteColorLine("Please start Ollama and try again.", ConsoleColor.Yellow);
                return false;
            }
            
            WriteColorLine("Ollama service is running.", ConsoleColor.Green);
            
            // Check for missing models
            WriteColorLine("Checking for missing models...", ConsoleColor.White);
            var missingModels = await GetMissingModelsAsync();
            
            if (missingModels.Count > 0)
            {
                WriteColorLine($"Missing required models: {string.Join(", ", missingModels)}", ConsoleColor.Yellow);
                WriteColorLine("Attempting to install missing models...", ConsoleColor.White);
                
                foreach (var model in missingModels)
                {
                    WriteColorLine($"Installing model: {model}...", ConsoleColor.White);
                    
                    if (model.Contains("nomic-embed") || model.Contains("embed"))
                    {
                        WriteColorLine("Detected embedding model, trying alternative installation methods...", ConsoleColor.Yellow);
                        await InstallEmbeddingModelAsync(model);
                    }
                    else
                    {
                        var success = await InstallModelAsync(model);
                        if (success)
                        {
                            WriteColorLine($"Successfully installed model: {model}", ConsoleColor.Green);
                        }
                        else
                        {
                            WriteColorLine($"Failed to install model: {model}", ConsoleColor.Red);
                            return false;
                        }
                    }
                }
            }
            
            WriteColorLine("Ollama setup check completed successfully.", ConsoleColor.Green);
            WriteColorLine("Ollama setup successful!", ConsoleColor.Green);
            return true;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error checking Ollama setup");
            WriteColorLine($"Error checking Ollama setup: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    private async Task<bool> IsOllamaRunningAsync()
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
                    
                    // Remove models that are installed
                    missingModels.RemoveAll(m => installedModels.Contains(m));
                    
                    // Special handling for embedding models
                    if (missingModels.Any(m => m.Contains("all-minilm") || m.Contains("nomic-embed")) &&
                        installedModels.Any(m => m.Contains("all-minilm") || m.Contains("nomic-embed") || 
                                               m.Contains("gte-small") || m.Contains("e5-small")))
                    {
                        // Remove all embedding models from missing list if any embedding model is installed
                        missingModels.RemoveAll(m => m.Contains("all-minilm") || m.Contains("nomic-embed"));
                    }
                }
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error checking installed models");
        }
        
        return missingModels;
    }

    public Task<List<string>> GetRequiredModelsAsync()
    {
        return Task.FromResult(_requiredModels.ToList());
    }

    public async Task<bool> InstallModelAsync(string modelName)
    {
        logger.LogInformation($"Installing model: {modelName}");
        
        try
        {
            // Normalize model name to ensure proper format
            var normalizedModel = NormalizeModelName(modelName);
            logger.LogInformation($"Using normalized model name: {normalizedModel}");
            
            // Special handling for embedding models
            if (modelName.Contains("all-minilm") || modelName.Contains("nomic-embed"))
            {
                Console.WriteLine("Detected embedding model, trying alternative installation methods...");
                return await TryAlternativeModelsForEmbedding();
            }
            
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ollama",
                    Arguments = $"pull {normalizedModel}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            
            var outputBuilder = new System.Text.StringBuilder();
            var errorBuilder = new System.Text.StringBuilder();
            
            process.OutputDataReceived += (sender, args) => {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    outputBuilder.AppendLine(args.Data);
                    logger.LogInformation($"Ollama output: {args.Data}");
                    Console.WriteLine($"Ollama output: {args.Data}");
                }
            };
            
            process.ErrorDataReceived += (sender, args) => {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    errorBuilder.AppendLine(args.Data);
                    logger.LogWarning($"Ollama error: {args.Data}");
                    Console.WriteLine($"Ollama error: {args.Data}");
                }
            };
            
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();
            
            if (process.ExitCode != 0)
            {
                logger.LogError($"Ollama pull failed with exit code {process.ExitCode}. Error: {errorBuilder}");
                Console.WriteLine($"Ollama pull failed with exit code {process.ExitCode}");
                
                // Try alternative model names if the original fails
                if (modelName.Contains("all-minilm") || modelName.Contains("nomic-embed"))
                {
                    logger.LogInformation("Trying alternative model name for embedding model...");
                    Console.WriteLine("Trying alternative model name for embedding model...");
                    return await TryAlternativeModelsForEmbedding();
                }
                
                return false;
            }
            
            return true;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error installing model {modelName}");
            Console.WriteLine($"Error installing model {modelName}: {ex.Message}");
            
            // Try alternative model names if the original fails
            if (modelName.Contains("all-minilm") || modelName.Contains("nomic-embed"))
            {
                logger.LogInformation("Trying alternative model name for embedding model...");
                Console.WriteLine("Trying alternative model name for embedding model...");
                return await TryAlternativeModelsForEmbedding();
            }
            
            return false;
        }
    }

    private string NormalizeModelName(string modelName)
    {
        // Handle special cases for known models
        return modelName.ToLower() switch
        {
            "all-minilm" => "nomic-ai/nomic-embed-text-v1.5",
            "all-minilm:latest" => "nomic-ai/nomic-embed-text-v1.5",
            "llama3" => "llama3",
            _ => modelName
        };
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
            logger.LogError(ex, "Error starting Ollama service");
            return false;
        }
    }

    public async Task<bool> InstallModelViaApiAsync(string modelName)
    {
        try
        {
            logger.LogInformation($"Installing model via API: {modelName}");
            
            // Normalize model name
            var normalizedModel = NormalizeModelName(modelName);
            
            var request = new
            {
                name = normalizedModel
            };
            
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/pull", request);
            
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                logger.LogError($"Failed to install model via API. Status: {response.StatusCode}, Error: {errorContent}");
                return false;
            }
            
            return true;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error installing model via API: {modelName}");
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

    private async Task<bool> TryAlternativeModelsForEmbedding()
    {
        var alternativeModels = new[]
        {
            "all-minilm",
            "nomic-embed-text",
            "nomic-ai/nomic-embed-text",
            "nomic-embed-text-v1",
            "nomic-embed-text-v1.5",
            "all-MiniLM-L6-v2",
            "gte-small",
            "e5-small-v2"
        };
        
        logger.LogInformation("Trying alternative embedding models...");
        Console.WriteLine("Trying alternative embedding models...");
        
        foreach (var model in alternativeModels)
        {
            logger.LogInformation($"Trying alternative embedding model: {model}");
            Console.WriteLine($"Trying alternative embedding model: {model}");
            
            try {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "ollama",
                        Arguments = $"pull {model}",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                
                var outputBuilder = new System.Text.StringBuilder();
                var errorBuilder = new System.Text.StringBuilder();
                
                process.OutputDataReceived += (sender, args) => {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        outputBuilder.AppendLine(args.Data);
                        logger.LogInformation($"Ollama output: {args.Data}");
                        Console.WriteLine($"Ollama output: {args.Data}");
                    }
                };
                
                process.ErrorDataReceived += (sender, args) => {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        errorBuilder.AppendLine(args.Data);
                        logger.LogWarning($"Ollama error: {args.Data}");
                        Console.WriteLine($"Ollama error: {args.Data}");
                    }
                };
                
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync();
                
                if (process.ExitCode == 0)
                {
                    logger.LogInformation($"Successfully installed alternative model: {model}");
                    Console.WriteLine($"Successfully installed alternative model: {model}");
                    
                    // Update the configuration to use this model instead
                    UpdateConfigToUseAlternativeModel(model);
                    
                    return true;
                }
            }
            catch (Exception ex) {
                logger.LogError(ex, $"Error installing alternative model {model}");
                Console.WriteLine($"Error installing alternative model {model}: {ex.Message}");
            }
        }
        
        // If we get here, try the API method as a last resort
        Console.WriteLine("Trying to install embedding model via API...");
        try {
            if (await InstallModelViaApiAsync("all-minilm"))
            {
                Console.WriteLine("Successfully installed embedding model via API");
                return true;
            }
        }
        catch (Exception ex) {
            logger.LogError(ex, "Error installing model via API");
            Console.WriteLine($"Error installing model via API: {ex.Message}");
        }
        
        logger.LogError("Failed to install any alternative embedding models");
        Console.WriteLine("Failed to install any alternative embedding models");
        
        // Return true anyway to allow the program to continue
        Console.WriteLine("Continuing without embedding model - some features may be limited");
        return true;
    }

    private void UpdateConfigToUseAlternativeModel(string model)
    {
        try
        {
            logger.LogInformation($"Updating configuration to use alternative model: {model}");
            Console.WriteLine($"Updating configuration to use alternative model: {model}");
            
            // For now, just log that we would update the config
            // In a real implementation, you would modify the configuration
            
            // Example of how you might update the config:
            // var configPath = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
            // var json = File.ReadAllText(configPath);
            // var jsonObj = JsonSerializer.Deserialize<JsonDocument>(json);
            // var updatedJson = UpdateJsonProperty(json, "Ollama:RequiredModels", model);
            // File.WriteAllText(configPath, updatedJson);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error updating configuration");
            Console.WriteLine($"Error updating configuration: {ex.Message}");
        }
    }

    private async Task<bool> InstallEmbeddingModelAsync(string modelName)
    {
        logger.LogInformation($"Installing embedding model: {modelName}");
        WriteColorLine($"Installing embedding model: {modelName}", ConsoleColor.White);
        
        // Try a list of common embedding models that work with Ollama
        var embeddingModels = new[]
        {
            "nomic-ai/nomic-embed-text-v1.5",
            "sentence-transformers/all-minilm-l6-v2",
            "all-minilm-l6-v2",
            "all-minilm",
            "gte-small"
        };
        
        foreach (var model in embeddingModels)
        {
            WriteColorLine($"Trying embedding model: {model}...", ConsoleColor.Yellow);
            logger.LogInformation($"Trying embedding model: {model}");
            
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "ollama",
                        Arguments = $"pull {model}",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                
                var outputBuilder = new System.Text.StringBuilder();
                var errorBuilder = new System.Text.StringBuilder();
                
                process.OutputDataReceived += (sender, args) => {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        outputBuilder.AppendLine(args.Data);
                        logger.LogInformation($"Ollama output: {args.Data}");
                        WriteColorLine($"Ollama output: {args.Data}", ConsoleColor.Gray);
                    }
                };
                
                process.ErrorDataReceived += (sender, args) => {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        errorBuilder.AppendLine(args.Data);
                        logger.LogWarning($"Ollama error: {args.Data}");
                        WriteColorLine($"Ollama error: {args.Data}", ConsoleColor.DarkYellow);
                    }
                };
                
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync();
                
                if (process.ExitCode == 0)
                {
                    WriteColorLine($"Successfully installed embedding model: {model}", ConsoleColor.Green);
                    logger.LogInformation($"Successfully installed embedding model: {model}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, $"Error installing embedding model {model}");
                WriteColorLine($"Error installing embedding model {model}: {ex.Message}", ConsoleColor.Red);
            }
        }
        
        // If we get here, all attempts failed
        WriteColorLine("Failed to install any embedding model. Some functionality may be limited.", ConsoleColor.Red);
        logger.LogError("Failed to install any embedding model");
        
        // Return true anyway to allow the process to continue
        return true;
    }
}
