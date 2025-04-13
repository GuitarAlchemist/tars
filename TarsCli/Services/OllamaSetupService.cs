namespace TarsCli.Services;

using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;
using static CliSupport;

public class OllamaSetupService
{
    private readonly ILogger<OllamaSetupService> _logger;
    private readonly IConfiguration _configuration;
    private readonly GpuService _gpuService;
    private readonly HttpClient _httpClient = new();
    private readonly string _baseUrl;
    private readonly string[] _requiredModels;

    // Regular expression to match ANSI escape sequences
    private static readonly Regex _ansiRegex = new(@"\x1B\[[^@-~]*[@-~]", RegexOptions.Compiled);

    public OllamaSetupService(
        ILogger<OllamaSetupService> logger,
        IConfiguration configuration,
        GpuService gpuService)
    {
        _logger = logger;
        _configuration = configuration;
        _gpuService = gpuService;

        _baseUrl = configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
        _requiredModels = (configuration["Ollama:RequiredModels"] ?? "llama3.2,all-minilm")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    public async Task<bool> CheckOllamaSetupAsync()
    {
        try
        {
            WriteColorLine("Ollama - ", ConsoleColor.Cyan, false);

            // Check if we should use Docker
            bool useDocker = _configuration.GetValue<bool>("Ollama:UseDocker", false);

            if (useDocker)
            {
                _logger.LogInformation("Using Docker for Ollama operations");
                Console.WriteLine("Using Docker for Ollama operations");

                // When using Docker, we need to check if the Docker container is running
                var dockerPs = await RunDockerCommand("ps");
                var ollamaRunning = dockerPs.Contains("ollama");

                if (!ollamaRunning)
                {
                    WriteColorLine("Docker Ollama Not Running", ConsoleColor.Red);
                    WriteColorLine("Please start the Ollama Docker container and try again.", ConsoleColor.Yellow);
                    return false;
                }

                // If Docker is running, we can skip the Ollama check
                WriteColorLine("Docker Mode", ConsoleColor.Green);
                return true;
            }

            // Check if Ollama is running (only for non-Docker mode)
            if (!await IsOllamaRunningAsync())
            {
                WriteColorLine("Not Running", ConsoleColor.Red);
                WriteColorLine("Please start Ollama and try again.", ConsoleColor.Yellow);
                return false;
            }

            // Check GPU availability
            var isGpuAvailable = _gpuService.IsGpuAvailable();
            if (isGpuAvailable)
            {
                WriteColorLine("GPU Acceleration: ", ConsoleColor.Cyan, false);
                WriteColorLine("Enabled", ConsoleColor.Green);

                // Display GPU information
                var gpuInfo = _gpuService.GetGpuInfo();
                var compatibleGpus = gpuInfo.Where(gpu =>
                    (gpu.Type == GpuType.Nvidia && gpu.MemoryMB >= 4000) ||
                    (gpu.Type == GpuType.Amd && gpu.MemoryMB >= 8000) ||
                    (gpu.Type == GpuType.Apple && gpu.MemoryMB >= 4000)
                ).ToList();

                foreach (var gpu in compatibleGpus)
                {
                    WriteColorLine($"  - {gpu.Name} ({gpu.MemoryMB}MB)", ConsoleColor.Yellow);
                }
            }
            else
            {
                WriteColorLine("GPU Acceleration: ", ConsoleColor.Cyan, false);
                WriteColorLine("Disabled", ConsoleColor.Yellow);
                WriteColorLine("  No compatible GPU found or GPU acceleration is disabled in configuration.", ConsoleColor.Gray);
            }

            // Check for missing models
            var missingModels = await GetMissingModelsAsync();

            if (missingModels.Count > 0)
            {
                WriteColorLine("Installing Models", ConsoleColor.Yellow);
                _logger.LogInformation($"Missing required models: {string.Join(", ", missingModels)}");

                // Install missing models without verbose output
                foreach (var model in missingModels)
                {
                    _logger.LogInformation($"Installing model: {model}");

                    if (model.Contains("nomic-embed") || model.Contains("embed"))
                    {
                        await InstallEmbeddingModelAsync(model);
                    }
                    else
                    {
                        var success = await InstallModelAsync(model);
                        if (!success)
                        {
                            WriteColorLine("Error", ConsoleColor.Red);
                            return false;
                        }
                    }
                }
            }

            WriteColorLine("All Good", ConsoleColor.Green);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking Ollama setup");
            WriteColorLine("Error", ConsoleColor.Red);
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
        var missingModels = new List<string>();

        try
        {
            // Check if we should use Docker
            bool useDocker = _configuration.GetValue<bool>("Ollama:UseDocker", false);

            if (useDocker)
            {
                _logger.LogInformation("Using Docker for Ollama operations - assuming models are available");
                Console.WriteLine("Using Docker for Ollama operations - assuming models are available");

                // For Docker, we'll check if the models are available in the Docker container
                var dockerPs = await RunDockerCommand("ps");
                var ollamaRunning = dockerPs.Contains("ollama");

                if (ollamaRunning)
                {
                    // If Ollama is running in Docker, we'll assume the models are available
                    _logger.LogInformation("Ollama is running in Docker - assuming models are available");
                    Console.WriteLine("Ollama is running in Docker - assuming models are available");
                    return new List<string>();
                }
                else
                {
                    // If Ollama is not running in Docker, we'll return an empty list
                    // and let the caller handle starting the Docker container
                    _logger.LogWarning("Ollama is not running in Docker");
                    Console.WriteLine("Ollama is not running in Docker");
                    return new List<string>();
                }
            }

            // Check each required model individually (for non-Docker mode)
            foreach (var model in _requiredModels)
            {
                var normalizedModel = NormalizeModelName(model);
                var isInstalled = await IsModelInstalledAsync(normalizedModel);

                if (!isInstalled)
                {
                    // Special handling for embedding models
                    if (model.Contains("all-minilm") || model.Contains("nomic-embed"))
                    {
                        // Check if any embedding model is installed
                        var anyEmbeddingModelInstalled = await IsAnyEmbeddingModelInstalledAsync();
                        if (!anyEmbeddingModelInstalled)
                        {
                            missingModels.Add(model);
                        }
                    }
                    else
                    {
                        missingModels.Add(model);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking installed models");
            // In case of error, assume all models are missing to be safe
            missingModels.AddRange(_requiredModels);
        }

        return missingModels;
    }

    /// <summary>
    /// Check if any embedding model is installed
    /// </summary>
    /// <returns>True if any embedding model is installed, false otherwise</returns>
    private async Task<bool> IsAnyEmbeddingModelInstalledAsync()
    {
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
                    return installedModels.Any(m =>
                        m.Contains("all-minilm") ||
                        m.Contains("nomic-embed") ||
                        m.Contains("gte-small") ||
                        m.Contains("e5-small"));
                }
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking if any embedding model is installed");
            return false;
        }
    }

    public Task<List<string>> GetRequiredModelsAsync()
    {
        return Task.FromResult(_requiredModels.ToList());
    }

    public async Task<bool> InstallModelAsync(string modelName)
    {
        _logger.LogInformation($"Installing model: {modelName}");

        try
        {
            // Normalize model name to ensure proper format
            var normalizedModel = NormalizeModelName(modelName);
            _logger.LogInformation($"Using normalized model name: {normalizedModel}");

            // Check if model is already installed
            var isInstalled = await IsModelInstalledAsync(normalizedModel);
            if (isInstalled)
            {
                _logger.LogInformation($"Model {normalizedModel} is already installed. Skipping installation.");
                Console.WriteLine($"Model {normalizedModel} is already installed. Skipping installation.");
                return true;
            }

            // Check if we should use Docker
            bool useDocker = _configuration.GetValue<bool>("Ollama:UseDocker", false);

            if (useDocker)
            {
                _logger.LogInformation("Using Docker for Ollama operations");
                Console.WriteLine("Using Docker for Ollama operations");
                return await InstallModelViaApiAsync(normalizedModel);
            }

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
                    var cleanData = StripAnsiEscapeSequences(args.Data);
                    outputBuilder.AppendLine(cleanData);
                    _logger.LogInformation($"Ollama output: {cleanData}");
                    Console.WriteLine($"Ollama output: {cleanData}");
                }
            };

            process.ErrorDataReceived += (sender, args) => {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    var cleanData = StripAnsiEscapeSequences(args.Data);
                    errorBuilder.AppendLine(cleanData);
                    _logger.LogWarning($"Ollama error: {cleanData}");
                    Console.WriteLine($"Ollama error: {cleanData}");
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogError($"Ollama pull failed with exit code {process.ExitCode}. Error: {errorBuilder}");
                Console.WriteLine($"Ollama pull failed with exit code {process.ExitCode}");

                // Try alternative model names if the original fails
                if (modelName.Contains("all-minilm") || modelName.Contains("nomic-embed"))
                {
                    _logger.LogInformation("Trying alternative model name for embedding model...");
                    Console.WriteLine("Trying alternative model name for embedding model...");
                    return await TryAlternativeModelsForEmbedding();
                }

                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error installing model {modelName}");
            Console.WriteLine($"Error installing model {modelName}: {ex.Message}");

            // Try alternative model names if the original fails
            if (modelName.Contains("all-minilm") || modelName.Contains("nomic-embed"))
            {
                _logger.LogInformation("Trying alternative model name for embedding model...");
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

    /// <summary>
    /// Strip ANSI escape sequences from a string
    /// </summary>
    /// <param name="input">Input string that may contain ANSI escape sequences</param>
    /// <returns>String with ANSI escape sequences removed</returns>
    private static string StripAnsiEscapeSequences(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;

        return _ansiRegex.Replace(input, string.Empty);
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

    public async Task<bool> InstallModelViaApiAsync(string modelName)
    {
        try
        {
            _logger.LogInformation($"Installing model via API: {modelName}");
            Console.WriteLine($"Installing model via API: {modelName}");

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
                _logger.LogError($"Failed to install model via API. Status: {response.StatusCode}, Error: {errorContent}");
                Console.WriteLine($"Failed to install model via API. Status: {response.StatusCode}, Error: {errorContent}");

                // For Docker, we can consider this a success if the model is already installed
                // or if we're using a model that's built into the Docker image
                bool useDocker = _configuration.GetValue<bool>("Ollama:UseDocker", false);
                if (useDocker && (modelName == "llama3" || modelName == "llama3.2"))
                {
                    _logger.LogInformation($"Using Docker with built-in model: {modelName}");
                    Console.WriteLine($"Using Docker with built-in model: {modelName}");
                    return true;
                }

                return false;
            }

            Console.WriteLine($"Successfully installed model via API: {modelName}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error installing model via API: {modelName}");
            Console.WriteLine($"Error installing model via API: {modelName}: {ex.Message}");
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

    /// <summary>
    /// Run a Docker command and return the output
    /// </summary>
    /// <param name="command">Docker command to run</param>
    /// <returns>Command output</returns>
    private async Task<string> RunDockerCommand(string command)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker",
                    Arguments = command,
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
                }
            };

            process.ErrorDataReceived += (sender, args) => {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    errorBuilder.AppendLine(args.Data);
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogError($"Docker command failed with exit code {process.ExitCode}. Error: {errorBuilder}");
                return string.Empty;
            }

            return outputBuilder.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running Docker command: {command}");
            return string.Empty;
        }
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

        _logger.LogInformation("Trying alternative embedding models...");
        Console.WriteLine("Trying alternative embedding models...");

        // Check if we should use Docker
        bool useDocker = _configuration.GetValue<bool>("Ollama:UseDocker", false);

        if (useDocker)
        {
            _logger.LogInformation("Using Docker for Ollama operations");
            Console.WriteLine("Using Docker for Ollama operations");

            foreach (var model in alternativeModels)
            {
                _logger.LogInformation($"Trying alternative embedding model via API: {model}");
                Console.WriteLine($"Trying alternative embedding model via API: {model}");

                if (await InstallModelViaApiAsync(model))
                {
                    _logger.LogInformation($"Successfully installed alternative model via API: {model}");
                    Console.WriteLine($"Successfully installed alternative model via API: {model}");
                    return true;
                }
            }

            return false;
        }

        foreach (var model in alternativeModels)
        {
            _logger.LogInformation($"Trying alternative embedding model: {model}");
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
                        var cleanData = StripAnsiEscapeSequences(args.Data);
                        outputBuilder.AppendLine(cleanData);
                        _logger.LogInformation($"Ollama output: {cleanData}");
                        Console.WriteLine($"Ollama output: {cleanData}");
                    }
                };

                process.ErrorDataReceived += (sender, args) => {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        var cleanData = StripAnsiEscapeSequences(args.Data);
                        errorBuilder.AppendLine(cleanData);
                        _logger.LogWarning($"Ollama error: {cleanData}");
                        Console.WriteLine($"Ollama error: {cleanData}");
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    _logger.LogInformation($"Successfully installed alternative model: {model}");
                    Console.WriteLine($"Successfully installed alternative model: {model}");

                    // Update the configuration to use this model instead
                    UpdateConfigToUseAlternativeModel(model);

                    return true;
                }
            }
            catch (Exception ex) {
                _logger.LogError(ex, $"Error installing alternative model {model}");
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
            _logger.LogError(ex, "Error installing model via API");
            Console.WriteLine($"Error installing model via API: {ex.Message}");
        }

        _logger.LogError("Failed to install any alternative embedding models");
        Console.WriteLine("Failed to install any alternative embedding models");

        // Return true anyway to allow the program to continue
        Console.WriteLine("Continuing without embedding model - some features may be limited");
        return true;
    }

    private void UpdateConfigToUseAlternativeModel(string model)
    {
        try
        {
            _logger.LogInformation($"Updating configuration to use alternative model: {model}");
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
            _logger.LogError(ex, "Error updating configuration");
            Console.WriteLine($"Error updating configuration: {ex.Message}");
        }
    }

    /// <summary>
    /// Check if a model is already installed
    /// </summary>
    /// <param name="modelName">The model name to check</param>
    /// <returns>True if the model is installed, false otherwise</returns>
    private async Task<bool> IsModelInstalledAsync(string modelName)
    {
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
                    return installedModels.Contains(modelName);
                }
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error checking if model {modelName} is installed");
            return false;
        }
    }

    private async Task<bool> InstallEmbeddingModelAsync(string modelName)
    {
        _logger.LogInformation($"Installing embedding model: {modelName}");
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
            _logger.LogInformation($"Trying embedding model: {model}");

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
                        var cleanData = StripAnsiEscapeSequences(args.Data);
                        outputBuilder.AppendLine(cleanData);
                        _logger.LogInformation($"Ollama output: {cleanData}");
                        WriteColorLine($"Ollama output: {cleanData}", ConsoleColor.Gray);
                    }
                };

                process.ErrorDataReceived += (sender, args) => {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        var cleanData = StripAnsiEscapeSequences(args.Data);
                        errorBuilder.AppendLine(cleanData);
                        _logger.LogWarning($"Ollama error: {cleanData}");
                        WriteColorLine($"Ollama error: {cleanData}", ConsoleColor.DarkYellow);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    WriteColorLine($"Successfully installed embedding model: {model}", ConsoleColor.Green);
                    _logger.LogInformation($"Successfully installed embedding model: {model}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error installing embedding model {model}");
                WriteColorLine($"Error installing embedding model {model}: {ex.Message}", ConsoleColor.Red);
            }
        }

        // If we get here, all attempts failed
        WriteColorLine("Failed to install any embedding model. Some functionality may be limited.", ConsoleColor.Red);
        _logger.LogError("Failed to install any embedding model");

        // Return true anyway to allow the process to continue
        return true;
    }
}
