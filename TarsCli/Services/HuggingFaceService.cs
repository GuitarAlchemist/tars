using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Configuration;
using System.Diagnostics;

namespace TarsCli.Services;

public class HuggingFaceService
{
    private readonly ILogger<HuggingFaceService> _logger;
    private readonly IConfiguration _configuration;
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl = "https://huggingface.co/api";
    private readonly string _modelCacheDir;
    private readonly OllamaSetupService _ollamaSetupService;
    private readonly UserInteractionService _userInteractionService;
    private readonly SecretsService _secretsService;
    private string? _apiKey;

    public HuggingFaceService(
        ILogger<HuggingFaceService> logger,
        IConfiguration configuration,
        OllamaSetupService ollamaSetupService,
        UserInteractionService userInteractionService,
        SecretsService secretsService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaSetupService = ollamaSetupService;
        _userInteractionService = userInteractionService;
        _secretsService = secretsService;

        // Initialize HttpClient
        _httpClient = new HttpClient();

        // Set up model cache directory
        _modelCacheDir = Path.Combine(
            _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory(),
            "models",
            "huggingface");

        Directory.CreateDirectory(_modelCacheDir);
    }

    /// <summary>
    /// Get the API key, prompting the user if necessary
    /// </summary>
    /// <param name="required">Whether the API key is required</param>
    /// <returns>The API key</returns>
    private async Task<string> GetApiKeyAsync(bool required = false)
    {
        if (_apiKey == null)
        {
            // First try to get from secrets
            _apiKey = await _secretsService.GetSecretAsync("HuggingFace:ApiKey");

            // If not found in secrets, try configuration
            if (string.IsNullOrEmpty(_apiKey))
            {
                _apiKey = _configuration["HuggingFace:ApiKey"] ?? string.Empty;
            }

            // If still not found, ask the user
            if (string.IsNullOrEmpty(_apiKey))
            {
                _apiKey = await _userInteractionService.AskForApiKeyAsync("HuggingFace", "ApiKey", required);
            }

            // Set the API key in the HTTP client
            if (!string.IsNullOrEmpty(_apiKey))
            {
                _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            }
        }

        return _apiKey ?? string.Empty;
    }

    /// <summary>
    /// Search for models on Hugging Face
    /// </summary>
    public async Task<List<HuggingFaceModel>> SearchModelsAsync(
        string query,
        string task = "text-generation",
        int limit = 10)
    {
        try
        {
            // Get API key (not required for search)
            await GetApiKeyAsync();

            _logger.LogInformation($"Searching for models with query: {query}, task: {task}");

            // Build the search URL
            var searchUrl = $"{_baseUrl}/models?search={Uri.EscapeDataString(query)}";

            if (!string.IsNullOrEmpty(task))
            {
                searchUrl += $"&filter={Uri.EscapeDataString(task)}";
            }

            searchUrl += $"&limit={limit}";

            // Make the API request
            var response = await _httpClient.GetAsync(searchUrl);
            response.EnsureSuccessStatusCode();

            var content = await response.Content.ReadAsStringAsync();
            var models = JsonSerializer.Deserialize<List<HuggingFaceModel>>(content);

            if (models == null || models.Count == 0)
            {
                _logger.LogWarning("No models found matching the search criteria");
                return new List<HuggingFaceModel>();
            }

            _logger.LogInformation($"Found {models.Count} models matching the search criteria");
            return models;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error searching for models on Hugging Face");
            return new List<HuggingFaceModel>();
        }
    }

    /// <summary>
    /// Get the best coding models from Hugging Face
    /// </summary>
    public async Task<List<HuggingFaceModel>> GetBestCodingModelsAsync(int limit = 10)
    {
        // Search for coding-specific models
        var codingModels = await SearchModelsAsync("code", "text-generation", limit);

        // Sort by downloads and likes
        return codingModels
            .OrderByDescending(m => m.Downloads)
            .ThenByDescending(m => m.Likes)
            .Take(limit)
            .ToList();
    }

    /// <summary>
    /// Get detailed information about a model
    /// </summary>
    public async Task<HuggingFaceModelDetails> GetModelDetailsAsync(string modelId)
    {
        try
        {
            // Get API key (not required for model details)
            await GetApiKeyAsync();

            _logger.LogInformation($"Getting details for model: {modelId}");

            var url = $"{_baseUrl}/models/{modelId}";
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var content = await response.Content.ReadAsStringAsync();
            var modelDetails = JsonSerializer.Deserialize<HuggingFaceModelDetails>(content);

            if (modelDetails == null)
            {
                _logger.LogWarning($"No details found for model: {modelId}");
                return new HuggingFaceModelDetails();
            }

            return modelDetails;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting details for model: {modelId}");
            return new HuggingFaceModelDetails();
        }
    }

    /// <summary>
    /// Download a model from Hugging Face
    /// </summary>
    public async Task<bool> DownloadModelAsync(string modelId)
    {
        try
        {
            // Get API key (required for downloading models)
            await GetApiKeyAsync(true);

            _logger.LogInformation($"Downloading model: {modelId}");
            CliSupport.WriteColorLine($"Downloading model: {modelId}", ConsoleColor.Cyan);

            // Create a directory for the model
            var modelDir = Path.Combine(_modelCacheDir, modelId.Replace("/", "_"));
            Directory.CreateDirectory(modelDir);

            // Use git to clone the model repository
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = $"clone https://huggingface.co/{modelId} \"{modelDir}\"",
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
                    _logger.LogInformation($"Git output: {args.Data}");
                }
            };

            process.ErrorDataReceived += (sender, args) => {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    errorBuilder.AppendLine(args.Data);
                    _logger.LogWarning($"Git error: {args.Data}");
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogError($"Git clone failed with exit code {process.ExitCode}. Error: {errorBuilder}");
                CliSupport.WriteColorLine($"Failed to download model: {modelId}", ConsoleColor.Red);
                return false;
            }

            CliSupport.WriteColorLine($"Successfully downloaded model: {modelId}", ConsoleColor.Green);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error downloading model: {modelId}");
            CliSupport.WriteColorLine($"Error downloading model: {modelId}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Convert a Hugging Face model to Ollama format
    /// </summary>
    public async Task<bool> ConvertToOllamaAsync(string modelId, string ollamaModelName = "")
    {
        try
        {
            _logger.LogInformation($"Converting model to Ollama format: {modelId}");
            CliSupport.WriteColorLine($"Converting model to Ollama format: {modelId}", ConsoleColor.Cyan);

            // If no Ollama model name is provided, use the model ID (replacing / with _)
            if (string.IsNullOrEmpty(ollamaModelName))
            {
                ollamaModelName = modelId.Replace("/", "_");
            }

            // Get the model directory
            var modelDir = Path.Combine(_modelCacheDir, modelId.Replace("/", "_"));

            if (!Directory.Exists(modelDir))
            {
                _logger.LogError($"Model directory not found: {modelDir}");
                CliSupport.WriteColorLine($"Model not found. Please download it first.", ConsoleColor.Red);
                return false;
            }

            // Create a Modelfile for Ollama
            var modelfilePath = Path.Combine(modelDir, "Modelfile");
            var modelfileContent = $@"FROM {modelDir}
TEMPLATE ""{{prompt}}""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER stop ""</s>""
PARAMETER stop ""<|endoftext|>""
PARAMETER stop ""<|end|>""";

            await File.WriteAllTextAsync(modelfilePath, modelfileContent);

            // Create the model in Ollama
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ollama",
                    Arguments = $"create {ollamaModelName} -f {modelfilePath}",
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
                    _logger.LogInformation($"Ollama output: {args.Data}");
                }
            };

            process.ErrorDataReceived += (sender, args) => {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    errorBuilder.AppendLine(args.Data);
                    _logger.LogWarning($"Ollama error: {args.Data}");
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogError($"Ollama create failed with exit code {process.ExitCode}. Error: {errorBuilder}");
                CliSupport.WriteColorLine($"Failed to convert model to Ollama format: {modelId}", ConsoleColor.Red);
                return false;
            }

            CliSupport.WriteColorLine($"Successfully converted model to Ollama format: {ollamaModelName}", ConsoleColor.Green);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error converting model to Ollama format: {modelId}");
            CliSupport.WriteColorLine($"Error converting model to Ollama format: {modelId}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Install a model from Hugging Face to Ollama
    /// </summary>
    public async Task<bool> InstallModelAsync(string modelId, string ollamaModelName = "")
    {
        try
        {
            // Step 1: Download the model
            var downloadSuccess = await DownloadModelAsync(modelId);
            if (!downloadSuccess)
            {
                return false;
            }

            // Step 2: Convert to Ollama format
            var convertSuccess = await ConvertToOllamaAsync(modelId, ollamaModelName);
            if (!convertSuccess)
            {
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error installing model: {modelId}");
            CliSupport.WriteColorLine($"Error installing model: {modelId}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Get a list of installed models
    /// </summary>
    public List<string> GetInstalledModels()
    {
        try
        {
            var installedModels = new List<string>();

            if (!Directory.Exists(_modelCacheDir))
            {
                return installedModels;
            }

            foreach (var dir in Directory.GetDirectories(_modelCacheDir))
            {
                var modelName = Path.GetFileName(dir);
                installedModels.Add(modelName);
            }

            return installedModels;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting installed models");
            return new List<string>();
        }
    }
}

public class HuggingFaceModel
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("modelId")]
    public string ModelId { get; set; } = string.Empty;

    [JsonPropertyName("author")]
    public string Author { get; set; } = string.Empty;

    [JsonPropertyName("downloads")]
    public int Downloads { get; set; }

    [JsonPropertyName("likes")]
    public int Likes { get; set; }

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new List<string>();

    [JsonPropertyName("pipeline_tag")]
    public string PipelineTag { get; set; } = string.Empty;

    [JsonPropertyName("lastModified")]
    public string LastModified { get; set; } = string.Empty;

    [JsonPropertyName("private")]
    public bool Private { get; set; }

    public override string ToString()
    {
        return $"{Id} by {Author} - Downloads: {Downloads}, Likes: {Likes}";
    }
}

public class HuggingFaceModelDetails
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("author")]
    public string Author { get; set; } = string.Empty;

    [JsonPropertyName("sha")]
    public string Sha { get; set; } = string.Empty;

    [JsonPropertyName("lastModified")]
    public string LastModified { get; set; } = string.Empty;

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new List<string>();

    [JsonPropertyName("pipeline_tag")]
    public string PipelineTag { get; set; } = string.Empty;

    [JsonPropertyName("downloads")]
    public int Downloads { get; set; }

    [JsonPropertyName("likes")]
    public int Likes { get; set; }

    [JsonPropertyName("private")]
    public bool Private { get; set; }

    [JsonPropertyName("siblings")]
    public List<ModelFile> Siblings { get; set; } = new List<ModelFile>();

    [JsonPropertyName("cardData")]
    public ModelCardData CardData { get; set; } = new ModelCardData();
}

public class ModelFile
{
    [JsonPropertyName("rfilename")]
    public string Filename { get; set; } = string.Empty;

    [JsonPropertyName("size")]
    public long Size { get; set; }

    [JsonPropertyName("blob_id")]
    public string BlobId { get; set; } = string.Empty;
}

public class ModelCardData
{
    [JsonPropertyName("license")]
    public string License { get; set; } = string.Empty;

    [JsonPropertyName("language")]
    public List<string> Languages { get; set; } = new List<string>();

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new List<string>();

    [JsonPropertyName("datasets")]
    public List<string> Datasets { get; set; } = new List<string>();

    [JsonPropertyName("metrics")]
    public List<string> Metrics { get; set; } = new List<string>();

    [JsonPropertyName("model-index")]
    public List<ModelIndexEntry> ModelIndex { get; set; } = new List<ModelIndexEntry>();
}

public class ModelIndexEntry
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("results")]
    public List<ModelResult> Results { get; set; } = new List<ModelResult>();
}

public class ModelResult
{
    [JsonPropertyName("task")]
    public ModelTask Task { get; set; } = new ModelTask();

    [JsonPropertyName("dataset")]
    public ModelDataset Dataset { get; set; } = new ModelDataset();

    [JsonPropertyName("metrics")]
    public List<ModelMetric> Metrics { get; set; } = new List<ModelMetric>();
}

public class ModelTask
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;
}

public class ModelDataset
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("config")]
    public string Config { get; set; } = string.Empty;

    [JsonPropertyName("split")]
    public string Split { get; set; } = string.Empty;
}

public class ModelMetric
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("value")]
    public double Value { get; set; }
}