using System.CommandLine.Invocation;
using System.Net.Http.Json;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Constants;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for interacting with Docker Model Runner
/// </summary>
public class DockerModelRunnerCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Create a new Docker Model Runner command
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public DockerModelRunnerCommand(IServiceProvider serviceProvider) : base("docker-model-runner", "Interact with Docker Model Runner")
    {
        _serviceProvider = serviceProvider;
        AddAlias("dmr");

        // Add subcommands
        AddCommand(new ListModelsCommand(_serviceProvider));
        AddCommand(new StatusCommand(_serviceProvider));
        AddCommand(new PullModelCommand(_serviceProvider));
        AddCommand(new RunModelCommand(_serviceProvider));
        AddCommand(new ConfigCommand(_serviceProvider));
        AddCommand(new StartCommand(_serviceProvider));
        AddCommand(new StopCommand(_serviceProvider));
    }

    /// <summary>
    /// Command to list available models
    /// </summary>
    private class ListModelsCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public ListModelsCommand(IServiceProvider serviceProvider) : base("list", "List available models")
        {
            _serviceProvider = serviceProvider;
            AddAlias("ls");

            this.SetHandler(async (context) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Fetching available models from Docker Model Runner...");

                    var isAvailable = await dockerModelRunnerService.IsAvailable();
                    if (!isAvailable)
                    {
                        consoleService.WriteError("Docker Model Runner is not available. Make sure it's running and accessible.");
                        context.ExitCode = 1;
                        return;
                    }

                    var models = await dockerModelRunnerService.GetAvailableModels();

                    if (models.Count == 0)
                    {
                        consoleService.WriteWarning("No models found. You may need to pull models first.");
                        context.ExitCode = 0;
                        return;
                    }

                    consoleService.WriteSuccess($"Found {models.Count} models:");

                    // Display models in a table format
                    consoleService.WriteTable(
                        ["ID", "Owner", "Created"],
                        models.Select(m => new[]
                        {
                            m.Id,
                            m.OwnedBy,
                            DateTimeOffset.FromUnixTimeSeconds(m.Created).ToString("yyyy-MM-dd HH:mm:ss")
                        })
                    );

                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error listing models");
                    consoleService.WriteError($"Error listing models: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to check the status of Docker Model Runner
    /// </summary>
    private class StatusCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StatusCommand(IServiceProvider serviceProvider) : base("status", "Check the status of Docker Model Runner")
        {
            _serviceProvider = serviceProvider;
            this.SetHandler((InvocationContext context) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var gpuService = _serviceProvider.GetRequiredService<GpuService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo("Checking Docker Model Runner status...");

                    var isAvailableTask = dockerModelRunnerService.IsAvailable();
                    isAvailableTask.Wait();
                    var isAvailable = isAvailableTask.Result;

                    if (isAvailable)
                    {
                        consoleService.WriteSuccess("Docker Model Runner is available");
                        consoleService.WriteInfo($"Base URL: {dockerModelRunnerService.BaseUrl}");
                        consoleService.WriteInfo($"Default model: {dockerModelRunnerService.DefaultModel}");

                        // Check GPU status
                        var isGpuAvailable = gpuService.IsGpuAvailable();
                        if (isGpuAvailable)
                        {
                            consoleService.WriteSuccess("GPU acceleration is available");

                            var gpuInfo = gpuService.GetGpuInfo();
                            foreach (var gpu in gpuInfo.Where(g => gpuService.IsGpuCompatible(g)))
                            {
                                consoleService.WriteInfo($"Compatible GPU: {gpu.Name} with {gpu.MemoryMB}MB memory");
                            }
                        }
                        else
                        {
                            consoleService.WriteWarning("GPU acceleration is not available");
                        }

                        // Get available models
                        var modelsTask = dockerModelRunnerService.GetAvailableModels();
                        modelsTask.Wait();
                        var models = modelsTask.Result;
                        consoleService.WriteInfo($"Available models: {models.Count}");

                        context.ExitCode = 0;
                    }
                    else
                    {
                        consoleService.WriteError("Docker Model Runner is not available");
                        consoleService.WriteInfo("Make sure Docker Desktop is running and Docker Model Runner is enabled");
                        consoleService.WriteInfo("You can enable Docker Model Runner in Docker Desktop settings");

                        context.ExitCode = 1;
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error checking Docker Model Runner status");
                    consoleService.WriteError($"Error checking Docker Model Runner status: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to pull a model from Docker Hub
    /// </summary>
    private class PullModelCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public PullModelCommand(IServiceProvider serviceProvider) : base("pull", "Pull a model from Docker Hub")
        {
            _serviceProvider = serviceProvider;
            var modelArgument = new Argument<string>("model", "The model to pull (e.g., llama3:8b)");
            AddArgument(modelArgument);

            this.SetHandler((InvocationContext context) =>
            {
                var model = context.ParseResult.GetValueForArgument(modelArgument);
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Pulling model {model} from Docker Hub...");

                    var isAvailableTask = dockerModelRunnerService.IsAvailable();
                    isAvailableTask.Wait();
                    var isAvailable = isAvailableTask.Result;
                    if (!isAvailable)
                    {
                        consoleService.WriteError("Docker Model Runner is not available. Make sure it's running and accessible.");
                        context.ExitCode = 1;
                        return;
                    }

                    // Create a pull request
                    var pullRequest = new { name = model };
                    using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(10));
                    var responseTask = dockerModelRunnerService.HttpClient.PostAsJsonAsync($"{dockerModelRunnerService.BaseUrl}/api/pull", pullRequest, cts.Token);
                    responseTask.Wait();
                    var response = responseTask.Result;

                    if (response.IsSuccessStatusCode)
                    {
                        consoleService.WriteSuccess($"Model {model} pulled successfully");
                        context.ExitCode = 0;
                    }
                    else
                    {
                        consoleService.WriteError($"Failed to pull model {model}: {response.StatusCode} - {response.ReasonPhrase}");
                        context.ExitCode = 1;
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, $"Error pulling model {model}");
                    consoleService.WriteError($"Error pulling model {model}: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to run a model
    /// </summary>
    private class RunModelCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public RunModelCommand(IServiceProvider serviceProvider) : base("run", "Run a model with a prompt")
        {
            _serviceProvider = serviceProvider;
            var modelArgument = new Argument<string>("model", "The model to run (e.g., llama3:8b)");
            var promptArgument = new Argument<string>("prompt", "The prompt to send to the model");
            var temperatureOption = new Option<float>("--temperature", () => 0.7f, "Temperature for sampling (0.0 to 1.0)");
            var maxTokensOption = new Option<int>("--max-tokens", () => 2048, "Maximum number of tokens to generate");
            var streamOption = new Option<bool>("--stream", () => false, "Stream the response");

            AddArgument(modelArgument);
            AddArgument(promptArgument);
            AddOption(temperatureOption);
            AddOption(maxTokensOption);
            AddOption(streamOption);

            this.SetHandler((InvocationContext context) =>
            {
                var model = context.ParseResult.GetValueForArgument(modelArgument);
                var prompt = context.ParseResult.GetValueForArgument(promptArgument);
                var temperature = context.ParseResult.GetValueForOption(temperatureOption);
                var maxTokens = context.ParseResult.GetValueForOption(maxTokensOption);
                var stream = context.ParseResult.GetValueForOption(streamOption);
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteInfo($"Running model {model} with prompt: {prompt}");

                    var isAvailableTask = dockerModelRunnerService.IsAvailable();
                    isAvailableTask.Wait();
                    var isAvailable = isAvailableTask.Result;
                    if (!isAvailable)
                    {
                        consoleService.WriteError("Docker Model Runner is not available. Make sure it's running and accessible.");
                        context.ExitCode = 1;
                        return;
                    }

                    var isModelAvailableTask = dockerModelRunnerService.IsModelAvailable(model);
                    isModelAvailableTask.Wait();
                    var isModelAvailable = isModelAvailableTask.Result;
                    if (!isModelAvailable)
                    {
                        consoleService.WriteWarning($"Model {model} is not available. Attempting to pull it...");

                        // Create a pull request
                        var pullRequest = new { name = model };
                        using var pullCts = new CancellationTokenSource(TimeSpan.FromMinutes(10));
                        var pullResponseTask = dockerModelRunnerService.HttpClient.PostAsJsonAsync($"{dockerModelRunnerService.BaseUrl}/api/pull", pullRequest, pullCts.Token);
                        pullResponseTask.Wait();
                        var pullResponse = pullResponseTask.Result;

                        if (!pullResponse.IsSuccessStatusCode)
                        {
                            consoleService.WriteError($"Failed to pull model {model}: {pullResponse.StatusCode} - {pullResponse.ReasonPhrase}");
                            context.ExitCode = 1;
                            return;
                        }

                        consoleService.WriteSuccess($"Model {model} pulled successfully");
                    }

                    // Create the request
                    var request = new Models.OllamaCompletionRequest
                    {
                        Model = model,
                        Prompt = prompt,
                        Options = new Models.OllamaOptions
                        {
                            Temperature = temperature,
                            NumPredict = maxTokens
                        }
                    };

                    using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));

                    if (stream)
                    {
                        // Stream the response
                        consoleService.WriteInfo("Streaming response:");
                        Console.WriteLine();

                        var streamRequest = new HttpRequestMessage(HttpMethod.Post, $"{dockerModelRunnerService.BaseUrl}/api/generate");
                        streamRequest.Content = JsonContent.Create(request);

                        var streamResponseTask = dockerModelRunnerService.HttpClient.SendAsync(streamRequest, HttpCompletionOption.ResponseHeadersRead, cts.Token);
                        streamResponseTask.Wait();
                        var streamResponse = streamResponseTask.Result;
                        streamResponse.EnsureSuccessStatusCode();

                        var streamTask = streamResponse.Content.ReadAsStreamAsync(cts.Token);
                        streamTask.Wait();
                        using var streamReader = new StreamReader(streamTask.Result);
                        string? line;
                        var fullResponse = new StringBuilder();

                        while ((line = streamReader.ReadLine()) != null)
                        {
                            if (string.IsNullOrWhiteSpace(line)) continue;

                            try
                            {
                                var jsonObject = System.Text.Json.JsonDocument.Parse(line);
                                if (jsonObject.RootElement.TryGetProperty("response", out var responseElement))
                                {
                                    var responseText = responseElement.GetString() ?? string.Empty;
                                    Console.Write(responseText);
                                    fullResponse.Append(responseText);
                                }

                                if (jsonObject.RootElement.TryGetProperty("done", out var doneElement) && doneElement.GetBoolean())
                                {
                                    break;
                                }
                            }
                            catch (Exception ex)
                            {
                                logger.LogWarning(ex, $"Error parsing streaming response line: {line}");
                            }
                        }

                        Console.WriteLine("\n");
                        consoleService.WriteSuccess("Response complete");
                    }
                    else
                    {
                        // Get the response as a single result
                        var responseTask = dockerModelRunnerService.GenerateCompletion(prompt, model);
                        responseTask.Wait();
                        var response = responseTask.Result;
                        Console.WriteLine();
                        Console.WriteLine(response);
                        Console.WriteLine();
                    }

                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, $"Error running model {model}");
                    consoleService.WriteError($"Error running model {model}: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to configure Docker Model Runner
    /// </summary>
    private class ConfigCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public ConfigCommand(IServiceProvider serviceProvider) : base("config", "Configure Docker Model Runner settings")
        {
            _serviceProvider = serviceProvider;
            var baseUrlOption = new Option<string>("--base-url", "Base URL for Docker Model Runner (e.g., http://localhost:8080)");
            var defaultModelOption = new Option<string>("--default-model", "Default model to use (e.g., llama3:8b)");

            AddOption(baseUrlOption);
            AddOption(defaultModelOption);

            this.SetHandler((InvocationContext context) =>
            {
                var baseUrl = context.ParseResult.GetValueForOption(baseUrlOption);
                var defaultModel = context.ParseResult.GetValueForOption(defaultModelOption);
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                var configurationService = _serviceProvider.GetRequiredService<ConfigurationService>();

                try
                {
                    // If no options provided, show current configuration
                    if (string.IsNullOrEmpty(baseUrl) && string.IsNullOrEmpty(defaultModel))
                    {
                        consoleService.WriteInfo("Current Docker Model Runner configuration:");
                        consoleService.WriteInfo($"Base URL: {dockerModelRunnerService.BaseUrl}");
                        consoleService.WriteInfo($"Default model: {dockerModelRunnerService.DefaultModel}");
                        context.ExitCode = 0;
                        return;
                    }

                    // Update configuration
                    if (!string.IsNullOrEmpty(baseUrl))
                    {
                        var baseUrlTask = configurationService.SetConfigurationValueAsync(ConfigurationKeys.DockerModelRunner.BaseUrl, baseUrl);
                        baseUrlTask.Wait();
                        consoleService.WriteSuccess($"Base URL updated to {baseUrl}");
                    }

                    if (!string.IsNullOrEmpty(defaultModel))
                    {
                        var defaultModelTask = configurationService.SetConfigurationValueAsync(ConfigurationKeys.DockerModelRunner.DefaultModel, defaultModel);
                        defaultModelTask.Wait();
                        consoleService.WriteSuccess($"Default model updated to {defaultModel}");
                    }

                    consoleService.WriteInfo("Configuration updated. Restart the application for changes to take effect.");
                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error configuring Docker Model Runner");
                    consoleService.WriteError($"Error configuring Docker Model Runner: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to start Docker Model Runner
    /// </summary>
    private class StartCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StartCommand(IServiceProvider serviceProvider) : base("start", "Start Docker Model Runner")
        {
            _serviceProvider = serviceProvider;
            this.SetHandler((InvocationContext context) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                var dockerService = _serviceProvider.GetRequiredService<DockerService>();

                try
                {
                    consoleService.WriteInfo("Starting Docker Model Runner...");

                    // Check if Docker is running
                    var isDockerRunningTask = dockerService.IsDockerRunning();
                    isDockerRunningTask.Wait();
                    if (!isDockerRunningTask.Result)
                    {
                        consoleService.WriteError("Docker is not running. Please start Docker Desktop first.");
                        context.ExitCode = 1;
                        return;
                    }

                    // Start the Docker Model Runner container
                    var resultTask = dockerService.StartContainer("docker-compose-model-runner.yml", "model-runner");
                    resultTask.Wait();
                    var result = resultTask.Result;

                    if (result)
                    {
                        consoleService.WriteSuccess("Docker Model Runner started successfully");
                        consoleService.WriteInfo("It may take a few moments for the service to be fully available");
                        context.ExitCode = 0;
                    }
                    else
                    {
                        consoleService.WriteError("Failed to start Docker Model Runner");
                        context.ExitCode = 1;
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error starting Docker Model Runner");
                    consoleService.WriteError($"Error starting Docker Model Runner: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to stop Docker Model Runner
    /// </summary>
    private class StopCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        public StopCommand(IServiceProvider serviceProvider) : base("stop", "Stop Docker Model Runner")
        {
            _serviceProvider = serviceProvider;
            this.SetHandler((InvocationContext context) =>
            {
                var logger = _serviceProvider.GetRequiredService<ILogger<DockerModelRunnerCommand>>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                var dockerService = _serviceProvider.GetRequiredService<DockerService>();

                try
                {
                    consoleService.WriteInfo("Stopping Docker Model Runner...");

                    // Check if Docker is running
                    var isDockerRunningTask = dockerService.IsDockerRunning();
                    isDockerRunningTask.Wait();
                    if (!isDockerRunningTask.Result)
                    {
                        consoleService.WriteError("Docker is not running. Please start Docker Desktop first.");
                        context.ExitCode = 1;
                        return;
                    }

                    // Stop the Docker Model Runner container
                    var resultTask = dockerService.StopContainer("docker-compose-model-runner.yml", "model-runner");
                    resultTask.Wait();
                    var result = resultTask.Result;

                    if (result)
                    {
                        consoleService.WriteSuccess("Docker Model Runner stopped successfully");
                        context.ExitCode = 0;
                    }
                    else
                    {
                        consoleService.WriteError("Failed to stop Docker Model Runner");
                        context.ExitCode = 1;
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error stopping Docker Model Runner");
                    consoleService.WriteError($"Error stopping Docker Model Runner: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }
}