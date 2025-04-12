using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Constants;
using TarsCli.Services;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for interacting with LLMs
    /// </summary>
    public class LlmCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="LlmCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public LlmCommand(IServiceProvider serviceProvider) : base("llm", "Interact with LLMs")
        {
            _serviceProvider = serviceProvider;

            // Add subcommands
            AddCommand(new ListModelsCommand(_serviceProvider));
            AddCommand(new ModelInfoCommand(_serviceProvider));
            AddCommand(new ChatCommand(_serviceProvider));
        }

        /// <summary>
        /// Command for listing available models
        /// </summary>
        private class ListModelsCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            /// <summary>
            /// Initializes a new instance of the <see cref="ListModelsCommand"/> class
            /// </summary>
            /// <param name="serviceProvider">The service provider</param>
            public ListModelsCommand(IServiceProvider serviceProvider) : base("list", "List available models")
            {
                _serviceProvider = serviceProvider;

                this.SetHandler(async (InvocationContext context) =>
                {
                    var logger = _serviceProvider.GetRequiredService<ILogger<LlmCommand>>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                    var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();

                    try
                    {
                        consoleService.WriteHeader("Available LLM Models");

                        // Check if Docker Model Runner is available
                        var isAvailable = await dockerModelRunnerService.IsAvailable();
                        if (!isAvailable)
                        {
                            consoleService.WriteWarning("Docker Model Runner is not available. Make sure it's running.");
                            return;
                        }

                        // Get available models
                        var models = await dockerModelRunnerService.GetAvailableModels();

                        if (models.Count == 0)
                        {
                            consoleService.WriteWarning("No models available. You may need to pull models first.");

                            // Show recommended models
                            consoleService.WriteLine();
                            consoleService.WriteHeader("Recommended Models");

                            var recommendedModels = dockerModelRunnerService.GetRecommendedModels();
                            foreach (var category in recommendedModels)
                            {
                                consoleService.WriteSubHeader(category.Key);
                                foreach (var model in category.Value)
                                {
                                    consoleService.WriteLine($"  - {model}");
                                }
                                consoleService.WriteLine();
                            }

                            consoleService.WriteLine("To pull a model, use: tarscli llm pull --model <model-name>");
                            return;
                        }

                        // Display available models
                        consoleService.WriteLine($"Found {models.Count} models:");
                        consoleService.WriteLine();

                        foreach (var model in models)
                        {
                            var modelInfo = dockerModelRunnerService.GetModelInfo(model.Id);

                            if (!string.IsNullOrEmpty(modelInfo.Provider))
                            {
                                consoleService.WriteSuccess($"{model.Id} ({modelInfo.Provider})");
                            }
                            else
                            {
                                consoleService.WriteSuccess(model.Id);
                            }

                            if (!string.IsNullOrEmpty(modelInfo.Description))
                            {
                                consoleService.WriteLine($"  {modelInfo.Description}");
                            }

                            if (modelInfo.ContextLength > 0)
                            {
                                consoleService.WriteLine($"  Context Length: {modelInfo.ContextLength:N0} tokens");
                            }

                            consoleService.WriteLine();
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error listing models");
                        consoleService.WriteError($"Error listing models: {ex.Message}");
                    }
                });
            }
        }

        /// <summary>
        /// Command for getting information about a model
        /// </summary>
        private class ModelInfoCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            /// <summary>
            /// Initializes a new instance of the <see cref="ModelInfoCommand"/> class
            /// </summary>
            /// <param name="serviceProvider">The service provider</param>
            public ModelInfoCommand(IServiceProvider serviceProvider) : base("info", "Get information about a model")
            {
                _serviceProvider = serviceProvider;

                // Add options
                var modelOption = new Option<string>(
                    "--model",
                    description: "The model to get information about")
                {
                    IsRequired = true
                };

                AddOption(modelOption);

                this.SetHandler(async (InvocationContext context) =>
                {
                    var model = context.ParseResult.GetValueForOption(modelOption);

                    var logger = _serviceProvider.GetRequiredService<ILogger<LlmCommand>>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                    var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();

                    try
                    {
                        consoleService.WriteHeader($"Model Information: {model}");

                        // Check if Docker Model Runner is available
                        var isAvailable = await dockerModelRunnerService.IsAvailable();
                        if (!isAvailable)
                        {
                            consoleService.WriteWarning("Docker Model Runner is not available. Make sure it's running.");
                            return;
                        }

                        // Check if model is available
                        var isModelAvailable = await dockerModelRunnerService.IsModelAvailable(model);
                        if (!isModelAvailable)
                        {
                            consoleService.WriteWarning($"Model {model} is not available. You may need to pull it first.");
                            consoleService.WriteLine();
                            consoleService.WriteLine($"To pull the model, use: tarscli llm pull --model {model}");
                            return;
                        }

                        // Get model information
                        var modelInfo = dockerModelRunnerService.GetModelInfo(model);

                        // Display model information
                        consoleService.WriteSuccess($"Model: {modelInfo.Id}");

                        if (!string.IsNullOrEmpty(modelInfo.Provider))
                        {
                            consoleService.WriteLine($"Provider: {modelInfo.Provider}");
                        }

                        if (!string.IsNullOrEmpty(modelInfo.Description))
                        {
                            consoleService.WriteLine($"Description: {modelInfo.Description}");
                        }

                        if (modelInfo.ContextLength > 0)
                        {
                            consoleService.WriteLine($"Context Length: {modelInfo.ContextLength:N0} tokens");
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error getting model information");
                        consoleService.WriteError($"Error getting model information: {ex.Message}");
                    }
                });
            }
        }

        /// <summary>
        /// Command for chatting with a model
        /// </summary>
        private class ChatCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            /// <summary>
            /// Initializes a new instance of the <see cref="ChatCommand"/> class
            /// </summary>
            /// <param name="serviceProvider">The service provider</param>
            public ChatCommand(IServiceProvider serviceProvider) : base("chat", "Chat with a model")
            {
                _serviceProvider = serviceProvider;

                // Add options
                var modelOption = new Option<string>(
                    "--model",
                    description: "The model to chat with",
                    getDefaultValue: () => "llama3:8b-instruct");

                var promptOption = new Option<string>(
                    "--prompt",
                    description: "The prompt to send to the model")
                {
                    IsRequired = true
                };

                AddOption(modelOption);
                AddOption(promptOption);

                this.SetHandler(async (InvocationContext context) =>
                {
                    var model = context.ParseResult.GetValueForOption(modelOption);
                    var prompt = context.ParseResult.GetValueForOption(promptOption);

                    var logger = _serviceProvider.GetRequiredService<ILogger<LlmCommand>>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                    var dockerModelRunnerService = _serviceProvider.GetRequiredService<DockerModelRunnerService>();
                    var operationSummaryService = _serviceProvider.GetRequiredService<OperationSummaryService>();

                    try
                    {
                        consoleService.WriteHeader($"Chat with {model}");

                        // Check if Docker Model Runner is available
                        var isAvailable = await dockerModelRunnerService.IsAvailable();
                        if (!isAvailable)
                        {
                            consoleService.WriteWarning("Docker Model Runner is not available. Make sure it's running.");
                            return;
                        }

                        // Check if model is available
                        var isModelAvailable = await dockerModelRunnerService.IsModelAvailable(model);
                        if (!isModelAvailable)
                        {
                            consoleService.WriteWarning($"Model {model} is not available. Attempting to pull it...");

                            var pullSuccess = await dockerModelRunnerService.PullModel(model);
                            if (!pullSuccess)
                            {
                                consoleService.WriteError($"Failed to pull model {model}. Please check the model name and try again.");
                                return;
                            }

                            consoleService.WriteSuccess($"Successfully pulled model {model}");
                        }

                        // Create chat messages
                        var messages = new List<ChatMessage>
                        {
                            new ChatMessage { Role = "user", Content = prompt }
                        };

                        // Generate chat completion
                        consoleService.WriteLine("Generating response...");
                        consoleService.WriteLine();

                        // Use live display for streaming response
                        var responseBuilder = new StringBuilder();
                        var startTime = DateTime.Now;

                        await consoleService.ShowSpinnerAsync("Thinking...", async () =>
                        {
                            await dockerModelRunnerService.GenerateChatCompletion(messages, model, chunk =>
                            {
                                responseBuilder.Append(chunk);
                            });
                        });

                        // Calculate duration
                        var duration = (DateTime.Now - startTime).TotalMilliseconds;

                        // Get the response
                        var response = responseBuilder.ToString();

                        // Display response
                        consoleService.WriteLine("Response:");
                        consoleService.WriteLine();
                        consoleService.WriteLine(response);

                        // Record the operation with full details
                        operationSummaryService.RecordLlmOperation(
                            model,
                            prompt,
                            response,
                            (long)duration);

                        // Save the summary
                        var summaryPath = operationSummaryService.SaveSummary();
                        if (!string.IsNullOrEmpty(summaryPath))
                        {
                            consoleService.WriteLine();
                            consoleService.WriteSuccess($"Operation summary saved to: {summaryPath}");
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error chatting with model");
                        consoleService.WriteError($"Error chatting with model: {ex.Message}");
                    }
                });
            }
        }
    }
}
