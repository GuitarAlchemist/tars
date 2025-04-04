using Microsoft.Extensions.DependencyInjection;
using TarsCli.Models;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for running demos
/// </summary>
public class DemoCommand : Command
{
    /// <summary>
    /// Create a new demo command
    /// </summary>
    public DemoCommand() : base("demo", "Run demos of TARS features")
    {
        // Add subcommands
        AddCommand(new ModelProvidersCommand());
        AddCommand(new AllFeaturesCommand());
    }

    /// <summary>
    /// Command to demo model providers
    /// </summary>
    private class ModelProvidersCommand : Command
    {
        public ModelProvidersCommand() : base("model-providers", "Demo model providers (Ollama and Docker Model Runner)")
        {
            this.SetHandler(async (context) =>
            {
                var serviceProvider = context.BindingContext.GetService<IServiceProvider>()
                    ?? throw new InvalidOperationException("Service provider not found");
                var logger = serviceProvider.GetRequiredService<ILogger<DemoCommand>>();
                var consoleService = serviceProvider.GetRequiredService<ConsoleService>();
                var modelProviderFactory = serviceProvider.GetRequiredService<ModelProviderFactory>();
                var ollamaService = serviceProvider.GetRequiredService<OllamaService>();
                var dockerModelRunnerService = serviceProvider.GetRequiredService<DockerModelRunnerService>();

                try
                {
                    consoleService.WriteHeader("TARS Model Providers Demo");
                    consoleService.WriteInfo("This demo showcases the different model providers available in TARS");

                    // Check Ollama availability
                    consoleService.WriteSubHeader("Checking Ollama availability...");
                    var isOllamaAvailable = await modelProviderFactory.IsProviderAvailable(ModelProvider.Ollama);

                    if (isOllamaAvailable)
                    {
                        consoleService.WriteSuccess("Ollama is available");
                        consoleService.WriteInfo($"Base URL: {ollamaService.BaseUrl}");
                        consoleService.WriteInfo($"Default model: {ollamaService.DefaultModel}");

                        // Get available models
                        var ollamaModels = await ollamaService.GetAvailableModels();
                        if (ollamaModels.Any())
                        {
                            consoleService.WriteInfo($"Available models: {string.Join(", ", ollamaModels)}");
                        }
                        else
                        {
                            consoleService.WriteWarning("No models available in Ollama");
                        }
                    }
                    else
                    {
                        consoleService.WriteWarning("Ollama is not available");
                        consoleService.WriteInfo("Make sure Ollama is running and accessible");
                    }

                    // Check Docker Model Runner availability
                    consoleService.WriteSubHeader("Checking Docker Model Runner availability...");
                    var isDmrAvailable = await modelProviderFactory.IsProviderAvailable(ModelProvider.DockerModelRunner);

                    if (isDmrAvailable)
                    {
                        consoleService.WriteSuccess("Docker Model Runner is available");
                        consoleService.WriteInfo($"Base URL: {dockerModelRunnerService.BaseUrl}");
                        consoleService.WriteInfo($"Default model: {dockerModelRunnerService.DefaultModel}");

                        // Get available models
                        var dmrModels = await dockerModelRunnerService.GetAvailableModels();
                        if (dmrModels.Any())
                        {
                            consoleService.WriteInfo($"Available models: {string.Join(", ", dmrModels.Select(m => m.Id))}");
                        }
                        else
                        {
                            consoleService.WriteWarning("No models available in Docker Model Runner");
                        }
                    }
                    else
                    {
                        consoleService.WriteWarning("Docker Model Runner is not available");
                        consoleService.WriteInfo("Make sure Docker Desktop is running and Docker Model Runner is enabled");
                    }

                    // Generate text using available providers
                    consoleService.WriteSubHeader("Generating text using available providers...");

                    var prompt = "Explain the benefits of using Docker for LLM inference in 3 sentences.";

                    if (isOllamaAvailable)
                    {
                        consoleService.WriteInfo("Generating text using Ollama...");
                        var ollamaResult = await modelProviderFactory.GenerateCompletion(prompt, provider: ModelProvider.Ollama);
                        consoleService.WriteSuccess("Ollama result:");
                        Console.WriteLine(ollamaResult);
                        Console.WriteLine();
                    }

                    if (isDmrAvailable)
                    {
                        consoleService.WriteInfo("Generating text using Docker Model Runner...");
                        var dmrResult = await modelProviderFactory.GenerateCompletion(prompt, provider: ModelProvider.DockerModelRunner);
                        consoleService.WriteSuccess("Docker Model Runner result:");
                        Console.WriteLine(dmrResult);
                        Console.WriteLine();
                    }

                    consoleService.WriteSuccess("Demo completed successfully");
                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error running model providers demo");
                    consoleService.WriteError($"Error running demo: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command to demo all features
    /// </summary>
    private class AllFeaturesCommand : Command
    {
        public AllFeaturesCommand() : base("all", "Demo all TARS features")
        {
            this.SetHandler(async (context) =>
            {
                var serviceProvider = context.BindingContext.GetService<IServiceProvider>()
                    ?? throw new InvalidOperationException("Service provider not found");
                var logger = serviceProvider.GetRequiredService<ILogger<DemoCommand>>();
                var consoleService = serviceProvider.GetRequiredService<ConsoleService>();
                var modelProviderFactory = serviceProvider.GetRequiredService<ModelProviderFactory>();

                try
                {
                    consoleService.WriteHeader("TARS All Features Demo");
                    consoleService.WriteInfo("This demo showcases all the features available in TARS");

                    // Demo model providers
                    consoleService.WriteSubHeader("Model Providers");
                    consoleService.WriteInfo("TARS supports multiple model providers:");
                    consoleService.WriteInfo("1. Ollama - Local LLM inference using Ollama");
                    consoleService.WriteInfo("2. Docker Model Runner - Local LLM inference using Docker");

                    // Check which providers are available
                    var isOllamaAvailable = await modelProviderFactory.IsProviderAvailable(ModelProvider.Ollama);
                    var isDmrAvailable = await modelProviderFactory.IsProviderAvailable(ModelProvider.DockerModelRunner);

                    consoleService.WriteInfo($"Ollama available: {(isOllamaAvailable ? "Yes" : "No")}");
                    consoleService.WriteInfo($"Docker Model Runner available: {(isDmrAvailable ? "Yes" : "No")}");

                    // Demo DSL
                    consoleService.WriteSubHeader("TARS DSL");
                    consoleService.WriteInfo("TARS includes a Domain-Specific Language (DSL) for defining AI tasks and workflows");
                    consoleService.WriteInfo("Example DSL:");
                    Console.WriteLine(@"
CONFIG {
    model: ""llama3""
    temperature: 0.7
    max_tokens: 1000
}

AGENT {
    name: ""researcher""
    description: ""A research agent that can find information on the web""
    capabilities: [""search"", ""summarize"", ""analyze""]
}

TASK {
    description: ""Find information about the history of Paris""
    agent: ""researcher""
}

ACTION {
    type: ""execute""
    task: ""Find information about the history of Paris""
}");

                    // Demo MCP
                    consoleService.WriteSubHeader("Model Context Protocol (MCP)");
                    consoleService.WriteInfo("TARS implements the Model Context Protocol (MCP) for AI assistant collaboration");
                    consoleService.WriteInfo("MCP allows TARS to interact with other AI assistants like Augment Code");
                    consoleService.WriteInfo("TARS can act as an MCP server or client");

                    // Demo Docker deployment
                    consoleService.WriteSubHeader("Docker Deployment");
                    consoleService.WriteInfo("TARS can be deployed in Docker containers for better isolation and scalability");
                    consoleService.WriteInfo("Docker deployment options:");
                    consoleService.WriteInfo("1. TARS with Docker Model Runner as MCP Server");
                    consoleService.WriteInfo("2. TARS Candidate in Docker for testing new versions");
                    consoleService.WriteInfo("3. TARS as MCP Server in Docker for collaboration with other AI assistants");

                    consoleService.WriteSuccess("Demo completed successfully");
                    context.ExitCode = 0;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error running all features demo");
                    consoleService.WriteError($"Error running demo: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }
}
