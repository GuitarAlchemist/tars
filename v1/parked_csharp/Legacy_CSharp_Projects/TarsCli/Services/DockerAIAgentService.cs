using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Models;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for interacting with Docker AI Agent
    /// </summary>
    public class DockerAIAgentService
    {
        private readonly ILogger<DockerAIAgentService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly DockerService _dockerService;
        private readonly string _dockerAIAgentComposeFile;
        private readonly string _dockerAIAgentContainerName;

        /// <summary>
        /// Constructor for DockerAIAgentService
        /// </summary>
        /// <param name="logger">Logger</param>
        /// <param name="configuration">Configuration</param>
        /// <param name="dockerService">Docker service</param>
        public DockerAIAgentService(
            ILogger<DockerAIAgentService> logger,
            IConfiguration configuration,
            DockerService dockerService)
        {
            _logger = logger;
            _configuration = configuration;
            _dockerService = dockerService;
            _httpClient = new HttpClient();
            
            // Set default values
            _dockerAIAgentComposeFile = "docker-compose-docker-ai-agent.yml";
            _dockerAIAgentContainerName = "tars-docker-ai-agent";
        }

        /// <summary>
        /// Start the Docker AI Agent
        /// </summary>
        /// <returns>True if the Docker AI Agent was started successfully, false otherwise</returns>
        public async Task<bool> StartDockerAIAgentAsync()
        {
            try
            {
                _logger.LogInformation("Starting Docker AI Agent...");
                return await _dockerService.StartContainer(_dockerAIAgentComposeFile, _dockerAIAgentContainerName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting Docker AI Agent");
                return false;
            }
        }

        /// <summary>
        /// Stop the Docker AI Agent
        /// </summary>
        /// <returns>True if the Docker AI Agent was stopped successfully, false otherwise</returns>
        public async Task<bool> StopDockerAIAgentAsync()
        {
            try
            {
                _logger.LogInformation("Stopping Docker AI Agent...");
                return await _dockerService.StopContainer(_dockerAIAgentComposeFile, _dockerAIAgentContainerName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Docker AI Agent");
                return false;
            }
        }

        /// <summary>
        /// Check if Docker AI Agent is available
        /// </summary>
        /// <returns>True if Docker AI Agent is available, false otherwise</returns>
        public async Task<bool> IsAvailableAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("http://localhost:8997/health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking Docker AI Agent availability");
                return false;
            }
        }

        /// <summary>
        /// Run a model using Docker AI Agent
        /// </summary>
        /// <param name="modelName">Name of the model to run</param>
        /// <returns>True if the model was started successfully, false otherwise</returns>
        public async Task<bool> RunModelAsync(string modelName)
        {
            try
            {
                _logger.LogInformation($"Running model {modelName} using Docker AI Agent...");
                
                var requestData = new
                {
                    model = modelName
                };
                
                var response = await _httpClient.PostAsJsonAsync("http://localhost:8997/api/models/run", requestData);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running model {modelName} using Docker AI Agent");
                return false;
            }
        }

        /// <summary>
        /// Generate text using Docker AI Agent
        /// </summary>
        /// <param name="prompt">Prompt to generate text from</param>
        /// <param name="modelName">Name of the model to use</param>
        /// <returns>Generated text</returns>
        public async Task<string> GenerateTextAsync(string prompt, string modelName = "")
        {
            try
            {
                _logger.LogInformation($"Generating text using Docker AI Agent with model {modelName}...");
                
                var requestData = new
                {
                    prompt = prompt,
                    model = string.IsNullOrEmpty(modelName) ? "default" : modelName
                };
                
                var response = await _httpClient.PostAsJsonAsync("http://localhost:8997/api/generate", requestData);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
                    return responseData["text"];
                }
                else
                {
                    _logger.LogError($"Error generating text: {response.StatusCode}");
                    return string.Empty;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating text using Docker AI Agent");
                return string.Empty;
            }
        }

        /// <summary>
        /// Execute a shell command using Docker AI Agent
        /// </summary>
        /// <param name="command">Command to execute</param>
        /// <returns>Command output</returns>
        public async Task<string> ExecuteShellCommandAsync(string command)
        {
            try
            {
                _logger.LogInformation($"Executing shell command using Docker AI Agent: {command}");
                
                var requestData = new
                {
                    command = command
                };
                
                var response = await _httpClient.PostAsJsonAsync("http://localhost:8997/api/shell/execute", requestData);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
                    return responseData["output"];
                }
                else
                {
                    _logger.LogError($"Error executing shell command: {response.StatusCode}");
                    return string.Empty;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error executing shell command using Docker AI Agent: {command}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Get available models from Docker AI Agent
        /// </summary>
        /// <returns>List of available models</returns>
        public async Task<List<string>> GetAvailableModelsAsync()
        {
            try
            {
                _logger.LogInformation("Getting available models from Docker AI Agent...");
                
                var response = await _httpClient.GetAsync("http://localhost:8997/api/models");
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadFromJsonAsync<Dictionary<string, List<string>>>();
                    return responseData["models"];
                }
                else
                {
                    _logger.LogError($"Error getting available models: {response.StatusCode}");
                    return new List<string>();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting available models from Docker AI Agent");
                return new List<string>();
            }
        }

        /// <summary>
        /// Bridge Docker AI Agent with MCP
        /// </summary>
        /// <param name="mcpUrl">MCP URL</param>
        /// <returns>True if the bridge was established successfully, false otherwise</returns>
        public async Task<bool> BridgeWithMcpAsync(string mcpUrl)
        {
            try
            {
                _logger.LogInformation($"Bridging Docker AI Agent with MCP at {mcpUrl}...");
                
                var requestData = new
                {
                    mcpUrl = mcpUrl
                };
                
                var response = await _httpClient.PostAsJsonAsync("http://localhost:8997/api/bridge/mcp", requestData);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error bridging Docker AI Agent with MCP at {mcpUrl}");
                return false;
            }
        }
    }
}
