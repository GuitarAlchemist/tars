using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;
using TarsEngine.Services.Models;

namespace TarsEngine.Services
{
    /// <summary>
    /// Service for interacting with Large Language Models
    /// </summary>
    public class LlmService : ILlmService
    {
        private readonly ILogger<LlmService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly string _ollamaEndpoint;
        private readonly string _defaultModel;

        public LlmService(
            ILogger<LlmService> logger,
            IConfiguration configuration,
            HttpClient httpClient)
        {
            _logger = logger;
            _configuration = configuration;
            _httpClient = httpClient;

            // Get configuration values
            _ollamaEndpoint = _configuration["Ollama:Endpoint"] ?? "http://localhost:11434/api/generate";
            _defaultModel = _configuration["Ollama:DefaultModel"] ?? "llama3";
        }

        /// <summary>
        /// Gets a completion from the LLM
        /// </summary>
        /// <param name="prompt">The prompt to send to the LLM</param>
        /// <param name="model">The model to use (defaults to configuration value)</param>
        /// <param name="temperature">The temperature to use (0.0 to 1.0)</param>
        /// <param name="maxTokens">The maximum number of tokens to generate</param>
        /// <returns>The generated text</returns>
        public virtual async Task<string> GetCompletionAsync(
            string prompt,
            string model = null,
            double temperature = 0.7,
            int maxTokens = 1000)
        {
            try
            {
                _logger.LogInformation($"Getting completion for prompt: {prompt.Substring(0, Math.Min(50, prompt.Length))}...");

                // Use the default model if none is provided
                model = model ?? _defaultModel;

                // Create the request payload
                var payload = new
                {
                    model = model,
                    prompt = prompt,
                    temperature = temperature,
                    max_tokens = maxTokens,
                    stream = false
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_ollamaEndpoint, content);

                // Check if the request was successful
                if (!response.IsSuccessStatusCode)
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error from LLM API: {errorContent}");
                    throw new Exception($"Error from LLM API: {response.StatusCode} - {errorContent}");
                }

                // Parse the response
                string responseContent = await response.Content.ReadAsStringAsync();
                var responseObject = JsonSerializer.Deserialize<OllamaResponse>(responseContent);

                return responseObject?.Response ?? string.Empty;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting completion: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Gets a chat completion from the LLM
        /// </summary>
        /// <param name="messages">The messages to send to the LLM</param>
        /// <param name="model">The model to use (defaults to configuration value)</param>
        /// <param name="temperature">The temperature to use (0.0 to 1.0)</param>
        /// <param name="maxTokens">The maximum number of tokens to generate</param>
        /// <returns>The generated text</returns>
        public virtual async Task<string> GetChatCompletionAsync(
            List<ChatMessage> messages,
            string model = null,
            double temperature = 0.7,
            int maxTokens = 1000)
        {
            try
            {
                _logger.LogInformation($"Getting chat completion with {messages.Count} messages");

                // Use the default model if none is provided
                model = model ?? _defaultModel;

                // Create the request payload
                var payload = new
                {
                    model = model,
                    messages = messages,
                    temperature = temperature,
                    max_tokens = maxTokens,
                    stream = false
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_ollamaEndpoint, content);

                // Check if the request was successful
                if (!response.IsSuccessStatusCode)
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error from LLM API: {errorContent}");
                    throw new Exception($"Error from LLM API: {response.StatusCode} - {errorContent}");
                }

                // Parse the response
                string responseContent = await response.Content.ReadAsStringAsync();
                var responseObject = JsonSerializer.Deserialize<OllamaResponse>(responseContent);

                return responseObject?.Response ?? string.Empty;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting chat completion: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Gets the available models from the LLM API
        /// </summary>
        /// <returns>A list of available models</returns>
        public virtual async Task<List<string>> GetAvailableModelsAsync()
        {
            try
            {
                _logger.LogInformation("Getting available models");

                // Send the request
                var response = await _httpClient.GetAsync("http://localhost:11434/api/tags");

                // Check if the request was successful
                if (!response.IsSuccessStatusCode)
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error from LLM API: {errorContent}");
                    throw new Exception($"Error from LLM API: {response.StatusCode} - {errorContent}");
                }

                // Parse the response
                string responseContent = await response.Content.ReadAsStringAsync();
                var responseObject = JsonSerializer.Deserialize<OllamaTagsResponse>(responseContent);

                return responseObject?.Models ?? new List<string>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting available models: {ex.Message}");
                throw;
            }
        }
    }
}
