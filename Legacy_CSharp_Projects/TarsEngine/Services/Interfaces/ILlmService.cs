using TarsEngine.Services.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the LLM service
/// </summary>
public interface ILlmService
{
    /// <summary>
    /// Gets a completion from the LLM
    /// </summary>
    /// <param name="prompt">The prompt to send to the LLM</param>
    /// <param name="model">The model to use (defaults to configuration value)</param>
    /// <param name="temperature">The temperature to use (0.0 to 1.0)</param>
    /// <param name="maxTokens">The maximum number of tokens to generate</param>
    /// <returns>The generated text</returns>
    Task<string> GetCompletionAsync(
        string prompt,
        string model = null,
        double temperature = 0.7,
        int maxTokens = 1000);

    /// <summary>
    /// Gets a chat completion from the LLM
    /// </summary>
    /// <param name="messages">The messages to send to the LLM</param>
    /// <param name="model">The model to use (defaults to configuration value)</param>
    /// <param name="temperature">The temperature to use (0.0 to 1.0)</param>
    /// <param name="maxTokens">The maximum number of tokens to generate</param>
    /// <returns>The generated text</returns>
    Task<string> GetChatCompletionAsync(
        List<ChatMessage> messages,
        string model = null,
        double temperature = 0.7,
        int maxTokens = 1000);

    /// <summary>
    /// Gets the available models from the LLM API
    /// </summary>
    /// <returns>A list of available models</returns>
    Task<List<string>> GetAvailableModelsAsync();
}