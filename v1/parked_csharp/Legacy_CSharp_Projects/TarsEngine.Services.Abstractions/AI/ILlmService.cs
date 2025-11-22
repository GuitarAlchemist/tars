using TarsEngine.Services.Abstractions.Common;

namespace TarsEngine.Services.Abstractions.AI
{
    /// <summary>
    /// Interface for services that interact with Large Language Models.
    /// </summary>
    public interface ILlmService : IService
    {
        /// <summary>
        /// Generates a completion for the given prompt.
        /// </summary>
        /// <param name="prompt">The prompt to generate a completion for.</param>
        /// <param name="maxTokens">The maximum number of tokens to generate.</param>
        /// <param name="temperature">The sampling temperature to use.</param>
        /// <param name="model">The model to use for generation.</param>
        /// <returns>The generated completion.</returns>
        Task<string> GenerateCompletionAsync(
            string prompt,
            int maxTokens = 1000,
            float temperature = 0.7f,
            string? model = null);

        /// <summary>
        /// Generates a chat completion for the given messages.
        /// </summary>
        /// <param name="messages">The messages to generate a completion for.</param>
        /// <param name="maxTokens">The maximum number of tokens to generate.</param>
        /// <param name="temperature">The sampling temperature to use.</param>
        /// <param name="model">The model to use for generation.</param>
        /// <returns>The generated chat completion.</returns>
        Task<string> GenerateChatCompletionAsync(
            IEnumerable<(string role, string content)> messages,
            int maxTokens = 1000,
            float temperature = 0.7f,
            string? model = null);
    }
}
