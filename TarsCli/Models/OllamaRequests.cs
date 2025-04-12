using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Request to generate a completion with Ollama
/// </summary>
public class OllamaCompletionRequest
{
    /// <summary>
    /// The model to use
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// The prompt to generate a completion for
    /// </summary>
    [JsonPropertyName("prompt")]
    public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// Options for the completion
    /// </summary>
    [JsonPropertyName("options")]
    public OllamaOptions? Options { get; set; }
}

/// <summary>
/// Request to generate a chat completion with Ollama
/// </summary>
public class OllamaChatRequest
{
    /// <summary>
    /// The model to use
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// The messages to generate a completion for
    /// </summary>
    [JsonPropertyName("messages")]
    public List<OllamaMessage> Messages { get; set; } = new List<OllamaMessage>();

    /// <summary>
    /// Options for the completion
    /// </summary>
    [JsonPropertyName("options")]
    public OllamaOptions? Options { get; set; }
}

/// <summary>
/// Options for Ollama completions
/// </summary>
public class OllamaOptions
{
    /// <summary>
    /// Temperature for sampling
    /// </summary>
    [JsonPropertyName("temperature")]
    public float Temperature { get; set; } = 0.7f;

    /// <summary>
    /// Number of tokens to predict
    /// </summary>
    [JsonPropertyName("num_predict")]
    public int NumPredict { get; set; } = 2048;
}

/// <summary>
/// Message for Ollama chat completions
/// </summary>
public class OllamaMessage
{
    /// <summary>
    /// Role of the message sender
    /// </summary>
    [JsonPropertyName("role")]
    public string Role { get; set; } = string.Empty;

    /// <summary>
    /// Content of the message
    /// </summary>
    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;
}
