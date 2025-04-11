using System.Text.Json.Serialization;

namespace TarsEngine.Services.Models;

/// <summary>
/// Represents a chat message for LLM interactions
/// </summary>
public class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;
}

/// <summary>
/// Represents a response from the Ollama API
/// </summary>
public class OllamaResponse
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("response")]
    public string Response { get; set; } = string.Empty;

    [JsonPropertyName("done")]
    public bool Done { get; set; }
}

/// <summary>
/// Represents a response from the Ollama tags API
/// </summary>
public class OllamaTagsResponse
{
    [JsonPropertyName("models")]
    public List<string> Models { get; set; } = new List<string>();
}