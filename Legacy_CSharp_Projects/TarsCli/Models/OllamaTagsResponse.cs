using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Response from Ollama API for tags
/// </summary>
public class OllamaTagsResponse
{
    /// <summary>
    /// List of models
    /// </summary>
    [JsonPropertyName("models")]
    public List<OllamaModel>? Models { get; set; }
}

/// <summary>
/// Ollama model information
/// </summary>
public class OllamaModel
{
    /// <summary>
    /// Model name
    /// </summary>
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Model size
    /// </summary>
    [JsonPropertyName("size")]
    public long Size { get; set; }

    /// <summary>
    /// Model modified date
    /// </summary>
    [JsonPropertyName("modified_at")]
    public string ModifiedAt { get; set; } = string.Empty;
}
