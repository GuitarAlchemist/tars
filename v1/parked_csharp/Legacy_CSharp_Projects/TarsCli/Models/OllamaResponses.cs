using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Response from Ollama API for completions
/// </summary>
public class OllamaCompletionResponse
{
    /// <summary>
    /// The model used
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// The generated completion
    /// </summary>
    [JsonPropertyName("response")]
    public string Response { get; set; } = string.Empty;

    /// <summary>
    /// The message for chat completions
    /// </summary>
    [JsonPropertyName("message")]
    public OllamaMessage? Message { get; set; }

    /// <summary>
    /// Whether the response was truncated
    /// </summary>
    [JsonPropertyName("done")]
    public bool Done { get; set; }

    /// <summary>
    /// Total duration in nanoseconds
    /// </summary>
    [JsonPropertyName("total_duration")]
    public long TotalDuration { get; set; }

    /// <summary>
    /// Load duration in nanoseconds
    /// </summary>
    [JsonPropertyName("load_duration")]
    public long LoadDuration { get; set; }

    /// <summary>
    /// Prompt evaluation duration in nanoseconds
    /// </summary>
    [JsonPropertyName("prompt_eval_duration")]
    public long PromptEvalDuration { get; set; }

    /// <summary>
    /// Evaluation count
    /// </summary>
    [JsonPropertyName("eval_count")]
    public int EvalCount { get; set; }

    /// <summary>
    /// Evaluation duration in nanoseconds
    /// </summary>
    [JsonPropertyName("eval_duration")]
    public long EvalDuration { get; set; }
}
