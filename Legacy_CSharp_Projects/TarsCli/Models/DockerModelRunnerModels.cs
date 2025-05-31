using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Information about a model
/// </summary>
public record ModelInfo
{
    /// <summary>
    /// The model ID
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    /// <summary>
    /// The object type
    /// </summary>
    [JsonPropertyName("object")]
    public string Object { get; init; } = string.Empty;

    /// <summary>
    /// The creation timestamp
    /// </summary>
    [JsonPropertyName("created")]
    public long Created { get; init; }

    /// <summary>
    /// The owner of the model
    /// </summary>
    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; init; } = string.Empty;

    /// <summary>
    /// The model provider (e.g., OpenAI, Anthropic, Meta, Google)
    /// </summary>
    [JsonPropertyName("provider")]
    public string Provider { get; init; } = string.Empty;

    /// <summary>
    /// The model description
    /// </summary>
    [JsonPropertyName("description")]
    public string Description { get; init; } = string.Empty;

    /// <summary>
    /// The model context length in tokens
    /// </summary>
    [JsonPropertyName("context_length")]
    public int ContextLength { get; init; }
}

/// <summary>
/// Response from the models endpoint
/// </summary>
public record ModelListResponse
{
    /// <summary>
    /// The list of models
    /// </summary>
    [JsonPropertyName("models")]
    public List<ModelInfo>? Models { get; init; }
}

/// <summary>
/// Request for the completions endpoint
/// </summary>
public record CompletionRequest
{
    /// <summary>
    /// The model to use
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    /// <summary>
    /// The prompt to generate a completion for
    /// </summary>
    [JsonPropertyName("prompt")]
    public string Prompt { get; init; } = string.Empty;

    /// <summary>
    /// The maximum number of tokens to generate
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; init; }

    /// <summary>
    /// The temperature to use for sampling
    /// </summary>
    [JsonPropertyName("temperature")]
    public float Temperature { get; init; }

    /// <summary>
    /// Whether to stream the response
    /// </summary>
    [JsonPropertyName("stream")]
    public bool Stream { get; init; }
}

/// <summary>
/// A completion choice
/// </summary>
public record CompletionChoice
{
    /// <summary>
    /// The generated text
    /// </summary>
    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// The index of the choice
    /// </summary>
    [JsonPropertyName("index")]
    public int Index { get; init; }

    /// <summary>
    /// The reason the completion finished
    /// </summary>
    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; init; } = string.Empty;
}

/// <summary>
/// Response from the completions endpoint
/// </summary>
public record CompletionResponse
{
    /// <summary>
    /// The response ID
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    /// <summary>
    /// The object type
    /// </summary>
    [JsonPropertyName("object")]
    public string Object { get; init; } = string.Empty;

    /// <summary>
    /// The creation timestamp
    /// </summary>
    [JsonPropertyName("created")]
    public long Created { get; init; }

    /// <summary>
    /// The model used
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    /// <summary>
    /// The completion choices
    /// </summary>
    [JsonPropertyName("choices")]
    public List<CompletionChoice>? Choices { get; init; }
}

/// <summary>
/// A chat message
/// </summary>
public record ChatMessageModel
{
    /// <summary>
    /// The role of the message sender
    /// </summary>
    [JsonPropertyName("role")]
    public string Role { get; init; } = string.Empty;

    /// <summary>
    /// The message content
    /// </summary>
    [JsonPropertyName("content")]
    public string Content { get; init; } = string.Empty;
}

/// <summary>
/// Request for the chat completions endpoint
/// </summary>
public record ChatCompletionRequest
{
    /// <summary>
    /// The model to use
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    /// <summary>
    /// The messages to generate a completion for
    /// </summary>
    [JsonPropertyName("messages")]
    public List<Dictionary<string, object>> Messages { get; init; } = new();

    /// <summary>
    /// The maximum number of tokens to generate
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; init; }

    /// <summary>
    /// The temperature to use for sampling
    /// </summary>
    [JsonPropertyName("temperature")]
    public float Temperature { get; init; }

    /// <summary>
    /// Whether to stream the response
    /// </summary>
    [JsonPropertyName("stream")]
    public bool Stream { get; init; }
}

/// <summary>
/// A chat completion choice
/// </summary>
public record ChatCompletionChoice
{
    /// <summary>
    /// The index of the choice
    /// </summary>
    [JsonPropertyName("index")]
    public int Index { get; init; }

    /// <summary>
    /// The message
    /// </summary>
    [JsonPropertyName("message")]
    public ChatMessageModel? Message { get; init; }

    /// <summary>
    /// The reason the completion finished
    /// </summary>
    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; init; } = string.Empty;
}

/// <summary>
/// Response from the chat completions endpoint
/// </summary>
public record ChatCompletionResponse
{
    /// <summary>
    /// The response ID
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    /// <summary>
    /// The object type
    /// </summary>
    [JsonPropertyName("object")]
    public string Object { get; init; } = string.Empty;

    /// <summary>
    /// The creation timestamp
    /// </summary>
    [JsonPropertyName("created")]
    public long Created { get; init; }

    /// <summary>
    /// The model used
    /// </summary>
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    /// <summary>
    /// The completion choices
    /// </summary>
    [JsonPropertyName("choices")]
    public List<ChatCompletionChoice>? Choices { get; init; }
}
