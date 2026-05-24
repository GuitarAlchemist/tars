namespace Tars.Llm

/// <summary>
/// LLM error types for type-safe error handling with AsyncResult
/// </summary>
/// LLM operation errors
type LlmError =
    | InvalidPrompt of reason: string
    | ModelNotFound of model: string
    | ApiTimeout of endpoint: string
    | NetworkError of message: string
    | ResponseParseError of message: string
    | RateLimitExceeded
    | InsufficientContext of required: int * available: int
    | UnknownError of exn

module LlmError =
    /// Convert exception to LlmError
    let fromException (ex: exn) : LlmError =
        match ex with
        | :? System.Net.Http.HttpRequestException as httpEx ->
            if httpEx.Message.Contains("timeout") then
                ApiTimeout "unknown"
            else
                NetworkError httpEx.Message
        | :? System.Text.Json.JsonException as jsonEx -> ResponseParseError jsonEx.Message
        | _ -> UnknownError ex

    /// Get user-friendly error message
    let toMessage (error: LlmError) : string =
        match error with
        | InvalidPrompt reason -> $"Invalid prompt: {reason}"
        | ModelNotFound model -> $"Model not found: {model}"
        | ApiTimeout endpoint -> $"API timeout at {endpoint}"
        | NetworkError msg -> $"Network error: {msg}"
        | ResponseParseError msg -> $"Failed to parse response: {msg}"
        | RateLimitExceeded -> "Rate limit exceeded, please retry later"
        | InsufficientContext(required, available) ->
            $"Insufficient context: required {required} tokens, available {available}"
        | UnknownError ex -> $"Unknown error: {ex.Message}"
