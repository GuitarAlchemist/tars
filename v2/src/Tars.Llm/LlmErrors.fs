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

/// A provider answered that it has no such model. Carries the name so the
/// functional path can report `ModelNotFound` instead of an opaque failure — a
/// plain `failwith` reaches `fromException` as `UnknownError`, which callers can
/// only log.
type ModelNotFoundException(model: string, detail: string) =
    inherit exn(detail)
    /// The model name the provider did not recognise.
    member _.Model = model

module LlmError =
    /// Convert exception to LlmError
    let rec fromException (ex: exn) : LlmError =
        match ex with
        | :? System.AggregateException as aggregate ->
            // Anything raised inside a Task arrives wrapped, so classifying the
            // wrapper turns every typed answer underneath it into UnknownError.
            match aggregate.Flatten().InnerExceptions |> Seq.tryHead with
            | Some inner -> fromException inner
            | None -> UnknownError ex
        | :? ModelNotFoundException as notFound -> ModelNotFound notFound.Model
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
