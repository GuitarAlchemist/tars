namespace Tars.Llm

open System.Threading.Tasks
open System.Text.Json
open Tars.Core
open Tars.Llm

/// <summary>
/// Fluent builder and executor for LLM requests.
/// Reduces boilerplate at call sites.
/// </summary>
module Prompt =

    // =========================================================================
    // Builders
    // =========================================================================

    /// <summary>Starts a new request with a user message.</summary>
    let ask (content: string) =
        { LlmRequest.Default with Messages = [ { Role = Role.User; Content = content } ] }

    /// <summary>Starts a new request with a system prompt.</summary>
    let system (content: string) =
        { LlmRequest.Default with SystemPrompt = Some content }

    /// <summary>Starts a new request with multiple messages.</summary>
    let ofMessages (messages: LlmMessage list) =
        { LlmRequest.Default with Messages = messages }

    /// <summary>Sets the system prompt.</summary>
    let withSystem (content: string) (req: LlmRequest) =
        { req with SystemPrompt = Some content }

    /// <summary>Sets the sampling temperature (0.0 to 1.0+).</summary>
    let withTemp (temp: float) (req: LlmRequest) =
        { req with Temperature = Some temp }

    /// <summary>Sets the sampling temperature optionally.</summary>
    let withOptTemp (temp: float option) (req: LlmRequest) =
        { req with Temperature = temp |> Option.orElse req.Temperature }

    /// <summary>Sets the model hint for routing (e.g., "code", "reasoning").</summary>
    let withHint (hint: string) (req: LlmRequest) =
        { req with ModelHint = Some hint }

    /// <summary>Sets the model hint optionally.</summary>
    let withOptHint (hint: string option) (req: LlmRequest) =
        { req with ModelHint = hint |> Option.orElse req.ModelHint }

    /// <summary>Sets the maximum tokens to generate.</summary>
    let withMaxTokens (max: int) (req: LlmRequest) =
        { req with MaxTokens = Some max }

    /// <summary>Sets the maximum tokens optionally.</summary>
    let withOptMaxTokens (max: int option) (req: LlmRequest) =
        { req with MaxTokens = max |> Option.orElse req.MaxTokens }

    /// <summary>Sets the specific model override.</summary>
    let withModel (model: string) (req: LlmRequest) =
        { req with Model = Some model }

    /// <summary>Sets the model override optionally.</summary>
    let withOptModel (model: string option) (req: LlmRequest) =
        { req with Model = model |> Option.orElse req.Model }

    /// <summary>Sets the context window size.</summary>
    let withContextWindow (size: int) (req: LlmRequest) =
        { req with ContextWindow = Some size }

    /// <summary>Sets the context window size optionally.</summary>
    let withOptContextWindow (size: int option) (req: LlmRequest) =
        { req with ContextWindow = size |> Option.orElse req.ContextWindow }

    /// <summary>Enables or disables streaming.</summary>
    let withStream (stream: bool) (req: LlmRequest) =
        { req with Stream = stream }

    /// <summary>Adds a message to the conversation.</summary>
    let withMessage (role: Role) (content: string) (req: LlmRequest) =
        { req with Messages = req.Messages @ [ { Role = role; Content = content } ] }

    /// <summary>Adds a user message to the conversation.</summary>
    let withUser (content: string) (req: LlmRequest) =
        withMessage Role.User content req

    /// <summary>Adds an assistant message to the conversation.</summary>
    let withAssistant (content: string) (req: LlmRequest) =
        withMessage Role.Assistant content req

    /// <summary>Enables JSON mode and sets response format to JSON.</summary>
    let withJson (req: LlmRequest) =
        { req with JsonMode = true; ResponseFormat = Some ResponseFormat.Json }

    /// <summary>Sets a constrained grammar for the response.</summary>
    let withConstrained (grammar: Grammar) (req: LlmRequest) =
        { req with ResponseFormat = Some (ResponseFormat.Constrained grammar) }

    /// <summary>Sets an EBNF grammar constraint.</summary>
    let withEbnf (grammar: string) (req: LlmRequest) =
        withConstrained (Grammar.Ebnf grammar) req

    /// <summary>Sets a JSON schema constraint.</summary>
    let withJsonSchema (schema: string) (req: LlmRequest) =
        withConstrained (Grammar.JsonSchema schema) req

    // =========================================================================
    // Executors
    // =========================================================================

    /// <summary>Executes the request and returns the full response.</summary>
    let complete (llm: ILlmService) (req: LlmRequest) : Task<LlmResponse> =
        llm.CompleteAsync req

    /// <summary>Executes the request and returns the generated text.</summary>
    let text (llm: ILlmService) (req: LlmRequest) : Task<string> =
        task {
            let! res = llm.CompleteAsync req
            return res.Text
        }

    /// <summary>
    /// Executes the request and parses the result as JSON.
    /// Reuses existing fenced-JSON extraction from JsonParsing.
    /// </summary>
    let json<'T> (llm: ILlmService) (req: LlmRequest) : Task<Result<'T, string>> =
        task {
            let! res = llm.CompleteAsync req
            match JsonParsing.tryParseElement res.Text with
            | Result.Ok elem ->
                try
                    let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                    let parsed = elem.Deserialize<'T>(options)
                    return Result.Ok parsed
                with ex ->
                    return Result.Error $"JSON deserialization failed: {ex.Message}. Content: {res.Text}"
            | Result.Error err -> return Result.Error $"JSON parsing failed: {err}. Content: {res.Text}"
        }
