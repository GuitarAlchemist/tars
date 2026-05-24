namespace Tars.Llm

/// AI Functions: functions evaluated by LLM agents with typed I/O and post-condition validation.
/// Inspired by Strands Agents' AI Functions — replaces "prompt and pray" with verified contracts.
///
/// Usage:
///   let summarize = AiFunction.create<string, Summary> {
///       Name = "summarize"
///       SystemPrompt = "Return JSON: {\"text\": \"...\", \"wordCount\": N}"
///       FormatInput = fun text -> $"Summarize:\n{text}"
///       PostConditions = [ fun r -> if r.WordCount <= 50 then Ok () else Error "too long" ]
///       MaxAttempts = 3
///       JsonSchema = None; ModelHint = None; Temperature = None
///   }
///   let! result = AiFunction.executeAsync llm summarize input

open System
open System.Text.Json

// ─── Types ───────────────────────────────────────────────────────────────────

/// Errors from AI Function execution.
type AiFunctionError =
    | DeserializationFailed of raw: string * error: string
    | PostConditionFailed of attempts: int * lastError: string
    | LlmCallFailed of exn: string
    | MaxAttemptsExhausted of attempts: int * lastViolation: string

/// Configuration for a single AI Function.
type AiFunctionConfig<'input, 'output> = {
    /// Human-readable name for tracing/logging
    Name: string
    /// System prompt instructing the LLM how to produce the output
    SystemPrompt: string
    /// Converts typed input to a user-message string
    FormatInput: 'input -> string
    /// Post-conditions: each returns Ok () or Error "reason"
    PostConditions: ('output -> Result<unit, string>) list
    /// Max attempts when post-conditions fail (default 3)
    MaxAttempts: int
    /// Optional JSON schema for constrained decoding
    JsonSchema: string option
    /// Optional model hint for routing
    ModelHint: string option
    /// Optional temperature override
    Temperature: float option
}

// ─── Core execution ──────────────────────────────────────────────────────────

module AiFunction =

    let private jsonOptions =
        lazy (
            let o = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
            o
        )

    /// Build an LLM request from config, input, and accumulated feedback.
    let internal buildRequest (cfg: AiFunctionConfig<'i, 'o>) (input: 'i) (feedback: string list) : LlmRequest =
        let userContent = cfg.FormatInput input
        let messages =
            [ { Role = User; Content = userContent } ]
            @ (feedback |> List.map (fun fb ->
                { Role = User; Content = $"Your previous output failed validation: {fb}\nPlease fix and try again. Output ONLY valid JSON." }))

        let baseReq =
            { LlmRequest.Default with
                SystemPrompt = Some cfg.SystemPrompt
                Messages = messages
                ModelHint = cfg.ModelHint
                Temperature = cfg.Temperature
                JsonMode = true }

        match cfg.JsonSchema with
        | Some schema -> ConstrainedDecoding.withJsonSchema schema baseReq
        | None -> baseReq

    /// Deserialize LLM text to 'output, handling fenced code blocks.
    let internal deserialize<'output> (text: string) : Result<'output, AiFunctionError> =
        match JsonParsing.tryParseElement text with
        | Error e -> Error (DeserializationFailed(text, e))
        | Ok elem ->
            try
                let json = elem.GetRawText()
                let result = JsonSerializer.Deserialize<'output>(json, jsonOptions.Value)
                Ok result
            with ex ->
                Error (DeserializationFailed(text, ex.Message))

    /// Validate output against all post-conditions (short-circuit on first failure).
    let internal validate (postConditions: ('o -> Result<unit, string>) list) (output: 'o) : Result<'o, string> =
        postConditions
        |> List.fold (fun acc pc ->
            match acc with
            | Error _ -> acc
            | Ok o -> match pc o with Ok () -> Ok o | Error reason -> Error reason
        ) (Ok output)

    /// Execute an AI function: call LLM, deserialize, validate, retry on post-condition failure.
    let executeAsync
        (llm: ILlmService)
        (cfg: AiFunctionConfig<'input, 'output>)
        (input: 'input)
        : Async<Result<'output, AiFunctionError>> =
        async {
            let mutable attempt = 0
            let mutable feedback: string list = []
            let mutable result: Result<'output, AiFunctionError> = Error (MaxAttemptsExhausted(0, "not started"))

            while attempt < cfg.MaxAttempts do
                attempt <- attempt + 1
                let req = buildRequest cfg input feedback

                let! resp =
                    async {
                        try
                            let! r = llm.CompleteAsync(req) |> Async.AwaitTask
                            return Ok r
                        with ex ->
                            return Error (LlmCallFailed ex.Message)
                    }

                match resp with
                | Error e ->
                    result <- Error e
                    attempt <- cfg.MaxAttempts // break
                | Ok llmResp ->
                    match deserialize<'output> llmResp.Text with
                    | Error e ->
                        let msg = match e with DeserializationFailed(_, m) -> m | _ -> "unknown"
                        feedback <- feedback @ [$"Deserialization failed: {msg}"]
                        result <- Error e
                    | Ok output ->
                        match validate cfg.PostConditions output with
                        | Ok valid ->
                            result <- Ok valid
                            attempt <- cfg.MaxAttempts // break
                        | Error reason ->
                            feedback <- feedback @ [$"Post-condition violated: {reason}"]
                            result <- Error (PostConditionFailed(attempt, reason))

            // Wrap trailing post-condition/deser failures as exhausted
            match result with
            | Error (PostConditionFailed(_, reason)) ->
                return Error (MaxAttemptsExhausted(cfg.MaxAttempts, reason))
            | Error (DeserializationFailed(raw, msg)) when attempt >= cfg.MaxAttempts ->
                return Error (MaxAttemptsExhausted(cfg.MaxAttempts, $"Deserialization: {msg}"))
            | other -> return other
        }

    /// Create a default config with name and prompt.
    let create<'i, 'o> (name: string) (systemPrompt: string) (formatInput: 'i -> string) : AiFunctionConfig<'i, 'o> =
        { Name = name
          SystemPrompt = systemPrompt
          FormatInput = formatInput
          PostConditions = []
          MaxAttempts = 3
          JsonSchema = None
          ModelHint = None
          Temperature = None }

    /// Add a post-condition to a config.
    let withPostCondition (name: string) (predicate: 'o -> bool) (cfg: AiFunctionConfig<'i, 'o>) : AiFunctionConfig<'i, 'o> =
        let pc output = if predicate output then Ok () else Error $"Condition '{name}' failed"
        { cfg with PostConditions = cfg.PostConditions @ [pc] }

    /// Set max retry attempts.
    let withMaxAttempts (n: int) (cfg: AiFunctionConfig<'i, 'o>) : AiFunctionConfig<'i, 'o> =
        { cfg with MaxAttempts = n }

    /// Set JSON schema for constrained decoding.
    let withSchema (schema: string) (cfg: AiFunctionConfig<'i, 'o>) : AiFunctionConfig<'i, 'o> =
        { cfg with JsonSchema = Some schema }

    /// Set model hint.
    let withModel (hint: string) (cfg: AiFunctionConfig<'i, 'o>) : AiFunctionConfig<'i, 'o> =
        { cfg with ModelHint = Some hint }

    /// Set temperature.
    let withTemperature (t: float) (cfg: AiFunctionConfig<'i, 'o>) : AiFunctionConfig<'i, 'o> =
        { cfg with Temperature = Some t }
