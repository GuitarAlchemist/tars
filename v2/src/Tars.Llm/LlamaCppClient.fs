namespace Tars.Llm

// Client for llama.cpp server (llama-server).
// Provides high-performance local inference with GGUF models.
// Uses OpenAI-compatible API with llama.cpp-specific extensions.
//
// llama.cpp server provides an OpenAI-compatible endpoint at /v1/chat/completions.
// This client adds support for llama.cpp-specific features like slot management,
// performance metrics, and flash attention configuration.

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Client module for llama.cpp server interactions.
/// </summary>
module LlamaCppClient =

    open Tars.Llm

    /// <summary>Performance metrics from llama.cpp inference.</summary>
    type LlamaCppMetrics =
        {
            /// <summary>Tokens per second for generation.</summary>
            TokensPerSecond: float
            /// <summary>Tokens per second for prompt processing.</summary>
            PromptTokensPerSecond: float
            /// <summary>Total inference time in milliseconds.</summary>
            TotalTimeMs: int64
        }

    /// <summary>Model information from llama.cpp server.</summary>
    type LlamaCppModelInfo =
        {
            /// <summary>Model architecture (e.g., "llama", "qwen2").</summary>
            Architecture: string option
            /// <summary>Number of parameters.</summary>
            ParameterCount: int64 option
            /// <summary>Context length.</summary>
            ContextLength: int option
        }

    /// <summary>DTO for llama.cpp message.</summary>
    [<CLIMutable>]
    type LlamaCppMessageDto = { role: string; content: string }

    /// <summary>DTO for response format.</summary>
    [<CLIMutable>]
    type LlamaCppResponseFormatDto = { ``type``: string }

    /// <summary>DTO for llama.cpp chat request with extended options.</summary>
    [<CLIMutable>]
    type LlamaCppRequestDto =
        { model: string
          messages: LlamaCppMessageDto[]
          max_tokens: int option
          temperature: float option
          stream: bool option
          response_format: LlamaCppResponseFormatDto option
          n_gpu_layers: int option
          n_ctx: int option
          n_parallel: int option
          flash_attn: bool option }

    /// <summary>DTO for response message.</summary>
    [<CLIMutable>]
    type LlamaCppChoiceMessageDto =
        { role: string
          content: string
          reasoning_content: string option }

    /// <summary>DTO for response choice.</summary>
    [<CLIMutable>]
    type LlamaCppChoiceDto =
        { index: int
          message: LlamaCppChoiceMessageDto
          finish_reason: string }

    /// <summary>DTO for usage statistics.</summary>
    [<CLIMutable>]
    type LlamaCppUsageDto =
        { prompt_tokens: int
          completion_tokens: int
          total_tokens: int }

    /// <summary>DTO for timing information (llama.cpp specific).</summary>
    [<CLIMutable>]
    type LlamaCppTimingsDto =
        { predicted_per_second: float option
          prompt_per_second: float option }

    /// <summary>DTO for llama.cpp chat response.</summary>
    [<CLIMutable>]
    type LlamaCppResponseDto =
        { id: string
          choices: LlamaCppChoiceDto[]
          usage: LlamaCppUsageDto option
          timings: LlamaCppTimingsDto option }

    /// <summary>DTO for health endpoint response.</summary>
    [<CLIMutable>]
    type LlamaCppHealthDto =
        { status: string
          slots_idle: int option
          slots_processing: int option }

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toLlamaCppRole =
        function
        | Role.System -> "system"
        | Role.User -> "user"
        | Role.Assistant -> "assistant"

    let private toLlamaCppMessages (systemPrompt: string option) (msgs: LlmMessage list) =
        let systemMsg =
            match systemPrompt with
            | Some p -> [ ({ role = "system"; content = p }: LlamaCppMessageDto) ]
            | None -> []

        let otherMsgs =
            msgs
            |> List.map (fun m ->
                { role = toLlamaCppRole m.Role
                  content = m.Content }
                : LlamaCppMessageDto)

        (systemMsg @ otherMsgs) |> List.toArray

    /// <summary>
    /// Checks if llama.cpp server is healthy and ready.
    /// </summary>
    /// <param name="http">The HttpClient to use.</param>
    /// <param name="baseUri">The base URI of the llama.cpp server.</param>
    /// <returns>True if server is healthy.</returns>
    let isHealthyAsync (http: HttpClient) (baseUri: Uri) : Task<bool> =
        task {
            try
                let uri = Uri(baseUri, "/health")
                let! resp = http.GetAsync(uri)
                return resp.IsSuccessStatusCode
            with _ ->
                return false
        }

    /// <summary>
    /// Sends a chat completion request to llama.cpp server.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the llama.cpp server.</param>
    /// <param name="model">The model name (informational for llama.cpp).</param>
    /// <param name="config">Optional llama.cpp-specific configuration.</param>
    /// <param name="req">The LLM request containing messages and parameters.</param>
    /// <returns>The LLM response with generated text.</returns>
    let sendChatAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (config: LlamaCppConfig option)
        (req: LlmRequest)
        : Task<LlmResponse> =
        task {
            let cfg = config |> Option.defaultValue LlamaCppConfig.Default

            let dto: LlamaCppRequestDto =
                { model = model
                  messages = toLlamaCppMessages req.SystemPrompt req.Messages
                  max_tokens = req.MaxTokens
                  temperature = req.Temperature
                  stream = Some false
                  response_format =
                    match req.ResponseFormat with
                    | Some ResponseFormat.Json -> Some { ``type`` = "json_object" }
                    | _ ->
                        if req.JsonMode then
                            Some { ``type`` = "json_object" }
                        else
                            None
                  n_gpu_layers = cfg.GpuLayers
                  n_ctx = cfg.ContextSize
                  n_parallel = cfg.NumParallel
                  flash_attn = if cfg.FlashAttention then Some true else None }

            // Ensure baseUri has trailing slash for proper path combination
            let normalizedBase =
                let s = baseUri.ToString()
                if s.EndsWith("/") then baseUri else Uri(s + "/")

            let uri = Uri(normalizedBase, "v1/chat/completions")
            // printfn "  [DEBUG LlamaCpp] Calling: %s" (uri.ToString())
            let content = JsonContent.Create(dto, options = jsonOptions)
            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)


            use! resp = http.SendAsync(requestMessage)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<LlamaCppResponseDto>(raw, jsonOptions)

            if isNull (box parsed) then
                return
                    { Text = ""
                      FinishReason = Some "parse_error"
                      Usage = None
                      Raw = Some raw }
            else
                let choice =
                    if parsed.choices = null then
                        None
                    else
                        parsed.choices |> Array.sortBy (fun c -> c.index) |> Array.tryHead

                match choice with
                | None ->
                    return
                        { Text = ""
                          FinishReason = Some "no_choices"
                          Usage = None
                          Raw = Some raw }
                | Some c ->
                    let usage =
                        match parsed.usage with
                        | Some u ->
                            Some
                                { PromptTokens = u.prompt_tokens
                                  CompletionTokens = u.completion_tokens
                                  TotalTokens = u.total_tokens }
                        | None -> None

                    let text =
                        if not (String.IsNullOrEmpty c.message.content) then
                            c.message.content
                        else
                            c.message.reasoning_content |> Option.defaultValue ""

                    return
                        { Text = text
                          FinishReason = Some c.finish_reason
                          Usage = usage
                          Raw = Some raw }
        }

    /// <summary>
    /// Sends a streaming chat completion request to llama.cpp server.
    /// </summary>
    let sendChatStreamAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (config: LlamaCppConfig option)
        (req: LlmRequest)
        (onToken: string -> unit)
        : Task<LlmResponse> =
        task {
            let cfg = config |> Option.defaultValue LlamaCppConfig.Default

            let dto: LlamaCppRequestDto =
                { model = model
                  messages = toLlamaCppMessages req.SystemPrompt req.Messages
                  max_tokens = req.MaxTokens
                  temperature = req.Temperature
                  stream = Some true
                  response_format =
                    match req.ResponseFormat with
                    | Some ResponseFormat.Json -> Some { ``type`` = "json_object" }
                    | _ ->
                        if req.JsonMode then
                            Some { ``type`` = "json_object" }
                        else
                            None
                  n_gpu_layers = cfg.GpuLayers
                  n_ctx = cfg.ContextSize
                  n_parallel = cfg.NumParallel
                  flash_attn = if cfg.FlashAttention then Some true else None }

            // Ensure baseUri has trailing slash for proper path combination
            let normalizedBase =
                let s = baseUri.ToString()
                if s.EndsWith("/") then baseUri else Uri(s + "/")

            let uri = Uri(normalizedBase, "v1/chat/completions")


            let content =
                new StringContent(
                    JsonSerializer.Serialize(dto, jsonOptions),
                    System.Text.Encoding.UTF8,
                    "application/json"
                )

            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)

            use! resp = http.SendAsync(requestMessage, HttpCompletionOption.ResponseHeadersRead)
            resp.EnsureSuccessStatusCode() |> ignore

            use! stream = resp.Content.ReadAsStreamAsync()
            use reader = new System.IO.StreamReader(stream)

            let mutable fullText = ""
            let mutable isDone = false
            let mutable finishReason = "unknown"

            while not isDone && not reader.EndOfStream do
                let! line = reader.ReadLineAsync()

                if not (String.IsNullOrWhiteSpace(line)) then
                    let dataLine =
                        if line.StartsWith("data: ") then
                            line.Substring(6)
                        else
                            line

                    if dataLine = "[DONE]" then
                        isDone <- true
                    elif not (String.IsNullOrWhiteSpace(dataLine)) then
                        try
                            let chunk = JsonSerializer.Deserialize<LlamaCppResponseDto>(dataLine, jsonOptions)

                            if
                                not (isNull (box chunk))
                                && not (isNull chunk.choices)
                                && chunk.choices.Length > 0
                            then
                                let choice = chunk.choices.[0]

                                if not (isNull (box choice.message)) && not (isNull choice.message.content) then
                                    let token = choice.message.content
                                    fullText <- fullText + token
                                    onToken token

                                if not (isNull choice.finish_reason) then
                                    finishReason <- choice.finish_reason
                                    isDone <- true
                        with _ ->
                            ()

            return
                { Text = fullText
                  FinishReason = Some finishReason
                  Usage = None
                  Raw = None }
        }
