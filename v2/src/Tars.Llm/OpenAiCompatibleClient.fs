namespace Tars.Llm

// Generic OpenAI-compatible client for various LLM providers.
// Works with any service that implements the OpenAI API format including
// OpenAI, Azure OpenAI, LocalAI, LM Studio, text-generation-inference, etc.
//
// This is the most versatile client as many LLM services implement
// the OpenAI API format for compatibility.

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Client module for interacting with OpenAI-compatible APIs.
/// Supports chat completions and embeddings.
/// </summary>
module OpenAiCompatibleClient =

    open Tars.Llm

    /// <summary>DTO for OpenAI message.</summary>
    [<CLIMutable>]
    type OpenAiMessageDto = { role: string; content: string }

    /// <summary>DTO for OpenAI chat request.</summary>
    [<CLIMutable>]
    type OpenAiRequestDto =
        { model: string
          messages: OpenAiMessageDto[]
          max_tokens: int option
          temperature: float option
          stream: bool option
          response_format: obj option
          /// vLLM >= 0.12 unified constraint field, emitted top-level. Replaces the
          /// old nested `extra_body.guided_decoding`, which was never a server-side
          /// API — `extra_body` is a Python-SDK client-side kwarg that merges into
          /// the body, so a literal field by that name was silently ignored.
          /// Gated on the resolved backend: OpenAI proper rejects unknown top-level
          /// params, so this stays None for anything but vLLM.
          structured_outputs: obj option }

    /// <summary>DTO for response message.</summary>
    [<CLIMutable>]
    type OpenAiChoiceMessageDto = { role: string; content: string }

    /// <summary>DTO for response choice.</summary>
    [<CLIMutable>]
    type OpenAiChoiceDto =
        { index: int
          message: OpenAiChoiceMessageDto
          finish_reason: string }

    /// <summary>DTO for token usage statistics.</summary>
    [<CLIMutable>]
    type OpenAiUsageDto =
        { prompt_tokens: int
          completion_tokens: int
          total_tokens: int }

    /// <summary>DTO for OpenAI chat response.</summary>
    [<CLIMutable>]
    type OpenAiResponseDto =
        { id: string
          choices: OpenAiChoiceDto[]
          usage: OpenAiUsageDto option }

    /// Exposed so serialization tests assert the exact bytes we put on the wire.
    let jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    /// Build the vLLM `structured_outputs` payload. Returns None unless the resolved
    /// backend is vLLM: `Vllm`, `OpenAI` and `DockerModelRunner` share this adapter,
    /// and OpenAI proper 400s on unknown top-level parameters.
    let private buildStructuredOutputs (vllmExtensions: bool) (req: LlmRequest) : obj option =
        if not vllmExtensions then
            None
        else
            match req.ResponseFormat with
            | Some(ResponseFormat.Constrained(Grammar.Ebnf grammar)) -> Some(box {| grammar = grammar |})
            | Some(ResponseFormat.Constrained(Grammar.Regex pattern)) -> Some(box {| regex = pattern |})
            | Some(ResponseFormat.Constrained(Grammar.JsonSchema schema)) ->
                Some(box {| json = JsonSerializer.Deserialize<JsonElement>(schema) |})
            | _ -> None

    let private toOpenAiRole =
        function
        | Role.System -> "system"
        | Role.User -> "user"
        | Role.Assistant -> "assistant"

    let private toOpenAiMessages (systemPrompt: string option) (msgs: LlmMessage list) =
        let systemMsg =
            match systemPrompt with
            | Some p -> [ ({ role = "system"; content = p }: OpenAiMessageDto) ]
            | None -> []

        let otherMsgs =
            msgs
            |> List.map (fun m ->
                { role = toOpenAiRole m.Role
                  content = m.Content }
                : OpenAiMessageDto)

        (systemMsg @ otherMsgs) |> List.toArray

    /// The single place an OpenAI-wire request DTO is shaped. Both the streaming and
    /// non-streaming paths go through here — they were byte-for-byte duplicates, and
    /// a wire contract that depends on whether you stream is a bug waiting to happen.
    /// Public so serialization tests can assert the exact payload; nothing else in
    /// the codebase asserted it before, which is how the dead `extra_body` shape
    /// survived.
    let buildRequestDto (vllmExtensions: bool) (model: string) (stream: bool) (req: LlmRequest) : OpenAiRequestDto =
        { model = model
          messages = toOpenAiMessages req.SystemPrompt req.Messages
          max_tokens = req.MaxTokens
          temperature = req.Temperature
          stream = Some stream
          structured_outputs = buildStructuredOutputs vllmExtensions req
          response_format =
            match req.ResponseFormat with
            | Some ResponseFormat.Json -> Some(box {| ``type`` = "json_object" |})
            | Some(ResponseFormat.Constrained(Grammar.JsonSchema schema)) ->
                Some(
                    box
                        {| ``type`` = "json_schema"
                           json_schema =
                            {| name = "output"
                               strict = true
                               schema = JsonSerializer.Deserialize<JsonElement>(schema) |} |}
                )
            | Some(ResponseFormat.Constrained(Grammar.Ebnf _)) -> None // carried in structured_outputs
            | Some(ResponseFormat.Constrained(Grammar.Regex _)) -> None // carried in structured_outputs
            | Some ResponseFormat.Text -> None
            | None ->
                if req.JsonMode then
                    Some(box {| ``type`` = "json_object" |})
                else
                    None }

    /// <summary>DTO for embedding request.</summary>
    [<CLIMutable>]
    type OpenAiEmbeddingRequestDto = { input: string; model: string }

    /// <summary>DTO for embedding data.</summary>
    [<CLIMutable>]
    type OpenAiEmbeddingDataDto = { embedding: float32[] }

    /// <summary>DTO for embedding response.</summary>
    [<CLIMutable>]
    type OpenAiEmbeddingResponseDto = { data: OpenAiEmbeddingDataDto[] }

    /// <summary>
    /// Sends a chat completion request to an OpenAI-compatible API.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the API server.</param>
    /// <param name="model">The model name (e.g., "gpt-4", "gpt-3.5-turbo").</param>
    /// <param name="req">The LLM request containing messages and parameters.</param>
    /// <param name="apiKey">Optional API key for authentication.</param>
    /// <returns>The LLM response with generated text and usage stats.</returns>
    let sendChatAsyncWith
        (vllmExtensions: bool)
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        : Task<LlmResponse> =
        task {
            let dto = buildRequestDto vllmExtensions model false req


            let uri = Uri(baseUri, "/v1/chat/completions")
            let content = JsonContent.Create(dto, options = jsonOptions)
            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)

            match apiKey with
            | Some key when not (String.IsNullOrWhiteSpace(key)) ->
                requestMessage.Headers.Authorization <- System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", key)
            | _ -> ()

            use! resp = http.SendAsync(requestMessage)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<OpenAiResponseDto>(raw, jsonOptions)

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

                    return
                        { Text = c.message.content
                          FinishReason = Some c.finish_reason
                          Usage = usage
                          Raw = Some raw }
        }

    /// <summary>
    /// Generates embeddings for the given text using an OpenAI-compatible API.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the API server.</param>
    /// <param name="model">The embedding model name (e.g., "text-embedding-ada-002").</param>
    /// <param name="text">The text to embed.</param>
    /// <param name="apiKey">Optional API key for authentication.</param>
    /// <returns>The embedding vector as float32 array.</returns>
    let getEmbeddingsAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (text: string)
        : Task<float32[]> =
        task {
            let dto: OpenAiEmbeddingRequestDto = { input = text; model = model }
            let uri = Uri(baseUri, "/v1/embeddings")
            let content = JsonContent.Create(dto, options = jsonOptions)
            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)

            match apiKey with
            | Some key when not (String.IsNullOrWhiteSpace(key)) ->
                requestMessage.Headers.Authorization <- System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", key)
            | _ -> ()

            use! resp = http.SendAsync(requestMessage)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()

            let parsed =
                JsonSerializer.Deserialize<OpenAiEmbeddingResponseDto>(raw, jsonOptions)

            if isNull (box parsed) || isNull parsed.data || parsed.data.Length = 0 then
                return [||]
            else
                return parsed.data[0].embedding
        }

    /// <summary>DTO for streaming response delta.</summary>
    [<CLIMutable>]
    type OpenAiStreamDeltaDto = { content: string }

    /// <summary>DTO for streaming choice.</summary>
    [<CLIMutable>]
    type OpenAiStreamChoiceDto =
        { index: int
          delta: OpenAiStreamDeltaDto
          finish_reason: string }

    /// <summary>DTO for streaming response chunk.</summary>
    [<CLIMutable>]
    type OpenAiStreamResponseDto =
        { id: string
          choices: OpenAiStreamChoiceDto[] }

    /// <summary>
    /// Sends a streaming chat completion request to an OpenAI-compatible API.
    /// Yields tokens as they are generated.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the API server.</param>
    /// <param name="model">The model name.</param>
    /// <param name="req">The LLM request.</param>
    /// <param name="onToken">Callback invoked for each token received.</param>
    /// <returns>The complete LLM response after streaming completes.</returns>
    let sendChatStreamAsyncWith
        (vllmExtensions: bool)
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        (onToken: string -> unit)
        : Task<LlmResponse> =
        task {
            let dto = buildRequestDto vllmExtensions model true req


            let uri = Uri(baseUri, "/v1/chat/completions")

            let content =
                new StringContent(
                    JsonSerializer.Serialize(dto, jsonOptions),
                    System.Text.Encoding.UTF8,
                    "application/json"
                )

            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)

            match apiKey with
            | Some key when not (String.IsNullOrWhiteSpace(key)) ->
                requestMessage.Headers.Authorization <- System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", key)
            | _ -> ()

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
                    // OpenAI streams data: prefix
                    let dataLine =
                        if line.StartsWith("data: ") then
                            line.Substring(6)
                        else
                            line

                    if dataLine = "[DONE]" then
                        isDone <- true
                    elif not (String.IsNullOrWhiteSpace(dataLine)) then
                        try
                            let chunk =
                                JsonSerializer.Deserialize<OpenAiStreamResponseDto>(dataLine, jsonOptions)

                            if
                                not (isNull (box chunk))
                                && not (isNull chunk.choices)
                                && chunk.choices.Length > 0
                            then
                                let choice = chunk.choices.[0]

                                if not (isNull (box choice.delta)) && not (isNull choice.delta.content) then
                                    let token = choice.delta.content
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

    /// Back-compat entry points. `vllmExtensions` defaults to false — the
    /// OpenAI-safe choice, since emitting vLLM-only top-level params against
    /// OpenAI proper is a 400. Backends.resolve opts in for the Vllm case.
    let sendChatAsync http baseUri model apiKey req =
        sendChatAsyncWith false http baseUri model apiKey req

    let sendChatStreamAsync http baseUri model apiKey req onToken =
        sendChatStreamAsyncWith false http baseUri model apiKey req onToken
