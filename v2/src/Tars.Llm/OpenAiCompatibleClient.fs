/// <summary>
/// Generic OpenAI-compatible client for various LLM providers.
/// Works with any service that implements the OpenAI API format including
/// OpenAI, Azure OpenAI, LocalAI, LM Studio, text-generation-inference, etc.
/// </summary>
/// <remarks>
/// This is the most versatile client as many LLM services implement
/// the OpenAI API format for compatibility.
/// </remarks>
namespace Tars.Llm

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
          stream: bool option }

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

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toOpenAiRole =
        function
        | Role.System -> "system"
        | Role.User -> "user"
        | Role.Assistant -> "assistant"

    let private toOpenAiMessages (msgs: LlmMessage list) =
        msgs
        |> List.map (fun m ->
            { role = toOpenAiRole m.Role
              content = m.Content }
            : OpenAiMessageDto)
        |> List.toArray

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
    /// <returns>The LLM response with generated text and usage stats.</returns>
    let sendChatAsync (http: HttpClient) (baseUri: Uri) (model: string) (req: LlmRequest) : Task<LlmResponse> =
        task {
            let dto: OpenAiRequestDto =
                { model = model
                  messages = toOpenAiMessages req.Messages
                  max_tokens = req.MaxTokens
                  temperature = req.Temperature
                  stream = Some false }

            let uri = Uri(baseUri, "/v1/chat/completions")
            use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
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
    /// <returns>The embedding vector as float32 array.</returns>
    let getEmbeddingsAsync (http: HttpClient) (baseUri: Uri) (model: string) (text: string) : Task<float32[]> =
        task {
            let dto: OpenAiEmbeddingRequestDto = { input = text; model = model }
            let uri = Uri(baseUri, "/v1/embeddings")
            use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
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
    type OpenAiStreamChoiceDto = { index: int; delta: OpenAiStreamDeltaDto; finish_reason: string }

    /// <summary>DTO for streaming response chunk.</summary>
    [<CLIMutable>]
    type OpenAiStreamResponseDto = { id: string; choices: OpenAiStreamChoiceDto[] }

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
    let sendChatStreamAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (req: LlmRequest)
        (onToken: string -> unit)
        : Task<LlmResponse> =
        task {
            let dto: OpenAiRequestDto =
                { model = model
                  messages = toOpenAiMessages req.Messages
                  max_tokens = req.MaxTokens
                  temperature = req.Temperature
                  stream = Some true }

            let uri = Uri(baseUri, "/v1/chat/completions")
            let content = new StringContent(JsonSerializer.Serialize(dto, jsonOptions), System.Text.Encoding.UTF8, "application/json")
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
                    // OpenAI streams data: prefix
                    let dataLine = if line.StartsWith("data: ") then line.Substring(6) else line
                    if dataLine = "[DONE]" then
                        isDone <- true
                    elif not (String.IsNullOrWhiteSpace(dataLine)) then
                        try
                            let chunk = JsonSerializer.Deserialize<OpenAiStreamResponseDto>(dataLine, jsonOptions)
                            if not (isNull (box chunk)) && not (isNull chunk.choices) && chunk.choices.Length > 0 then
                                let choice = chunk.choices.[0]
                                if not (isNull (box choice.delta)) && not (isNull choice.delta.content) then
                                    let token = choice.delta.content
                                    fullText <- fullText + token
                                    onToken token
                                if not (isNull choice.finish_reason) then
                                    finishReason <- choice.finish_reason
                                    isDone <- true
                        with _ -> ()

            return
                { Text = fullText
                  FinishReason = Some finishReason
                  Usage = None
                  Raw = None }
        }
