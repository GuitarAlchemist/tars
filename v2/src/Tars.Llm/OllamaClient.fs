/// <summary>
/// Ollama LLM client for local inference.
/// Provides chat completions, embeddings, and model listing via the Ollama API.
/// </summary>
/// <remarks>
/// Ollama runs locally and supports various open-source models like Llama, Mistral, etc.
/// Default endpoint: http://localhost:11434
/// </remarks>
namespace Tars.Llm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Client module for interacting with the Ollama API.
/// All functions are stateless and take HttpClient as a parameter.
/// </summary>
module OllamaClient =

    open Tars.Llm

    /// <summary>DTO for Ollama chat message.</summary>
    [<CLIMutable>]
    type OllamaMessageDto = { role: string; content: string }

    type OllamaOptionsDto =
        { stop: string[] option
          seed: int option
          num_predict: int option
          temperature: float option }

    /// <summary>DTO for Ollama chat request.</summary>
    [<CLIMutable>]
    type OllamaRequestDto =
        { model: string
          messages: OllamaMessageDto[]
          stream: bool
          format: string option
          options: OllamaOptionsDto option }

    /// <summary>DTO for Ollama response message.</summary>
    [<CLIMutable>]
    type OllamaResponseMessageDto = { role: string; content: string }

    /// <summary>DTO for Ollama chat response.</summary>
    [<CLIMutable>]
    type OllamaResponseDto =
        { model: string
          message: OllamaResponseMessageDto
          [<JsonPropertyName("done")>]
          isDone: bool
          eval_count: int option
          prompt_eval_count: int option }

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toOllamaRole =
        function
        | Role.System -> "system"
        | Role.User -> "user"
        | Role.Assistant -> "assistant"

    let private toOllamaMessages (systemPrompt: string option) (msgs: LlmMessage list) : OllamaMessageDto[] =
        let systemMsg =
            match systemPrompt with
            | Some p -> [ ({ role = "system"; content = p }: OllamaMessageDto) ]
            | None -> []

        let otherMsgs =
            msgs
            |> List.map (fun m ->
                { role = toOllamaRole m.Role
                  content = m.Content }
                : OllamaMessageDto)

        (systemMsg @ otherMsgs) |> List.toArray

    /// <summary>DTO for Ollama embedding request.</summary>
    [<CLIMutable>]
    type OllamaEmbeddingRequestDto = { model: string; prompt: string }

    /// <summary>DTO for Ollama embedding response.</summary>
    [<CLIMutable>]
    type OllamaEmbeddingResponseDto = { embedding: float32[] }

    /// <summary>DTO for Ollama model info.</summary>
    [<CLIMutable>]
    type OllamaModelDto = { name: string; model: string }

    /// <summary>DTO for Ollama tags (model list) response.</summary>
    [<CLIMutable>]
    type OllamaTagsResponseDto = { models: OllamaModelDto list }

    /// <summary>
    /// Sends a chat completion request to Ollama.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the Ollama server (e.g., http://localhost:11434).</param>
    /// <param name="model">The model name (e.g., "llama3.2", "mistral").</param>
    /// <param name="req">The LLM request containing messages and parameters.</param>
    /// <returns>The LLM response with generated text and usage stats.</returns>
    let sendChatAsync (http: HttpClient) (baseUri: Uri) (model: string) (req: LlmRequest) : Task<LlmResponse> =
        task {
            let options =
                { stop =
                    if List.isEmpty req.Stop then
                        None
                    else
                        Some(List.toArray req.Stop)
                  seed = req.Seed
                  num_predict = req.MaxTokens
                  temperature = req.Temperature }

            let dto: OllamaRequestDto =
                { model = model
                  messages = toOllamaMessages req.SystemPrompt req.Messages
                  stream = false
                  format = if req.JsonMode then Some "json" else None
                  options = Some options }

            let uri = Uri(baseUri, "api/chat")
            use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<OllamaResponseDto>(raw, jsonOptions)

            if isNull (box parsed) then
                return
                    { Text = ""
                      FinishReason = Some "parse_error"
                      Usage = None
                      Raw = Some raw }
            else
                let usage =
                    match parsed.prompt_eval_count, parsed.eval_count with
                    | Some p, Some c ->
                        Some
                            { PromptTokens = p
                              CompletionTokens = c
                              TotalTokens = p + c }
                    | _ -> None

                return
                    { Text = parsed.message.content
                      FinishReason = Some(if parsed.isDone then "done" else "unknown")
                      Usage = usage
                      Raw = Some raw }
        }

    /// <summary>
    /// Generates embeddings for the given text using Ollama.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the Ollama server.</param>
    /// <param name="model">The embedding model name (e.g., "nomic-embed-text").</param>
    /// <param name="text">The text to embed.</param>
    /// <returns>The embedding vector as float32 array.</returns>
    let getEmbeddingsAsync (http: HttpClient) (baseUri: Uri) (model: string) (text: string) : Task<float32[]> =
        task {
            let dto: OllamaEmbeddingRequestDto = { model = model; prompt = text }
            let uri = Uri(baseUri, "api/embeddings")
            use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()

            let parsed =
                JsonSerializer.Deserialize<OllamaEmbeddingResponseDto>(raw, jsonOptions)

            if isNull (box parsed) || isNull parsed.embedding then
                return [||]
            else
                return parsed.embedding
        }

    /// <summary>
    /// Lists all available models on the Ollama server.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the Ollama server.</param>
    /// <returns>List of model names.</returns>
    let getTagsAsync (http: HttpClient) (baseUri: Uri) : Task<string list> =
        task {
            let uri = Uri(baseUri, "api/tags")
            let! resp = http.GetAsync(uri)
            resp.EnsureSuccessStatusCode() |> ignore
            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<OllamaTagsResponseDto>(raw, jsonOptions)

            if isNull (box parsed) || isNull (box parsed.models) then
                return []
            else
                return parsed.models |> List.map (fun m -> m.name)
        }

    /// <summary>
    /// Sends a streaming chat completion request to Ollama.
    /// Yields tokens as they are generated.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the Ollama server.</param>
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
            let options =
                { stop =
                    if List.isEmpty req.Stop then
                        None
                    else
                        Some(List.toArray req.Stop)
                  seed = req.Seed
                  num_predict = req.MaxTokens
                  temperature = req.Temperature }

            let dto: OllamaRequestDto =
                { model = model
                  messages = toOllamaMessages req.SystemPrompt req.Messages
                  stream = true
                  format = if req.JsonMode then Some "json" else None
                  options = Some options }

            let uri = Uri(baseUri, "api/chat")

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
            let mutable promptTokens = 0
            let mutable completionTokens = 0
            let mutable isDone = false

            while not isDone && not reader.EndOfStream do
                let! line = reader.ReadLineAsync()

                if not (String.IsNullOrWhiteSpace(line)) then
                    try
                        let chunk = JsonSerializer.Deserialize<OllamaResponseDto>(line, jsonOptions)

                        if not (isNull (box chunk)) then
                            if not (isNull (box chunk.message)) && not (isNull chunk.message.content) then
                                let token = chunk.message.content
                                fullText <- fullText + token
                                onToken token

                            if chunk.isDone then
                                isDone <- true

                                match chunk.prompt_eval_count with
                                | Some p -> promptTokens <- p
                                | None -> ()

                                match chunk.eval_count with
                                | Some c -> completionTokens <- c
                                | None -> ()
                    with _ ->
                        ()

            let usage =
                if promptTokens > 0 || completionTokens > 0 then
                    Some
                        { PromptTokens = promptTokens
                          CompletionTokens = completionTokens
                          TotalTokens = promptTokens + completionTokens }
                else
                    None

            return
                { Text = fullText
                  FinishReason = Some "done"
                  Usage = usage
                  Raw = None }
        }
