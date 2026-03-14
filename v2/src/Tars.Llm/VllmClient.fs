namespace Tars.Llm

// vLLM client for high-performance LLM inference.
// vLLM uses PagedAttention for efficient memory management and supports
// the OpenAI-compatible API format.
//
// vLLM is typically used for production deployments requiring high throughput.
// See: https://github.com/vllm-project/vllm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Client module for interacting with vLLM servers via the OpenAI-compatible API.
/// </summary>
module VllmClient =

    open Tars.Llm

    /// <summary>DTO for OpenAI-compatible message.</summary>
    [<CLIMutable>]
    type OpenAiMessageDto = { role: string; content: string }

    /// <summary>DTO for OpenAI-compatible request.</summary>
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

    /// <summary>DTO for OpenAI-compatible response.</summary>
    [<CLIMutable>]
    type OpenAiResponseDto =
        { id: string
          choices: OpenAiChoiceDto[] }

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

    /// <summary>
    /// Sends a chat completion request to a vLLM server.
    /// </summary>
    /// <param name="http">The HttpClient to use for the request.</param>
    /// <param name="baseUri">The base URI of the vLLM server.</param>
    /// <param name="model">The model name.</param>
    /// <param name="req">The LLM request containing messages and parameters.</param>
    /// <returns>The LLM response with generated text.</returns>
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
                    return
                        { Text = c.message.content
                          FinishReason = Some c.finish_reason
                          Usage = None
                          Raw = Some raw }
        }
