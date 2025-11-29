// Adapted from conversation: ChatGPT-What is vLLM.md
// Original Author: Stephane Pareilleux
// Date: 2025-11-26

namespace Tars.Llm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

module OllamaClient =

    open Tars.Llm

    [<CLIMutable>]
    type OllamaMessageDto = { role: string; content: string }

    [<CLIMutable>]
    type OllamaRequestDto =
        { model: string
          messages: OllamaMessageDto[]
          stream: bool
          temperature: float option }

    [<CLIMutable>]
    type OllamaResponseMessageDto = { role: string; content: string }

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

    let private toOllamaMessages (msgs: LlmMessage list) =
        msgs
        |> List.map (fun m ->
            { role = toOllamaRole m.Role
              content = m.Content }
            : OllamaMessageDto)
        |> List.toArray

    [<CLIMutable>]
    type OllamaEmbeddingRequestDto = { model: string; prompt: string }

    [<CLIMutable>]
    type OllamaEmbeddingResponseDto = { embedding: float32[] }

    [<CLIMutable>]
    type OllamaModelDto = { name: string; model: string }

    [<CLIMutable>]
    type OllamaTagsResponseDto = { models: OllamaModelDto list }

    let sendChatAsync (http: HttpClient) (baseUri: Uri) (model: string) (req: LlmRequest) : Task<LlmResponse> =
        task {
            let dto: OllamaRequestDto =
                { model = model
                  messages = toOllamaMessages req.Messages
                  stream = false
                  temperature = req.Temperature }

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
