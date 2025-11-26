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
          isDone: bool }

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

    let sendChatAsync (http: HttpClient) (baseUri: Uri) (model: string) (req: LlmRequest) : Task<LlmResponse> =
        task {
            let dto: OllamaRequestDto =
                { model = model
                  messages = toOllamaMessages req.Messages
                  stream = false
                  temperature = req.Temperature }

            let uri = Uri(baseUri, "/api/chat")
            use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<OllamaResponseDto>(raw, jsonOptions)

            if isNull (box parsed) then
                return
                    { Text = ""
                      FinishReason = Some "parse_error"
                      Raw = Some raw }
            else
                return
                    { Text = parsed.message.content
                      FinishReason = Some(if parsed.isDone then "done" else "unknown")
                      Raw = Some raw }
        }
