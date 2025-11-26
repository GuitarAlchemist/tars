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

module OpenAiCompatibleClient =

    open Tars.Llm

    [<CLIMutable>]
    type OpenAiMessageDto = { role: string; content: string }

    [<CLIMutable>]
    type OpenAiRequestDto =
        { model: string
          messages: OpenAiMessageDto[]
          max_tokens: int option
          temperature: float option
          stream: bool option }

    [<CLIMutable>]
    type OpenAiChoiceMessageDto = { role: string; content: string }

    [<CLIMutable>]
    type OpenAiChoiceDto =
        { index: int
          message: OpenAiChoiceMessageDto
          finish_reason: string }

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
                          Raw = Some raw }
                | Some c ->
                    return
                        { Text = c.message.content
                          FinishReason = Some c.finish_reason
                          Raw = Some raw }
        }
