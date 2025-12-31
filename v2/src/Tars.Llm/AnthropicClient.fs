namespace Tars.Llm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Client for Anthropic Messages API.
/// </summary>
module AnthropicClient =

    open Tars.Llm

    [<Literal>]
    let private ApiVersion = "2023-06-01"

    [<CLIMutable>]
    type AnthropicContentBlockDto = { ``type``: string; text: string }

    [<CLIMutable>]
    type AnthropicMessageDto = { role: string; content: AnthropicContentBlockDto[] }

    [<CLIMutable>]
    type AnthropicRequestDto =
        { model: string
          max_tokens: int
          messages: AnthropicMessageDto[]
          system: string option
          temperature: float option
          stop_sequences: string[] option
          stream: bool option }

    [<CLIMutable>]
    type AnthropicUsageDto =
        { input_tokens: int
          output_tokens: int }

    [<CLIMutable>]
    type AnthropicResponseDto =
        { content: AnthropicContentBlockDto[]
          stop_reason: string option
          usage: AnthropicUsageDto option }

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toAnthropicRole =
        function
        | Role.User -> "user"
        | Role.Assistant -> "assistant"
        | Role.System -> "user"

    let private buildSystemPrompt (req: LlmRequest) =
        let systemMessages =
            req.Messages
            |> List.choose (fun m ->
                if m.Role = Role.System then Some m.Content else None)

        let baseSystem =
            match req.SystemPrompt, systemMessages with
            | None, [] -> None
            | Some p, [] -> Some p
            | None, msgs -> Some (String.Join("\n", msgs))
            | Some p, msgs -> Some (String.Join("\n", p :: msgs))

        let jsonHint =
            match req.ResponseFormat with
            | Some ResponseFormat.Json -> Some "Respond with valid JSON only."
            | Some(ResponseFormat.Constrained(Grammar.JsonSchema _)) -> Some "Respond with valid JSON only."
            | Some(ResponseFormat.Constrained(Grammar.Regex pattern)) -> Some $"Respond matching regex: {pattern}"
            | Some ResponseFormat.Text -> None
            | None -> if req.JsonMode then Some "Respond with valid JSON only." else None

        match baseSystem, jsonHint with
        | None, None -> None
        | Some s, None -> Some s
        | None, Some h -> Some h
        | Some s, Some h -> Some (s + "\n" + h)

    let private toAnthropicMessages (req: LlmRequest) =
        req.Messages
        |> List.filter (fun m -> m.Role <> Role.System)
        |> List.map (fun m ->
            { role = toAnthropicRole m.Role
              content = [| { ``type`` = "text"; text = m.Content } |] }
            : AnthropicMessageDto)
        |> List.toArray

    let private emitChunks (text: string) (chunkSize: int) (onToken: string -> unit) =
        if not (String.IsNullOrEmpty(text)) then
            let mutable i = 0
            while i < text.Length do
                let len = Math.Min(chunkSize, text.Length - i)
                onToken (text.Substring(i, len))
                i <- i + len

    /// <summary>
    /// Sends a Messages API request to Anthropic.
    /// </summary>
    let sendMessageAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        : Task<LlmResponse> =
        task {
            let maxTokens = req.MaxTokens |> Option.defaultValue 1024
            let uri = Uri(baseUri, "/v1/messages")

            let dto: AnthropicRequestDto =
                { model = model
                  max_tokens = maxTokens
                  messages = toAnthropicMessages req
                  system = buildSystemPrompt req
                  temperature = req.Temperature
                  stop_sequences =
                    if req.Stop.IsEmpty then None
                    else Some (req.Stop |> List.toArray)
                  stream = Some false }

            let content = JsonContent.Create(dto, options = jsonOptions)
            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)

            requestMessage.Headers.Add("anthropic-version", ApiVersion)

            match apiKey with
            | Some key when not (String.IsNullOrWhiteSpace(key)) ->
                requestMessage.Headers.Add("x-api-key", key)
            | _ -> ()

            use! resp = http.SendAsync(requestMessage)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<AnthropicResponseDto>(raw, jsonOptions)

            if isNull (box parsed) || isNull parsed.content || parsed.content.Length = 0 then
                return
                    { Text = ""
                      FinishReason = Some "no_content"
                      Usage = None
                      Raw = Some raw }
            else
                let text =
                    parsed.content
                    |> Array.choose (fun part ->
                        if part.``type`` = "text" then Some part.text else None)
                    |> String.concat ""

                let usage =
                    match parsed.usage with
                    | Some u ->
                        Some
                            { PromptTokens = u.input_tokens
                              CompletionTokens = u.output_tokens
                              TotalTokens = u.input_tokens + u.output_tokens }
                    | None -> None

                return
                    { Text = text
                      FinishReason = parsed.stop_reason
                      Usage = usage
                      Raw = Some raw }
        }

    /// <summary>
    /// Streams content by chunking a non-streamed response.
    /// </summary>
    let sendMessageStreamAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        (onToken: string -> unit)
        : Task<LlmResponse> =
        task {
            let! response = sendMessageAsync http baseUri model apiKey req
            emitChunks response.Text 64 onToken
            return response
        }
