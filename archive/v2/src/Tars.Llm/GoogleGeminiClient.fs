namespace Tars.Llm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Client for Google Gemini API (v1beta).
/// </summary>
module GoogleGeminiClient =

    open Tars.Llm

    // --- DTOs for Gemini API ---

    [<CLIMutable>]
    type GeminiPartDto = { text: string }

    [<CLIMutable>]
    type GeminiContentDto =
        { role: string; parts: GeminiPartDto[] }

    [<CLIMutable>]
    type GeminiRequestDto =
        { contents: GeminiContentDto[]
          generationConfig: GeminiGenerationConfigDto option }

    and [<CLIMutable>] GeminiGenerationConfigDto =
        { temperature: float option
          maxOutputTokens: int option
          responseMimeType: string option
          responseSchema: obj option }

    [<CLIMutable>]
    type GeminiCandidateDto =
        { content: GeminiContentDto
          finishReason: string
          index: int }

    [<CLIMutable>]
    type GeminiUsageMetadataDto =
        { promptTokenCount: int
          candidatesTokenCount: int
          totalTokenCount: int }

    [<CLIMutable>]
    type GeminiResponseDto =
        { candidates: GeminiCandidateDto[]
          usageMetadata: GeminiUsageMetadataDto option }

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toGeminiRole =
        function
        | Role.System -> "user" // Gemini (v1) treats system prompts as user, or specific "system" instructions in 1.5
        | Role.User -> "user"
        | Role.Assistant -> "model"

    let private toGeminiContent (msgs: LlmMessage list) =
        msgs
        |> List.map (fun m ->
            { role = toGeminiRole m.Role
              parts = [| { text = m.Content } |] }
            : GeminiContentDto)
        |> List.toArray

    /// <summary>
    /// Sends a generateContent request to Google Gemini API.
    /// </summary>
    let generateContentAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        : Task<LlmResponse> =
        task {
            // endpoint: /v1beta/models/{model}:generateContent
            let endpoint = $"/v1beta/models/{model}:generateContent"

            // Add API key as query param if present choice (or header)
            // Google recommends x-goog-api-key header or key query param.
            // We'll use query param for simplicity with Uri builder, or header.
            // Let's use header for cleaner logs.

            let uri = Uri(baseUri, endpoint)

            let genConfig =
                { temperature = req.Temperature
                  maxOutputTokens = req.MaxTokens
                  responseMimeType =
                    match req.ResponseFormat with
                    | Some ResponseFormat.Json -> Some "application/json"
                    | Some(ResponseFormat.Constrained _) -> Some "application/json"
                    | Some ResponseFormat.Text -> None
                    | None -> if req.JsonMode then Some "application/json" else None
                  responseSchema =
                    match req.ResponseFormat with
                    | Some(ResponseFormat.Constrained(Grammar.JsonSchema schema)) ->
                        try
                            Some(JsonSerializer.Deserialize<obj>(schema))
                        with _ ->
                            None
                    | _ -> None }

            let dto: GeminiRequestDto =
                { contents = toGeminiContent req.Messages
                  generationConfig = Some genConfig }

            let content = JsonContent.Create(dto, options = jsonOptions)
            use requestMessage = new HttpRequestMessage(HttpMethod.Post, uri, Content = content)

            match apiKey with
            | Some key when not (String.IsNullOrWhiteSpace(key)) -> requestMessage.Headers.Add("x-goog-api-key", key)
            | _ -> ()

            use! resp = http.SendAsync(requestMessage)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<GeminiResponseDto>(raw, jsonOptions)

            if isNull (box parsed) || isNull parsed.candidates || parsed.candidates.Length = 0 then
                return
                    { Text = ""
                      FinishReason = Some "no_candidates"
                      Usage = None
                      Raw = Some raw }
            else
                let candidate = parsed.candidates.[0]

                let text =
                    if isNull candidate.content.parts || candidate.content.parts.Length = 0 then
                        ""
                    else
                        candidate.content.parts.[0].text

                let usage =
                    match parsed.usageMetadata with
                    | Some u ->
                        Some
                            { PromptTokens = u.promptTokenCount
                              CompletionTokens = u.candidatesTokenCount
                              TotalTokens = u.totalTokenCount }
                    | None -> None

                return
                    { Text = text
                      FinishReason = Some candidate.finishReason
                      Usage = usage
                      Raw = Some raw }
        }
