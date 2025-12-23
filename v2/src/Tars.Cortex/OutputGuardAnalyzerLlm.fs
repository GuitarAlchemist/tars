namespace Tars.Cortex

open System
open System.Text.Json
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// LLM-backed output guard analyzer. Expects the model to return JSON:
/// { "risk": 0.0-1.0, "action": "accept|ask|retry|reject|fallback", "reasons": ["..."] }
type LlmOutputGuardAnalyzer(llm: ILlmServiceFunctional, ?modelHint: string, ?temperature: float, ?maxTokens: int) =

    let parseResult (text: string) =
        try
            use doc = JsonDocument.Parse(text)
            let root = doc.RootElement

            let risk =
                match root.TryGetProperty("risk") with
                | true, v when v.ValueKind = JsonValueKind.Number ->
                    match v.TryGetDouble() with
                    | true, d -> Math.Max(0.0, Math.Min(1.0, d))
                    | _ -> 0.5
                | _ -> 0.5

            let action =
                match root.TryGetProperty("action") with
                | true, v when v.ValueKind = JsonValueKind.String ->
                    match v.GetString().ToLowerInvariant() with
                    | "reject" -> GuardAction.Reject "LLM analysis flagged risk"
                    | "retry" -> GuardAction.RetryWithHint "LLM analysis suggests retry with tighter constraints"
                    | "ask" -> GuardAction.AskForEvidence "Provide citations/tool outputs to proceed"
                    | "fallback" -> GuardAction.Fallback "Fallback to minimal safe response"
                    | _ -> GuardAction.Accept
                | _ -> GuardAction.Accept

            let reasons =
                match root.TryGetProperty("reasons") with
                | true, v when v.ValueKind = JsonValueKind.Array ->
                    v.EnumerateArray()
                    |> Seq.choose (fun e ->
                        if e.ValueKind = JsonValueKind.String then
                            Some(e.GetString())
                        else
                            None)
                    |> Seq.toList
                | _ -> []

            Some
                { Risk = risk
                  Action = action
                  Messages = reasons }
        with _ ->
            None

    let buildPrompt (input: GuardInput) =
        let fields =
            match input.ExpectedJsonFields with
            | None -> "[]"
            | Some xs -> JsonSerializer.Serialize(xs)

        let citations =
            match input.Citations with
            | None -> "[]"
            | Some xs -> JsonSerializer.Serialize(xs)

        $"""You are an output guard. Classify the response for hallucination, cargo-cult code, or fabrication risk.
Respond ONLY with JSON: {{"risk": <0-1>, "action": "accept|ask|retry|reject|fallback", "reasons": ["..."]}}
Response: ```{input.ResponseText}```
ExpectedJsonFields: {fields}
RequireCitations: {input.RequireCitations}
CitationsProvided: {citations}
AllowExtraFields: {input.AllowExtraFields}
GrammarProvided: {input.Grammar.IsSome}
"""

    interface IOutputGuardAnalyzer with
        member _.Analyze(input: GuardInput) : Async<GuardResult option> =
            async {
                try
                    let prompt = buildPrompt input

                    let req: LlmRequest =
                        { ModelHint = modelHint
                          Model = None
                          SystemPrompt = None
                          MaxTokens = maxTokens
                          Temperature = temperature
                          Stop = []
                          Messages =
                            [ { Role = Role.System
                                Content = "You are a strict output guard. Return only JSON with risk/action/reasons." }
                              { Role = Role.User; Content = prompt } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = None
                          Stream = false
                          JsonMode = true
                          Seed = None }

                    let! res = llm.CompleteAsync req

                    match res with
                    | Microsoft.FSharp.Core.Result.Ok response ->
                        match parseResult response.Text with
                        | Some r -> return Some r
                        | None -> return None
                    | Microsoft.FSharp.Core.Result.Error _ -> return None
                with _ ->
                    return None
            }

/// Factory helpers to build LLM-based analyzers (e.g., for Ollama)
module OutputGuardAnalyzerFactory =
    open System.Net.Http
    open Tars.Llm.Routing

    let private defaultUri fallback (value: string option) =
        match value with
        | Some v when not (String.IsNullOrWhiteSpace v) -> Uri(v)
        | _ -> Uri(fallback)

    /// Create an analyzer using an Ollama endpoint and model.
    let createOllamaAnalyzer (baseUri: Uri) (model: string) : IOutputGuardAnalyzer =
        let routingCfg: RoutingConfig =
            { OllamaBaseUri = baseUri
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = model
              DefaultVllmModel = model
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text"

              OllamaKey = None
              VllmKey = None
              OpenAIKey = None
              GoogleGeminiKey = None
              AnthropicKey = None
              DockerModelRunnerBaseUri = None
              LlamaCppBaseUri = None
              DefaultDockerModelRunnerModel = None
              DefaultLlamaCppModel = None
              DockerModelRunnerKey = None
              LlamaCppKey = None }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        let httpClient = new HttpClient(Timeout = TimeSpan.FromSeconds(30.0))
        let llm = DefaultLlmService(httpClient, svcCfg) :> ILlmServiceFunctional
        LlmOutputGuardAnalyzer(llm, modelHint = "cheap", temperature = 0.2) :> IOutputGuardAnalyzer

    /// Create an analyzer using environment variables (OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL).
    let createFromEnv () : IOutputGuardAnalyzer =
        let baseUrl = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL") |> Option.ofObj
        // default to a common coding-capable model if unset
        let model =
            Environment.GetEnvironmentVariable("DEFAULT_OLLAMA_MODEL")
            |> Option.ofObj
            |> Option.defaultValue "qwen2.5-coder:latest"

        let uri = defaultUri "http://localhost:11434/" baseUrl
        createOllamaAnalyzer uri model
