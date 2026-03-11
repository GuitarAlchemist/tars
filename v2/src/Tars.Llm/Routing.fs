/// <summary>
/// LLM request routing module.
/// Routes requests to appropriate backends based on model hints and configuration.
/// </summary>
module Tars.Llm.Routing

open System
open Tars.Llm

/// <summary>
/// Configuration for LLM routing.
/// Defines endpoints and default models for each backend.
/// </summary>
type RoutingConfig =
    { OllamaBaseUri: Uri
      VllmBaseUri: Uri
      OpenAIBaseUri: Uri
      GoogleGeminiBaseUri: Uri
      AnthropicBaseUri: Uri
      DockerModelRunnerBaseUri: Uri option
      LlamaCppBaseUri: Uri option
      DefaultOllamaModel: string
      DefaultVllmModel: string
      DefaultOpenAIModel: string
      DefaultGoogleGeminiModel: string
      DefaultAnthropicModel: string
      DefaultDockerModelRunnerModel: string option
      DefaultLlamaCppModel: string option
      DefaultEmbeddingModel: string
      ReasoningModel: string option
      CodingModel: string option
      FastModel: string option
      OllamaKey: string option
      VllmKey: string option
      OpenAIKey: string option
      GoogleGeminiKey: string option
      AnthropicKey: string option
      DockerModelRunnerKey: string option
      LlamaCppKey: string option
      LlamaSharpModelPath: string option
      DefaultContextWindow: int option
      DefaultTemperature: float option
      PreferredProvider: string }

    static member Default =
        { OllamaBaseUri = Uri("http://localhost:11434")
          VllmBaseUri = Uri("http://localhost:8000")
          OpenAIBaseUri = Uri("https://api.openai.com")
          GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com")
          AnthropicBaseUri = Uri("https://api.anthropic.com")
          DockerModelRunnerBaseUri = None
          LlamaCppBaseUri = None
          DefaultOllamaModel = "llama3"
          DefaultVllmModel = "llama3"
          DefaultOpenAIModel = "gpt-4"
          DefaultGoogleGeminiModel = "gemini-1.5-flash"
          DefaultAnthropicModel = "claude-3-5-sonnet-latest"
          DefaultDockerModelRunnerModel = None
          DefaultLlamaCppModel = None
          DefaultEmbeddingModel = "nomic-embed-text"
          ReasoningModel = None
          CodingModel = None
          FastModel = None
          OllamaKey = None
          VllmKey = None
          OpenAIKey = None
          GoogleGeminiKey = None
          AnthropicKey = None
          DockerModelRunnerKey = None
          LlamaCppKey = None
          LlamaSharpModelPath = None
          DefaultContextWindow = None
          DefaultTemperature = None
          PreferredProvider = "Ollama" }

/// <summary>
/// Result of a routing decision.
/// </summary>
type RoutedBackend =
    { Backend: LlmBackend
      Endpoint: Uri
      ApiKey: string option }

/// <summary>
/// Routes an LLM request to the appropriate backend based on model hints.
/// </summary>
/// <param name="cfg">Routing configuration.</param>
/// <param name="req">The LLM request to route.</param>
/// <returns>The routed backend with endpoint.</returns>
let chooseBackend (cfg: RoutingConfig) (req: LlmRequest) : RoutedBackend =
    let llamaSharpFallback () =
        match cfg.LlamaSharpModelPath with
        | Some modelPath ->
            Some
                { Backend = LlamaSharp modelPath
                  Endpoint = Uri("local://llamasharp")
                  ApiKey = None }
        | None -> None

    let orLlamaSharp fallback =
        match llamaSharpFallback () with
        | Some routed -> routed
        | None -> fallback

    match req.Model with
    | Some model ->
        // If model is explicitly set, try to guess backend or default to Ollama
        if model.Contains("gpt", StringComparison.OrdinalIgnoreCase) then
            { Backend = OpenAI model
              Endpoint = cfg.OpenAIBaseUri
              ApiKey = cfg.OpenAIKey }
        elif model.Contains("claude", StringComparison.OrdinalIgnoreCase) then
            { Backend = Anthropic model
              Endpoint = cfg.AnthropicBaseUri
              ApiKey = cfg.AnthropicKey }
        elif model.Contains("gemini", StringComparison.OrdinalIgnoreCase) then
            { Backend = GoogleGemini model
              Endpoint = cfg.GoogleGeminiBaseUri
              ApiKey = cfg.GoogleGeminiKey }
        else
            // Check if it matches configured llama.cpp model
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel, cfg.LlamaSharpModelPath with
            | Some llamaUri, Some llamaModel, _ when
                String.Equals(model, llamaModel, StringComparison.OrdinalIgnoreCase)
                || model.Contains("magistral", StringComparison.OrdinalIgnoreCase)
                ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = llamaUri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                // Default to Ollama for local models
                orLlamaSharp
                    { Backend = Ollama model
                      Endpoint = cfg.OllamaBaseUri
                      ApiKey = cfg.OllamaKey }
    | None ->
        let hint = req.ModelHint |> Option.defaultValue ""

        // Helper to route to preferred local backend
        let localRoute model =
            if cfg.PreferredProvider = "Ollama" then
                orLlamaSharp
                    { Backend = Ollama model
                      Endpoint = cfg.OllamaBaseUri
                      ApiKey = cfg.OllamaKey }
            else if
                // If VLLM is configured and not default, use it, otherwise fallback to Ollama
                cfg.VllmBaseUri.Host <> "localhost" || cfg.VllmBaseUri.Port <> 8000
            then
                { Backend = Vllm model
                  Endpoint = cfg.VllmBaseUri
                  ApiKey = cfg.VllmKey }
            else
                orLlamaSharp
                    { Backend = Ollama model
                      Endpoint = cfg.OllamaBaseUri
                      ApiKey = cfg.OllamaKey }

        match hint with
        | h when h.Contains("code", StringComparison.OrdinalIgnoreCase) ->
            localRoute (cfg.CodingModel |> Option.defaultValue cfg.DefaultOllamaModel)

        | h when h.Contains("cheap", StringComparison.OrdinalIgnoreCase) -> localRoute cfg.DefaultOllamaModel

        | h when
            h.Contains("reason", StringComparison.OrdinalIgnoreCase)
            || h.Contains("analysis", StringComparison.OrdinalIgnoreCase)
            || h.Contains("think", StringComparison.OrdinalIgnoreCase)
            || h.Contains("math", StringComparison.OrdinalIgnoreCase)
            || h.Contains("complex", StringComparison.OrdinalIgnoreCase)
            || h.Contains("step", StringComparison.OrdinalIgnoreCase)
            || h.Contains("smart", StringComparison.OrdinalIgnoreCase)
            ->
            localRoute (cfg.ReasoningModel |> Option.defaultValue cfg.DefaultOllamaModel)

        | h when h.Contains("docker", StringComparison.OrdinalIgnoreCase) ->
            match cfg.DockerModelRunnerBaseUri, cfg.DefaultDockerModelRunnerModel with
            | Some uri, Some model ->
                { Backend = DockerModelRunner model
                  Endpoint = uri
                  ApiKey = cfg.DockerModelRunnerKey }
            | _ -> localRoute cfg.DefaultOllamaModel

        | h when
            h.Contains("llamacpp", StringComparison.OrdinalIgnoreCase)
            || h.Contains("perf", StringComparison.OrdinalIgnoreCase)
            || h.Contains("gguf", StringComparison.OrdinalIgnoreCase)
            ->
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ -> localRoute cfg.DefaultOllamaModel

        | h when
            h.Contains("fast", StringComparison.OrdinalIgnoreCase)
            || h.Contains("quick", StringComparison.OrdinalIgnoreCase)
            ->
            localRoute (cfg.FastModel |> Option.defaultValue cfg.DefaultOllamaModel)

        | _ -> localRoute cfg.DefaultOllamaModel

module RoutingConfig =
    /// <summary>
    /// Creates a RoutingConfig from a TarsConfig.
    /// maps standard TARS configuration to LLM routing settings.
    /// </summary>
    let fromTarsConfig (tarsCfg: Tars.Core.TarsConfig) : RoutingConfig =
        let baseUri =
            tarsCfg.Llm.BaseUrl |> Option.defaultValue "http://localhost:11434" |> Uri

        let llamaCppUri = tarsCfg.Llm.LlamaCppUrl |> Option.map Uri

        { OllamaBaseUri = baseUri
          VllmBaseUri = Uri("http://localhost:8000/")
          OpenAIBaseUri = Uri("https://api.openai.com/")
          GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
          AnthropicBaseUri = Uri("https://api.anthropic.com/")
          DockerModelRunnerBaseUri = None
          LlamaCppBaseUri = llamaCppUri

          DefaultOllamaModel = tarsCfg.Llm.Model
          DefaultVllmModel = "llama3"
          DefaultOpenAIModel = "gpt-4o"
          DefaultGoogleGeminiModel = "gemini-1.5-flash"
          DefaultAnthropicModel = "claude-3-5-sonnet-latest"
          DefaultDockerModelRunnerModel = None
          DefaultLlamaCppModel = if llamaCppUri.IsSome then Some tarsCfg.Llm.Model else None
          DefaultEmbeddingModel = tarsCfg.Llm.EmbeddingModel

          ReasoningModel = tarsCfg.Llm.ReasoningModel
          CodingModel = tarsCfg.Llm.CodingModel
          FastModel = tarsCfg.Llm.FastModel

          OllamaKey = tarsCfg.Llm.ApiKey
          VllmKey = tarsCfg.Llm.ApiKey
          OpenAIKey = tarsCfg.Llm.ApiKey
          GoogleGeminiKey = tarsCfg.Llm.ApiKey
          AnthropicKey = tarsCfg.Llm.ApiKey
          DockerModelRunnerKey = tarsCfg.Llm.ApiKey
          LlamaCppKey = tarsCfg.Llm.ApiKey
          LlamaSharpModelPath = tarsCfg.Llm.LlamaSharpModelPath
          DefaultContextWindow = Some tarsCfg.Llm.ContextWindow
          DefaultTemperature = Some tarsCfg.Llm.Temperature
          PreferredProvider = tarsCfg.Llm.Provider }
