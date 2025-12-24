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
    {
        /// <summary>Ollama server URI (e.g., http://localhost:11434/).</summary>
        OllamaBaseUri: Uri
        /// <summary>vLLM server URI (e.g., http://localhost:8000/).</summary>
        VllmBaseUri: Uri
        /// <summary>OpenAI API URI (e.g., https://api.openai.com/).</summary>
        OpenAIBaseUri: Uri
        /// <summary>Google Gemini API URI.</summary>
        GoogleGeminiBaseUri: Uri
        /// <summary>Anthropic API URI.</summary>
        AnthropicBaseUri: Uri
        /// <summary>Docker Model Runner URI (e.g., http://localhost:12434/v1). Optional.</summary>
        DockerModelRunnerBaseUri: Uri option
        /// <summary>llama.cpp server URI (e.g., http://localhost:8080). Optional.</summary>
        LlamaCppBaseUri: Uri option

        /// <summary>Default model for Ollama (e.g., "llama3.2").</summary>
        DefaultOllamaModel: string
        /// <summary>Default model for vLLM.</summary>
        DefaultVllmModel: string
        /// <summary>Default model for OpenAI (e.g., "gpt-4").</summary>
        DefaultOpenAIModel: string
        /// <summary>Default model for Google Gemini.</summary>
        DefaultGoogleGeminiModel: string
        /// <summary>Default model for Anthropic.</summary>
        DefaultAnthropicModel: string
        /// <summary>Default model for Docker Model Runner. Optional.</summary>
        DefaultDockerModelRunnerModel: string option
        /// <summary>Default model/GGUF file for llama.cpp. Optional.</summary>
        DefaultLlamaCppModel: string option
        /// <summary>Default embedding model.</summary>
        DefaultEmbeddingModel: string

        // --- API Keys ---
        OllamaKey: string option
        VllmKey: string option
        OpenAIKey: string option
        GoogleGeminiKey: string option
        AnthropicKey: string option
        /// <summary>Docker Model Runner API key (usually not needed).</summary>
        DockerModelRunnerKey: string option
        /// <summary>llama.cpp API key (usually not needed).</summary>
        LlamaCppKey: string option
    }

    /// <summary>
    /// Default Configuration
    /// </summary>
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
          OllamaKey = None
          VllmKey = None
          OpenAIKey = None
          GoogleGeminiKey = None
          AnthropicKey = None
          DockerModelRunnerKey = None
          LlamaCppKey = None }

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
/// <remarks>
/// Routing logic:
/// - "code" or "cheap" hints → Ollama (local, fast)
/// - "reason" or "analysis" hints → vLLM (high performance)
/// - Model name hints (llama, qwen, mistral) → Ollama
/// - Default → Ollama (safest for local dev)
/// </remarks>
let chooseBackend (cfg: RoutingConfig) (req: LlmRequest) : RoutedBackend =
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
            match cfg.DefaultLlamaCppModel, cfg.LlamaCppBaseUri with
            | Some llamaModel, Some llamaUri when String.Equals(model, llamaModel, StringComparison.OrdinalIgnoreCase) ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = llamaUri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                // Default to Ollama for local models
                { Backend = Ollama model
                  Endpoint = cfg.OllamaBaseUri
                  ApiKey = cfg.OllamaKey }
    | None ->
        match req.ModelHint |> Option.defaultValue "" with
        | hint when hint.Contains("code", StringComparison.OrdinalIgnoreCase) ->
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri
              ApiKey = cfg.OllamaKey }
        | hint when hint.Contains("cheap", StringComparison.OrdinalIgnoreCase) ->
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri
              ApiKey = cfg.OllamaKey }
        | hint when
            hint.Contains("reason", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("analysis", StringComparison.OrdinalIgnoreCase)
            ->
            // Prefer llama.cpp for reasoning (faster local inference)
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                { Backend = Vllm cfg.DefaultVllmModel
                  Endpoint = cfg.VllmBaseUri
                  ApiKey = cfg.VllmKey }

        | hint when
            (hint.Contains("llama", StringComparison.OrdinalIgnoreCase)
             && not (hint.Contains("llamacpp", StringComparison.OrdinalIgnoreCase)))
            || hint.Contains("qwen", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("mistral", StringComparison.OrdinalIgnoreCase)
            ->
            // detailed check for LlamaCpp preference
            match cfg.DefaultLlamaCppModel, cfg.LlamaCppBaseUri with
            | Some llamaModel, Some llamaUri when
                String.Equals(hint, llamaModel, StringComparison.OrdinalIgnoreCase)
                || llamaModel.Contains(hint, StringComparison.OrdinalIgnoreCase)
                ->
                { Backend = LlamaCpp(llamaModel, Some LlamaCppConfig.Default)
                  Endpoint = llamaUri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                { Backend = Ollama cfg.DefaultOllamaModel
                  Endpoint = cfg.OllamaBaseUri
                  ApiKey = cfg.OllamaKey }
        // Docker Model Runner hints
        | hint when hint.Contains("docker", StringComparison.OrdinalIgnoreCase) ->
            match cfg.DockerModelRunnerBaseUri, cfg.DefaultDockerModelRunnerModel with
            | Some uri, Some model ->
                { Backend = DockerModelRunner model
                  Endpoint = uri
                  ApiKey = cfg.DockerModelRunnerKey }
            | _ ->
                // Fallback to Ollama if Docker Model Runner not configured
                { Backend = Ollama cfg.DefaultOllamaModel
                  Endpoint = cfg.OllamaBaseUri
                  ApiKey = cfg.OllamaKey }
        // llama.cpp hints for performance-critical scenarios
        | hint when
            hint.Contains("llamacpp", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("perf", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("gguf", StringComparison.OrdinalIgnoreCase)
            ->
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                // Fallback to Ollama if llama.cpp not configured
                { Backend = Ollama cfg.DefaultOllamaModel
                  Endpoint = cfg.OllamaBaseUri
                  ApiKey = cfg.OllamaKey }
        // Smart/tool hints for function calling - prefer configured defaults
        | hint when
            hint.Contains("smart", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("tool", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("function", StringComparison.OrdinalIgnoreCase)
            ->
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri
              ApiKey = cfg.OllamaKey }
        // Thinking/reasoning hints - prefer configured reasoning defaults
        | hint when
            hint.Contains("think", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("reason", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("math", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("complex", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("step", StringComparison.OrdinalIgnoreCase)
            ->
            // Prefer llama.cpp if available, else use configured defaults
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                { Backend = Vllm cfg.DefaultVllmModel
                  Endpoint = cfg.VllmBaseUri
                  ApiKey = cfg.VllmKey }

        // Fast/quick hints - use configured defaults
        | hint when
            hint.Contains("fast", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("quick", StringComparison.OrdinalIgnoreCase)
            ->
            // llama.cpp is fastest, else use configured defaults
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                { Backend = Ollama cfg.DefaultOllamaModel
                  Endpoint = cfg.OllamaBaseUri
                  ApiKey = cfg.OllamaKey }

        | _ ->
            // Default: prefer llama.cpp (1.8x faster) if available, else Ollama
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                { Backend = Ollama cfg.DefaultOllamaModel
                  Endpoint = cfg.OllamaBaseUri
                  ApiKey = cfg.OllamaKey }

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

          OllamaKey = tarsCfg.Llm.ApiKey
          VllmKey = None
          OpenAIKey = None
          GoogleGeminiKey = None
          AnthropicKey = None
          DockerModelRunnerKey = None
          LlamaCppKey = None }
