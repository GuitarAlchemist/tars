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
        /// <summary>Default embedding model.</summary>
        DefaultEmbeddingModel: string
    }

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
              Endpoint = cfg.OpenAIBaseUri }
        elif model.Contains("claude", StringComparison.OrdinalIgnoreCase) then
            { Backend = Anthropic model
              Endpoint = cfg.AnthropicBaseUri }
        elif model.Contains("gemini", StringComparison.OrdinalIgnoreCase) then
            { Backend = GoogleGemini model
              Endpoint = cfg.GoogleGeminiBaseUri }
        else
            // Default to Ollama for local models
            { Backend = Ollama model
              Endpoint = cfg.OllamaBaseUri }
    | None ->
        match req.ModelHint |> Option.defaultValue "" with
        | hint when hint.Contains("code", StringComparison.OrdinalIgnoreCase) ->
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri }
        | hint when hint.Contains("cheap", StringComparison.OrdinalIgnoreCase) ->
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri }
        | hint when
            hint.Contains("reason", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("analysis", StringComparison.OrdinalIgnoreCase)
            ->
            { Backend = Vllm cfg.DefaultVllmModel
              Endpoint = cfg.VllmBaseUri }
        | hint when
            hint.Contains("llama", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("qwen", StringComparison.OrdinalIgnoreCase)
            || hint.Contains("mistral", StringComparison.OrdinalIgnoreCase)
            ->
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri }
        | _ ->
            // Default: send to Ollama (safest for local dev)
            { Backend = Ollama cfg.DefaultOllamaModel
              Endpoint = cfg.OllamaBaseUri }
