// Adapted from conversation: ChatGPT-What is vLLM.md
// Original Author: Stephane Pareilleux
// Date: 2025-11-26

module Tars.Llm.Routing

open System
open Tars.Llm

type RoutingConfig =
    { OllamaBaseUri: Uri // e.g. http://localhost:11434/
      VllmBaseUri: Uri // e.g. http://localhost:8000/ (OpenAI-compatible)
      OpenAIBaseUri: Uri // e.g. https://api.openai.com/
      GoogleGeminiBaseUri: Uri // e.g. https://generativelanguage.googleapis.com/
      AnthropicBaseUri: Uri // e.g. https://api.anthropic.com/
      DefaultOllamaModel: string
      DefaultVllmModel: string
      DefaultOpenAIModel: string
      DefaultGoogleGeminiModel: string
      DefaultAnthropicModel: string
      DefaultEmbeddingModel: string }

/// Super simple, easy-to-evolve routing logic
let chooseBackend (cfg: RoutingConfig) (req: LlmRequest) : RoutedBackend =
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
