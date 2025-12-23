namespace Tars.Llm

// Core domain types for the TARS LLM abstraction layer.
// Provides backend-agnostic types for LLM requests, responses, and routing.

open System

/// <summary>Message role in a conversation.</summary>
type Role =
    /// <summary>System prompt that sets context/behavior.</summary>
    | System
    /// <summary>User input message.</summary>
    | User
    /// <summary>Assistant (LLM) response.</summary>
    | Assistant

/// <summary>A single message in a conversation.</summary>
type LlmMessage = { Role: Role; Content: string }

/// <summary>Grammar constraint for structured output.</summary>
type Grammar =
    | JsonSchema of schema: string
    | Regex of pattern: string

/// <summary>Format of the expected response.</summary>
type ResponseFormat =
    | Json
    | Text
    | Constrained of Grammar

/// <summary>
/// High-level request from TARS – independent of specific backend.
/// </summary>
type LlmRequest =
    {
        /// <summary>Hint for routing (e.g., "code", "reasoning", "cheap", "deep").</summary>
        ModelHint: string option
        /// <summary>Specific model override (optional).</summary>
        Model: string option
        /// <summary>System prompt override (optional).</summary>
        SystemPrompt: string option
        /// <summary>Maximum tokens to generate.</summary>
        MaxTokens: int option
        /// <summary>Sampling temperature (0.0 = deterministic, 1.0+ = creative).</summary>
        Temperature: float option
        /// <summary>Stop sequences.</summary>
        Stop: string list
        /// <summary>Conversation messages.</summary>
        Messages: LlmMessage list
        /// <summary>Tool definitions (optional).</summary>
        Tools: obj list
        /// <summary>Tool choice (optional).</summary>
        ToolChoice: obj option
        /// <summary>Response format (e.g. json_object).</summary>
        ResponseFormat: ResponseFormat option
        /// <summary>Whether to stream the response.</summary>
        Stream: bool
        /// <summary>Whether to enforce JSON mode.</summary>
        JsonMode: bool
        /// <summary>Random seed (optional).</summary>
        Seed: int option
    }

/// <summary>Token usage statistics from an LLM response.</summary>
type TokenUsage =
    {
        /// <summary>Tokens in the prompt.</summary>
        PromptTokens: int
        /// <summary>Tokens generated in the response.</summary>
        CompletionTokens: int
        /// <summary>Total tokens (prompt + completion).</summary>
        TotalTokens: int
    }

/// <summary>Unified response type from any LLM backend.</summary>
type LlmResponse =
    {
        /// <summary>Generated text content.</summary>
        Text: string
        /// <summary>Reason for completion (e.g., "stop", "length").</summary>
        FinishReason: string option
        /// <summary>Token usage statistics if available.</summary>
        Usage: TokenUsage option
        /// <summary>Raw JSON response for debugging.</summary>
        Raw: string option
    }

/// <summary>Configuration options for llama.cpp backend.</summary>
type LlamaCppConfig =
    {
        /// <summary>Number of GPU layers to offload (-1 = all, 0 = CPU only).</summary>
        GpuLayers: int option
        /// <summary>Context size in tokens (default: 2048).</summary>
        ContextSize: int option
        /// <summary>Number of parallel inference slots (default: 1).</summary>
        NumParallel: int option
        /// <summary>Enable flash attention for faster inference.</summary>
        FlashAttention: bool
    }

/// <summary>Default llama.cpp configuration.</summary>
module LlamaCppConfig =
    let Default =
        { GpuLayers = None
          ContextSize = None
          NumParallel = None
          FlashAttention = true }

/// <summary>Supported LLM backends.</summary>
type LlmBackend =
    /// <summary>Local Ollama server.</summary>
    | Ollama of model: string
    /// <summary>vLLM high-performance server.</summary>
    | Vllm of model: string
    /// <summary>OpenAI API.</summary>
    | OpenAI of model: string
    /// <summary>Google Gemini API.</summary>
    | GoogleGemini of model: string
    /// <summary>Anthropic Claude API.</summary>
    | Anthropic of model: string
    /// <summary>Docker Model Runner (OpenAI-compatible, container-native).</summary>
    | DockerModelRunner of model: string
    /// <summary>llama.cpp server (high-performance local inference with GGUF models).</summary>
    | LlamaCpp of model: string * config: LlamaCppConfig option

/// <summary>Result of routing policy - which backend and endpoint to use.</summary>
type RoutedBackend = { Backend: LlmBackend; Endpoint: Uri }
