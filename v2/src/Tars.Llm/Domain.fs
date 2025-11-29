// Adapted from conversation: ChatGPT-What is vLLM.md
// Original Author: Stephane Pareilleux
// Date: 2025-11-26

namespace Tars.Llm

open System

type Role =
    | System
    | User
    | Assistant

type LlmMessage = { Role: Role; Content: string }

/// High-level request from TARS – independent of specific backend.
type LlmRequest =
    { ModelHint: string option // e.g. "code", "reasoning", "cheap", "deep"
      MaxTokens: int option
      Temperature: float option
      Messages: LlmMessage list }

type TokenUsage =
    { PromptTokens: int
      CompletionTokens: int
      TotalTokens: int }

/// Unified response type
type LlmResponse =
    { Text: string
      FinishReason: string option
      Usage: TokenUsage option
      Raw: string option } // raw JSON if you want to log/debug

type LlmBackend =
    | Ollama of model: string
    | Vllm of model: string
    | OpenAI of model: string
    | GoogleGemini of model: string
    | Anthropic of model: string

/// Routing policy result
type RoutedBackend = { Backend: LlmBackend; Endpoint: Uri }
