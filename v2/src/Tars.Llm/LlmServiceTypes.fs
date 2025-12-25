namespace Tars.Llm

open System.Threading
open System.Threading.Tasks
open Tars.Core
open Tars.Llm.Routing

/// <summary>Configuration for the LLM service.</summary>
type LlmServiceConfig =
    { Routing: RoutingConfig }

    static member Default = { Routing = RoutingConfig.Default }

/// <summary>
/// Interface for LLM operations.
/// Abstracts over different LLM backends (Ollama, vLLM, OpenAI, etc.).
/// </summary>
type ILlmService =
    /// <summary>Sends a chat completion request.</summary>
    abstract member CompleteAsync: LlmRequest -> Task<LlmResponse>
    /// <summary>Generates embeddings for text.</summary>
    abstract member EmbedAsync: text: string -> Task<float32[]>
    /// <summary>Sends a streaming chat completion request.</summary>
    abstract member CompleteStreamAsync: LlmRequest * (string -> unit) -> Task<LlmResponse>
    /// <summary>Routes a request to a backend without executing it.</summary>
    abstract member RouteAsync: LlmRequest -> Task<RoutedBackend>

/// <summary>
/// Optional cancellable interface for LLM operations.
/// Implementations may short-circuit when the token is cancelled.
/// </summary>
type ICancellableLlmService =
    abstract member CompleteAsync: LlmRequest * CancellationToken -> Task<LlmResponse>
    abstract member EmbedAsync: text: string * CancellationToken -> Task<float32[]>
    abstract member CompleteStreamAsync: LlmRequest * (string -> unit) * CancellationToken -> Task<LlmResponse>
    abstract member RouteAsync: LlmRequest * CancellationToken -> Task<RoutedBackend>

/// <summary>
/// Functional interface for LLM operations using AsyncResult.
/// </summary>
type ILlmServiceFunctional =
    /// <summary>Sends a chat completion request with functional error handling.</summary>
    abstract member CompleteAsync: LlmRequest -> AsyncResult<LlmResponse, LlmError>
    /// <summary>Generates embeddings for text with functional error handling.</summary>
    abstract member EmbedAsync: text: string -> AsyncResult<float32[], LlmError>
    /// <summary>Routes a request to a backend with functional error handling.</summary>
    abstract member RouteAsync: LlmRequest -> AsyncResult<RoutedBackend, LlmError>
