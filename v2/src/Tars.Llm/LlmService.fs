/// <summary>
/// High-level LLM service that abstracts over multiple backends.
/// Provides a unified interface for chat completions and embeddings.
/// </summary>
namespace Tars.Llm

open System
open System.Net.Http
open System.Threading.Tasks

/// <summary>
/// LLM service module providing the main service interface and implementation.
/// </summary>
module LlmService =

    open Tars.Llm
    open Tars.Llm.Routing

    open Tars.Core

    let inline private rerror<'a> (e: LlmError) : Result<'a, LlmError> = Result<'a, LlmError>.Error e

    /// <summary>Configuration for the LLM service.</summary>
    type LlmServiceConfig = { Routing: RoutingConfig }

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

    /// <summary>
    /// Functional interface for LLM operations using AsyncResult.
    /// </summary>
    type ILlmServiceFunctional =
        /// <summary>Sends a chat completion request with functional error handling.</summary>
        abstract member CompleteAsync: LlmRequest -> AsyncResult<LlmResponse, LlmError>
        /// <summary>Generates embeddings for text with functional error handling.</summary>
        abstract member EmbedAsync: text: string -> AsyncResult<float32[], LlmError>

    /// <summary>
    /// Default implementation of ILlmService and ILlmServiceFunctional.
    /// Routes requests to appropriate backends based on configuration.
    /// </summary>
    type DefaultLlmService(httpClient: HttpClient, cfg: LlmServiceConfig) =

        interface ILlmServiceFunctional with
            member _.CompleteAsync(req: LlmRequest) : AsyncResult<LlmResponse, LlmError> =
                asyncResult {
                    let routed = chooseBackend cfg.Routing req

                    match routed.Backend with
                    | Ollama model -> return! OllamaClientAsync.generateValidated httpClient routed.Endpoint model req
                    | Vllm model ->
                        try
                            let! res =
                                OpenAiCompatibleClient.sendChatAsync httpClient routed.Endpoint model req
                                |> Async.AwaitTask
                                |> AsyncResult.ofAsync

                            return res
                        with ex ->
                            return! AsyncResult.ofResult (rerror<LlmResponse> (LlmError.fromException ex))
                    | OpenAI model ->
                        try
                            let! res =
                                OpenAiCompatibleClient.sendChatAsync httpClient routed.Endpoint model req
                                |> Async.AwaitTask
                                |> AsyncResult.ofAsync

                            return res
                        with ex ->
                            return! AsyncResult.ofResult (rerror<LlmResponse> (LlmError.fromException ex))
                    | GoogleGemini _ ->
                        return! AsyncResult.ofResult (rerror<LlmResponse> (LlmError.ModelNotFound "Google Gemini not implemented"))
                    | Anthropic _ ->
                        return! AsyncResult.ofResult (rerror<LlmResponse> (LlmError.ModelNotFound "Anthropic not implemented"))
                }

            member _.EmbedAsync(text: string) : AsyncResult<float32[], LlmError> =
                asyncResult {
                    let model = cfg.Routing.DefaultEmbeddingModel

                    if
                        model.Contains("nomic")
                        || model.Contains("mxbai")
                        || model.Contains("llama")
                        || model.Contains("qwen")
                    then
                        return! OllamaClientAsync.getEmbeddingsAsync httpClient cfg.Routing.OllamaBaseUri model text
                    else
                        try
                            let! res =
                                OpenAiCompatibleClient.getEmbeddingsAsync
                                    httpClient
                                    cfg.Routing.OpenAIBaseUri
                                    model
                                    text
                                |> Async.AwaitTask
                                |> AsyncResult.ofAsync

                            return res
                        with ex ->
                            return! AsyncResult.ofResult (rerror<float32[]> (LlmError.fromException ex))
                }

        interface ILlmService with
            member _.CompleteAsync(req: LlmRequest) : Task<LlmResponse> =
                task {
                    let routed = chooseBackend cfg.Routing req

                    match routed.Backend with
                    | Ollama model -> return! OllamaClient.sendChatAsync httpClient routed.Endpoint model req
                    | Vllm model -> return! OpenAiCompatibleClient.sendChatAsync httpClient routed.Endpoint model req
                    | OpenAI model -> return! OpenAiCompatibleClient.sendChatAsync httpClient routed.Endpoint model req
                    | GoogleGemini _ -> return raise (NotImplementedException("Google Gemini not implemented"))
                    | Anthropic _ -> return raise (NotImplementedException("Anthropic not implemented"))
                }

            member _.CompleteStreamAsync(req: LlmRequest, onToken: string -> unit) : Task<LlmResponse> =
                task {
                    let routed = chooseBackend cfg.Routing req

                    match routed.Backend with
                    | Ollama model ->
                        return! OllamaClient.sendChatStreamAsync httpClient routed.Endpoint model req onToken
                    | Vllm model ->
                        return! OpenAiCompatibleClient.sendChatStreamAsync httpClient routed.Endpoint model req onToken
                    | OpenAI model ->
                        return! OpenAiCompatibleClient.sendChatStreamAsync httpClient routed.Endpoint model req onToken
                    | GoogleGemini _ ->
                        return raise (NotImplementedException("Google Gemini streaming not implemented"))
                    | Anthropic _ -> return raise (NotImplementedException("Anthropic streaming not implemented"))
                }

            member _.EmbedAsync(text: string) : Task<float32[]> =
                task {
                    let model = cfg.Routing.DefaultEmbeddingModel

                    if
                        model.Contains("nomic")
                        || model.Contains("mxbai")
                        || model.Contains("llama")
                        || model.Contains("qwen")
                    then
                        return! OllamaClient.getEmbeddingsAsync httpClient cfg.Routing.OllamaBaseUri model text
                    else
                        return!
                            OpenAiCompatibleClient.getEmbeddingsAsync httpClient cfg.Routing.OpenAIBaseUri model text
                }
