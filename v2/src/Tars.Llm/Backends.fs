namespace Tars.Llm

open System
open System.Net.Http
open System.Threading.Tasks
open Tars.Llm.Routing

/// A chat-completion backend resolved to a single provider/endpoint/model/key.
/// This is the one place provider dispatch lives: routing decides *which* backend,
/// and the adapters wrap the existing per-provider client functions unchanged.
type ILlmBackend =
    /// Non-streaming chat completion.
    abstract member Complete: LlmRequest -> Task<LlmResponse>
    /// Streaming chat completion; tokens are delivered to onToken as they arrive.
    abstract member Stream: LlmRequest * (string -> unit) -> Task<LlmResponse>

/// Resolves a routed backend to its adapter.
/// The single `resolve` match replaces the per-interface dispatch that used to be
/// duplicated across DefaultLlmService.
module Backends =

    let private openAiCompatible (http: HttpClient) (endpoint: Uri) (model: string) (apiKey: string option) =
        { new ILlmBackend with
            member _.Complete req =
                OpenAiCompatibleClient.sendChatAsync http endpoint model apiKey req

            member _.Stream(req, onToken) =
                OpenAiCompatibleClient.sendChatStreamAsync http endpoint model apiKey req onToken }

    let private ollama (http: HttpClient) (endpoint: Uri) (model: string) (apiKey: string option) =
        { new ILlmBackend with
            member _.Complete req =
                OllamaClient.sendChatAsync http endpoint model apiKey req

            member _.Stream(req, onToken) =
                OllamaClient.sendChatStreamAsync http endpoint model apiKey req onToken }

    let private gemini (http: HttpClient) (endpoint: Uri) (model: string) (apiKey: string option) =
        { new ILlmBackend with
            member _.Complete req =
                GoogleGeminiClient.generateContentAsync http endpoint model apiKey req

            member _.Stream(_req, _onToken) =
                raise (NotImplementedException("Google Gemini streaming not implemented")) }

    let private anthropic (http: HttpClient) (endpoint: Uri) (model: string) (apiKey: string option) =
        { new ILlmBackend with
            member _.Complete req =
                AnthropicClient.sendMessageAsync http endpoint model apiKey req

            member _.Stream(req, onToken) =
                AnthropicClient.sendMessageStreamAsync http endpoint model apiKey req onToken }

    let private llamaCpp (http: HttpClient) (endpoint: Uri) (model: string) config (apiKey: string option) =
        { new ILlmBackend with
            member _.Complete req =
                LlamaCppClient.sendChatAsync http endpoint model config apiKey req

            member _.Stream(req, onToken) =
                LlamaCppClient.sendChatStreamAsync http endpoint model config apiKey req onToken }

    let private llamaSharp (cfg: LlmServiceConfig) (modelPath: string) (apiKey: string option) =
        let svc = LlamaSharpFactory.getService cfg apiKey modelPath

        { new ILlmBackend with
            member _.Complete req = svc.CompleteAsync req
            member _.Stream(req, onToken) = svc.CompleteStreamAsync(req, onToken) }

    /// The single provider-dispatch match: a routed backend becomes its adapter.
    let resolve (cfg: LlmServiceConfig) (http: HttpClient) (routed: RoutedBackend) : ILlmBackend =
        match routed.Backend with
        | Ollama model -> ollama http routed.Endpoint model routed.ApiKey
        | Vllm model
        | OpenAI model
        | DockerModelRunner model -> openAiCompatible http routed.Endpoint model routed.ApiKey
        | GoogleGemini model -> gemini http routed.Endpoint model routed.ApiKey
        | Anthropic model -> anthropic http routed.Endpoint model routed.ApiKey
        | LlamaCpp(model, config) -> llamaCpp http routed.Endpoint model config routed.ApiKey
        | LlamaSharp modelPath -> llamaSharp cfg modelPath routed.ApiKey

/// Embedding routing is independent of chat-backend routing: it selects by the
/// configured embedding-model name, not by the routed chat backend.
module Embedder =

    let private isOllamaEmbedding (model: string) =
        model.Contains("nomic")
        || model.Contains("mxbai")
        || model.Contains("llama")
        || model.Contains("qwen")

    /// Select the embedding backend by model name and run it (Task flavor).
    let embed (http: HttpClient) (routing: RoutingConfig) (text: string) : Task<float32[]> =
        let model = routing.DefaultEmbeddingModel

        if isOllamaEmbedding model then
            OllamaClient.getEmbeddingsAsync http routing.OllamaBaseUri model text
        else
            OpenAiCompatibleClient.getEmbeddingsAsync http routing.OpenAIBaseUri model routing.OpenAIKey text
