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

    /// `vllmExtensions` gates vLLM-only top-level request params. Vllm, OpenAI and
    /// DockerModelRunner all resolve to this adapter, but OpenAI proper rejects
    /// unknown top-level parameters — so only the Vllm case opts in.
    let private openAiCompatible
        (vllmExtensions: bool)
        (http: HttpClient)
        (endpoint: Uri)
        (model: string)
        (apiKey: string option)
        =
        { new ILlmBackend with
            member _.Complete req =
                OpenAiCompatibleClient.sendChatAsyncWith vllmExtensions http endpoint model apiKey req

            member _.Stream(req, onToken) =
                OpenAiCompatibleClient.sendChatStreamAsyncWith vllmExtensions http endpoint model apiKey req onToken }

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
        | Vllm model -> openAiCompatible true http routed.Endpoint model routed.ApiKey
        | OpenAI model
        | DockerModelRunner model -> openAiCompatible false http routed.Endpoint model routed.ApiKey
        | GoogleGemini model -> gemini http routed.Endpoint model routed.ApiKey
        | Anthropic model -> anthropic http routed.Endpoint model routed.ApiKey
        | LlamaCpp(model, config) -> llamaCpp http routed.Endpoint model config routed.ApiKey
        | LlamaSharp modelPath -> llamaSharp cfg modelPath routed.ApiKey

/// Embedding routing is independent of chat-backend routing: it selects by the
/// configured provider and key, not by the routed chat backend.
module Embedder =

    /// Does this embedding request belong to OpenAI?
    ///
    /// It used to be decided by asking whether the model's name contained one of
    /// four substrings, with everything else treated as OpenAI's. `bge-m3`,
    /// `all-minilm`, `snowflake-arctic-embed` and `granite-embedding` are ordinary
    /// `ollama pull` models and match none of them, so the caller's text was posted
    /// to api.openai.com — with no key, for a 401 whose message named Ollama. Text
    /// leaving the machine is not something a substring check should decide.
    ///
    /// So: `Llm.EmbeddingProvider` decides when it is set, either way — it is the one
    /// unambiguous statement of where the text is meant to go, and it is the answer
    /// for a local alias that happens to be named `text-embedding-something`. When it
    /// is not set, the model's own name is the only evidence there is, and only
    /// OpenAI's own `text-embedding-*` family counts.
    ///
    /// Either way there must be a key. A blank one does not count: `getEmbeddingsAsync`
    /// omits the authorization header for a blank key, so the text would leave the
    /// machine only to come back 401 — the egress this is here to prevent. The key is
    /// a weak signal on its own, since `RoutingConfig.fromTarsConfig` copies the single
    /// `Llm.ApiKey` into every provider's slot; it can veto, never elect.
    ///
    /// `PreferredProvider` deliberately does not enter into it. It is the *chat*
    /// provider, and embeddings are configured separately (`Llm.Provider` vs
    /// `Llm.EmbeddingModel`), so a mixed setup — OpenAI for chat, the default
    /// `nomic-embed-text` locally for embeddings — is both ordinary and exactly the
    /// case where letting the chat provider decide would post a local model name to
    /// api.openai.com.
    let usesOpenAi (routing: RoutingConfig) =
        let hasKey = routing.OpenAIKey |> Option.exists (String.IsNullOrWhiteSpace >> not)

        let wantsOpenAi =
            match routing.EmbeddingProvider with
            | Some declared -> declared.Trim().Equals("OpenAI", StringComparison.OrdinalIgnoreCase)
            | None ->
                routing.DefaultEmbeddingModel.StartsWith("text-embedding", StringComparison.OrdinalIgnoreCase)

        wantsOpenAi && hasKey

    /// Select the embedding backend and run it (Task flavor).
    let embed (http: HttpClient) (routing: RoutingConfig) (text: string) : Task<float32[]> =
        let model = routing.DefaultEmbeddingModel

        if usesOpenAi routing then
            OpenAiCompatibleClient.getEmbeddingsAsync http routing.OpenAIBaseUri model routing.OpenAIKey text
        else
            OllamaClient.getEmbeddingsAsync http routing.OllamaBaseUri model text
