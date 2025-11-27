// Adapted from conversation: ChatGPT-What is vLLM.md
// Original Author: Stephane Pareilleux
// Date: 2025-11-26

namespace Tars.Llm

open System
open System.Net.Http
open System.Threading.Tasks

module LlmService =

    open Tars.Llm
    open Tars.Llm.Routing

    type LlmServiceConfig = { Routing: RoutingConfig }

    type ILlmService =
        abstract member CompleteAsync: LlmRequest -> Task<LlmResponse>
        abstract member EmbedAsync: text: string -> Task<float32[]>

    type DefaultLlmService(httpClient: HttpClient, cfg: LlmServiceConfig) =

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

            member _.EmbedAsync(text: string) : Task<float32[]> =
                task {
                    // For now, assume embeddings always go to Ollama or OpenAI based on config
                    // We'll use the DefaultEmbeddingModel from config
                    let model = cfg.Routing.DefaultEmbeddingModel

                    // Simple routing for embeddings: if model name contains "nomic" or "mxbai", use Ollama
                    // otherwise try OpenAI compatible
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
