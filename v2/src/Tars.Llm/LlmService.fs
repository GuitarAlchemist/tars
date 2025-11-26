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
