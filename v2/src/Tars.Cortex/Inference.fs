namespace Tars.Cortex

// Inference providers for TARS cognitive operations.
// Uses Microsoft.Extensions.AI abstractions (IChatClient, IEmbeddingGenerator).

open Microsoft.Extensions.AI
open Tars.Kernel

/// <summary>
/// Microsoft.Extensions.AI-based cognitive provider.
/// Implements ICognitiveProvider using IChatClient and IEmbeddingGenerator.
/// </summary>
/// <param name="chatClient">An IChatClient implementation (e.g., OpenAI, Ollama).</param>
/// <param name="embeddingGenerator">An IEmbeddingGenerator implementation.</param>
type ExtensionsAIProvider(chatClient: IChatClient, embeddingGenerator: IEmbeddingGenerator<string, Embedding<float32>>) =

    interface ICognitiveProvider with
        member this.AskAsync(prompt: string) =
            task {
                let! response = chatClient.GetResponseAsync(prompt)
                return response.Text
            }

        member this.GetEmbeddingsAsync(texts: string list) =
            task {
                let! embeddings = embeddingGenerator.GenerateAsync(texts)
                return embeddings |> Seq.map (fun x -> x.Vector.ToArray()) |> Seq.toArray
            }

/// <summary>
/// Convenience constructor for OpenAI-backed provider.
/// </summary>
type OpenAIProvider(apiKey: string, modelId: string, embeddingModelId: string) =
    let chatClient =
        OpenAI.Chat.ChatClient(modelId, apiKey).AsIChatClient()

    let embeddingGenerator =
        OpenAI.Embeddings.EmbeddingClient(embeddingModelId, apiKey).AsIEmbeddingGenerator()

    let inner = ExtensionsAIProvider(chatClient, embeddingGenerator)

    interface ICognitiveProvider with
        member this.AskAsync(prompt) = (inner :> ICognitiveProvider).AskAsync(prompt)
        member this.GetEmbeddingsAsync(texts) = (inner :> ICognitiveProvider).GetEmbeddingsAsync(texts)
