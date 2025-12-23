namespace Tars.Cortex

// Inference providers for TARS cognitive operations.
// Wraps Semantic Kernel for LLM and embedding operations.

open System.Threading.Tasks
open Microsoft.SemanticKernel
open Microsoft.Extensions.AI
open Tars.Kernel

/// <summary>
/// Semantic Kernel-based cognitive provider.
/// Implements ICognitiveProvider using Microsoft Semantic Kernel for LLM operations.
/// </summary>
/// <param name="apiKey">OpenAI API key.</param>
/// <param name="modelId">Chat model ID (e.g., "gpt-4").</param>
/// <param name="embeddingModelId">Embedding model ID (e.g., "text-embedding-ada-002").</param>
type SemanticKernelProvider(apiKey: string, modelId: string, embeddingModelId: string) =
    let kernel =
        Kernel
            .CreateBuilder()
            .AddOpenAIChatCompletion(modelId, apiKey)
            .AddOpenAIEmbeddingGenerator(embeddingModelId, apiKey)
            .Build()

    let embeddingService =
        kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float32>>>()

    interface ICognitiveProvider with
        member this.AskAsync(prompt: string) =
            task {
                let! result = kernel.InvokePromptAsync(prompt)
                return result.ToString()
            }

        member this.GetEmbeddingsAsync(texts: string list) =
            task {
                let! embeddings = embeddingService.GenerateAsync(texts)
                return embeddings |> Seq.map (fun x -> x.Vector.ToArray()) |> Seq.toArray
            }
