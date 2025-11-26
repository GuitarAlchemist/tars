namespace Tars.Cortex

open System.Threading.Tasks
open Microsoft.SemanticKernel
open Microsoft.Extensions.AI
open Tars.Kernel

type SemanticKernelProvider(apiKey: string, modelId: string, embeddingModelId: string) =
    let kernel =
        Kernel.CreateBuilder()
            .AddOpenAIChatCompletion(modelId, apiKey)
            .AddOpenAIEmbeddingGenerator(embeddingModelId, apiKey)
            .Build()

    let embeddingService = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float32>>>()

    interface ICognitiveProvider with
        member this.AskAsync(prompt: string) =
            task {
                let! result = kernel.InvokePromptAsync(prompt)
                return result.ToString()
            }

        member this.GetEmbeddingsAsync(texts: string list) =
            task {
                let! embeddings = embeddingService.GenerateAsync(texts)
                return embeddings 
                       |> Seq.map (fun x -> x.Vector.ToArray()) 
                       |> Seq.toArray
            }
