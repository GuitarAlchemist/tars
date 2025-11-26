namespace Tars.Cortex

open System.Threading.Tasks
open Microsoft.SemanticKernel
open Microsoft.SemanticKernel.Embeddings
open Tars.Kernel

type SemanticKernelProvider(apiKey: string, modelId: string, embeddingModelId: string) =
    let kernel =
        Kernel.CreateBuilder()
            .AddOpenAIChatCompletion(modelId, apiKey)
            .AddOpenAITextEmbeddingGeneration(embeddingModelId, apiKey)
            .Build()

    let embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>()

    interface ICognitiveProvider with
        member this.AskAsync(prompt: string) =
            task {
                let! result = kernel.InvokePromptAsync(prompt)
                return result.ToString()
            }

        member this.GetEmbeddingsAsync(texts: string list) =
            task {
                let input = ResizeArray(texts)
                let! embeddings = embeddingService.GenerateEmbeddingsAsync(input)
                return embeddings 
                       |> Seq.map (fun x -> x.ToArray()) 
                       |> Seq.toArray
            }
