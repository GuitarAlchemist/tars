module Tars.Interface.Cli.Commands.Memory

open System
open Tars.Core
open Tars.Kernel
open Tars.Cortex

let add (coll: string) (id: string) (text: string) =
    task {
        let apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")

        if String.IsNullOrEmpty(apiKey) then
            printfn "Error: OPENAI_API_KEY not set."
            return 1
        else
            let modelId = "gpt-4o"
            let embeddingId = "text-embedding-3-small"

            let provider =
                new OpenAIProvider(apiKey, modelId, embeddingId) :> ICognitiveProvider

            let vectorStore = new ChromaVectorStore("http://localhost:8000") :> IVectorStore

            try
                let! embeddings = provider.GetEmbeddingsAsync([ text ])
                let vector = embeddings[0]
                do! vectorStore.SaveAsync(coll, id, vector, Map [ "text", text ])
                printfn $"Stored %s{text} in %s{coll} with id %s{id}"
                return 0
            with ex ->
                printfn $"Error: %s{ex.Message}"
                return 1
    }

let search (coll: string) (text: string) =
    task {
        let apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")

        if String.IsNullOrEmpty(apiKey) then
            printfn "Error: OPENAI_API_KEY not set."
            return 1
        else
            let modelId = "gpt-4o"
            let embeddingId = "text-embedding-3-small"

            let provider =
                new OpenAIProvider(apiKey, modelId, embeddingId) :> ICognitiveProvider

            let vectorStore = new ChromaVectorStore("http://localhost:8000") :> IVectorStore

            try
                let! embeddings = provider.GetEmbeddingsAsync([ text ])
                let vector = embeddings[0]
                let! results = vectorStore.SearchAsync(coll, vector, 5)
                printfn $"Found %d{results.Length} results:"

                for (id, dist, meta) in results do
                    printfn $"  [%s{id}] (dist: %f{dist}) %A{meta}"

                return 0
            with ex ->
                printfn $"Error: %s{ex.Message}"
                return 1
    }
