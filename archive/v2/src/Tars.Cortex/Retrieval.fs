namespace Tars.Cortex

open System
open System.Collections.Generic
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// Advanced retrieval algorithms and strategies
module Retrieval = 

    /// Simple TF-IDF / BM25-like keyword scoring
    let computeKeywordScore (query: string) (content: string) (idfProvider: string -> float32) = 
        if String.IsNullOrWhiteSpace query || String.IsNullOrWhiteSpace content then
            0.0f
        else
            let queryTerms =
                query
                    .ToLowerInvariant()
                    .Split([| ' '; ','; '.'; '!'; '?'; ';'; ':' |], StringSplitOptions.RemoveEmptyEntries)
                |> Array.distinct

            let contentLower = content.ToLowerInvariant()
            let contentLen = float32 content.Length
            let avgDocLen = 500.0f
            let k1 = 1.2f
            let b = 0.75f

            let mutable score = 0.0f

            for term in queryTerms do
                let mutable tf = 0
                let mutable idx = 0

                while idx >= 0 do
                    idx <- contentLower.IndexOf(term, idx)
                    if idx >= 0 then
                        tf <- tf + 1
                        idx <- idx + 1

                if tf > 0 then
                    let idf = idfProvider term
                        
                    let tfNorm =
                        (float32 tf * (k1 + 1.0f))
                        / (float32 tf + k1 * (1.0f - b + b * contentLen / avgDocLen))

                    score <- score + (tfNorm * idf)

            if queryTerms.Length > 0 then score / float32 queryTerms.Length
            else 0.0f

    /// Combine semantic and keyword scores
    let hybridScore (semanticScore: float32) (keywordScore: float32) (semanticWeight: float32) = 
        let normalizedKeyword = min 1.0f (keywordScore / 2.0f)
        semanticWeight * semanticScore + (1.0f - semanticWeight) * normalizedKeyword

    /// Rerank results using LLM (expensive but accurate)
    let rerankWithLlm
        (llm: ILlmService)
        (query: string)
        (results: (string * float32 * Map<string, string>) list) = 
        task {
            if List.isEmpty results then return results
            else
                let docsText =
                    results
                    |> List.mapi (fun i (id, _, payload) ->
                        let content = payload |> Map.tryFind "content" |> Option.defaultValue ""
                        let truncated = if content.Length > 300 then content.[..297] + "..." else content
                        sprintf "[%d] %s" (i + 1) truncated) 
                    |> String.concat "\n\n"

                let prompt =
                    sprintf "Given the query: \"%s\"\n\nRank these documents by relevance (most relevant first). Return ONLY a comma-separated list of document numbers.\nExample: 3,1,4,2\n\nDocuments:\n%s\n\nRanking:" query docsText

                let req =
                    { ModelHint = Some "fast"
                      Model = None
                      SystemPrompt = Some "You are a ranking assistant. Return only comma-separated indices."
                      MaxTokens = Some 50
                      Temperature = Some 0.0
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None }

                try
                    let! response = llm.CompleteAsync req
                    let rankStr = response.Text.Trim()

                    let indices =
                        rankStr.Split(',')
                        |> Array.choose (fun s ->
                            match Int32.TryParse(s.Trim()) with
                            | true, v when v >= 1 && v <= results.Length -> Some(v - 1)
                            | _ -> None)
                        |> Array.distinct
                        |> Array.toList

                    let reranked = indices |> List.choose (fun i -> results |> List.tryItem i)
                    let remaining =
                        results
                        |> List.indexed
                        |> List.filter (fun (i, _) -> not (List.contains i indices))
                        |> List.map snd

                    return reranked @ remaining
                with _ ->
                    return results
        }

    /// Reciprocal Rank Fusion
    let reciprocalRankFusion (resultLists: (string * float32 * Map<string, string>) list list) (k: int) = 
        let scores = Dictionary<string, float32>()
        let payloads = Dictionary<string, Map<string, string>>()

        for results in resultLists do
            results
            |> List.iteri (fun rank (id, _, payload) ->
                let rrfScore = 1.0f / (float32 k + float32 (rank + 1))
                match scores.TryGetValue(id) with
                | true, existing -> scores.[id] <- existing + rrfScore
                | false, _ -> scores.[id] <- rrfScore
                payloads.[id] <- payload)

        scores
        |> Seq.map (fun kv -> (kv.Key, 1.0f - kv.Value, payloads.[kv.Key]))
        |> Seq.sortBy (fun (_, dist, _) -> dist)
        |> Seq.toList