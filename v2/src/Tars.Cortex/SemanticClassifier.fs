namespace Tars.Cortex

open System
open Tars.Core
open Tars.Llm

/// Generic category for semantic classification
type SemanticCategory<'T> =
    { Label: string
      Value: 'T
      Sentinels: string list }

/// Semantic Classifier - Maps text to categories using embeddings and cosine similarity
type SemanticClassifier<'T>(llm: ILlmService, categories: SemanticCategory<'T> list) =

    // Cache for sentinel embeddings
    let mutable sentinelEmbeddings: (SemanticCategory<'T> * float32[] list) list option =
        None

    let ensureEmbeddings () =
        async {
            match sentinelEmbeddings with
            | Some cached -> return cached
            | None ->
                let! results =
                    categories
                    |> List.map (fun cat ->
                        async {
                            let! embeds =
                                cat.Sentinels
                                |> List.map (fun s -> llm.EmbedAsync s |> Async.AwaitTask)
                                |> Async.Parallel

                            return (cat, embeds |> Array.toList)
                        })
                    |> Async.Parallel

                let cached = results |> Array.toList
                sentinelEmbeddings <- Some cached
                return cached
        }

    /// Classify text into one of the categories
    member this.ClassifyAsync(text: string, ?threshold: float) =
        async {
            let threshold = defaultArg threshold 0.7

            // 1. Get embedding for the input text
            let! inputEmbed = llm.EmbedAsync text |> Async.AwaitTask

            // 2. Load/Ensure sentinel embeddings
            let! sentinels = ensureEmbeddings ()

            // 3. Find the best match
            let matches =
                sentinels
                |> List.map (fun (cat, embeds) ->
                    let maxSim =
                        embeds
                        |> List.map (fun e -> MetricSpace.cosineSimilarity inputEmbed e)
                        |> List.max

                    (cat, maxSim))
                |> List.filter (fun (_, sim) -> float sim >= threshold)
                |> List.sortByDescending snd

            match matches with
            | [] -> return None
            | (bestCat, score) :: _ -> return Some(bestCat.Value, float score)
        }

module SemanticClassifierFactory =

    /// Create a classifier for Agent Speech Acts (Performatives)
    let createSpeechActClassifier (llm: ILlmService) =
        let categories =
            [ { Label = "Request"
                Value = Performative.Request
                Sentinels = [ "do this"; "please execute"; "perform action"; "request" ] }
              { Label = "Inform"
                Value = Performative.Inform
                Sentinels = [ "here is information"; "sharing fact"; "tell you"; "provide data" ] }
              { Label = "Query"
                Value = Performative.Query
                Sentinels = [ "what is"; "tell me about"; "how many"; "search for" ] }
              { Label = "Propose"
                Value = Performative.Propose
                Sentinels = [ "I suggest"; "how about we"; "I can do this"; "proposal" ] }
              { Label = "Refuse"
                Value = Performative.Refuse
                Sentinels = [ "I cannot"; "refused"; "violated policy"; "won't do that" ] }
              { Label = "Failure"
                Value = Performative.Failure
                Sentinels = [ "execution failed"; "error happened"; "something went wrong"; "crash" ] } ]

        SemanticClassifier(llm, categories)

    /// Create a classifier for Agent Domains (Intents)
    let createDomainClassifier (llm: ILlmService) =
        let categories =
            [ { Label = "Coding"
                Value = AgentDomain.Coding
                Sentinels =
                  [ "write code"
                    "refactor function"
                    "debug error"
                    "implement feature"
                    "programming" ] }
              { Label = "Planning"
                Value = AgentDomain.Planning
                Sentinels = [ "what should I do next"; "create roadmap"; "step by step"; "strategy" ] }
              { Label = "Reasoning"
                Value = AgentDomain.Reasoning
                Sentinels = [ "why did this happen"; "logical analysis"; "solve puzzle"; "think deep" ] }
              { Label = "Chat"
                Value = AgentDomain.Chat
                Sentinels = [ "hello"; "how are you"; "just talking"; "general assistance" ] } ]

        SemanticClassifier(llm, categories)

    /// Attempts to parse an intent from a wire format string semantically.
    /// Format: "ACT: <IntentName>: <Content>"
    /// Uses semantic classification for the IntentName part.
    let tryParseSemanticAsync (llm: ILlmService) (text: string) =
        async {
            if not (text.StartsWith("ACT:")) then
                return None
            else
                let parts = text.Split(':', 3)

                if parts.Length < 3 then
                    return None
                else
                    let intentName = parts.[1].Trim()
                    let content = parts.[2].Trim()

                    // Use semantic classifier for the intent name
                    let classifier = createSpeechActClassifier llm
                    let! matchResult = classifier.ClassifyAsync(intentName, 0.4) // Lax threshold for names

                    match matchResult with
                    | Some(performative, _) ->
                        let intent =
                            match performative with
                            | Performative.Request -> AgentIntent.Ask content
                            | Performative.Inform -> AgentIntent.Tell content
                            | Performative.Query -> AgentIntent.Ask content
                            | Performative.Propose -> AgentIntent.Propose content
                            | Performative.Refuse -> AgentIntent.Reject(Guid.Empty, content)
                            | Performative.Failure -> AgentIntent.Error content
                            | Performative.NotUnderstood -> AgentIntent.Error "NotUnderstood"
                            | Performative.Event -> AgentIntent.Event("system", content)

                        return Some(intent, content)
                    | None -> return None
        }
