module Tars.Tests.CapabilityStoreTests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Cortex

/// Embeds a few known phrases as fixed 2-D vectors so similarity is deterministic.
type private PhraseEmbeddings() =
    let vectorFor (text: string) =
        match text with
        | t when t.Contains "summar" -> [| 1.0f; 0.05f |]
        | t when t.Contains "search" -> [| 0.05f; 1.0f |]
        | _ -> [| 0.7f; 0.7f |]

    interface ILlmService with
        member _.CompleteAsync(_) = failwith "not used"
        member _.CompleteStreamAsync(_, _) = failwith "not used"
        member _.EmbedAsync(text) = Task.FromResult(vectorFor text)
        member _.RouteAsync(_) = failwith "not used"

let private capability kind description : Capability =
    { Kind = kind
      Description = description
      InputSchema = None
      OutputSchema = None
      Confidence = None
      Reputation = None }

// Issue #244: IVectorStore returns a distance (lower is closer), but FindAgentsAsync blended
// it as if it were a similarity, so routing preferred the least similar agent.
[<Fact>]
let ``FindAgentsAsync scores the closer capability higher`` () =
    task {
        let store = CapabilityStore(InMemoryVectorStore(), PhraseEmbeddings())
        let summarizer = AgentId(Guid.NewGuid())
        let searcher = AgentId(Guid.NewGuid())

        do! store.RegisterAsync(summarizer, capability CapabilityKind.Summarization "summarize long text")
        do! store.RegisterAsync(searcher, capability CapabilityKind.WebSearch "search the web")

        let! hits = store.FindAgentsAsync("summarize this report", 2)

        let scoreOf agent =
            hits |> List.find (fun (id, _, _) -> id = agent) |> (fun (_, _, s) -> s)

        Assert.Equal(2, hits.Length)

        Assert.True(
            scoreOf summarizer > scoreOf searcher,
            $"summarizer %f{scoreOf summarizer} <= searcher %f{scoreOf searcher}"
        )
    }
