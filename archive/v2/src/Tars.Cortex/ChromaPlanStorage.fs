/// ChromaDB-based Plan Storage for Semantic Similarity Search
/// "Find plans similar to this goal - we've solved this before"
namespace Tars.Cortex

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open Tars.Knowledge

// ============================================================================
// ChromaDB DTOs
// ============================================================================

[<CLIMutable>]
type ChromaCollectionDto =
    { [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("metadata")>]
      Metadata: Map<string, obj> option }

[<CLIMutable>]
type ChromaAddRequestDto =
    { [<JsonPropertyName("ids")>]
      Ids: string array
      [<JsonPropertyName("embeddings")>]
      Embeddings: float array array option
      [<JsonPropertyName("documents")>]
      Documents: string array option
      [<JsonPropertyName("metadatas")>]
      Metadatas: Map<string, obj> array option }

[<CLIMutable>]
type ChromaQueryRequestDto =
    { [<JsonPropertyName("query_texts")>]
      QueryTexts: string array option
      [<JsonPropertyName("query_embeddings")>]
      QueryEmbeddings: float array array option
      [<JsonPropertyName("n_results")>]
      NResults: int
      [<JsonPropertyName("where")>]
      Where: Map<string, obj> option }

[<CLIMutable>]
type ChromaQueryResultDto =
    { [<JsonPropertyName("ids")>]
      Ids: string array array
      [<JsonPropertyName("distances")>]
      Distances: float array array option
      [<JsonPropertyName("documents")>]
      Documents: string array array option
      [<JsonPropertyName("metadatas")>]
      Metadatas: Map<string, obj> array array option }

// ============================================================================
// Simple ChromaDB Client
// ============================================================================

type ChromaClient(baseUrl: string) =
    let http = new HttpClient()
    let jsonOptions = JsonSerializerOptions(WriteIndented = false)

    do http.BaseAddress <- Uri(baseUrl)

    /// Create or get collection
    member this.GetOrCreateCollection(name: string) =
        task {
            try
                let! response = http.PostAsJsonAsync($"/api/v1/collections", {| name = name |})
                response.EnsureSuccessStatusCode() |> ignore
                return FSharp.Core.Result.Ok()
            with ex ->
                return FSharp.Core.Result.Error ex.Message
        }

    /// Add documents to collection
    member this.Add(collection: string, request: ChromaAddRequestDto) =
        task {
            try
                let! response = http.PostAsJsonAsync($"/api/v1/collections/{collection}/add", request)
                response.EnsureSuccessStatusCode() |> ignore
                return FSharp.Core.Result.Ok()
            with ex ->
                return FSharp.Core.Result.Error ex.Message
        }

    /// Query collection by text
    member this.Query(collection: string, request: ChromaQueryRequestDto) =
        task {
            try
                let! result = http.PostAsJsonAsync($"/api/v1/collections/{collection}/query", request)
                result.EnsureSuccessStatusCode() |> ignore
                let! queryResult = result.Content.ReadFromJsonAsync<ChromaQueryResultDto>()
                return FSharp.Core.Result.Ok queryResult
            with ex ->
                return FSharp.Core.Result.Error ex.Message
        }

// ============================================================================
// ChromaDB Plan Storage
// ============================================================================

/// ChromaDB implementation of IPlanStorage
/// Enables semantic search: "Find plans similar to this goal"
type ChromaPlanStorage(chromaUrl: string, ?collectionName: string) =
    let client = new ChromaClient(chromaUrl)
    let collection = defaultArg collectionName "tars_plans"
    let jsonOptions = JsonSerializerOptions(WriteIndented = false)

    // Initialize collection
    do
        task {
            let! _ = client.GetOrCreateCollection(collection)
            ()
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    /// Convert plan to searchable document
    let planToDocument (plan: Plan) : string =
        let stepsText = plan.Steps |> List.map (fun s -> s.Description) |> String.concat " "

        // Concatenate goal + steps for semantic embedding
        $"{plan.Goal} {stepsText}"

    /// Convert plan to metadata
    let planToMetadata (plan: Plan) : Map<string, obj> =
        Map.ofList
            [ "plan_id", box (plan.Id.Value.ToString())
              "goal", box plan.Goal
              "status", box (plan.Status.ToString())
              "version", box plan.Version
              "created_at", box (plan.CreatedAt.ToString("o"))
              "created_by", box plan.CreatedBy.Value
              "step_count", box plan.Steps.Length
              "has_assumptions", box (not plan.Assumptions.IsEmpty) ]

    interface IPlanStorage with
        member _.SavePlan(plan) =
            task {
                try
                    let doc = planToDocument plan
                    let metadata = planToMetadata plan

                    let request: ChromaAddRequestDto =
                        { Ids = [| plan.Id.Value.ToString() |]
                          Embeddings = None // Chroma will auto-embed
                          Documents = Some [| doc |]
                          Metadatas = Some [| metadata |] }

                    let! result = client.Add(collection, request)
                    return result
                with ex ->
                    return FSharp.Core.Result.Error $"ChromaDB SavePlan exception: {ex.Message}"
            }

        member _.UpdatePlan(plan) =
            task {
                // ChromaDB doesn't have update - we can add with same ID to replace
                // Or we could delete + add, but for now just add (upsert behavior)
                try
                    let doc = planToDocument plan
                    let metadata = planToMetadata plan

                    let request: ChromaAddRequestDto =
                        { Ids = [| plan.Id.Value.ToString() |]
                          Embeddings = None
                          Documents = Some [| doc |]
                          Metadatas = Some [| metadata |] }

                    let! result = client.Add(collection, request)
                    return result
                with ex ->
                    return FSharp.Core.Result.Error $"ChromaDB UpdatePlan exception: {ex.Message}"
            }

        member _.GetPlan(planId) =
            task {
                // ChromaDB is optimized for similarity search, not direct lookup
                // Return None (use PostgreSQL for direct queries)
                return None
            }

        member _.GetPlansByStatus(status) =
            task {
                // We could filter by metadata, but ChromaDB is not great for this
                // Return empty list (use PostgreSQL for status queries)
                return []
            }

        member _.AppendEvent(event) =
            task {
                // Events don't need to be in ChromaDB (they're not searchable goals)
                // Return Ok to avoid breaking the interface
                return FSharp.Core.Result.Ok()
            }

    /// Search for plans similar to a goal description
    member this.FindSimilarPlans
        (goalDescription: string, topK: int)
        : Task<Result<(PlanId * float * Map<string, obj>) list, string>> =
        task {
            try
                let request: ChromaQueryRequestDto =
                    { QueryTexts = Some [| goalDescription |]
                      QueryEmbeddings = None
                      NResults = topK
                      Where = None }

                let! result = client.Query(collection, request)

                match result with
                | Ok queryResult ->
                    // Extract plan IDs and distances
                    let results =
                        if queryResult.Ids.Length > 0 && queryResult.Ids.[0].Length > 0 then
                            let ids = queryResult.Ids.[0]

                            let distances =
                                queryResult.Distances
                                |> Option.map (fun d -> d.[0])
                                |> Option.defaultValue (Array.create ids.Length 0.0)

                            let metadatas =
                                queryResult.Metadatas
                                |> Option.map (fun m -> m.[0])
                                |> Option.defaultValue (Array.create ids.Length Map.empty)

                            ids
                            |> Array.mapi (fun i id ->
                                match Guid.TryParse(id) with
                                | true, guid -> Some(PlanId(guid), distances.[i], metadatas.[i])
                                | _ -> None)
                            |> Array.choose id
                            |> Array.toList
                        else
                            []

                    return FSharp.Core.Result.Ok results
                | FSharp.Core.Result.Error e -> return FSharp.Core.Result.Error e
            with ex ->
                return FSharp.Core.Result.Error $"ChromaDB FindSimilarPlans exception: {ex.Message}"
        }

/// Module for creating ChromaDB plan storage
module ChromaPlanStorage =
    let create (url: string) = ChromaPlanStorage(url)

    let createWithCollection (url: string) (collectionName: string) = ChromaPlanStorage(url, collectionName)
