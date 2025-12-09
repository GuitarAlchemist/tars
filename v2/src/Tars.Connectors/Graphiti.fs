/// Graphiti Temporal Knowledge Graph Client
/// HTTP client for interacting with the Graphiti REST API
module Tars.Connectors.Graphiti

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

// ============================================================================
// DTOs for Graphiti API
// ============================================================================

/// Episode represents raw input data (messages, events, code changes)
[<CLIMutable>]
type EpisodeDto =
    { [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("content")>]
      Content: string
      [<JsonPropertyName("source")>]
      Source: string
      [<JsonPropertyName("source_description")>]
      SourceDescription: string option
      [<JsonPropertyName("reference_time")>]
      ReferenceTime: DateTime option }

/// Entity extracted from episodes
[<CLIMutable>]
type EntityDto =
    { [<JsonPropertyName("uuid")>]
      Uuid: string
      [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("entity_type")>]
      EntityType: string
      [<JsonPropertyName("summary")>]
      Summary: string option
      [<JsonPropertyName("created_at")>]
      CreatedAt: DateTime }

/// Fact/relationship between entities
[<CLIMutable>]
type FactDto =
    { [<JsonPropertyName("uuid")>]
      Uuid: string
      [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("fact")>]
      Fact: string
      [<JsonPropertyName("source_node_uuid")>]
      SourceNodeUuid: string
      [<JsonPropertyName("target_node_uuid")>]
      TargetNodeUuid: string
      [<JsonPropertyName("valid_at")>]
      ValidAt: DateTime option
      [<JsonPropertyName("invalid_at")>]
      InvalidAt: DateTime option
      [<JsonPropertyName("created_at")>]
      CreatedAt: DateTime }

/// Search request
[<CLIMutable>]
type SearchRequestDto =
    { [<JsonPropertyName("query")>]
      Query: string
      [<JsonPropertyName("num_results")>]
      NumResults: int
      [<JsonPropertyName("center_node_uuid")>]
      CenterNodeUuid: string option }

/// Search result item
[<CLIMutable>]
type SearchResultDto =
    { [<JsonPropertyName("uuid")>]
      Uuid: string
      [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("fact")>]
      Fact: string option
      [<JsonPropertyName("score")>]
      Score: float }

/// Community/cluster of related entities
[<CLIMutable>]
type CommunityDto =
    { [<JsonPropertyName("uuid")>]
      Uuid: string
      [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("summary")>]
      Summary: string
      [<JsonPropertyName("member_count")>]
      MemberCount: int }

/// Health check response
[<CLIMutable>]
type HealthCheckDto =
    { [<JsonPropertyName("status")>]
      Status: string }

// ============================================================================
// Graphiti Client
// ============================================================================

type GraphitiClient(baseUri: Uri, ?httpClient: HttpClient) =
    let client = defaultArg httpClient (new HttpClient())

    let jsonOptions =
        new JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    /// Check if Graphiti server is healthy
    member _.HealthCheckAsync() : Task<Result<string, string>> =
        task {
            try
                let uri = Uri(baseUri, "/healthcheck")
                let! response = client.GetAsync(uri)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadFromJsonAsync<HealthCheckDto>()
                    return Ok content.Status
                else
                    return Error $"Health check failed: {response.StatusCode}"
            with ex ->
                return Error ex.Message
        }

    /// Add a new episode (raw data) to the knowledge graph
    member _.AddEpisodeAsync(episode: EpisodeDto, ?groupId: string) : Task<Result<string, string>> =
        task {
            try
                let gid = defaultArg groupId "tars"
                let uri = Uri(baseUri, $"/episodes/{gid}")
                let! response = client.PostAsJsonAsync(uri, episode, jsonOptions)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    return Ok content
                else
                    let! error = response.Content.ReadAsStringAsync()
                    return Error $"Add episode failed: {response.StatusCode} - {error}"
            with ex ->
                return Error ex.Message
        }

    /// Search the knowledge graph using hybrid search (semantic + BM25 + graph)
    member _.SearchAsync
        (query: string, ?numResults: int, ?centerNodeUuid: string)
        : Task<Result<SearchResultDto list, string>> =
        task {
            try
                let uri = Uri(baseUri, "/search")

                let request =
                    { Query = query
                      NumResults = defaultArg numResults 10
                      CenterNodeUuid = centerNodeUuid }

                let! response = client.PostAsJsonAsync(uri, request, jsonOptions)

                if response.IsSuccessStatusCode then
                    let! results = response.Content.ReadFromJsonAsync<SearchResultDto[]>(jsonOptions)
                    return Ok(results |> Array.toList)
                else
                    let! error = response.Content.ReadAsStringAsync()
                    return Error $"Search failed: {response.StatusCode} - {error}"
            with ex ->
                return Error ex.Message
        }

    /// Get all entities in the knowledge graph
    member _.GetEntitiesAsync() : Task<Result<EntityDto list, string>> =
        task {
            try
                let uri = Uri(baseUri, "/entity-node")
                let! response = client.GetAsync(uri)

                if response.IsSuccessStatusCode then
                    let! entities = response.Content.ReadFromJsonAsync<EntityDto[]>(jsonOptions)
                    return Ok(entities |> Array.toList)
                else
                    let! error = response.Content.ReadAsStringAsync()
                    return Error $"Get entities failed: {response.StatusCode} - {error}"
            with ex ->
                return Error ex.Message
        }

    /// Get all facts (relationships) in the knowledge graph
    member _.GetFactsAsync() : Task<Result<FactDto list, string>> =
        task {
            try
                let uri = Uri(baseUri, "/v1/facts")
                let! response = client.GetAsync(uri)

                if response.IsSuccessStatusCode then
                    let! facts = response.Content.ReadFromJsonAsync<FactDto[]>(jsonOptions)
                    return Ok(facts |> Array.toList)
                else
                    let! error = response.Content.ReadAsStringAsync()
                    return Error $"Get facts failed: {response.StatusCode} - {error}"
            with ex ->
                return Error ex.Message
        }

    /// Get detected communities/clusters
    member _.GetCommunitiesAsync() : Task<Result<CommunityDto list, string>> =
        task {
            try
                let uri = Uri(baseUri, "/v1/communities")
                let! response = client.GetAsync(uri)

                if response.IsSuccessStatusCode then
                    let! communities = response.Content.ReadFromJsonAsync<CommunityDto[]>(jsonOptions)
                    return Ok(communities |> Array.toList)
                else
                    let! error = response.Content.ReadAsStringAsync()
                    return Error $"Get communities failed: {response.StatusCode} - {error}"
            with ex ->
                return Error ex.Message
        }

    interface IDisposable with
        member _.Dispose() =
            if httpClient.IsNone then
                client.Dispose()

// ============================================================================
// Helper functions
// ============================================================================

/// Create a client from environment or default URL
let createClient () =
    let url =
        match Environment.GetEnvironmentVariable("GRAPHITI_URL") with
        | null
        | "" -> "http://localhost:8001"
        | url -> url

    new GraphitiClient(Uri(url))

/// Create an episode from a code change
let createCodeEpisode (filePath: string) (content: string) (changeType: string) =
    { Name = $"{changeType}: {IO.Path.GetFileName(filePath)}"
      Content = content
      Source = "tars_code"
      SourceDescription = Some $"F# code file: {filePath}"
      ReferenceTime = Some DateTime.UtcNow }

/// Create an episode from an agent interaction
let createAgentEpisode (agentName: string) (userInput: string) (response: string) =
    { Name = $"Chat with {agentName}"
      Content = $"User: {userInput}\n\nAgent: {response}"
      Source = "tars_agent"
      SourceDescription = Some $"Agent interaction with {agentName}"
      ReferenceTime = Some DateTime.UtcNow }

/// Create an episode from a reflection/belief
let createBeliefEpisode (statement: string) (confidence: float) =
    { Name = $"Belief (confidence: {confidence:F2})"
      Content = statement
      Source = "tars_reflection"
      SourceDescription = Some "Agent reflection/belief"
      ReferenceTime = Some DateTime.UtcNow }
