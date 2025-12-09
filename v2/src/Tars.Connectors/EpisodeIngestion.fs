/// Episode Ingestion Service
/// Bridges TARS internal events with the Graphiti temporal knowledge graph server
module Tars.Connectors.EpisodeIngestion

open System
open System.Threading.Tasks
open Tars.Connectors.Graphiti

// Import Episode type explicitly to avoid AgentState.Error shadowing Result.Error
type Episode = Tars.Core.Episode

/// Configuration for the ingestion service
type IngestionConfig =
    { GraphitiUrl: string
      BatchSize: int
      AutoFlushInterval: TimeSpan option }

module IngestionConfig =
    let defaults =
        { GraphitiUrl = "http://localhost:8001"
          BatchSize = 10
          AutoFlushInterval = Some(TimeSpan.FromSeconds(30.0)) }

/// Convert TARS Episode to Graphiti EpisodeDto
let toGraphitiEpisode (episode: Episode) : EpisodeDto =
    match episode with
    | Episode.AgentInteraction(agentId, input, output, ts) ->
        { Name = $"Agent Interaction: {agentId}"
          Content = $"User: {input}\n\nAgent: {output}"
          Source = "tars_agent"
          SourceDescription = Some $"Interaction with agent {agentId}"
          ReferenceTime = Some ts }

    | Episode.CodeChange(file, diff, author, ts) ->
        { Name = $"Code Change: {IO.Path.GetFileName(file)}"
          Content = $"File: {file}\nAuthor: {author}\n\n{diff}"
          Source = "tars_code"
          SourceDescription = Some $"Code modification by {author}"
          ReferenceTime = Some ts }

    | Episode.Reflection(agentId, content, ts) ->
        { Name = $"Reflection: {agentId}"
          Content = content
          Source = "tars_reflection"
          SourceDescription = Some $"Agent {agentId} reflection"
          ReferenceTime = Some ts }

    | Episode.UserMessage(content, metadata, ts) ->
        let metaStr =
            metadata
            |> Map.toSeq
            |> Seq.map (fun (k, v) -> $"{k}={v}")
            |> String.concat ", "

        { Name = "User Message"
          Content = content
          Source = "tars_user"
          SourceDescription = if metaStr = "" then None else Some metaStr
          ReferenceTime = Some ts }

    | Episode.ToolCall(name, args, result, ts) ->
        let argsStr =
            args |> Map.toSeq |> Seq.map (fun (k, v) -> $"{k}={v}") |> String.concat ", "

        { Name = $"Tool Call: {name}"
          Content = $"Tool: {name}\nArgs: {argsStr}\n\nResult:\n{result}"
          Source = "tars_tool"
          SourceDescription = Some $"Tool execution: {name}"
          ReferenceTime = Some ts }

    | Episode.BeliefUpdate(agentId, belief, confidence, ts) ->
        { Name = $"Belief Update: {agentId}"
          Content = $"Belief: {belief}\nConfidence: {confidence:F2}"
          Source = "tars_belief"
          SourceDescription = Some $"Agent {agentId} belief update"
          ReferenceTime = Some ts }

    | Episode.PatternDetected(patternType, location, details, ts) ->
        { Name = $"Pattern: {patternType}"
          Content = $"Type: {patternType}\nLocation: {location}\n\n{details}"
          Source = "tars_pattern"
          SourceDescription = Some $"Pattern detection at {location}"
          ReferenceTime = Some ts }

/// Episode ingestion service that manages sync with Graphiti
type EpisodeIngestionService(config: IngestionConfig) =
    let client = new GraphitiClient(Uri(config.GraphitiUrl))
    let mutable pendingEpisodes = ResizeArray<Episode>()
    let mutable totalIngested = 0
    let mutable lastFlush = DateTime.UtcNow

    /// Add an episode to the pending queue
    member _.Queue(episode: Episode) =
        lock pendingEpisodes (fun () -> pendingEpisodes.Add(episode))

    /// Flush pending episodes to Graphiti
    member this.FlushAsync() : Task<Result<int, string>> =
        task {
            let episodesToFlush =
                lock pendingEpisodes (fun () ->
                    let eps = pendingEpisodes.ToArray()
                    pendingEpisodes.Clear()
                    eps)

            if episodesToFlush.Length = 0 then
                return Ok 0
            else
                let mutable successCount = 0
                let mutable errors = ResizeArray<string>()

                for episode in episodesToFlush do
                    let dto = toGraphitiEpisode episode
                    let! result = client.AddEpisodeAsync(dto)

                    match result with
                    | Ok _ -> successCount <- successCount + 1
                    | Error err -> errors.Add(err)

                totalIngested <- totalIngested + successCount
                lastFlush <- DateTime.UtcNow

                if errors.Count > 0 then
                    return Error(String.concat "; " errors)
                else
                    return Ok successCount
        }

    /// Ingest a single episode immediately
    member _.IngestAsync(episode: Episode) : Task<Result<unit, string>> =
        task {
            let dto = toGraphitiEpisode episode
            let! result = client.AddEpisodeAsync(dto)

            match result with
            | Ok _ ->
                totalIngested <- totalIngested + 1
                return Ok()
            | Error err -> return Error err
        }

    /// Get ingestion statistics
    member _.GetStats() =
        {| TotalIngested = totalIngested
           PendingCount = pendingEpisodes.Count
           LastFlush = lastFlush |}

    /// Check Graphiti server health
    member _.HealthCheckAsync() = client.HealthCheckAsync()

    /// Search Graphiti knowledge graph
    member _.SearchAsync(query: string, ?numResults: int) =
        client.SearchAsync(query, ?numResults = numResults)

    /// Get all entities from Graphiti
    member _.GetEntitiesAsync() = client.GetEntitiesAsync()

    /// Get all facts from Graphiti
    member _.GetFactsAsync() = client.GetFactsAsync()

    /// Get communities from Graphiti
    member _.GetCommunitiesAsync() = client.GetCommunitiesAsync()

    interface IDisposable with
        member _.Dispose() = (client :> IDisposable).Dispose()

/// Create an ingestion service with default config
let createService () =
    new EpisodeIngestionService(IngestionConfig.defaults)

/// Create an ingestion service with custom Graphiti URL
let createServiceWithUrl (url: string) =
    new EpisodeIngestionService(
        { IngestionConfig.defaults with
            GraphitiUrl = url }
    )
