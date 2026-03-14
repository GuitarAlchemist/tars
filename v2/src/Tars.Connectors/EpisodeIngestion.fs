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

    | Episode.CognitiveStateUpdate(runId, mode, entropy, stability, ts) ->
        { Name = $"Cognitive State: {runId}"
          Content = $"Mode: {mode}\nEntropy: {entropy}\nStability: {stability}"
          Source = "tars_cognitive"
          SourceDescription = Some $"Cognitive state update during run {runId}"
          ReferenceTime = Some ts }

/// Convert TARS Episode to Graphiti MessageDto (for /messages endpoint)
let toGraphitiMessage (episode: Episode) : MessageDto =
    let content, roleType, role, sourceDesc, ts =
        match episode with
        | Episode.AgentInteraction(agentId, input, output, ts) ->
            $"User: {input}\n\nAgent: {output}", "assistant", agentId, Some "agent_interaction", ts
        | Episode.CodeChange(file, _, author, ts) -> $"Code change in {file}", "system", author, Some "code_change", ts
        | Episode.Reflection(agentId, content, ts) -> content, "assistant", agentId, Some "reflection", ts
        | Episode.UserMessage(content, _, ts) -> content, "user", "user", Some "user_message", ts
        | Episode.ToolCall(name, _, result, ts) -> $"Tool {name}: {result}", "system", "tool", Some "tool_call", ts
        | Episode.BeliefUpdate(agentId, belief, _, ts) -> belief, "assistant", agentId, Some "belief_update", ts
        | Episode.PatternDetected(patternType, location, _, ts) ->
            $"{patternType} at {location}", "system", "pattern_detector", Some "pattern", ts
        | Episode.CognitiveStateUpdate(runId, mode, entropy, _, ts) ->
            $"Cognitive State {runId}: {mode} (Entropy {entropy:F2})", "system", "cortex", Some "cognitive_state", ts

    { Content = content
      RoleType = roleType
      Role = role
      Timestamp = Some ts
      SourceDescription = sourceDesc
      Uuid = None }

/// Interface for episode ingestion services
type IEpisodeIngestionService =
    inherit IDisposable
    abstract member Queue: Episode -> unit
    abstract member FlushAsync: unit -> Task<Result<int, string>>
    abstract member IngestAsync: Episode -> Task<Result<unit, string>>
    abstract member SearchAsync: string * int option -> Task<Result<SearchResultDto list, string>>
    abstract member HealthCheckAsync: unit -> Task<Result<string, string>>
    // For full compatibility, though Local might mock these
    abstract member GetEntitiesAsync: unit -> Task<Result<EntityDto list, string>>
    abstract member GetFactsAsync: unit -> Task<Result<FactDto list, string>>
    abstract member GetCommunitiesAsync: unit -> Task<Result<CommunityDto list, string>>

/// Graphiti implementation
type GraphitiIngestionService(config: IngestionConfig) =
    let client = new GraphitiClient(Uri(config.GraphitiUrl))
    let mutable pendingEpisodes = ResizeArray<Episode>()
    let mutable totalIngested = 0
    let mutable lastFlush = DateTime.UtcNow

    interface IEpisodeIngestionService with
        member _.Queue(episode: Episode) =
            lock pendingEpisodes (fun () -> pendingEpisodes.Add(episode))

        member this.FlushAsync() =
            task {
                let episodesToFlush =
                    lock pendingEpisodes (fun () ->
                        let eps = pendingEpisodes.ToArray()
                        pendingEpisodes.Clear()
                        eps)

                if episodesToFlush.Length = 0 then
                    return Ok 0
                else
                    // Convert episodes to messages and send in batch
                    let messages = episodesToFlush |> Array.map toGraphitiMessage
                    let! result = client.AddMessagesAsync("tars", messages)

                    match result with
                    | Ok _ ->
                        totalIngested <- totalIngested + messages.Length
                        lastFlush <- DateTime.UtcNow
                        return Ok messages.Length
                    | Error err -> return Error err
            }

        member _.IngestAsync(episode: Episode) =
            task {
                let message = toGraphitiMessage episode
                let! result = client.AddMessagesAsync("tars", [| message |])

                match result with
                | Ok _ ->
                    totalIngested <- totalIngested + 1
                    return Ok()
                | Error err -> return Error err
            }

        member _.SearchAsync(query: string, numResults: int option) =
            client.SearchAsync(query, ?numResults = numResults)

        member _.HealthCheckAsync() = client.HealthCheckAsync()
        member _.GetEntitiesAsync() = client.GetEntitiesAsync()
        member _.GetFactsAsync() = client.GetFactsAsync()
        member _.GetCommunitiesAsync() = client.GetCommunitiesAsync()
        member _.Dispose() = (client :> IDisposable).Dispose()

open Tars.Core

/// Local implementation using InternalGraphService
type LocalIngestionService(graphService: IGraphService) =
    let mutable pendingEpisodes = ResizeArray<Episode>()

    interface IEpisodeIngestionService with
        member _.Queue(episode: Episode) =
            lock pendingEpisodes (fun () -> pendingEpisodes.Add(episode))

        member this.FlushAsync() =
            task {
                let episodesToFlush =
                    lock pendingEpisodes (fun () ->
                        let eps = pendingEpisodes.ToArray()
                        pendingEpisodes.Clear()
                        eps)

                for ep in episodesToFlush do
                    let! _ = graphService.AddEpisodeAsync(ep)
                    ()

                return Ok episodesToFlush.Length
            }

        member _.IngestAsync(episode: Episode) =
            task {
                let! _ = graphService.AddEpisodeAsync(episode)
                return Ok()
            }

        member _.SearchAsync(query: string, numResults: int option) =
            task {
                let! facts = graphService.QueryAsync(query)

                let results =
                    facts
                    |> List.map (fun f ->
                        { Uuid = Guid.NewGuid().ToString() // Dummy
                          Name = (TarsFact.source f |> TarsEntity.getId)
                          Fact = Some $"%A{f}"
                          Score = 1.0 })

                return Ok results
            }

        member _.HealthCheckAsync() =
            Task.FromResult(Ok "Local Graph Service Operational")

        member _.GetEntitiesAsync() = Task.FromResult(Ok []) // Not fully implemented

        member _.GetFactsAsync() =
            task {
                let! facts = graphService.QueryAsync("*")
                // Map to FactDto if needed, or just partial
                return Ok []
            }

        member _.GetCommunitiesAsync() = Task.FromResult(Ok [])
        member _.Dispose() = ()

/// Create an ingestion service (Graphiti)
let createGraphitiService (config: IngestionConfig) =
    new GraphitiIngestionService(config) :> IEpisodeIngestionService

/// Create a local service
let createLocalService (graphService: IGraphService) =
    new LocalIngestionService(graphService) :> IEpisodeIngestionService

// Legacy factory for backward compatibility (defaults to Graphiti unless we change it globally later)
let createService () =
    createGraphitiService IngestionConfig.defaults

let createServiceWithUrl (url: string) =
    createGraphitiService
        { IngestionConfig.defaults with
            GraphitiUrl = url }
