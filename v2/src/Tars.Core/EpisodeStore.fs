namespace Tars.Core

// Episode storage for Graphiti-style non-lossy data ingestion

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

/// Episode store for persisting raw episodes (Graphiti's Gₑ layer)
type EpisodeStore(basePath: string) =
    let episodes = List<Guid * Episode>()
    let indexByType = Dictionary<string, List<Guid>>()
    let indexByTime = SortedList<DateTime, List<Guid>>()

    let jsonOptions =
        let opts = JsonSerializerOptions()
        opts.WriteIndented <- true
        opts.Converters.Add(JsonFSharpConverter())
        opts

    let ensureDirectory () =
        if not (Directory.Exists basePath) then
            Directory.CreateDirectory basePath |> ignore

    let episodesFile = Path.Combine(basePath, "episodes.json")

    let addToTimeIndex (id: Guid) (timestamp: DateTime) =
        let key = timestamp.Date // Index by day for efficiency

        if not (indexByTime.ContainsKey key) then
            indexByTime.[key] <- List<Guid>()

        indexByTime.[key].Add(id)

    let addToTypeIndex (id: Guid) (typeTag: string) =
        if not (indexByType.ContainsKey typeTag) then
            indexByType.[typeTag] <- List<Guid>()

        indexByType.[typeTag].Add(id)

    do
        ensureDirectory ()
        // Load existing episodes
        if File.Exists episodesFile then
            try
                let json = File.ReadAllText episodesFile
                let loaded = JsonSerializer.Deserialize<(Guid * Episode) list>(json, jsonOptions)

                for (id, ep) in loaded do
                    episodes.Add((id, ep))
                    addToTypeIndex id (Episode.typeTag ep)
                    addToTimeIndex id (Episode.timestamp ep)
            with ex ->
                printfn $"Warning: Could not load episodes: {ex.Message}"

    /// Ingest a new episode, returns its ID
    member this.Ingest(episode: Episode) : Guid =
        let id = Guid.NewGuid()
        episodes.Add((id, episode))
        addToTypeIndex id (Episode.typeTag episode)
        addToTimeIndex id (Episode.timestamp episode)
        id

    /// Get episode by ID
    member this.Get(id: Guid) : Episode option =
        episodes |> Seq.tryFind (fun (i, _) -> i = id) |> Option.map snd

    /// Get all episodes
    member this.GetAll() : (Guid * Episode) list = episodes |> Seq.toList

    /// Get episodes by time range
    member this.GetByTimeRange(startTime: DateTime, endTime: DateTime) : (Guid * Episode) list =
        episodes
        |> Seq.filter (fun (_, ep) ->
            let ts = Episode.timestamp ep
            ts >= startTime && ts <= endTime)
        |> Seq.toList

    /// Get episodes by type
    member this.GetByType(typeTag: string) : (Guid * Episode) list =
        match indexByType.TryGetValue typeTag with
        | true, ids ->
            ids
            |> Seq.choose (fun id -> this.Get id |> Option.map (fun ep -> (id, ep)))
            |> Seq.toList
        | false, _ -> []

    /// Get recent episodes (last N)
    member this.GetRecent(count: int) : (Guid * Episode) list =
        episodes
        |> Seq.sortByDescending (fun (_, ep) -> Episode.timestamp ep)
        |> Seq.truncate count
        |> Seq.toList

    /// Count of all episodes
    member this.Count = episodes.Count

    /// Save all episodes to disk
    member this.Save() =
        ensureDirectory ()
        let json = JsonSerializer.Serialize(episodes |> Seq.toList, jsonOptions)
        File.WriteAllText(episodesFile, json)

    /// Clear all episodes (for testing)
    member this.Clear() =
        episodes.Clear()
        indexByType.Clear()
        indexByTime.Clear()

/// Module with helper functions for EpisodeStore
[<AutoOpen>]
module EpisodeStoreHelpers =
    /// Create a store at the default location
    let createDefaultEpisodeStore () =
        let path = Path.Combine(Environment.CurrentDirectory, ".tars", "episodes")
        EpisodeStore(path)

    /// Create a store at a specific path
    let createEpisodeStore (path: string) = EpisodeStore(path)

    /// Ingest an agent interaction episode
    let ingestAgentInteraction (store: EpisodeStore) agentId input output =
        AgentInteraction(agentId, input, output, DateTime.UtcNow) |> store.Ingest

    /// Ingest a code change episode
    let ingestCodeChange (store: EpisodeStore) file diff author =
        CodeChange(file, diff, author, DateTime.UtcNow) |> store.Ingest

    /// Ingest a reflection episode
    let ingestReflection (store: EpisodeStore) agentId content =
        Reflection(agentId, content, DateTime.UtcNow) |> store.Ingest

    /// Ingest a tool call episode
    let ingestToolCall (store: EpisodeStore) name args result =
        ToolCall(name, args, result, DateTime.UtcNow) |> store.Ingest

    /// Ingest a belief update episode
    let ingestBeliefUpdate (store: EpisodeStore) agentId belief confidence =
        BeliefUpdate(agentId, belief, confidence, DateTime.UtcNow) |> store.Ingest
