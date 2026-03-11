namespace Tars.Core

open System
open System.IO
open Tars.Core.TemporalKnowledgeGraph

/// Internal implementation of the Knowledge Graph Service
type InternalGraphService(storagePath: string) =
    let graph = TemporalGraph()
    let graphFile = Path.Combine(storagePath, "knowledge_graph.json")

    do
        // Ensure directory exists
        if not (Directory.Exists storagePath) then
            Directory.CreateDirectory storagePath |> ignore

        // Try to load existing graph
        if File.Exists graphFile then
            let loaded = graph.Load(graphFile)

            if loaded then
                printfn $"[GraphService] Loaded knowledge graph from %s{graphFile}"
            else
                printfn "[GraphService] Failed to load graph, starting fresh."
    
    member this.Graph = graph

    interface IGraphService with
        member this.AddNodeAsync(entity: TarsEntity) =
            task {
                let id = graph.AddNode(entity)
                graph.Save(graphFile)
                return id
            }

        member this.AddFactAsync(fact: TarsFact) =
            task {
                let id = graph.AddFact(fact)
                // Autosave for now
                graph.Save(graphFile)
                return id
            }

        member this.AddEpisodeAsync(episode: Episode) =
            task {
                let id = EpisodeIngestor.ingestEpisode graph episode
                graph.Save(graphFile)
                return id
            }

        member this.QueryAsync(query: string) =
            task {
                // Determine query intent
                if query = "*" then
                    return graph.GetCurrentFacts()
                else
                    // Basic entity search (naive)
                    let facts = graph.GetCurrentFacts()

                    return
                        facts
                        |> List.filter (fun f ->
                            let s = TarsFact.source f

                            match s with
                            | ConceptE c -> c.Name.Contains(query, StringComparison.OrdinalIgnoreCase)
                            | CodeModuleE m -> m.Path.Contains(query, StringComparison.OrdinalIgnoreCase)
                            | EpisodeE e ->
                                // Simplistic search in episode content
                                match e with
                                | Episode.UserMessage(content, _, _) ->
                                    content.Contains(query, StringComparison.OrdinalIgnoreCase)
                                | Episode.AgentInteraction(_, content, output, _) ->
                                    content.Contains(query, StringComparison.OrdinalIgnoreCase)
                                    || output.Contains(query, StringComparison.OrdinalIgnoreCase)
                                | _ -> false
                            | _ -> false)
            }

        member this.PersistAsync() = task { graph.Save(graphFile) }
