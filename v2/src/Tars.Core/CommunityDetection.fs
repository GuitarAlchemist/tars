namespace Tars.Core

open System
open Tars.Core.TemporalKnowledgeGraph

module CommunityDetection =

    /// Run Label Propagation Algorithm to detect communities
    /// Returns a map of NodeId -> CommunityId
    let labelPropagation (graph: TemporalGraph) (maxIterations: int) (snapshotTime: DateTime) : Map<string, string> =
        // 1. Get snapshot of valid nodes and edges
        let facts = graph.GetSnapshot(snapshotTime)
        
        // Build adjacency list (undirected for community detection)
        let adjacency = 
            facts 
            |> List.fold (fun acc fact ->
                let sourceId = TarsEntity.getId (TarsFact.source fact)
                let targetIdOpt = TarsFact.target fact |> Option.map TarsEntity.getId
                
                match targetIdOpt with
                | Some targetId ->
                    let acc1 = 
                        match Map.tryFind sourceId acc with
                        | Some neighbors -> Map.add sourceId (targetId :: neighbors) acc
                        | None -> Map.add sourceId [targetId] acc
                    
                    match Map.tryFind targetId acc1 with
                    | Some neighbors -> Map.add targetId (sourceId :: neighbors) acc1
                    | None -> Map.add targetId [sourceId] acc1
                | None -> acc
            ) Map.empty<string, string list>

        // 2. Initialize labels (each node is its own community)
        let allNodes = adjacency.Keys |> Seq.toList
        let initialLabels = allNodes |> List.map (fun n -> n, n) |> Map.ofList

        // 3. Iterate
        let rec iterate currentLabels iteration =
            if iteration >= maxIterations then
                currentLabels
            else
                // Shuffle nodes to prevent oscillation
                let rng = Random(iteration)
                let shuffledNodes = allNodes |> List.sortBy (fun _ -> rng.Next())

                let (newLabels, changed) =
                    shuffledNodes
                    |> List.fold (fun (labels, changed) nodeId ->
                        match Map.tryFind nodeId adjacency with
                        | Some neighbors ->
                            // Count neighbor labels
                            let neighborLabels = 
                                neighbors 
                                |> List.choose (fun n -> Map.tryFind n labels)
                            
                            if List.isEmpty neighborLabels then
                                (labels, changed)
                            else
                                // Find most frequent label
                                let mostFrequent =
                                    neighborLabels
                                    |> List.countBy id
                                    |> List.sortByDescending snd
                                    |> List.head
                                    |> fst
                                
                                let currentLabel = Map.find nodeId labels
                                if currentLabel <> mostFrequent then
                                    (Map.add nodeId mostFrequent labels, true)
                                else
                                    (labels, changed)
                        | None -> (labels, changed)
                    ) (currentLabels, false)

                if not changed then
                    newLabels
                else
                    iterate newLabels (iteration + 1)

        iterate initialLabels 0
