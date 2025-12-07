namespace Tars.Core

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open FSharp.SystemTextJson

[<Obsolete("Use Tars.Core.TemporalKnowledgeGraph instead")>]
module LegacyKnowledgeGraph =
    type NodeId = string

    /// Get a stable ID for a GraphNode
    let getNodeId (node: GraphNode) : NodeId =
        match node with
        | Concept name -> $"concept:{name.ToLowerInvariant()}"
        | AgentNode(AgentId id) -> $"agent:{id}"
        | FileNode path -> $"file:{path.ToLowerInvariant().GetHashCode():x8}"
        | TaskNode id -> $"task:{id}"
        | BeliefNode(id, _) -> $"belief:{id}"
        | ModuleNode name -> $"module:{name}"
        | TypeNode name -> $"type:{name}"
        | FunctionNode name -> $"func:{name}"

    type EdgeId = string

    type GraphEdgeWrapper =
        { Id: EdgeId
          Source: NodeId
          Target: NodeId
          Type: GraphEdge
          Created: DateTime }

    type GraphNodeWrapper =
        { Id: NodeId
          Data: GraphNode
          ValidFrom: DateTime
          InvalidAt: DateTime option }

    type GraphEvent =
        | NodeAdded of GraphNodeWrapper
        | NodeRemoved of NodeId * DateTime
        | EdgeAdded of GraphEdgeWrapper
        | EdgeRemoved of EdgeId * DateTime

    type TemporalGraph() =
        let events = List<GraphEvent>()
        let mutable currentNodeState = Map.empty<NodeId, GraphNodeWrapper>
        let mutable currentEdgeState = Map.empty<EdgeId, GraphEdgeWrapper>
        let mutable adjacency = Map.empty<NodeId, EdgeId list>

        let updateAdjacency (edge: GraphEdgeWrapper) (add: bool) =
            let current = adjacency |> Map.tryFind edge.Source |> Option.defaultValue []

            let updated =
                if add then
                    edge.Id :: current
                else
                    current |> List.filter ((<>) edge.Id)

            adjacency <- adjacency |> Map.add edge.Source updated

        member this.NodeCount = currentNodeState.Count
        member this.EdgeCount = currentEdgeState.Count

        member this.Apply(event: GraphEvent) =
            events.Add(event)

            match event with
            | NodeAdded node -> currentNodeState <- currentNodeState |> Map.add node.Id node
            | NodeRemoved(id, _) -> currentNodeState <- currentNodeState |> Map.remove id
            // Also remove incident edges? For now, keep it simple.
            | EdgeAdded edge ->
                currentEdgeState <- currentEdgeState |> Map.add edge.Id edge
                updateAdjacency edge true
            | EdgeRemoved(id, _) ->
                match currentEdgeState.TryFind id with
                | Some edge ->
                    updateAdjacency edge false
                    currentEdgeState <- currentEdgeState |> Map.remove id
                | None -> ()

        member this.AddNode(node: GraphNode) =
            let id = getNodeId node

            if not (currentNodeState.ContainsKey id) then
                let wrapper =
                    { Id = id
                      Data = node
                      ValidFrom = DateTime.UtcNow
                      InvalidAt = None }

                this.Apply(NodeAdded wrapper)

        member this.AddEdge(source: GraphNode, target: GraphNode, edgeType: GraphEdge) =
            let sourceId = getNodeId source
            let targetId = getNodeId target

            // Ensure nodes exist
            this.AddNode source
            this.AddNode target

            let edge =
                { Id = Guid.NewGuid().ToString("N")
                  Source = sourceId
                  Target = targetId
                  Type = edgeType
                  Created = DateTime.UtcNow }

            this.Apply(EdgeAdded edge)

        member this.GetNodes() =
            currentNodeState.Values |> Seq.map (fun w -> w.Data) |> Seq.toList

        member this.GetEdges() = currentEdgeState.Values |> Seq.toList

        member this.GetNeighbors(node: GraphNode) =
            let id = getNodeId node

            match adjacency.TryFind id with
            | Some edgeIds ->
                edgeIds
                |> List.choose (fun eid -> currentEdgeState.TryFind eid)
                |> List.choose (fun edge ->
                    match currentNodeState.TryFind edge.Target with
                    | Some targetNode -> Some(targetNode.Data, edge.Type)
                    | None -> None)
            | None -> []

        member this.GetSnapshot(at: DateTime) =
            // Replay events up to 'at'
            let snapshotNodes = Dictionary<NodeId, GraphNodeWrapper>()
            let snapshotEdges = Dictionary<EdgeId, GraphEdgeWrapper>()

            for event in events do
                match event with
                | NodeAdded node when node.ValidFrom <= at -> snapshotNodes[node.Id] <- node
                | NodeRemoved(id, ts) when ts <= at -> snapshotNodes.Remove(id) |> ignore
                | EdgeAdded edge when edge.Created <= at -> snapshotEdges[edge.Id] <- edge
                | EdgeRemoved(id, ts) when ts <= at -> snapshotEdges.Remove(id) |> ignore
                | _ -> ()

            (snapshotNodes.Values |> Seq.map (fun n -> n.Data) |> Seq.toList,
             snapshotEdges.Values |> Seq.map (fun e -> e.Type) |> Seq.toList)

        member this.MultiHopTraverse(startNode: GraphNode, maxHops: int) =
            let visited = Dictionary<NodeId, int>()
            let results = ResizeArray<GraphNode * int * (GraphNode * GraphEdge) list>()
            let queue = Queue<GraphNode * int * (GraphNode * GraphEdge) list>()

            let startId = getNodeId startNode

            if currentNodeState.ContainsKey startId then
                queue.Enqueue((startNode, 0, []))
                visited.[startId] <- 0
                results.Add((startNode, 0, []))

                while queue.Count > 0 do
                    let (current, depth, path) = queue.Dequeue()

                    if depth < maxHops then
                        for (neighbor, edgeType) in this.GetNeighbors(current) do
                            let neighborId = getNodeId neighbor
                            let newPath = path @ [ (neighbor, edgeType) ]
                            let newDepth = depth + 1

                            match visited.TryGetValue(neighborId) with
                            | true, existingDepth when existingDepth <= newDepth -> ()
                            | _ ->
                                visited.[neighborId] <- newDepth
                                results.Add((neighbor, newDepth, newPath))
                                queue.Enqueue((neighbor, newDepth, newPath))

            results |> Seq.toList

        member this.IngestEpisode(trace: MemoryTrace) =
            let taskId =
                match Guid.TryParse trace.TaskId with
                | true, g -> g
                | _ -> Guid.NewGuid()

            let taskNode = TaskNode taskId
            this.AddNode taskNode

            match trace.Variables |> Map.tryFind "code_structure" with
            | Some(:? CodeStructure as cs) ->
                for m in cs.Modules do
                    this.AddEdge(taskNode, ModuleNode m, RelatesTo 1.0)

                for t in cs.Types do
                    this.AddEdge(taskNode, TypeNode t, RelatesTo 1.0)

                for f in cs.Functions do
                    this.AddEdge(taskNode, FunctionNode f, RelatesTo 1.0)
            | _ -> ()
