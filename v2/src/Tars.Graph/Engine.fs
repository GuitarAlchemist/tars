namespace Tars.Graph

open System.Collections.Concurrent
open System.IO
open System.Text.Json
open System.Threading.Tasks
open System.Collections.Generic

type GraphData =
    { Nodes: GraphNode list
      Edges: (GraphNode * GraphNode * GraphEdge) list }

type Graph() =
    let adjacency =
        ConcurrentDictionary<GraphNode, ConcurrentBag<GraphNode * GraphEdge>>()

    member this.AddNode(node: GraphNode) =
        adjacency.TryAdd(node, ConcurrentBag()) |> ignore

    member this.AddEdge(fromNode: GraphNode, toNode: GraphNode, edge: GraphEdge) =
        this.AddNode(fromNode)
        this.AddNode(toNode)

        match adjacency.TryGetValue(fromNode) with
        | true, edges -> edges.Add((toNode, edge))
        | false, _ -> ()

    member this.GetNeighbors(node: GraphNode) =
        match adjacency.TryGetValue(node) with
        | true, edges -> edges |> Seq.toList
        | false, _ -> []

    member this.Nodes = adjacency.Keys |> Seq.toList

    member this.FindPath(startNode: GraphNode, endNode: GraphNode) =
        let visited = HashSet<GraphNode>()
        let queue = Queue<GraphNode * GraphNode list>()
        queue.Enqueue((startNode, [ startNode ]))
        visited.Add(startNode) |> ignore

        let rec bfs () =
            if queue.Count = 0 then
                None
            else
                let (current, path) = queue.Dequeue()

                if current = endNode then
                    Some path
                else
                    let neighbors = this.GetNeighbors(current)

                    for (neighbor, _) in neighbors do
                        if visited.Add(neighbor) then
                            queue.Enqueue((neighbor, path @ [ neighbor ]))

                    bfs ()

        bfs ()

    member this.SaveGraphAsync(path: string) =
        task {
            let edges =
                adjacency
                |> Seq.collect (fun kvp -> kvp.Value |> Seq.map (fun (target, edge) -> kvp.Key, target, edge))
                |> Seq.toList

            let data = { Nodes = this.Nodes; Edges = edges }
            let options = JsonSerializerOptions(WriteIndented = true)
            let json = JsonSerializer.Serialize(data, options)
            do! File.WriteAllTextAsync(path, json)
        }

    member this.LoadGraphAsync(path: string) =
        task {
            if File.Exists(path) then
                let! json = File.ReadAllTextAsync(path)

                try
                    let data = JsonSerializer.Deserialize<GraphData>(json)
                    adjacency.Clear()

                    for node in data.Nodes do
                        this.AddNode(node)

                    for (fromNode, toNode, edge) in data.Edges do
                        this.AddEdge(fromNode, toNode, edge)

                    return true
                with _ ->
                    return false
            else
                return false
        }
