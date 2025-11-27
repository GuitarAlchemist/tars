namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Tars.Core
open System.Text.Json.Serialization

type KnowledgeGraph() =
    let adjacencyList =
        ConcurrentDictionary<GraphNode, ConcurrentBag<GraphNode * GraphEdge>>()

    member this.AddNode(node: GraphNode) =
        adjacencyList.TryAdd(node, ConcurrentBag()) |> ignore

    member this.AddEdge(source: GraphNode, target: GraphNode, edge: GraphEdge) =
        this.AddNode(source)
        this.AddNode(target)

        match adjacencyList.TryGetValue(source) with
        | true, edges -> edges.Add((target, edge))
        | false, _ -> ()

    member this.GetNeighbors(node: GraphNode) =
        match adjacencyList.TryGetValue(node) with
        | true, edges -> edges |> Seq.toList
        | false, _ -> []

    member this.FindPath(startNode: GraphNode, endNode: GraphNode) =
        let visited = System.Collections.Generic.HashSet<GraphNode>()

        let queue =
            System.Collections.Generic.Queue<GraphNode * List<GraphNode * GraphEdge>>()

        queue.Enqueue(startNode, [])
        visited.Add(startNode) |> ignore

        let mutable foundPath = None

        while queue.Count > 0 && foundPath.IsNone do
            let (current, path) = queue.Dequeue()

            if current = endNode then
                foundPath <- Some path
            else
                for (neighbor, edge) in this.GetNeighbors(current) do
                    if not (visited.Contains(neighbor)) then
                        visited.Add(neighbor) |> ignore
                        queue.Enqueue(neighbor, path @ [ (neighbor, edge) ])

        foundPath

    member this.PersistToFileAsync(path: string) =
        task {
            let data =
                adjacencyList
                |> Seq.map (fun kvp -> kvp.Key, kvp.Value |> Seq.toList)
                |> Seq.toList

            let options = JsonSerializerOptions(WriteIndented = true)
            options.Converters.Add(JsonFSharpConverter())
            let json = JsonSerializer.Serialize(data, options)
            do! File.WriteAllTextAsync(path, json)
        }

    member this.LoadFromFileAsync(path: string) =
        task {
            if File.Exists(path) then
                let! json = File.ReadAllTextAsync(path)
                let options = JsonSerializerOptions()
                options.Converters.Add(JsonFSharpConverter())

                let data =
                    JsonSerializer.Deserialize<List<GraphNode * List<GraphNode * GraphEdge>>>(json, options)

                adjacencyList.Clear()

                for (node, edges) in data do
                    let bag = ConcurrentBag(edges)
                    adjacencyList.TryAdd(node, bag) |> ignore

                return true
            else
                return false
        }
