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

    /// <summary>Get all nodes in the graph</summary>
    member this.GetAllNodes() = adjacencyList.Keys |> Seq.toList

    /// <summary>Get node count</summary>
    member this.NodeCount = adjacencyList.Count

    /// <summary>Get edge count</summary>
    member this.EdgeCount =
        adjacencyList.Values
        |> Seq.sumBy (fun bag -> bag.Count)

    /// <summary>Find all nodes within N hops from a starting node</summary>
    /// <param name="startNode">The starting node</param>
    /// <param name="maxHops">Maximum number of hops to traverse</param>
    /// <returns>List of (node, hop distance, path) tuples</returns>
    member this.MultiHopTraverse(startNode: GraphNode, maxHops: int) =
        let visited = System.Collections.Generic.Dictionary<GraphNode, int>()
        let results = ResizeArray<GraphNode * int * (GraphNode * GraphEdge) list>()
        let queue = System.Collections.Generic.Queue<GraphNode * int * (GraphNode * GraphEdge) list>()

        queue.Enqueue((startNode, 0, []))
        visited.[startNode] <- 0
        results.Add((startNode, 0, []))

        while queue.Count > 0 do
            let (current, depth, path) = queue.Dequeue()

            if depth < maxHops then
                for (neighbor, edge) in this.GetNeighbors(current) do
                    let newPath = path @ [(neighbor, edge)]
                    let newDepth = depth + 1

                    // Add if not visited or found at shorter depth
                    match visited.TryGetValue(neighbor) with
                    | true, existingDepth when existingDepth <= newDepth -> ()
                    | _ ->
                        visited.[neighbor] <- newDepth
                        results.Add((neighbor, newDepth, newPath))
                        queue.Enqueue((neighbor, newDepth, newPath))

        results |> Seq.toList

    /// <summary>Check if node is a Concept</summary>
    member private _.IsConcept(node: GraphNode) =
        match node with
        | Concept _ -> true
        | _ -> false

    /// <summary>Check if node is a FileNode</summary>
    member private _.IsFileNode(node: GraphNode) =
        match node with
        | FileNode _ -> true
        | _ -> false

    /// <summary>Check if edge is RelatesTo</summary>
    member private _.IsRelatesTo(edge: GraphEdge) =
        match edge with
        | RelatesTo _ -> true
        | _ -> false

    /// <summary>Find concept nodes within N hops</summary>
    member this.FindRelatedConcepts(startNode: GraphNode, maxHops: int) =
        this.MultiHopTraverse(startNode, maxHops)
        |> List.filter (fun (node, _, _) -> this.IsConcept(node))

    /// <summary>Find file nodes within N hops</summary>
    member this.FindRelatedFiles(startNode: GraphNode, maxHops: int) =
        this.MultiHopTraverse(startNode, maxHops)
        |> List.filter (fun (node, _, _) -> this.IsFileNode(node))

    /// <summary>Find nodes connected by RelatesTo edges within N hops</summary>
    member this.FindByRelatesTo(startNode: GraphNode, maxHops: int) =
        this.MultiHopTraverse(startNode, maxHops)
        |> List.filter (fun (_, _, path) ->
            path |> List.exists (fun (_, edge) -> this.IsRelatesTo(edge)))

    /// <summary>Get strongly connected concepts (nodes with many connections)</summary>
    member this.GetHubNodes(minConnections: int) =
        adjacencyList
        |> Seq.filter (fun kvp -> kvp.Value.Count >= minConnections)
        |> Seq.map (fun kvp -> kvp.Key)
        |> Seq.toList

    /// <summary>Find all paths between two nodes up to max length</summary>
    member this.FindAllPaths(startNode: GraphNode, endNode: GraphNode, maxLength: int) =
        let allPaths = ResizeArray<(GraphNode * GraphEdge) list>()

        let rec dfs current path visited =
            if current = endNode then
                allPaths.Add(path |> List.rev)
            elif path.Length < maxLength && not (Set.contains current visited) then
                let newVisited = Set.add current visited
                for (neighbor, edge) in this.GetNeighbors(current) do
                    dfs neighbor ((neighbor, edge) :: path) newVisited

        dfs startNode [] Set.empty
        allPaths |> Seq.toList

    /// <summary>Build graph from document relationships</summary>
    /// <param name="documents">List of (id, title, related_concepts)</param>
    member this.IndexDocuments(documents: (string * string * string list) list) =
        for (docId, _, concepts) in documents do
            let docNode = FileNode docId
            this.AddNode(docNode)

            for concept in concepts do
                let conceptNode = Concept concept
                this.AddNode(conceptNode)

                // Document -> RelatesTo -> Concept
                let edge = RelatesTo 1.0
                this.AddEdge(docNode, conceptNode, edge)

        // Link related concepts (concepts that appear in the same document)
        let conceptDocs = System.Collections.Generic.Dictionary<string, ResizeArray<string>>()
        for (docId, _, concepts) in documents do
            for concept in concepts do
                match conceptDocs.TryGetValue(concept) with
                | true, docs -> docs.Add(docId)
                | false, _ ->
                    let docs = ResizeArray<string>()
                    docs.Add(docId)
                    conceptDocs.[concept] <- docs

        // Create edges between concepts that co-occur
        let concepts = conceptDocs.Keys |> Seq.toArray
        for i in 0 .. concepts.Length - 2 do
            for j in i + 1 .. concepts.Length - 1 do
                let c1, c2 = concepts.[i], concepts.[j]
                let docs1 = Set.ofSeq conceptDocs.[c1]
                let docs2 = Set.ofSeq conceptDocs.[c2]
                let overlap = Set.intersect docs1 docs2
                if overlap.Count > 0 then
                    let n1 = Concept c1
                    let n2 = Concept c2
                    let weight = float overlap.Count / float (max docs1.Count docs2.Count)
                    let edge = RelatesTo weight
                    this.AddEdge(n1, n2, edge)
                    this.AddEdge(n2, n1, edge)
