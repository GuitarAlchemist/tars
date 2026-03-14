namespace Tars.Tools.Graph

open System.Text.Json
open Tars.Tools

/// Tools for graph operations and knowledge queries
module GraphTools =

    // In-memory graph store for demo/testing
    let private nodes =
        System.Collections.Concurrent.ConcurrentDictionary<string, JsonElement>()

    let private edges =
        System.Collections.Concurrent.ConcurrentDictionary<string, JsonElement>()

    [<TarsToolAttribute("graph_add_node",
                        "Adds a node to the knowledge graph. Input JSON: { \"id\": \"b:101\", \"type\": \"Belief\", \"label\": \"X causes Y\", \"confidence\": 0.85 }")>]
    let graphAddNode (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let id = root.GetProperty("id").GetString()
                nodes.[id] <- root.Clone()

                printfn $"➕ Added node: {id}"
                return $"Node added: {id}"
            with ex ->
                return $"graph_add_node error: {ex.Message}"
        }

    [<TarsToolAttribute("graph_add_edge",
                        "Adds an edge to the knowledge graph. Input JSON: { \"source\": \"b:101\", \"target\": \"c:hazard\", \"type\": \"mentions\", \"weight\": 1.0 }")>]
    let graphAddEdge (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let source = root.GetProperty("source").GetString()
                let target = root.GetProperty("target").GetString()
                let edgeId = $"{source}->{target}"
                edges.[edgeId] <- root.Clone()

                printfn $"🔗 Added edge: {source} -> {target}"
                return $"Edge added: {edgeId}"
            with ex ->
                return $"graph_add_edge error: {ex.Message}"
        }

    [<TarsToolAttribute("graph_get_neighborhood",
                        "Gets the neighborhood around a node. Input JSON: { \"id\": \"b:101\", \"depth\": 2, \"limit\": 50 }")>]
    let graphGetNeighborhood (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let id = root.GetProperty("id").GetString()

                let depth =
                    let mutable p = Unchecked.defaultof<JsonElement>
                    if root.TryGetProperty("depth", &p) then p.GetInt32() else 2

                let limit =
                    let mutable p = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("limit", &p) then
                        p.GetInt32()
                    else
                        50

                printfn $"🔍 Getting neighborhood: {id} (depth={depth}, limit={limit})"

                // Find connected nodes (simplified BFS)
                let visited = System.Collections.Generic.HashSet<string>()
                let queue = System.Collections.Generic.Queue<string * int>()
                queue.Enqueue((id, 0))

                let resultNodes = ResizeArray<JsonElement>()
                let resultEdges = ResizeArray<JsonElement>()

                while queue.Count > 0 && resultNodes.Count < limit do
                    let (nodeId, d) = queue.Dequeue()

                    if not (visited.Contains nodeId) && d <= depth then
                        visited.Add(nodeId) |> ignore

                        // Add node if exists
                        let mutable node = Unchecked.defaultof<JsonElement>

                        if nodes.TryGetValue(nodeId, &node) then
                            resultNodes.Add(node)

                        // Find connected edges
                        for kvp in edges do
                            let edge = kvp.Value
                            let mutable srcProp = Unchecked.defaultof<JsonElement>
                            let mutable tgtProp = Unchecked.defaultof<JsonElement>

                            if
                                edge.TryGetProperty("source", &srcProp)
                                && edge.TryGetProperty("target", &tgtProp)
                            then
                                let src = srcProp.GetString()
                                let tgt = tgtProp.GetString()

                                if src = nodeId then
                                    resultEdges.Add(edge)

                                    if d < depth then
                                        queue.Enqueue((tgt, d + 1))
                                elif tgt = nodeId then
                                    resultEdges.Add(edge)

                                    if d < depth then
                                        queue.Enqueue((src, d + 1))

                // Build response
                let nodesJson =
                    resultNodes |> Seq.map (fun n -> n.GetRawText()) |> String.concat ","

                let edgesJson =
                    resultEdges
                    |> Seq.distinct
                    |> Seq.map (fun e -> e.GetRawText())
                    |> String.concat ","

                return $"{{\"nodes\":[{nodesJson}],\"edges\":[{edgesJson}]}}"
            with ex ->
                return $"graph_get_neighborhood error: {ex.Message}"
        }

    [<TarsToolAttribute("graph_query",
                        "Queries the knowledge graph. Input JSON: { \"filter\": { \"type\": \"Belief\" }, \"limit\": 20 }")>]
    let graphQuery (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let limit =
                    let mutable p = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("limit", &p) then
                        p.GetInt32()
                    else
                        20

                let mutable filterProp = Unchecked.defaultof<JsonElement>
                let hasFilter = root.TryGetProperty("filter", &filterProp)

                let matchesFilter (node: JsonElement) =
                    if not hasFilter then
                        true
                    else
                        let mutable matches = true

                        for prop in filterProp.EnumerateObject() do
                            let mutable nodeProp = Unchecked.defaultof<JsonElement>

                            if node.TryGetProperty(prop.Name, &nodeProp) then
                                if nodeProp.GetRawText() <> prop.Value.GetRawText() then
                                    matches <- false
                            else
                                matches <- false

                        matches

                let results =
                    nodes.Values
                    |> Seq.filter matchesFilter
                    |> Seq.truncate limit
                    |> Seq.map (fun n -> n.GetRawText())
                    |> String.concat ","

                return $"{{\"nodes\":[{results}],\"count\":{nodes.Count}}}"
            with ex ->
                return $"graph_query error: {ex.Message}"
        }

    [<TarsToolAttribute("graph_stats", "Gets statistics about the knowledge graph. No input required.")>]
    let graphStats (_: string) =
        task {
            let nodeTypes =
                nodes.Values
                |> Seq.choose (fun n ->
                    let mutable p = Unchecked.defaultof<JsonElement>

                    if n.TryGetProperty("type", &p) then
                        Some(p.GetString())
                    else
                        None)
                |> Seq.countBy id
                |> Seq.map (fun (t, c) -> $"  - {t}: {c}")
                |> String.concat "\n"

            let edgeTypes =
                edges.Values
                |> Seq.choose (fun e ->
                    let mutable p = Unchecked.defaultof<JsonElement>

                    if e.TryGetProperty("type", &p) then
                        Some(p.GetString())
                    else
                        None)
                |> Seq.countBy id
                |> Seq.map (fun (t, c) -> $"  - {t}: {c}")
                |> String.concat "\n"

            return
                $"# Knowledge Graph Statistics\n\n**Nodes:** {nodes.Count}\n{nodeTypes}\n\n**Edges:** {edges.Count}\n{edgeTypes}"
        }

    [<TarsToolAttribute("graph_export_json",
                        "Exports the entire graph as JSON for visualization. Input JSON: { \"format\": \"3d-force\" }")>]
    let graphExportJson (args: string) =
        task {
            try
                let nodesJson =
                    nodes |> Seq.map (fun kvp -> kvp.Value.GetRawText()) |> String.concat ","

                let edgesJson =
                    edges |> Seq.map (fun kvp -> kvp.Value.GetRawText()) |> String.concat ","

                return $"{{\"nodes\":[{nodesJson}],\"edges\":[{edgesJson}]}}"
            with ex ->
                return $"graph_export_json error: {ex.Message}"
        }

    [<TarsToolAttribute("graph_find_contradictions",
                        "Finds beliefs that contradict each other. Input JSON: { \"threshold\": 0.5 }")>]
    let graphFindContradictions (args: string) =
        task {
            try
                // Find edges of type "contradicts"
                let contradictions =
                    edges.Values
                    |> Seq.choose (fun e ->
                        let mutable typeProp = Unchecked.defaultof<JsonElement>

                        if e.TryGetProperty("type", &typeProp) && typeProp.GetString() = "contradicts" then
                            let mutable srcProp = Unchecked.defaultof<JsonElement>
                            let mutable tgtProp = Unchecked.defaultof<JsonElement>

                            if e.TryGetProperty("source", &srcProp) && e.TryGetProperty("target", &tgtProp) then
                                Some(srcProp.GetString(), tgtProp.GetString())
                            else
                                None
                        else
                            None)
                    |> Seq.toList

                if contradictions.IsEmpty then
                    return "No contradictions found in the graph"
                else
                    let report =
                        contradictions
                        |> List.mapi (fun i (src, tgt) -> $"{i + 1}. {src} ⚡ contradicts ⚡ {tgt}")
                        |> String.concat "\n"

                    return $"# Contradictions Found ({contradictions.Length})\n\n{report}"
            with ex ->
                return $"graph_find_contradictions error: {ex.Message}"
        }

    [<TarsToolAttribute("graph_clear", "Clears the in-memory graph. Use with caution! No input required.")>]
    let graphClear (_: string) =
        task {
            nodes.Clear()
            edges.Clear()
            printfn "🗑️ Graph cleared"
            return "Graph cleared successfully"
        }
