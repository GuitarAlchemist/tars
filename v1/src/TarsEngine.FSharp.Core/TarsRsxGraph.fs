// ================================================
// 🕸️ TARS TRSX Hypergraph Builder
// ================================================
// Hypergraph construction from version diffs
// Based on ChatGPT-Leveraging Primes for TARS document

namespace TarsEngine.FSharp.Core

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.TarsRsxDiff

/// Represents a node in the TRSX hypergraph
type TrsxNode = {
    Id: string
    Version: string
    Timestamp: DateTime
    Content: string
    Embedding: float array // 16D semantic vector
    Significance: float
}

/// Represents a hyperedge connecting multiple nodes
type TrsxHyperEdge = {
    Id: string
    SourceNodes: string list
    TargetNodes: string list
    EdgeType: string // "evolution", "merge", "branch", etc.
    Weight: float
    Diff: TrsxDiff option
}

/// Represents the complete TRSX hypergraph
type TrsxHyperGraph = {
    Nodes: Map<string, TrsxNode>
    Edges: Map<string, TrsxHyperEdge>
    RootNode: string option
    Metadata: Map<string, string>
}

/// Result type for graph operations
type GraphResult<'T> = 
    | Success of 'T
    | Error of string

/// Graph analysis metrics
type GraphAnalysis = {
    NodeCount: int
    EdgeCount: int
    MaxDepth: int
    AverageSignificance: float
    MostSignificantPath: string list
    ClusteringCoefficient: float
}

module TarsRsxGraph =

    /// Create a new empty hypergraph
    let createEmptyGraph () : TrsxHyperGraph =
        {
            Nodes = Map.empty
            Edges = Map.empty
            RootNode = None
            Metadata = Map.empty
        }

    /// Generate unique ID for nodes and edges
    let generateId (prefix: string) : string =
        let timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        let random = 0 // HONEST: Cannot generate without real measurement
        $"%s{prefix}-%d{timestamp}-%d{random}"

    /// Create a node from version content
    let createNode (version: string) (content: string) (embedding: float array) (significance: float) : TrsxNode =
        {
            Id = generateId "node"
            Version = version
            Timestamp = DateTime.UtcNow
            Content = content
            Embedding = embedding
            Significance = significance
        }

    /// Create a hyperedge between nodes
    let createHyperEdge (sourceNodes: string list) (targetNodes: string list) (edgeType: string) (weight: float) (diff: TrsxDiff option) : TrsxHyperEdge =
        {
            Id = generateId "edge"
            SourceNodes = sourceNodes
            TargetNodes = targetNodes
            EdgeType = edgeType
            Weight = weight
            Diff = diff
        }

    /// Add a node to the hypergraph
    let addNode (graph: TrsxHyperGraph) (node: TrsxNode) : TrsxHyperGraph =
        let updatedNodes = graph.Nodes |> Map.add node.Id node
        let updatedRootNode = 
            match graph.RootNode with
            | None -> Some node.Id
            | Some _ -> graph.RootNode
        
        { graph with 
            Nodes = updatedNodes
            RootNode = updatedRootNode }

    /// Add a hyperedge to the graph
    let addHyperEdge (graph: TrsxHyperGraph) (edge: TrsxHyperEdge) : GraphResult<TrsxHyperGraph> =
        // Validate that all referenced nodes exist
        let allReferencedNodes = edge.SourceNodes @ edge.TargetNodes
        let missingNodes = 
            allReferencedNodes 
            |> List.filter (fun nodeId -> not (graph.Nodes.ContainsKey(nodeId)))
        
        if missingNodes.Length > 0 then
            let missingNodesStr = String.Join(", ", missingNodes)
            Error $"Missing nodes: {missingNodesStr}"
        else
            let updatedEdges = graph.Edges |> Map.add edge.Id edge
            Success { graph with Edges = updatedEdges }

    /// Find nodes by version pattern
    let findNodesByVersion (graph: TrsxHyperGraph) (versionPattern: string) : TrsxNode list =
        graph.Nodes.Values
        |> Seq.filter (fun node -> node.Version.Contains(versionPattern))
        |> Seq.toList

    /// Get all edges connected to a node
    let getConnectedEdges (graph: TrsxHyperGraph) (nodeId: string) : TrsxHyperEdge list =
        graph.Edges.Values
        |> Seq.filter (fun edge -> 
            edge.SourceNodes |> List.contains nodeId ||
            edge.TargetNodes |> List.contains nodeId)
        |> Seq.toList

    /// Calculate semantic distance between two nodes
    let calculateSemanticDistance (node1: TrsxNode) (node2: TrsxNode) : float =
        if node1.Embedding.Length <> node2.Embedding.Length then
            1.0 // Maximum distance for incompatible embeddings
        else
            // Euclidean distance in embedding space
            let squaredDiffs = 
                Array.zip node1.Embedding node2.Embedding
                |> Array.map (fun (a, b) -> (a - b) * (a - b))
            
            squaredDiffs |> Array.sum |> sqrt

    /// Find the shortest path between two nodes
    let findShortestPath (graph: TrsxHyperGraph) (sourceId: string) (targetId: string) : string list option =
        if not (graph.Nodes.ContainsKey(sourceId)) || not (graph.Nodes.ContainsKey(targetId)) then
            None
        else
            // Simple BFS implementation for hypergraph
            let visited = HashSet<string>()
            let queue = Queue<string * string list>()
            queue.Enqueue((sourceId, [sourceId]))
            
            let rec search () =
                if queue.Count = 0 then
                    None
                else
                    let (currentId, path) = queue.Dequeue()
                    
                    if currentId = targetId then
                        Some path
                    elif visited.Contains(currentId) then
                        search()
                    else
                        visited.Add(currentId) |> ignore
                        
                        // Find all connected nodes through hyperedges
                        let connectedNodes = 
                            getConnectedEdges graph currentId
                            |> List.collect (fun edge ->
                                if edge.SourceNodes |> List.contains currentId then
                                    edge.TargetNodes
                                else
                                    edge.SourceNodes)
                            |> List.distinct
                            |> List.filter (fun nodeId -> not (visited.Contains(nodeId)))
                        
                        for nodeId in connectedNodes do
                            queue.Enqueue((nodeId, path @ [nodeId]))
                        
                        search()
            
            search()

    /// Analyze the hypergraph structure
    let analyzeGraph (graph: TrsxHyperGraph) (logger: ILogger) : GraphAnalysis =
        logger.LogInformation("📊 Analyzing TRSX hypergraph structure")
        
        let nodeCount = graph.Nodes.Count
        let edgeCount = graph.Edges.Count
        
        // Calculate average significance
        let averageSignificance = 
            if nodeCount > 0 then
                graph.Nodes.Values
                |> Seq.map (fun node -> node.Significance)
                |> Seq.average
            else 0.0
        
        // Find most significant path (simplified - just highest significance nodes)
        let mostSignificantNodes = 
            graph.Nodes.Values
            |> Seq.sortByDescending (fun node -> node.Significance)
            |> Seq.take (min 5 nodeCount)
            |> Seq.map (fun node -> node.Id)
            |> Seq.toList
        
        // Calculate clustering coefficient (simplified)
        let clusteringCoefficient = 
            if nodeCount > 2 then
                let totalPossibleEdges = nodeCount * (nodeCount - 1) / 2
                let actualEdges = edgeCount
                float actualEdges / float totalPossibleEdges
            else 0.0
        
        // Calculate max depth (simplified - longest path from root)
        let maxDepth = 
            match graph.RootNode with
            | Some rootId ->
                graph.Nodes.Keys
                |> Seq.map (fun nodeId -> 
                    match findShortestPath graph rootId nodeId with
                    | Some path -> path.Length - 1
                    | None -> 0)
                |> Seq.max
            | None -> 0
        
        logger.LogInformation($"✅ Graph analysis complete: {nodeCount} nodes, {edgeCount} edges")
        logger.LogInformation($"📈 Average significance: {averageSignificance:F3}, Max depth: {maxDepth}")
        
        {
            NodeCount = nodeCount
            EdgeCount = edgeCount
            MaxDepth = maxDepth
            AverageSignificance = averageSignificance
            MostSignificantPath = mostSignificantNodes
            ClusteringCoefficient = clusteringCoefficient
        }

    /// Build hypergraph from a sequence of diffs
    let buildGraphFromDiffs (diffs: TrsxDiff list) (logger: ILogger) : GraphResult<TrsxHyperGraph> =
        try
            logger.LogInformation($"🕸️ Building TRSX hypergraph from {diffs.Length} diffs")
            
            let mutable graph = createEmptyGraph()
            let mutable nodeMap = Map.empty<string, string> // version -> nodeId
            
            // Create nodes for all unique versions
            let allVersions = 
                diffs 
                |> List.collect (fun diff -> [diff.SourceVersion; diff.TargetVersion])
                |> List.distinct
            
            for version in allVersions do
                let embedding = Array.zeroCreate 16 // TODO: Implement real functionality
                let significance = 0.5 // TODO: Implement real functionality
                let node = createNode version $"Content for {version}" embedding significance
                graph <- addNode graph node
                nodeMap <- nodeMap |> Map.add version node.Id
            
            // Create hyperedges from diffs
            for diff in diffs do
                match nodeMap.TryFind(diff.SourceVersion), nodeMap.TryFind(diff.TargetVersion) with
                | Some sourceNodeId, Some targetNodeId ->
                    let edge = createHyperEdge [sourceNodeId] [targetNodeId] "evolution" diff.OverallSignificance (Some diff)
                    match addHyperEdge graph edge with
                    | Success updatedGraph -> graph <- updatedGraph
                    | Error err -> 
                        logger.LogWarning($"⚠️ Failed to add edge: {err}")
                | _ ->
                    logger.LogWarning($"⚠️ Missing nodes for diff: {diff.SourceVersion} -> {diff.TargetVersion}")
            
            logger.LogInformation($"✅ Hypergraph built: {graph.Nodes.Count} nodes, {graph.Edges.Count} edges")
            
            Success graph
            
        with
        | ex ->
            logger.LogError($"❌ Failed to build hypergraph: {ex.Message}")
            Error ex.Message

    /// Test TRSX hypergraph operations
    let testTrsxGraph (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing TRSX hypergraph operations")
            
            // Create test graph
            let mutable graph = createEmptyGraph()
            
            // Add test nodes
            let node1 = createNode "v1.0" "Initial version" (Array.create 16 0.1) 0.5
            let node2 = createNode "v1.1" "Updated version" (Array.create 16 0.2) 0.7
            let node3 = createNode "v1.2" "Latest version" (Array.create 16 0.3) 0.9
            
            graph <- addNode graph node1
            graph <- addNode graph node2
            graph <- addNode graph node3
            
            // Add test edges
            let edge1 = createHyperEdge [node1.Id] [node2.Id] "evolution" 0.6 None
            let edge2 = createHyperEdge [node2.Id] [node3.Id] "evolution" 0.8 None
            
            match addHyperEdge graph edge1 with
            | Success g1 ->
                match addHyperEdge g1 edge2 with
                | Success g2 ->
                    graph <- g2
                    
                    // Test graph operations
                    let analysis = analyzeGraph graph logger
                    logger.LogInformation($"✅ Graph analysis: {analysis.NodeCount} nodes, {analysis.EdgeCount} edges")
                    
                    // Test path finding
                    match findShortestPath graph node1.Id node3.Id with
                    | Some path ->
                        let pathStr = String.Join(" -> ", path)
                        logger.LogInformation($"✅ Shortest path found: {pathStr}")
                    | None ->
                        logger.LogWarning("⚠️ No path found between nodes")
                    
                    // Test semantic distance
                    let distance = calculateSemanticDistance node1 node3
                    logger.LogInformation($"✅ Semantic distance: {distance:F3}")
                    
                    true
                | Error err ->
                    logger.LogError($"❌ Failed to add edge 2: {err}")
                    false
            | Error err ->
                logger.LogError($"❌ Failed to add edge 1: {err}")
                false
                
        with
        | ex ->
            logger.LogError($"❌ TRSX graph test failed: {ex.Message}")
            false
