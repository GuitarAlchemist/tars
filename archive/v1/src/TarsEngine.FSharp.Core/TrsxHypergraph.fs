namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Text.Json
open TarsEngine.FSharp.Core.HurwitzQuaternions

/// TRSX Hypergraph System for Version Tracking and Semantic Embedding
/// Implements advanced cognitive architecture for TARS metascript evolution
module TrsxHypergraph =

    /// TRSX file version node
    type TrsxNode = {
        Id: string
        FilePath: string
        Content: string
        Timestamp: DateTime
        Version: int
        Hash: string
        Metadata: Map<string, string>
    }

    /// Semantic difference between TRSX versions
    type SemanticDiff = {
        FromNode: string
        ToNode: string
        DiffType: string
        Changes: string list
        Similarity: float
        SemanticVector: float array  // 16D embedding
        Quaternion: HurwitzQuaternion
    }

    /// Hypergraph edge connecting multiple nodes
    type HypergraphEdge = {
        Id: string
        Nodes: string list
        EdgeType: string
        Weight: float
        Properties: Map<string, obj>
        Timestamp: DateTime
    }

    /// Complete TRSX hypergraph structure
    type TrsxHypergraph = {
        Nodes: Map<string, TrsxNode>
        Edges: Map<string, HypergraphEdge>
        SemanticDiffs: SemanticDiff list
        Embeddings: Map<string, float array>
        QuaternionMappings: Map<string, HurwitzQuaternion>
        EvolutionHistory: (DateTime * string) list
    }

    /// TRSX file parsing and analysis
    module TrsxParser =
        
        /// Parse TRSX file content
        let parseTrsxFile filePath =
            try
                let content = File.ReadAllText(filePath)
                let hash = content.GetHashCode().ToString()
                let fileInfo = FileInfo(filePath)
                
                Some {
                    Id = Path.GetFileNameWithoutExtension(filePath) + "_" + hash.[..7]
                    FilePath = filePath
                    Content = content
                    Timestamp = fileInfo.LastWriteTime
                    Version = 1  // Will be updated during graph building
                    Hash = hash
                    Metadata = Map.empty
                }
            with
            | ex -> 
                printfn "Error parsing TRSX file %s: %s" filePath ex.Message
                None
        
        /// Extract semantic features from TRSX content
        let extractSemanticFeatures (content: string) =
            let lines = content.Split('\n')
            let keywords = ["DESCRIBE"; "PATTERN"; "FSHARP"; "CSHARP"; "PYTHON"; "TASK"; "AGENT"]
            let features = Array.zeroCreate 16
            
            // Simple feature extraction (can be enhanced with ML)
            features.[0] <- float lines.Length / 100.0  // Content length
            features.[1] <- float (content.Split(' ').Length) / 1000.0  // Word count
            
            // Keyword presence
            keywords |> List.iteri (fun i keyword ->
                if i < 14 then  // Leave room for other features
                    features.[i + 2] <- if content.Contains(keyword) then 1.0 else 0.0
            )
            
            features
        
        /// Load all TRSX files from directory
        let loadTrsxDirectory directoryPath =
            if Directory.Exists(directoryPath) then
                Directory.GetFiles(directoryPath, "*.trsx", SearchOption.AllDirectories)
                |> Array.choose parseTrsxFile
                |> Array.toList
            else
                []

    /// Semantic difference computation
    module SemanticDiffEngine =
        
        /// Calculate Levenshtein distance between strings
        let levenshteinDistance (s1: string) (s2: string) =
            let len1, len2 = s1.Length, s2.Length
            let matrix = Array2D.zeroCreate (len1 + 1) (len2 + 1)
            
            for i in 0..len1 do matrix.[i, 0] <- i
            for j in 0..len2 do matrix.[0, j] <- j
            
            for i in 1..len1 do
                for j in 1..len2 do
                    let cost = if s1.[i-1] = s2.[j-1] then 0 else 1
                    matrix.[i, j] <- min (min (matrix.[i-1, j] + 1) (matrix.[i, j-1] + 1)) (matrix.[i-1, j-1] + cost)
            
            matrix.[len1, len2]
        
        /// Calculate semantic similarity between two TRSX nodes
        let calculateSimilarity node1 node2 =
            let distance = levenshteinDistance node1.Content node2.Content
            let maxLength = max node1.Content.Length node2.Content.Length
            if maxLength = 0 then 1.0 else 1.0 - (float distance / float maxLength)
        
        /// Generate semantic diff between two nodes
        let generateSemanticDiff node1 node2 =
            let similarity = calculateSimilarity node1 node2
            let features1 = TrsxParser.extractSemanticFeatures node1.Content
            let features2 = TrsxParser.extractSemanticFeatures node2.Content
            
            // Calculate difference vector
            let diffVector = Array.zip features1 features2 |> Array.map (fun (a, b) -> b - a)
            
            // Convert to quaternion for geometric reasoning
            let quat = HurwitzQuaternions.BeliefEncoding.encodeBelief 
                        (sprintf "diff_%s_%s" node1.Id node2.Id) 
                        similarity 
                        "semantic_diff"
            
            {
                FromNode = node1.Id
                ToNode = node2.Id
                DiffType = if similarity > 0.8 then "minor" elif similarity > 0.5 then "moderate" else "major"
                Changes = []  // Can be enhanced with detailed change detection
                Similarity = similarity
                SemanticVector = diffVector
                Quaternion = quat.Quaternion
            }
        
        /// Compute all pairwise semantic diffs
        let computeAllDiffs nodes =
            nodes
            |> List.allPairs nodes
            |> List.filter (fun (n1, n2) -> n1.Id <> n2.Id)
            |> List.map (fun (n1, n2) -> generateSemanticDiff n1 n2)

    /// 16D BSP (Binary Space Partitioning) for sedenion space
    module SedenionPartitioner =
        
        /// 16D point in sedenion space
        type SedenionPoint = {
            Id: string
            Coordinates: float array  // 16D coordinates
            Quaternion: HurwitzQuaternion
            Metadata: Map<string, obj>
        }
        
        /// BSP tree node for 16D space
        type BSPNode =
            | Leaf of points: SedenionPoint list
            | Branch of 
                dimension: int * 
                threshold: float * 
                left: BSPNode * 
                right: BSPNode
        
        /// Create sedenion point from TRSX node
        let createSedenionPoint (node: TrsxNode) =
            let features = TrsxParser.extractSemanticFeatures node.Content
            let quat = HurwitzQuaternions.BeliefEncoding.encodeBelief node.Content 1.0 "trsx_node"
            
            {
                Id = node.Id
                Coordinates = features
                Quaternion = quat.Quaternion
                Metadata = Map.ofList [("filePath", node.FilePath :> obj); ("timestamp", node.Timestamp :> obj)]
            }
        
        /// Build BSP tree for 16D points
        let rec buildBSPTree (points: SedenionPoint list) (depth: int) (maxDepth: int) =
            if points.Length <= 1 || depth >= maxDepth then
                Leaf points
            else
                let dimension = depth % 16
                let sortedPoints = points |> List.sortBy (fun p -> p.Coordinates.[dimension])
                let medianIndex = points.Length / 2
                let threshold = sortedPoints.[medianIndex].Coordinates.[dimension]
                
                let leftPoints = sortedPoints |> List.take medianIndex
                let rightPoints = sortedPoints |> List.skip medianIndex
                
                Branch (
                    dimension,
                    threshold,
                    buildBSPTree leftPoints (depth + 1) maxDepth,
                    buildBSPTree rightPoints (depth + 1) maxDepth
                )
        
        /// Query BSP tree for nearest neighbors
        let rec queryNearestNeighbors (tree: BSPNode) (queryPoint: SedenionPoint) (k: int) =
            let euclideanDistance p1 p2 =
                Array.zip p1.Coordinates p2.Coordinates
                |> Array.map (fun (a, b) -> (a - b) * (a - b))
                |> Array.sum
                |> sqrt
            
            match tree with
            | Leaf points ->
                points
                |> List.map (fun p -> (p, euclideanDistance p queryPoint))
                |> List.sortBy snd
                |> List.take (min k points.Length)
                |> List.map fst
            | Branch (dim, threshold, left, right) ->
                let goLeft = queryPoint.Coordinates.[dim] <= threshold
                let primaryResults = 
                    if goLeft then queryNearestNeighbors left queryPoint k
                    else queryNearestNeighbors right queryPoint k
                
                // Could also check other branch if needed for better results
                primaryResults

    /// Hypergraph construction and analysis
    module HypergraphBuilder =
        
        /// Build TRSX hypergraph from directory
        let buildHypergraphFromDirectory directoryPath =
            let nodes = TrsxParser.loadTrsxDirectory directoryPath
            let nodeMap = nodes |> List.map (fun n -> (n.Id, n)) |> Map.ofList
            
            // Generate semantic diffs
            let diffs = SemanticDiffEngine.computeAllDiffs nodes
            
            // Create embeddings
            let embeddings = 
                nodes 
                |> List.map (fun n -> (n.Id, TrsxParser.extractSemanticFeatures n.Content))
                |> Map.ofList
            
            // Create quaternion mappings
            let quaternionMappings =
                nodes
                |> List.map (fun n -> 
                    let quat = HurwitzQuaternions.BeliefEncoding.encodeBelief n.Content 1.0 "trsx_node"
                    (n.Id, quat.Quaternion))
                |> Map.ofList
            
            // Create hypergraph edges based on similarity
            let edges =
                diffs
                |> List.filter (fun d -> d.Similarity > 0.3)  // Only connect similar nodes
                |> List.mapi (fun i diff ->
                    let edgeId = sprintf "edge_%d" i
                    (edgeId, {
                        Id = edgeId
                        Nodes = [diff.FromNode; diff.ToNode]
                        EdgeType = "semantic_similarity"
                        Weight = diff.Similarity
                        Properties = Map.ofList [("diffType", diff.DiffType :> obj)]
                        Timestamp = DateTime.UtcNow
                    }))
                |> Map.ofList
            
            {
                Nodes = nodeMap
                Edges = edges
                SemanticDiffs = diffs
                Embeddings = embeddings
                QuaternionMappings = quaternionMappings
                EvolutionHistory = [(DateTime.UtcNow, "hypergraph_created")]
            }
        
        /// Analyze hypergraph patterns
        let analyzeHypergraphPatterns hypergraph =
            let nodeCount = hypergraph.Nodes.Count
            let edgeCount = hypergraph.Edges.Count
            let avgSimilarity = 
                if hypergraph.SemanticDiffs.IsEmpty then 0.0
                else hypergraph.SemanticDiffs |> List.averageBy (fun d -> d.Similarity)
            
            let primeQuaternions = 
                hypergraph.QuaternionMappings.Values
                |> Seq.filter HurwitzQuaternions.PrimeTesting.hasPrimeNorm
                |> Seq.length
            
            {|
                NodeCount = nodeCount
                EdgeCount = edgeCount
                AverageSimilarity = avgSimilarity
                PrimeQuaternions = primeQuaternions
                PrimeRatio = if nodeCount = 0 then 0.0 else float primeQuaternions / float nodeCount
                ConnectivityRatio = if nodeCount = 0 then 0.0 else float edgeCount / float nodeCount
            |}

    /// Integration with Guitar Alchemist music theory
    module MusicalHypergraph =
        
        /// Musical TRSX node with harmonic properties
        type MusicalTrsxNode = {
            TrsxNode: TrsxNode
            MusicalQuaternion: HurwitzQuaternions.MusicalQuaternions.MusicalQuaternion option
            HarmonicSeries: float list
            MusicalContext: string
        }
        
        /// Extract musical properties from TRSX content
        let extractMusicalProperties (node: TrsxNode) =
            // Look for musical keywords and patterns in TRSX content
            let content = node.Content.ToLower()
            let musicalKeywords = ["chord"; "scale"; "note"; "frequency"; "harmony"; "interval"]
            let hasMusicalContent = musicalKeywords |> List.exists content.Contains
            
            if hasMusicalContent then
                // Extract frequency if present (simplified)
                let frequency = 440.0  // Default A440, could be enhanced with regex parsing
                let interval = "unknown"
                Some (HurwitzQuaternions.MusicalQuaternions.encodeMusicalInterval interval frequency)
            else
                None
        
        /// Create musical TRSX node
        let createMusicalTrsxNode (node: TrsxNode) =
            {
                TrsxNode = node
                MusicalQuaternion = extractMusicalProperties node
                HarmonicSeries = [1.0; 2.0; 3.0; 4.0; 5.0]  // Basic harmonic series
                MusicalContext = "guitar_alchemist"
            }
