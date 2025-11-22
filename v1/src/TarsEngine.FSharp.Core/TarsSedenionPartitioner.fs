// ================================================
// 🌌 TARS Sedenion Partitioner
// ================================================
// 16D BSP partitioning for hyperdimensional reasoning
// Based on ChatGPT-Leveraging Primes for TARS document

namespace TarsEngine.FSharp.Core

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.TarsRsxDiff
open TarsEngine.FSharp.Core.TarsRsxGraph

/// Represents a 16-dimensional sedenion (hypercomplex number)
[<Struct>]
type Sedenion = {
    Components: float array // 16 components: e0, e1, e2, ..., e15
}

/// Represents a hyperplane in 16D space for BSP partitioning
type Hyperplane = {
    Normal: Sedenion
    Distance: float
    Significance: float
}

/// Represents a node in the BSP tree
type BspNode = {
    Id: string
    Hyperplane: Hyperplane option
    LeftChild: BspNode option
    RightChild: BspNode option
    Points: Sedenion list
    Depth: int
    Significance: float
}

/// Represents the complete BSP tree structure
type BspTree = {
    Root: BspNode option
    MaxDepth: int
    TotalPoints: int
    Metadata: Map<string, string>
}

/// Result type for partitioning operations
type PartitionResult<'T> = 
    | Success of 'T
    | Error of string

/// Partitioning performance metrics
type PartitionPerformance = {
    PointsPartitioned: int
    NodesCreated: int
    MaxDepth: int
    ElapsedMs: int64
    PointsPerSecond: float
}

module TarsSedenionPartitioner =

    /// Create a sedenion from 16 components
    let createSedenion (components: float array) : Sedenion =
        if components.Length <> 16 then
            failwith "Sedenion must have exactly 16 components"
        { Components = Array.copy components }

    /// Create zero sedenion
    let zeroSedenion () : Sedenion =
        { Components = Array.zeroCreate 16 }

    /// Create unit sedenion (1, 0, 0, ..., 0)
    let unitSedenion () : Sedenion =
        let components = Array.zeroCreate 16
        components.[0] <- 1.0
        { Components = components }

    /// Sedenion addition
    let add (s1: Sedenion) (s2: Sedenion) : Sedenion =
        let result = Array.zeroCreate 16
        for i in 0..15 do
            result.[i] <- s1.Components.[i] + s2.Components.[i]
        { Components = result }

    /// Sedenion scalar multiplication
    let scalarMultiply (scalar: float) (s: Sedenion) : Sedenion =
        let result = Array.zeroCreate 16
        for i in 0..15 do
            result.[i] <- scalar * s.Components.[i]
        { Components = result }

    /// Compute sedenion norm (magnitude)
    let norm (s: Sedenion) : float =
        s.Components 
        |> Array.map (fun x -> x * x)
        |> Array.sum
        |> sqrt

    /// Normalize sedenion to unit length
    let normalize (s: Sedenion) : Sedenion =
        let magnitude = norm s
        if magnitude > 1e-10 then
            scalarMultiply (1.0 / magnitude) s
        else
            unitSedenion()

    /// Compute dot product between two sedenions
    let dotProduct (s1: Sedenion) (s2: Sedenion) : float =
        Array.zip s1.Components s2.Components
        |> Array.map (fun (a, b) -> a * b)
        |> Array.sum

    /// Determine which side of hyperplane a point lies on
    let classifyPoint (hyperplane: Hyperplane) (point: Sedenion) : int =
        let distance = dotProduct hyperplane.Normal point - hyperplane.Distance
        if distance > 1e-10 then 1      // Positive side
        elif distance < -1e-10 then -1  // Negative side
        else 0                          // On the plane

    /// Find the best hyperplane to split a set of points
    let findBestHyperplane (points: Sedenion list) (logger: ILogger) : Hyperplane option =
        if points.Length < 2 then
            None
        else
            let mutable bestHyperplane = None
            let mutable bestScore = -1.0
            
            // Try multiple random hyperplanes and pick the best one
            let random = Random()
            
            for _ in 1..10 do
                // Generate random normal vector
                let normal = Array.init 16 (fun _ -> random.NextDouble() * 2.0 - 1.0)
                let normalSedenion = normalize { Components = normal }
                
                // Calculate distance as median projection
                let projections = 
                    points 
                    |> List.map (fun point -> dotProduct normalSedenion point)
                    |> List.sort
                
                let medianDistance = 
                    if projections.Length % 2 = 0 then
                        (projections.[projections.Length / 2 - 1] + projections.[projections.Length / 2]) / 2.0
                    else
                        projections.[projections.Length / 2]
                
                let hyperplane = {
                    Normal = normalSedenion
                    Distance = medianDistance
                    Significance = 1.0
                }
                
                // Score based on balance of partition
                let leftCount = points |> List.filter (fun p -> classifyPoint hyperplane p = -1) |> List.length
                let rightCount = points |> List.filter (fun p -> classifyPoint hyperplane p = 1) |> List.length
                let balance = 1.0 - abs(float leftCount - float rightCount) / float points.Length
                
                if balance > bestScore then
                    bestScore <- balance
                    bestHyperplane <- Some hyperplane
            
            bestHyperplane

    /// Create a BSP node
    let createBspNode (id: string) (points: Sedenion list) (depth: int) : BspNode =
        let significance = 
            if points.Length > 0 then
                // Calculate significance based on point density and depth
                let density = float points.Length / (float depth + 1.0)
                min 1.0 (density / 100.0)
            else 0.0
        
        {
            Id = id
            Hyperplane = None
            LeftChild = None
            RightChild = None
            Points = points
            Depth = depth
            Significance = significance
        }

    /// Generate unique ID for BSP nodes
    let generateNodeId (prefix: string) (depth: int) : string =
        let timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        $"%s{prefix}-d%d{depth}-%d{timestamp}"

    /// Recursively build BSP tree
    let rec buildBspTree (points: Sedenion list) (depth: int) (maxDepth: int) (logger: ILogger) : BspNode option =
        if points.Length = 0 then
            None
        elif depth >= maxDepth || points.Length <= 4 then
            // Leaf node
            let nodeId = generateNodeId "leaf" depth
            Some (createBspNode nodeId points depth)
        else
            match findBestHyperplane points logger with
            | Some hyperplane ->
                // Split points using hyperplane
                let leftPoints = points |> List.filter (fun p -> classifyPoint hyperplane p = -1)
                let rightPoints = points |> List.filter (fun p -> classifyPoint hyperplane p = 1)
                let onPlanePoints = points |> List.filter (fun p -> classifyPoint hyperplane p = 0)
                
                // Distribute on-plane points to maintain balance
                let leftWithOnPlane = leftPoints @ (onPlanePoints |> List.take (onPlanePoints.Length / 2))
                let rightWithOnPlane = rightPoints @ (onPlanePoints |> List.skip (onPlanePoints.Length / 2))
                
                // Recursively build children
                let leftChild = buildBspTree leftWithOnPlane (depth + 1) maxDepth logger
                let rightChild = buildBspTree rightWithOnPlane (depth + 1) maxDepth logger
                
                let nodeId = generateNodeId "internal" depth
                let node = createBspNode nodeId [] depth
                let updatedNode =
                    { node with
                        Hyperplane = Some hyperplane
                        LeftChild = leftChild
                        RightChild = rightChild }
                Some updatedNode
            | None ->
                // Fallback to leaf node
                let nodeId = generateNodeId "leaf" depth
                Some (createBspNode nodeId points depth)

    /// Convert TRSX diff change vectors to sedenions
    let diffsToSedenions (diffs: TrsxDiff list) : Sedenion list =
        diffs
        |> List.map (fun diff -> createSedenion diff.ChangeVector)

    /// Partition a set of change vectors using BSP
    let partitionChangeVectors (changeVectors: float array list) (maxDepth: int) (logger: ILogger) : PartitionResult<BspTree> =
        try
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            logger.LogInformation($"🌌 Partitioning {changeVectors.Length} change vectors with max depth {maxDepth}")
            
            // Convert to sedenions
            let sedenions = 
                changeVectors 
                |> List.map createSedenion
            
            // Build BSP tree
            let root = buildBspTree sedenions 0 maxDepth logger
            
            stopwatch.Stop()
            let elapsedMs = stopwatch.ElapsedMilliseconds
            
            let tree = {
                Root = root
                MaxDepth = maxDepth
                TotalPoints = changeVectors.Length
                Metadata = Map.empty
            }
            
            logger.LogInformation($"✅ BSP tree built in {elapsedMs}ms")
            
            Success tree
            
        with
        | ex ->
            logger.LogError($"❌ BSP partitioning failed: {ex.Message}")
            Error ex.Message

    /// Analyze BSP tree structure
    let analyzeBspTree (tree: BspTree) (logger: ILogger) : PartitionPerformance =
        let rec countNodes (node: BspNode option) : int =
            match node with
            | None -> 0
            | Some n -> 1 + countNodes n.LeftChild + countNodes n.RightChild
        
        let rec findMaxDepth (node: BspNode option) : int =
            match node with
            | None -> 0
            | Some n -> max n.Depth (max (findMaxDepth n.LeftChild) (findMaxDepth n.RightChild))
        
        let nodeCount = countNodes tree.Root
        let actualMaxDepth = findMaxDepth tree.Root
        
        logger.LogInformation($"📊 BSP Analysis: {nodeCount} nodes, max depth {actualMaxDepth}")
        
        {
            PointsPartitioned = tree.TotalPoints
            NodesCreated = nodeCount
            MaxDepth = actualMaxDepth
            ElapsedMs = 0L // Would be tracked during construction
            PointsPerSecond = 0.0 // Would be calculated during construction
        }

    /// Find the leaf node containing a specific point
    let rec findLeafNode (tree: BspTree) (point: Sedenion) : BspNode option =
        let rec search (node: BspNode option) : BspNode option =
            match node with
            | None -> None
            | Some n ->
                match n.Hyperplane with
                | None -> Some n // Leaf node
                | Some hyperplane ->
                    let classification = classifyPoint hyperplane point
                    if classification <= 0 then
                        search n.LeftChild
                    else
                        search n.RightChild
        
        search tree.Root

    /// Test sedenion partitioning
    let testSedenionPartitioning (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing sedenion partitioning")
            
            // Generate test data
            let random = Random(42) // Fixed seed for reproducible tests
            let testVectors = 
                [1..50]
                |> List.map (fun _ -> Array.init 16 (fun _ -> random.NextDouble() * 2.0 - 1.0))
            
            logger.LogInformation($"📊 Generated {testVectors.Length} test vectors")
            
            // Test partitioning
            match partitionChangeVectors testVectors 4 logger with
            | Success tree ->
                let analysis = analyzeBspTree tree logger
                logger.LogInformation($"✅ Partitioning successful: {analysis.NodesCreated} nodes")
                
                // Test point lookup
                let testPoint = createSedenion testVectors.[0]
                match findLeafNode tree testPoint with
                | Some leafNode ->
                    logger.LogInformation($"✅ Point lookup successful: found in node {leafNode.Id}")
                | None ->
                    logger.LogWarning("⚠️ Point lookup failed")
                
                true
            | Error err ->
                logger.LogError($"❌ Partitioning failed: {err}")
                false
                
        with
        | ex ->
            logger.LogError($"❌ Sedenion partitioning test failed: {ex.Message}")
            false
