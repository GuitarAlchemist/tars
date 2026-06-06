// ================================================
// 🌌 BSP Code Analysis Demo
// ================================================
// Real sedenion BSP partitioning for code clustering

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Net.Http
open System.Text
open Spectre.Console

module BspCodeAnalysisDemo =

    // Real code file analysis structure
    type CodeFile = {
        FilePath: string
        FileName: string
        LineCount: int
        FunctionCount: int
        TypeCount: int
        ModuleReferences: string list
        SemanticFeatures: float array // 16D semantic embedding
    }

    type CodeCluster = {
        ClusterName: string
        Files: CodeFile list
        CentroidFeatures: float array
        ClusterType: string
        ArchitecturalRole: string
    }

    // Real sedenion structure for code analysis
    type Sedenion = {
        Components: float array // 16 components representing code semantics
    }

    type Hyperplane = {
        Normal: Sedenion
        Distance: float
        Significance: float
    }

    type BspNode = {
        Id: string
        Hyperplane: Hyperplane option
        LeftChild: BspNode option
        RightChild: BspNode option
        CodeFiles: CodeFile list
        Depth: int
        Significance: float
    }

    // Real code analysis functions
    let analyzeCodeFile (filePath: string) : CodeFile =
        let content = File.ReadAllText(filePath)
        let lines = content.Split('\n')
        let lineCount = lines.Length
        
        // Extract semantic features from code
        let functionCount = content.Split("let ").Length - 1
        let typeCount = content.Split("type ").Length - 1
        let moduleRefs = 
            lines 
            |> Array.filter (fun line -> line.Trim().StartsWith("open "))
            |> Array.map (fun line -> line.Trim().Substring(5).Trim())
            |> Array.toList
        
        // Generate 16D semantic embedding based on code characteristics
        let semanticFeatures = Array.zeroCreate 16
        semanticFeatures.[0] <- float lineCount / 1000.0  // File size
        semanticFeatures.[1] <- float functionCount / 50.0  // Function density
        semanticFeatures.[2] <- float typeCount / 20.0  // Type density
        semanticFeatures.[3] <- float moduleRefs.Length / 10.0  // Dependency count
        
        // Semantic analysis based on file name and content
        let fileName = Path.GetFileNameWithoutExtension(filePath).ToLower()
        semanticFeatures.[4] <- if fileName.Contains("ai") || fileName.Contains("inference") then 1.0 else 0.0
        semanticFeatures.[5] <- if fileName.Contains("cli") || fileName.Contains("command") then 1.0 else 0.0
        semanticFeatures.[6] <- if fileName.Contains("core") || fileName.Contains("engine") then 1.0 else 0.0
        semanticFeatures.[7] <- if fileName.Contains("cuda") || fileName.Contains("gpu") then 1.0 else 0.0
        semanticFeatures.[8] <- if fileName.Contains("test") || fileName.Contains("demo") then 1.0 else 0.0
        semanticFeatures.[9] <- if fileName.Contains("flux") || fileName.Contains("metascript") then 1.0 else 0.0
        semanticFeatures.[10] <- if content.Contains("async") then 1.0 else 0.0
        semanticFeatures.[11] <- if content.Contains("Result<") then 1.0 else 0.0
        semanticFeatures.[12] <- if content.Contains("ILogger") then 1.0 else 0.0
        semanticFeatures.[13] <- if content.Contains("AnsiConsole") then 1.0 else 0.0
        semanticFeatures.[14] <- if content.Contains("CUDA") then 1.0 else 0.0
        semanticFeatures.[15] <- if content.Contains("vector") || content.Contains("Vector") then 1.0 else 0.0
        
        {
            FilePath = filePath
            FileName = Path.GetFileName(filePath)
            LineCount = lineCount
            FunctionCount = functionCount
            TypeCount = typeCount
            ModuleReferences = moduleRefs
            SemanticFeatures = semanticFeatures
        }

    // Real sedenion operations for code analysis
    let createSedenion (components: float array) : Sedenion =
        if components.Length <> 16 then
            failwith "Sedenion must have exactly 16 components"
        { Components = Array.copy components }

    let norm (s: Sedenion) : float =
        s.Components |> Array.map (fun x -> x * x) |> Array.sum |> sqrt

    let dotProduct (s1: Sedenion) (s2: Sedenion) : float =
        Array.zip s1.Components s2.Components |> Array.map (fun (a, b) -> a * b) |> Array.sum

    // Real sedenion hyperplane calculation for 16D partitioning
    let findBestHyperplane (points: Sedenion list) : Hyperplane option =
        if points.Length < 2 then None
        else
            let random = Random(42)
            let mutable bestHyperplane = None
            let mutable bestScore = -1.0
            
            // Try multiple random hyperplanes and pick the best one
            for _ in 1..5 do
                // Generate random normal vector in 16D
                let normal = Array.init 16 (fun _ -> random.NextDouble() * 2.0 - 1.0)
                let normalSedenion = createSedenion normal
                let normalMagnitude = norm normalSedenion
                let unitNormal = createSedenion (normal |> Array.map (fun x -> x / normalMagnitude))
                
                // Calculate distance as median projection
                let projections = 
                    points 
                    |> List.map (fun point -> dotProduct unitNormal point)
                    |> List.sort
                
                let medianDistance = 
                    if projections.Length % 2 = 0 then
                        (projections.[projections.Length / 2 - 1] + projections.[projections.Length / 2]) / 2.0
                    else
                        projections.[projections.Length / 2]
                
                let hyperplane = {
                    Normal = unitNormal
                    Distance = medianDistance
                    Significance = 1.0
                }
                
                // Score based on balance of partition
                let (left, right) = points |> List.partition (fun p -> dotProduct unitNormal p >= medianDistance)
                let balance = 1.0 - abs(float left.Length - float right.Length) / float points.Length
                
                if balance > bestScore then
                    bestScore <- balance
                    bestHyperplane <- Some hyperplane
            
            bestHyperplane

    // Real sedenion point classification
    let classifyPoint (hyperplane: Hyperplane) (point: Sedenion) : int =
        let distance = dotProduct hyperplane.Normal point - hyperplane.Distance
        if distance > 1e-10 then 1      // Positive side
        elif distance < -1e-10 then -1  // Negative side
        else 0                          // On the hyperplane

    // Real sedenion BSP tree building for code clustering
    let rec buildCodeBspTree (files: CodeFile list) (sedenions: Sedenion list) (depth: int) (nodeId: string) : BspNode =
        if files.Length <= 2 || depth > 6 then
            // Create leaf node with code cluster
            {
                Id = nodeId
                Hyperplane = None
                LeftChild = None
                RightChild = None
                CodeFiles = files
                Depth = depth
                Significance = float files.Length
            }
        else
            match findBestHyperplane sedenions with
            | None -> 
                {
                    Id = nodeId
                    Hyperplane = None
                    LeftChild = None
                    RightChild = None
                    CodeFiles = files
                    Depth = depth
                    Significance = float files.Length
                }
            | Some hyperplane ->
                // Partition code files using sedenion hyperplane
                let fileSedenionPairs = List.zip files sedenions
                let (leftPairs, rightPairs) = 
                    fileSedenionPairs 
                    |> List.partition (fun (_, sedenion) -> classifyPoint hyperplane sedenion >= 0)
                
                let leftFiles = leftPairs |> List.map fst
                let rightFiles = rightPairs |> List.map fst
                let leftSedenions = leftPairs |> List.map snd
                let rightSedenions = rightPairs |> List.map snd
                
                if leftFiles.Length = 0 || rightFiles.Length = 0 then
                    {
                        Id = nodeId
                        Hyperplane = None
                        LeftChild = None
                        RightChild = None
                        CodeFiles = files
                        Depth = depth
                        Significance = float files.Length
                    }
                else
                    let leftChild = buildCodeBspTree leftFiles leftSedenions (depth + 1) (nodeId + "L")
                    let rightChild = buildCodeBspTree rightFiles rightSedenions (depth + 1) (nodeId + "R")
                    {
                        Id = nodeId
                        Hyperplane = Some hyperplane
                        LeftChild = Some leftChild
                        RightChild = Some rightChild
                        CodeFiles = []
                        Depth = depth
                        Significance = float files.Length
                    }

    // Real code clustering analysis
    let rec analyzeCodeBspTree (node: BspNode) : (int * int * int * float) =
        match node.LeftChild, node.RightChild with
        | None, None -> (1, 0, node.CodeFiles.Length, node.Significance)
        | Some left, Some right ->
            let (leftLeaves, leftNodes, leftFiles, leftSig) = analyzeCodeBspTree left
            let (rightLeaves, rightNodes, rightFiles, rightSig) = analyzeCodeBspTree right
            (leftLeaves + rightLeaves, leftNodes + rightNodes + 1, leftFiles + rightFiles, leftSig + rightSig)
        | _ -> (1, 0, node.CodeFiles.Length, node.Significance)

    // Real code cluster extraction and analysis
    let rec extractCodeClusters (node: BspNode) : CodeCluster list =
        match node.LeftChild, node.RightChild with
        | None, None when node.CodeFiles.Length > 0 ->
            // This is a leaf with actual code files - analyze cluster characteristics
            let files = node.CodeFiles
            
            // Determine cluster type based on semantic features
            let avgFeatures = Array.zeroCreate 16
            for file in files do
                for i in 0..15 do
                    avgFeatures.[i] <- avgFeatures.[i] + file.SemanticFeatures.[i]
            for i in 0..15 do
                avgFeatures.[i] <- avgFeatures.[i] / float files.Length
            
            let clusterName, architecturalRole = 
                if avgFeatures.[4] > 0.3 then ("AI/ML Components", "Machine Learning & Inference")
                elif avgFeatures.[5] > 0.3 then ("CLI Components", "User Interface & Commands")
                elif avgFeatures.[6] > 0.3 then ("Core Engine", "Core Infrastructure")
                elif avgFeatures.[7] > 0.3 then ("CUDA/GPU", "High-Performance Computing")
                elif avgFeatures.[8] > 0.3 then ("Testing/Demo", "Quality Assurance & Examples")
                elif avgFeatures.[9] > 0.3 then ("FLUX/Metascript", "Domain-Specific Language")
                else ("Utility/Support", "Supporting Infrastructure")
            
            let clusterType = 
                if files.Length >= 4 then "Major Component Cluster"
                elif files.Length >= 2 then "Component Cluster"
                else "Isolated Component"
            
            [{
                ClusterName = clusterName
                Files = files
                CentroidFeatures = avgFeatures
                ClusterType = clusterType
                ArchitecturalRole = architecturalRole
            }]
        | Some left, Some right ->
            (extractCodeClusters left) @ (extractCodeClusters right)
        | _ -> []
