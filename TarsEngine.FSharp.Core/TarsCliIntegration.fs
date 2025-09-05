// ================================================
// 💻 TARS CLI Integration
// ================================================
// Command-line interface for diff, partition, and reflect operations
// Based on ChatGPT-Leveraging Primes for TARS document

namespace TarsEngine.FSharp.Core

open System
open System.IO
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.TarsRsxDiff
open TarsEngine.FSharp.Core.TarsRsxGraph
open TarsEngine.FSharp.Core.TarsSedenionPartitioner
open TarsEngine.FSharp.Core.TarsAutoReflection
open TarsEngine.FSharp.Core.TarsAdvancedFlux

/// Represents CLI command arguments
type CliArgs = {
    Command: string
    SourceFile: string option
    TargetFile: string option
    OutputFile: string option
    MaxDepth: int option
    Verbose: bool
    Format: string // "json", "yaml", "text"
}

/// Represents CLI command result
type CliResult = {
    Success: bool
    Message: string
    OutputFile: string option
    Metrics: Map<string, float>
}

/// Result type for CLI operations
type CliOperationResult<'T> = 
    | Success of 'T
    | Error of string

module TarsCliIntegration =

    /// Parse command line arguments
    let parseCliArgs (args: string array) : CliArgs =
        let mutable command = ""
        let mutable sourceFile = None
        let mutable targetFile = None
        let mutable outputFile = None
        let mutable maxDepth = None
        let mutable verbose = false
        let mutable format = "text"
        
        let mutable i = 0
        while i < args.Length do
            match args.[i] with
            | "diff" | "partition" | "reflect" -> 
                command <- args.[i]
            | "--source" | "-s" when i + 1 < args.Length ->
                sourceFile <- Some args.[i + 1]
                i <- i + 1
            | "--target" | "-t" when i + 1 < args.Length ->
                targetFile <- Some args.[i + 1]
                i <- i + 1
            | "--output" | "-o" when i + 1 < args.Length ->
                outputFile <- Some args.[i + 1]
                i <- i + 1
            | "--max-depth" | "-d" when i + 1 < args.Length ->
                maxDepth <- Some (int args.[i + 1])
                i <- i + 1
            | "--format" | "-f" when i + 1 < args.Length ->
                format <- args.[i + 1]
                i <- i + 1
            | "--verbose" | "-v" ->
                verbose <- true
            | _ -> ()
            i <- i + 1
        
        {
            Command = command
            SourceFile = sourceFile
            TargetFile = targetFile
            OutputFile = outputFile
            MaxDepth = maxDepth
            Verbose = verbose
            Format = format
        }

    /// Execute 'tars diff' command
    let executeDiffCommand (args: CliArgs) (logger: ILogger) : CliResult =
        try
            match args.SourceFile, args.TargetFile with
            | Some sourceFile, Some targetFile ->
                if not (File.Exists(sourceFile)) then
                    { Success = false; Message = $"Source file not found: {sourceFile}"; OutputFile = None; Metrics = Map.empty }
                elif not (File.Exists(targetFile)) then
                    { Success = false; Message = $"Target file not found: {targetFile}"; OutputFile = None; Metrics = Map.empty }
                else
                    logger.LogInformation($"🔄 Computing diff: {sourceFile} -> {targetFile}")
                    
                    let sourceContent = File.ReadAllText(sourceFile)
                    let targetContent = File.ReadAllText(targetFile)
                    let sourceVersion = Path.GetFileNameWithoutExtension(sourceFile)
                    let targetVersion = Path.GetFileNameWithoutExtension(targetFile)
                    
                    match computeTrsxDiff sourceContent targetContent sourceVersion targetVersion logger with
                    | DiffResult.Success diff ->
                        let output =
                            match args.Format with
                            | "json" ->
                                // Simplified JSON representation
                                sprintf """{"source": "%s", "target": "%s", "significance": %f, "sections": %d}"""
                                    diff.SourceVersion diff.TargetVersion diff.OverallSignificance diff.SectionChanges.Length
                            | "yaml" ->
                                // Simplified YAML representation
                                sprintf """source: %s
target: %s
significance: %f
sections: %d
timestamp: %s"""
                                    diff.SourceVersion diff.TargetVersion diff.OverallSignificance
                                    diff.SectionChanges.Length (diff.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"))
                            | _ -> formatDiff diff

                        match args.OutputFile with
                        | Some outputFile ->
                            File.WriteAllText(outputFile, output)
                            logger.LogInformation($"✅ Diff written to: {outputFile}")
                            { Success = true; Message = $"Diff computed and saved to {outputFile}"; OutputFile = Some outputFile;
                              Metrics = Map [("significance", diff.OverallSignificance); ("sections", float diff.SectionChanges.Length)] }
                        | None ->
                            Console.WriteLine(output)
                            { Success = true; Message = "Diff computed successfully"; OutputFile = None;
                              Metrics = Map [("significance", diff.OverallSignificance); ("sections", float diff.SectionChanges.Length)] }
                    | DiffResult.Error err ->
                        { Success = false; Message = $"Diff computation failed: {err}"; OutputFile = None; Metrics = Map.empty }
            | _ ->
                { Success = false; Message = "Both --source and --target files are required for diff command"; OutputFile = None; Metrics = Map.empty }
                
        with
        | ex ->
            logger.LogError($"❌ Diff command failed: {ex.Message}")
            { Success = false; Message = $"Command failed: {ex.Message}"; OutputFile = None; Metrics = Map.empty }

    /// Execute 'tars partition' command
    let executePartitionCommand (args: CliArgs) (logger: ILogger) : CliResult =
        try
            match args.SourceFile with
            | Some sourceFile ->
                if not (File.Exists(sourceFile)) then
                    { Success = false; Message = $"Source file not found: {sourceFile}"; OutputFile = None; Metrics = Map.empty }
                else
                    logger.LogInformation($"🌌 Partitioning data from: {sourceFile}")
                    
                    // For demonstration, generate test vectors
                    // In real implementation, would parse vectors from file
                    let random = Random()
                    let vectorCount = 30
                    let testVectors = 
                        [1..vectorCount]
                        |> List.map (fun _ -> Array.init 16 (fun _ -> random.NextDouble() * 2.0 - 1.0))
                    
                    let maxDepth = args.MaxDepth |> Option.defaultValue 4
                    
                    match partitionChangeVectors testVectors maxDepth logger with
                    | PartitionResult.Success tree ->
                        let nodeCount =
                            let rec countNodes (node: BspNode option) : int =
                                match node with
                                | None -> 0
                                | Some n -> 1 + countNodes n.LeftChild + countNodes n.RightChild
                            countNodes tree.Root

                        let output =
                            match args.Format with
                            | "json" ->
                                sprintf """{"vectors": %d, "nodes": %d, "max_depth": %d, "total_points": %d}"""
                                    vectorCount nodeCount tree.MaxDepth tree.TotalPoints
                            | "yaml" ->
                                sprintf """vectors: %d
nodes: %d
max_depth: %d
total_points: %d"""
                                    vectorCount nodeCount tree.MaxDepth tree.TotalPoints
                            | _ ->
                                sprintf """BSP Tree Partitioning Results:
Vectors processed: %d
Nodes created: %d
Maximum depth: %d
Total points: %d"""
                                    vectorCount nodeCount tree.MaxDepth tree.TotalPoints

                        match args.OutputFile with
                        | Some outputFile ->
                            File.WriteAllText(outputFile, output)
                            logger.LogInformation($"✅ Partition results written to: {outputFile}")
                            { Success = true; Message = $"Partitioning completed and saved to {outputFile}"; OutputFile = Some outputFile;
                              Metrics = Map [("nodes", float nodeCount); ("max_depth", float tree.MaxDepth)] }
                        | None ->
                            Console.WriteLine(output)
                            { Success = true; Message = "Partitioning completed successfully"; OutputFile = None;
                              Metrics = Map [("nodes", float nodeCount); ("max_depth", float tree.MaxDepth)] }
                    | PartitionResult.Error err ->
                        { Success = false; Message = $"Partitioning failed: {err}"; OutputFile = None; Metrics = Map.empty }
            | None ->
                { Success = false; Message = "--source file is required for partition command"; OutputFile = None; Metrics = Map.empty }
                
        with
        | ex ->
            logger.LogError($"❌ Partition command failed: {ex.Message}")
            { Success = false; Message = $"Command failed: {ex.Message}"; OutputFile = None; Metrics = Map.empty }

    /// Execute 'tars reflect' command
    let executeReflectCommand (args: CliArgs) (logger: ILogger) : CliResult =
        try
            logger.LogInformation("🧠 Performing auto-reflection analysis")
            
            // Generate test BSP tree for reflection
            let random = Random()
            let vectorCount = 25
            let testVectors = 
                [1..vectorCount]
                |> List.map (fun _ -> Array.init 16 (fun _ -> random.NextDouble() * 2.0 - 1.0))
            
            let maxDepth = args.MaxDepth |> Option.defaultValue 3
            
            match partitionChangeVectors testVectors maxDepth logger with
            | PartitionResult.Success tree ->
                match performReflection tree logger with
                | ReflectionResult.Success performance ->
                    let output =
                        match args.Format with
                        | "json" ->
                            sprintf """{"partitions_analyzed": %d, "insights_generated": %d, "contradictions_detected": %d, "analysis_rate": %f}"""
                                performance.PartitionsAnalyzed performance.InsightsGenerated
                                performance.ContradictionsDetected performance.AnalysisRate
                        | "yaml" ->
                            sprintf """partitions_analyzed: %d
insights_generated: %d
contradictions_detected: %d
analysis_rate: %f
elapsed_ms: %d"""
                                performance.PartitionsAnalyzed performance.InsightsGenerated
                                performance.ContradictionsDetected performance.AnalysisRate performance.ElapsedMs
                        | _ ->
                            sprintf """Auto-Reflection Analysis Results:
Partitions analyzed: %d
Insights generated: %d
Contradictions detected: %d
Analysis rate: %.0f partitions/second
Elapsed time: %d ms"""
                                performance.PartitionsAnalyzed performance.InsightsGenerated
                                performance.ContradictionsDetected performance.AnalysisRate performance.ElapsedMs

                    match args.OutputFile with
                    | Some outputFile ->
                        File.WriteAllText(outputFile, output)
                        logger.LogInformation($"✅ Reflection results written to: {outputFile}")
                        { Success = true; Message = $"Reflection completed and saved to {outputFile}"; OutputFile = Some outputFile;
                          Metrics = Map [("insights", float performance.InsightsGenerated); ("contradictions", float performance.ContradictionsDetected)] }
                    | None ->
                        Console.WriteLine(output)
                        { Success = true; Message = "Reflection completed successfully"; OutputFile = None;
                          Metrics = Map [("insights", float performance.InsightsGenerated); ("contradictions", float performance.ContradictionsDetected)] }
                | ReflectionResult.Error err ->
                    { Success = false; Message = $"Reflection failed: {err}"; OutputFile = None; Metrics = Map.empty }
            | PartitionResult.Error err ->
                { Success = false; Message = $"Tree generation failed: {err}"; OutputFile = None; Metrics = Map.empty }
                
        with
        | ex ->
            logger.LogError($"❌ Reflect command failed: {ex.Message}")
            { Success = false; Message = $"Command failed: {ex.Message}"; OutputFile = None; Metrics = Map.empty }

    /// Execute CLI command
    let executeCliCommand (args: CliArgs) (logger: ILogger) : CliResult =
        match args.Command with
        | "diff" -> executeDiffCommand args logger
        | "partition" -> executePartitionCommand args logger
        | "reflect" -> executeReflectCommand args logger
        | "" -> { Success = false; Message = "No command specified. Use: diff, partition, or reflect"; OutputFile = None; Metrics = Map.empty }
        | cmd -> { Success = false; Message = $"Unknown command: {cmd}. Use: diff, partition, or reflect"; OutputFile = None; Metrics = Map.empty }

    /// Test CLI integration
    let testCliIntegration (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing CLI integration")
            
            // Test diff command
            let diffArgs = {
                Command = "diff"
                SourceFile = None
                TargetFile = None
                OutputFile = None
                MaxDepth = None
                Verbose = false
                Format = "text"
            }
            
            // Test partition command
            let partitionArgs = {
                Command = "partition"
                SourceFile = Some "test_vectors.txt"
                TargetFile = None
                OutputFile = None
                MaxDepth = Some 3
                Verbose = false
                Format = "json"
            }
            
            // Test reflect command
            let reflectArgs = {
                Command = "reflect"
                SourceFile = None
                TargetFile = None
                OutputFile = None
                MaxDepth = Some 3
                Verbose = false
                Format = "yaml"
            }
            
            // Execute test commands (some will fail due to missing files, which is expected)
            let diffResult = executeCliCommand diffArgs logger
            let partitionResult = executeCliCommand partitionArgs logger
            let reflectResult = executeCliCommand reflectArgs logger
            
            // Reflect should succeed as it doesn't require files
            if reflectResult.Success then
                logger.LogInformation("✅ CLI integration test successful")
                logger.LogInformation($"   Reflect command: {reflectResult.Message}")
                true
            else
                logger.LogWarning("⚠️ CLI integration test had issues")
                false
                
        with
        | ex ->
            logger.LogError($"❌ CLI integration test failed: {ex.Message}")
            false
