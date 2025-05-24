namespace TarsEngine.DSL

open System
open System.IO
open System.Text
open Ast

/// <summary>
/// Module for testing the incremental parser.
/// </summary>
module TestIncrementalParser =
    /// <summary>
    /// Generate a large test file.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The generated test content.</returns>
    let generateLargeTestFile (blockCount: int) =
        let sb = StringBuilder()
        
        // Add a header
        sb.AppendLine("// This is a large test file for the incremental parser")
        sb.AppendLine("// It contains a large number of blocks to test parsing performance")
        sb.AppendLine()
        
        // Add blocks
        for i in 1 .. blockCount do
            sb.AppendLine(sprintf "CONFIG config_%d {" i)
            sb.AppendLine(sprintf "    name: \"Config %d\"," i)
            sb.AppendLine(sprintf "    version: \"1.0\"," i)
            sb.AppendLine(sprintf "    description: \"This is config %d\"" i)
            sb.AppendLine("}")
            sb.AppendLine()
            
            sb.AppendLine(sprintf "PROMPT prompt_%d {" i)
            sb.AppendLine(sprintf "    content: \"This is prompt %d\"," i)
            sb.AppendLine(sprintf "    content_type: \"text\"" i)
            sb.AppendLine("}")
            sb.AppendLine()
            
            sb.AppendLine(sprintf "AGENT agent_%d {" i)
            sb.AppendLine(sprintf "    name: \"Agent %d\"," i)
            sb.AppendLine(sprintf "    agent_type: \"standard\"," i)
            sb.AppendLine(sprintf "    description: \"This is agent %d\"," i)
            sb.AppendLine()
            sb.AppendLine(sprintf "    TASK task_%d {" i)
            sb.AppendLine(sprintf "        name: \"Task %d\"," i)
            sb.AppendLine(sprintf "        description: \"This is task %d\"," i)
            sb.AppendLine()
            sb.AppendLine(sprintf "        ACTION action_%d {" i)
            sb.AppendLine(sprintf "            name: \"Action %d\"," i)
            sb.AppendLine(sprintf "            description: \"This is action %d\"" i)
            sb.AppendLine("        }")
            sb.AppendLine("    }")
            sb.AppendLine("}")
            sb.AppendLine()
            
            sb.AppendLine(sprintf "VARIABLE variable_%d {" i)
            sb.AppendLine(sprintf "    name: \"Variable %d\"," i)
            sb.AppendLine(sprintf "    value: %d" i)
            sb.AppendLine("}")
            sb.AppendLine()
        
        // Add some references between blocks
        for i in 1 .. blockCount do
            let j = (i % blockCount) + 1
            
            sb.AppendLine(sprintf "VARIABLE reference_%d {" i)
            sb.AppendLine(sprintf "    name: \"Reference %d\"," i)
            sb.AppendLine(sprintf "    value: @variable_%d" j)
            sb.AppendLine("}")
            sb.AppendLine()
        
        sb.ToString()
    
    /// <summary>
    /// Test the incremental parser with a large file.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <param name="chunkSize">The chunk size to use.</param>
    /// <param name="useCache">Whether to use caching.</param>
    /// <returns>The parse result.</returns>
    let testIncrementalParser (blockCount: int) (chunkSize: int) (useCache: bool) =
        // Generate a large test file
        let content = generateLargeTestFile blockCount
        
        // Create a configuration
        let config = {
            ChunkSize = chunkSize
            UseParallelParsing = true
            UseCache = useCache
            CollectTelemetry = true
        }
        
        // Parse the content incrementally
        let result = IncrementalParser.parseStringIncrementally content Parser.parse config
        
        // Print telemetry
        match result.Telemetry with
        | Some telemetry ->
            printfn "Incremental parsing telemetry:"
            printfn "  Total parse time: %d ms" telemetry.TotalParseTimeMs
            printfn "  Chunking time: %d ms" telemetry.ChunkingTimeMs
            printfn "  Chunk parsing time: %d ms" telemetry.ChunkParsingTimeMs
            printfn "  Chunk combining time: %d ms" telemetry.ChunkCombiningTimeMs
            printfn "  Chunk count: %d" telemetry.ChunkCount
            printfn "  Cached chunk count: %d" telemetry.CachedChunkCount
            printfn "  Block count: %d" telemetry.BlockCount
            printfn "  Diagnostic count: %d" telemetry.DiagnosticCount
            printfn "  Unresolved reference count: %d" telemetry.UnresolvedReferenceCount
        | None ->
            printfn "No telemetry collected."
        
        // Print diagnostics
        if result.Diagnostics.Length > 0 then
            printfn "Diagnostics:"
            
            for diagnostic in result.Diagnostics do
                printfn "  %A: %s" diagnostic.Code diagnostic.Message
                printfn "    Severity: %A" diagnostic.Severity
                printfn "    Line: %d, Column: %d" diagnostic.Line diagnostic.Column
                printfn "    Suggestions:"
                
                for suggestion in diagnostic.Suggestions do
                    printfn "      - %s" suggestion
                
                printfn ""
        
        // Return the result
        result
    
    /// <summary>
    /// Compare the performance of incremental parsing with regular parsing.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <param name="chunkSize">The chunk size to use.</param>
    /// <returns>A tuple of (incrementalResult, regularResult).</returns>
    let compareParsingPerformance (blockCount: int) (chunkSize: int) =
        // Generate a large test file
        let content = generateLargeTestFile blockCount
        
        // Create a configuration
        let config = {
            ChunkSize = chunkSize
            UseParallelParsing = true
            UseCache = true
            CollectTelemetry = true
        }
        
        // Clear the cache
        ChunkCache.clearCache()
        
        // Parse the content incrementally
        printfn "Parsing incrementally..."
        let incrementalResult = IncrementalParser.parseStringIncrementally content Parser.parse config
        
        // Parse the content regularly
        printfn "Parsing regularly..."
        let regularStartTime = DateTime.Now
        let regularResult = Parser.parse content
        let regularEndTime = DateTime.Now
        let regularParseTime = (regularEndTime - regularStartTime).TotalMilliseconds
        
        // Print comparison
        printfn "Performance comparison:"
        
        match incrementalResult.Telemetry with
        | Some telemetry ->
            printfn "  Incremental parsing time: %d ms" telemetry.TotalParseTimeMs
            printfn "  Regular parsing time: %.2f ms" regularParseTime
            printfn "  Speedup: %.2fx" (regularParseTime / float telemetry.TotalParseTimeMs)
        | None ->
            printfn "  No telemetry collected for incremental parsing."
            printfn "  Regular parsing time: %.2f ms" regularParseTime
        
        // Compare results
        printfn "Result comparison:"
        printfn "  Incremental blocks: %d" incrementalResult.Program.Blocks.Length
        printfn "  Regular blocks: %d" regularResult.Blocks.Length
        
        // Return the results
        (incrementalResult, regularResult)
    
    /// <summary>
    /// Test the effect of chunk size on parsing performance.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <param name="chunkSizes">The chunk sizes to test.</param>
    /// <returns>A list of (chunkSize, parseTime) tuples.</returns>
    let testChunkSizes (blockCount: int) (chunkSizes: int list) =
        // Generate a large test file
        let content = generateLargeTestFile blockCount
        
        // Test each chunk size
        let results = 
            chunkSizes
            |> List.map (fun chunkSize ->
                // Create a configuration
                let config = {
                    ChunkSize = chunkSize
                    UseParallelParsing = true
                    UseCache = false // Disable cache to get accurate measurements
                    CollectTelemetry = true
                }
                
                // Parse the content incrementally
                printfn "Testing chunk size %d..." chunkSize
                let result = IncrementalParser.parseStringIncrementally content Parser.parse config
                
                // Return the chunk size and parse time
                match result.Telemetry with
                | Some telemetry -> (chunkSize, telemetry.TotalParseTimeMs)
                | None -> (chunkSize, 0L)
            )
        
        // Print results
        printfn "Chunk size performance:"
        
        for (chunkSize, parseTime) in results do
            printfn "  Chunk size %d: %d ms" chunkSize parseTime
        
        // Return the results
        results
    
    /// <summary>
    /// Test the effect of caching on parsing performance.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <param name="chunkSize">The chunk size to use.</param>
    /// <param name="iterations">The number of iterations to run.</param>
    /// <returns>A list of (iteration, parseTime) tuples.</returns>
    let testCaching (blockCount: int) (chunkSize: int) (iterations: int) =
        // Generate a large test file
        let content = generateLargeTestFile blockCount
        
        // Create a configuration
        let config = {
            ChunkSize = chunkSize
            UseParallelParsing = true
            UseCache = true
            CollectTelemetry = true
        }
        
        // Clear the cache
        ChunkCache.clearCache()
        
        // Run multiple iterations
        let results = 
            [1 .. iterations]
            |> List.map (fun iteration ->
                // Parse the content incrementally
                printfn "Iteration %d..." iteration
                let result = IncrementalParser.parseStringIncrementally content Parser.parse config
                
                // Return the iteration and parse time
                match result.Telemetry with
                | Some telemetry -> 
                    (iteration, telemetry.TotalParseTimeMs, telemetry.CachedChunkCount)
                | None -> 
                    (iteration, 0L, 0)
            )
        
        // Print results
        printfn "Caching performance:"
        
        for (iteration, parseTime, cachedChunkCount) in results do
            printfn "  Iteration %d: %d ms, %d cached chunks" iteration parseTime cachedChunkCount
        
        // Return the results
        results
