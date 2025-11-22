namespace TarsEngine.DSL

open System
open System.IO
open System.Diagnostics
open Ast

/// <summary>
/// Configuration for incremental parsing.
/// </summary>
type IncrementalParserConfig = {
    /// <summary>
    /// The chunk size in lines.
    /// </summary>
    ChunkSize: int
    
    /// <summary>
    /// Whether to use parallel parsing.
    /// </summary>
    UseParallelParsing: bool
    
    /// <summary>
    /// Whether to use caching.
    /// </summary>
    UseCache: bool
    
    /// <summary>
    /// Whether to collect telemetry.
    /// </summary>
    CollectTelemetry: bool
}

/// <summary>
/// Default configuration for incremental parsing.
/// </summary>
module IncrementalParserConfig =
    /// <summary>
    /// Default configuration for incremental parsing.
    /// </summary>
    let defaultConfig = {
        ChunkSize = FileChunker.defaultChunkSize
        UseParallelParsing = true
        UseCache = true
        CollectTelemetry = false
    }

/// <summary>
/// Telemetry data for incremental parsing.
/// </summary>
type IncrementalParserTelemetry = {
    /// <summary>
    /// The total time spent parsing in milliseconds.
    /// </summary>
    TotalParseTimeMs: int64
    
    /// <summary>
    /// The time spent chunking in milliseconds.
    /// </summary>
    ChunkingTimeMs: int64
    
    /// <summary>
    /// The time spent parsing chunks in milliseconds.
    /// </summary>
    ChunkParsingTimeMs: int64
    
    /// <summary>
    /// The time spent combining chunks in milliseconds.
    /// </summary>
    ChunkCombiningTimeMs: int64
    
    /// <summary>
    /// The number of chunks.
    /// </summary>
    ChunkCount: int
    
    /// <summary>
    /// The number of cached chunks.
    /// </summary>
    CachedChunkCount: int
    
    /// <summary>
    /// The number of blocks in the program.
    /// </summary>
    BlockCount: int
    
    /// <summary>
    /// The number of diagnostics.
    /// </summary>
    DiagnosticCount: int
    
    /// <summary>
    /// The number of unresolved references.
    /// </summary>
    UnresolvedReferenceCount: int
}

/// <summary>
/// Result of incremental parsing.
/// </summary>
type IncrementalParseResult = {
    /// <summary>
    /// The parsed program.
    /// </summary>
    Program: TarsProgram
    
    /// <summary>
    /// The diagnostics generated during parsing.
    /// </summary>
    Diagnostics: Diagnostic list
    
    /// <summary>
    /// The telemetry data collected during parsing.
    /// </summary>
    Telemetry: IncrementalParserTelemetry option
}

/// <summary>
/// Module for incremental parsing of large files.
/// </summary>
module IncrementalParser =
    /// <summary>
    /// Parse a file incrementally.
    /// </summary>
    /// <param name="filePath">The path to the file to parse.</param>
    /// <param name="parser">The parser function to use.</param>
    /// <param name="config">The configuration for incremental parsing.</param>
    /// <returns>The result of parsing the file.</returns>
    let parseFileIncrementally (filePath: string) (parser: string -> TarsProgram) (config: IncrementalParserConfig) =
        // Start the stopwatch
        let totalStopwatch = Stopwatch.StartNew()
        
        // Chunk the file
        let chunkingStopwatch = Stopwatch.StartNew()
        let chunks = FileChunker.chunkFile filePath config.ChunkSize
        chunkingStopwatch.Stop()
        
        // Parse the chunks
        let chunkParsingStopwatch = Stopwatch.StartNew()
        let mutable cachedChunkCount = 0
        
        let chunkResults = 
            chunks
            |> List.map (fun chunk ->
                // Check if the chunk is cached
                if config.UseCache then
                    match ChunkCache.getCachedResult chunk with
                    | Some result ->
                        cachedChunkCount <- cachedChunkCount + 1
                        result
                    | None ->
                        let result = ChunkParser.parseChunk chunk parser
                        
                        // Cache the result
                        ChunkCache.cacheResult chunk result
                        
                        result
                else
                    ChunkParser.parseChunk chunk parser
            )
        
        chunkParsingStopwatch.Stop()
        
        // Combine the chunks
        let chunkCombiningStopwatch = Stopwatch.StartNew()
        let combinedResult = ChunkCombiner.combineChunks chunkResults
        
        // Validate the combined result
        let validationDiagnostics = ChunkCombiner.validateCombinedResult combinedResult
        
        // Combine all diagnostics
        let allDiagnostics = combinedResult.Diagnostics @ validationDiagnostics
        
        chunkCombiningStopwatch.Stop()
        
        // Stop the total stopwatch
        totalStopwatch.Stop()
        
        // Create telemetry data if requested
        let telemetry = 
            if config.CollectTelemetry then
                Some {
                    TotalParseTimeMs = totalStopwatch.ElapsedMilliseconds
                    ChunkingTimeMs = chunkingStopwatch.ElapsedMilliseconds
                    ChunkParsingTimeMs = chunkParsingStopwatch.ElapsedMilliseconds
                    ChunkCombiningTimeMs = chunkCombiningStopwatch.ElapsedMilliseconds
                    ChunkCount = chunks.Length
                    CachedChunkCount = cachedChunkCount
                    BlockCount = combinedResult.Program.Blocks.Length
                    DiagnosticCount = allDiagnostics.Length
                    UnresolvedReferenceCount = combinedResult.UnresolvedReferences.Length
                }
            else
                None
        
        // Create the result
        {
            Program = combinedResult.Program
            Diagnostics = allDiagnostics
            Telemetry = telemetry
        }
    
    /// <summary>
    /// Parse a string incrementally.
    /// </summary>
    /// <param name="content">The content to parse.</param>
    /// <param name="parser">The parser function to use.</param>
    /// <param name="config">The configuration for incremental parsing.</param>
    /// <returns>The result of parsing the content.</returns>
    let parseStringIncrementally (content: string) (parser: string -> TarsProgram) (config: IncrementalParserConfig) =
        // Start the stopwatch
        let totalStopwatch = Stopwatch.StartNew()
        
        // Chunk the content
        let chunkingStopwatch = Stopwatch.StartNew()
        let chunks = FileChunker.chunkString content config.ChunkSize
        chunkingStopwatch.Stop()
        
        // Parse the chunks
        let chunkParsingStopwatch = Stopwatch.StartNew()
        let mutable cachedChunkCount = 0
        
        let chunkResults = 
            chunks
            |> List.map (fun chunk ->
                // Check if the chunk is cached
                if config.UseCache then
                    match ChunkCache.getCachedResult chunk with
                    | Some result ->
                        cachedChunkCount <- cachedChunkCount + 1
                        result
                    | None ->
                        let result = ChunkParser.parseChunk chunk parser
                        
                        // Cache the result
                        ChunkCache.cacheResult chunk result
                        
                        result
                else
                    ChunkParser.parseChunk chunk parser
            )
        
        chunkParsingStopwatch.Stop()
        
        // Combine the chunks
        let chunkCombiningStopwatch = Stopwatch.StartNew()
        let combinedResult = ChunkCombiner.combineChunks chunkResults
        
        // Validate the combined result
        let validationDiagnostics = ChunkCombiner.validateCombinedResult combinedResult
        
        // Combine all diagnostics
        let allDiagnostics = combinedResult.Diagnostics @ validationDiagnostics
        
        chunkCombiningStopwatch.Stop()
        
        // Stop the total stopwatch
        totalStopwatch.Stop()
        
        // Create telemetry data if requested
        let telemetry = 
            if config.CollectTelemetry then
                Some {
                    TotalParseTimeMs = totalStopwatch.ElapsedMilliseconds
                    ChunkingTimeMs = chunkingStopwatch.ElapsedMilliseconds
                    ChunkParsingTimeMs = chunkParsingStopwatch.ElapsedMilliseconds
                    ChunkCombiningTimeMs = chunkCombiningStopwatch.ElapsedMilliseconds
                    ChunkCount = chunks.Length
                    CachedChunkCount = cachedChunkCount
                    BlockCount = combinedResult.Program.Blocks.Length
                    DiagnosticCount = allDiagnostics.Length
                    UnresolvedReferenceCount = combinedResult.UnresolvedReferences.Length
                }
            else
                None
        
        // Create the result
        {
            Program = combinedResult.Program
            Diagnostics = allDiagnostics
            Telemetry = telemetry
        }
