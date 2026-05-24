namespace TarsEngine.DSL

open System
open System.Diagnostics
open System.Collections.Generic
open Ast

/// <summary>
/// Module for integrating telemetry collection with the parsers.
/// </summary>
module TelemetryService =
    /// <summary>
    /// Whether telemetry collection is enabled.
    /// </summary>
    let mutable private telemetryEnabled = true
    
    /// <summary>
    /// A dictionary to store performance metrics during parsing.
    /// </summary>
    let private performanceMetrics = Dictionary<string, int64>()
    
    /// <summary>
    /// A dictionary to store diagnostics during parsing.
    /// </summary>
    let private diagnosticsCollection = Dictionary<string, Diagnostic list>()
    
    /// <summary>
    /// A dictionary to store suppressed warnings during parsing.
    /// </summary>
    let private suppressedWarningsCollection = Dictionary<string, (int * WarningCode) list>()
    
    /// <summary>
    /// Initialize the telemetry service.
    /// </summary>
    let initialize() =
        // Load the telemetry configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        telemetryEnabled <- config.Enabled
        performanceMetrics.Clear()
        diagnosticsCollection.Clear()
        suppressedWarningsCollection.Clear()
    
    /// <summary>
    /// Check if telemetry collection is enabled.
    /// </summary>
    /// <returns>True if telemetry collection is enabled, false otherwise.</returns>
    let isTelemetryEnabled() =
        telemetryEnabled
    
    /// <summary>
    /// Enable telemetry collection.
    /// </summary>
    let enableTelemetry() =
        telemetryEnabled <- true
        TelemetryConfiguration.enableTelemetry TelemetryConfiguration.defaultConfigFilePath |> ignore
    
    /// <summary>
    /// Disable telemetry collection.
    /// </summary>
    let disableTelemetry() =
        telemetryEnabled <- false
        TelemetryConfiguration.disableTelemetry TelemetryConfiguration.defaultConfigFilePath |> ignore
    
    /// <summary>
    /// Start measuring a performance metric.
    /// </summary>
    /// <param name="metricName">The name of the metric to measure.</param>
    /// <returns>A stopwatch for measuring the metric.</returns>
    let startMeasuring (metricName: string) =
        if telemetryEnabled then
            let stopwatch = Stopwatch.StartNew()
            (metricName, stopwatch)
        else
            (metricName, null)
    
    /// <summary>
    /// Stop measuring a performance metric.
    /// </summary>
    /// <param name="metricTuple">The tuple containing the metric name and stopwatch.</param>
    let stopMeasuring (metricName: string, stopwatch: Stopwatch) =
        if telemetryEnabled && stopwatch <> null then
            stopwatch.Stop()
            let elapsedMs = stopwatch.ElapsedMilliseconds
            
            // If the metric already exists, add to it
            if performanceMetrics.ContainsKey(metricName) then
                performanceMetrics.[metricName] <- performanceMetrics.[metricName] + elapsedMs
            else
                performanceMetrics.Add(metricName, elapsedMs)
    
    /// <summary>
    /// Record diagnostics for telemetry.
    /// </summary>
    /// <param name="source">The source of the diagnostics.</param>
    /// <param name="diagnostics">The diagnostics to record.</param>
    let recordDiagnostics (source: string) (diagnostics: Diagnostic list) =
        if telemetryEnabled then
            diagnosticsCollection.[source] <- diagnostics
    
    /// <summary>
    /// Record suppressed warnings for telemetry.
    /// </summary>
    /// <param name="source">The source of the suppressed warnings.</param>
    /// <param name="suppressedWarnings">The suppressed warnings to record.</param>
    let recordSuppressedWarnings (source: string) (suppressedWarnings: (int * WarningCode) list) =
        if telemetryEnabled then
            suppressedWarningsCollection.[source] <- suppressedWarnings
    
    /// <summary>
    /// Parse a TARS program string with telemetry collection.
    /// </summary>
    /// <param name="parserType">The parser type used.</param>
    /// <param name="parseFunction">The parse function to use.</param>
    /// <param name="code">The TARS program string to parse.</param>
    /// <returns>The parsed TarsProgram.</returns>
    let parseWithTelemetry (parserType: string) (parseFunction: string -> TarsProgram) (code: string) =
        if not telemetryEnabled then
            // If telemetry is disabled, just call the parse function directly
            parseFunction code
        else
            // Load the telemetry configuration
            let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
            
            // Clear performance metrics
            performanceMetrics.Clear()
            
            // Start the overall stopwatch
            let startTime = DateTime.UtcNow
            let overallStopwatch = Stopwatch.StartNew()
            
            // Parse the code
            let program = parseFunction code
            
            // Stop the overall stopwatch
            overallStopwatch.Stop()
            let endTime = DateTime.UtcNow
            
            // Collect telemetry data
            if config.CollectUsageTelemetry || config.CollectPerformanceTelemetry then
                // Extract performance metrics
                let tokenizingTimeMs = 
                    if performanceMetrics.ContainsKey("Tokenizing") then
                        Some performanceMetrics.["Tokenizing"]
                    else
                        None
                
                let blockParsingTimeMs = 
                    if performanceMetrics.ContainsKey("BlockParsing") then
                        Some performanceMetrics.["BlockParsing"]
                    else
                        None
                
                let propertyParsingTimeMs = 
                    if performanceMetrics.ContainsKey("PropertyParsing") then
                        Some performanceMetrics.["PropertyParsing"]
                    else
                        None
                
                let nestedBlockParsingTimeMs = 
                    if performanceMetrics.ContainsKey("NestedBlockParsing") then
                        Some performanceMetrics.["NestedBlockParsing"]
                    else
                        None
                
                let chunkingTimeMs = 
                    if performanceMetrics.ContainsKey("Chunking") then
                        Some performanceMetrics.["Chunking"]
                    else
                        None
                
                let chunkParsingTimeMs = 
                    if performanceMetrics.ContainsKey("ChunkParsing") then
                        Some performanceMetrics.["ChunkParsing"]
                    else
                        None
                
                let chunkCombiningTimeMs = 
                    if performanceMetrics.ContainsKey("ChunkCombining") then
                        Some performanceMetrics.["ChunkCombining"]
                    else
                        None
                
                let chunkCount = 
                    if performanceMetrics.ContainsKey("ChunkCount") then
                        Some (int performanceMetrics.["ChunkCount"])
                    else
                        None
                
                let cachedChunkCount = 
                    if performanceMetrics.ContainsKey("CachedChunkCount") then
                        Some (int performanceMetrics.["CachedChunkCount"])
                    else
                        None
                
                let peakMemoryUsageBytes = 
                    if performanceMetrics.ContainsKey("PeakMemoryUsage") then
                        Some performanceMetrics.["PeakMemoryUsage"]
                    else
                        None
                
                // Extract diagnostics and suppressed warnings
                let diagnostics = 
                    if diagnosticsCollection.ContainsKey(parserType) then
                        Some diagnosticsCollection.[parserType]
                    else
                        None
                
                let suppressedWarnings = 
                    if suppressedWarningsCollection.ContainsKey(parserType) then
                        Some suppressedWarningsCollection.[parserType]
                    else
                        None
                
                let telemetry = 
                    TelemetryCollector.collectTelemetry 
                        parserType 
                        code 
                        program 
                        startTime 
                        endTime 
                        tokenizingTimeMs
                        blockParsingTimeMs
                        propertyParsingTimeMs
                        nestedBlockParsingTimeMs
                        chunkingTimeMs
                        chunkParsingTimeMs
                        chunkCombiningTimeMs
                        chunkCount
                        cachedChunkCount
                        peakMemoryUsageBytes
                        diagnostics
                        suppressedWarnings
                
                // Anonymize telemetry data if configured
                let telemetryToStore = 
                    if config.Anonymize then
                        TelemetryPrivacy.anonymizeTelemetry telemetry
                    else
                        telemetry
                
                // Store telemetry data
                TelemetryStorage.storeTelemetryToFile config.TelemetryDirectory telemetryToStore |> ignore
            
            // Return the parsed program
            program
    
    /// <summary>
    /// Parse a TARS program file with telemetry collection.
    /// </summary>
    /// <param name="parserType">The parser type used.</param>
    /// <param name="parseFileFunction">The parse file function to use.</param>
    /// <param name="filePath">The path to the file containing the TARS program.</param>
    /// <returns>The parsed TarsProgram.</returns>
    let parseFileWithTelemetry (parserType: string) (parseFileFunction: string -> TarsProgram) (filePath: string) =
        if not telemetryEnabled then
            // If telemetry is disabled, just call the parse file function directly
            parseFileFunction filePath
        else
            // Load the telemetry configuration
            let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
            
            // Read the file content
            let code = System.IO.File.ReadAllText(filePath)
            
            // Clear performance metrics
            performanceMetrics.Clear()
            
            // Start the overall stopwatch
            let startTime = DateTime.UtcNow
            let overallStopwatch = Stopwatch.StartNew()
            
            // Parse the code
            let program = parseFileFunction filePath
            
            // Stop the overall stopwatch
            overallStopwatch.Stop()
            let endTime = DateTime.UtcNow
            
            // Collect telemetry data
            if config.CollectUsageTelemetry || config.CollectPerformanceTelemetry then
                // Extract performance metrics
                let tokenizingTimeMs = 
                    if performanceMetrics.ContainsKey("Tokenizing") then
                        Some performanceMetrics.["Tokenizing"]
                    else
                        None
                
                let blockParsingTimeMs = 
                    if performanceMetrics.ContainsKey("BlockParsing") then
                        Some performanceMetrics.["BlockParsing"]
                    else
                        None
                
                let propertyParsingTimeMs = 
                    if performanceMetrics.ContainsKey("PropertyParsing") then
                        Some performanceMetrics.["PropertyParsing"]
                    else
                        None
                
                let nestedBlockParsingTimeMs = 
                    if performanceMetrics.ContainsKey("NestedBlockParsing") then
                        Some performanceMetrics.["NestedBlockParsing"]
                    else
                        None
                
                let chunkingTimeMs = 
                    if performanceMetrics.ContainsKey("Chunking") then
                        Some performanceMetrics.["Chunking"]
                    else
                        None
                
                let chunkParsingTimeMs = 
                    if performanceMetrics.ContainsKey("ChunkParsing") then
                        Some performanceMetrics.["ChunkParsing"]
                    else
                        None
                
                let chunkCombiningTimeMs = 
                    if performanceMetrics.ContainsKey("ChunkCombining") then
                        Some performanceMetrics.["ChunkCombining"]
                    else
                        None
                
                let chunkCount = 
                    if performanceMetrics.ContainsKey("ChunkCount") then
                        Some (int performanceMetrics.["ChunkCount"])
                    else
                        None
                
                let cachedChunkCount = 
                    if performanceMetrics.ContainsKey("CachedChunkCount") then
                        Some (int performanceMetrics.["CachedChunkCount"])
                    else
                        None
                
                let peakMemoryUsageBytes = 
                    if performanceMetrics.ContainsKey("PeakMemoryUsage") then
                        Some performanceMetrics.["PeakMemoryUsage"]
                    else
                        None
                
                // Extract diagnostics and suppressed warnings
                let diagnostics = 
                    if diagnosticsCollection.ContainsKey(parserType) then
                        Some diagnosticsCollection.[parserType]
                    else
                        None
                
                let suppressedWarnings = 
                    if suppressedWarningsCollection.ContainsKey(parserType) then
                        Some suppressedWarningsCollection.[parserType]
                    else
                        None
                
                let telemetry = 
                    TelemetryCollector.collectTelemetry 
                        parserType 
                        code 
                        program 
                        startTime 
                        endTime 
                        tokenizingTimeMs
                        blockParsingTimeMs
                        propertyParsingTimeMs
                        nestedBlockParsingTimeMs
                        chunkingTimeMs
                        chunkParsingTimeMs
                        chunkCombiningTimeMs
                        chunkCount
                        cachedChunkCount
                        peakMemoryUsageBytes
                        diagnostics
                        suppressedWarnings
                
                // Anonymize telemetry data if configured
                let telemetryToStore = 
                    if config.Anonymize then
                        TelemetryPrivacy.anonymizeTelemetry telemetry
                    else
                        telemetry
                
                // Store telemetry data
                TelemetryStorage.storeTelemetryToFile config.TelemetryDirectory telemetryToStore |> ignore
            
            // Return the parsed program
            program
    
    /// <summary>
    /// Record a non-time metric for telemetry.
    /// </summary>
    /// <param name="metricName">The name of the metric.</param>
    /// <param name="value">The value of the metric.</param>
    let recordMetric (metricName: string) (value: int64) =
        if telemetryEnabled then
            performanceMetrics.[metricName] <- value
    
    /// <summary>
    /// Get the current process memory usage in bytes.
    /// </summary>
    /// <returns>The current process memory usage in bytes.</returns>
    let getCurrentMemoryUsage() =
        let process = System.Diagnostics.Process.GetCurrentProcess()
        process.WorkingSet64
