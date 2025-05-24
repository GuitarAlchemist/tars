namespace TarsEngine.DSL

open System
open System.Diagnostics
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
    /// Initialize the telemetry service.
    /// </summary>
    let initialize() =
        // Load the telemetry configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        telemetryEnabled <- config.Enabled
    
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
            
            // Start the stopwatch
            let startTime = DateTime.UtcNow
            let stopwatch = Stopwatch.StartNew()
            
            // Parse the code
            let program = parseFunction code
            
            // Stop the stopwatch
            stopwatch.Stop()
            let endTime = DateTime.UtcNow
            
            // Collect telemetry data
            if config.CollectUsageTelemetry || config.CollectPerformanceTelemetry then
                let telemetry = 
                    TelemetryCollector.collectTelemetry 
                        parserType 
                        code 
                        program 
                        startTime 
                        endTime 
                        None // tokenizingTimeMs
                        None // blockParsingTimeMs
                        None // propertyParsingTimeMs
                        None // nestedBlockParsingTimeMs
                        None // chunkingTimeMs
                        None // chunkParsingTimeMs
                        None // chunkCombiningTimeMs
                        None // chunkCount
                        None // cachedChunkCount
                        None // peakMemoryUsageBytes
                        None // diagnostics
                        None // suppressedWarnings
                
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
            
            // Start the stopwatch
            let startTime = DateTime.UtcNow
            let stopwatch = Stopwatch.StartNew()
            
            // Parse the code
            let program = parseFileFunction filePath
            
            // Stop the stopwatch
            stopwatch.Stop()
            let endTime = DateTime.UtcNow
            
            // Collect telemetry data
            if config.CollectUsageTelemetry || config.CollectPerformanceTelemetry then
                let telemetry = 
                    TelemetryCollector.collectTelemetry 
                        parserType 
                        code 
                        program 
                        startTime 
                        endTime 
                        None // tokenizingTimeMs
                        None // blockParsingTimeMs
                        None // propertyParsingTimeMs
                        None // nestedBlockParsingTimeMs
                        None // chunkingTimeMs
                        None // chunkParsingTimeMs
                        None // chunkCombiningTimeMs
                        None // chunkCount
                        None // cachedChunkCount
                        None // peakMemoryUsageBytes
                        None // diagnostics
                        None // suppressedWarnings
                
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
