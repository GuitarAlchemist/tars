namespace TarsEngine.DSL

open System
open System.IO
open Ast

/// <summary>
/// Module demonstrating how to integrate the enhanced telemetry service with a parser.
/// </summary>
module ParserWithEnhancedTelemetry =
    /// <summary>
    /// Parse a TARS program string into a structured TarsProgram with enhanced telemetry collection
    /// </summary>
    /// <param name="code">The TARS program string to parse</param>
    /// <returns>The parsed TarsProgram</returns>
    let parse (code: string) =
        // Check if telemetry is enabled
        if not (TelemetryService.isTelemetryEnabled()) then
            // If telemetry is disabled, just call the parse function directly
            Parser.parseInternal code
        else
            // Start overall parsing measurement
            let overallMetric = TelemetryService.startMeasuring "Overall"
            
            // Start tokenizing measurement
            let tokenizingMetric = TelemetryService.startMeasuring "Tokenizing"
            
            // Tokenize the code (simulated here)
            let tokens = code.Split([|' '; '\n'; '\r'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
            
            // Stop tokenizing measurement
            TelemetryService.stopMeasuring tokenizingMetric
            
            // Start block parsing measurement
            let blockParsingMetric = TelemetryService.startMeasuring "BlockParsing"
            
            // Parse blocks (actual parsing happens here)
            let program = Parser.parseInternal code
            
            // Stop block parsing measurement
            TelemetryService.stopMeasuring blockParsingMetric
            
            // Record memory usage
            TelemetryService.recordMetric "PeakMemoryUsage" (TelemetryService.getCurrentMemoryUsage())
            
            // Record diagnostics if available
            let diagnostics = 
                // In a real implementation, you would extract diagnostics from the parsing process
                // For this example, we'll create some sample diagnostics
                [
                    {
                        Severity = DiagnosticSeverity.Warning
                        Code = WarningCode.DeprecatedBlockType
                        Message = "Block type 'OLD_BLOCK' is deprecated"
                        Line = 10
                        Column = 1
                        Length = 9
                    }
                    {
                        Severity = DiagnosticSeverity.Info
                        Code = WarningCode.NamingConvention
                        Message = "Block name should be in PascalCase"
                        Line = 15
                        Column = 1
                        Length = 12
                    }
                ]
            
            // Record the diagnostics
            TelemetryService.recordDiagnostics "EnhancedParser" diagnostics
            
            // Record suppressed warnings if available
            let suppressedWarnings = 
                // In a real implementation, you would extract suppressed warnings from the parsing process
                // For this example, we'll create some sample suppressed warnings
                [
                    (20, WarningCode.DeepNesting)
                    (25, WarningCode.ProblematicNesting)
                ]
            
            // Record the suppressed warnings
            TelemetryService.recordSuppressedWarnings "EnhancedParser" suppressedWarnings
            
            // Stop overall parsing measurement
            TelemetryService.stopMeasuring overallMetric
            
            // Use the telemetry service to collect and store telemetry data
            TelemetryService.parseWithTelemetry "EnhancedParser" Parser.parseInternal code
    
    /// <summary>
    /// Parse a TARS program file into a structured TarsProgram with enhanced telemetry collection
    /// </summary>
    /// <param name="filePath">The path to the file containing the TARS program</param>
    /// <returns>The parsed TarsProgram</returns>
    let parseFile (filePath: string) =
        let code = File.ReadAllText(filePath)
        parse code
    
    /// <summary>
    /// Example of using the enhanced telemetry service with the incremental parser
    /// </summary>
    /// <param name="code">The TARS program string to parse</param>
    /// <returns>The parsed TarsProgram</returns>
    let parseIncrementally (code: string) =
        // Check if telemetry is enabled
        if not (TelemetryService.isTelemetryEnabled()) then
            // If telemetry is disabled, just call the parse function directly
            let config = {
                ChunkSize = 100
                UseParallelParsing = true
                UseCache = true
                CollectTelemetry = false
            }
            IncrementalParser.parseStringIncrementally code Parser.parseInternal config
        else
            // Start overall parsing measurement
            let overallMetric = TelemetryService.startMeasuring "Overall"
            
            // Start chunking measurement
            let chunkingMetric = TelemetryService.startMeasuring "Chunking"
            
            // Chunk the code (simulated here)
            let chunkSize = 100
            let chunks = 
                [0 .. chunkSize .. (code.Length - 1)]
                |> List.map (fun i -> 
                    let length = min chunkSize (code.Length - i)
                    code.Substring(i, length))
            
            // Stop chunking measurement
            TelemetryService.stopMeasuring chunkingMetric
            
            // Record chunk count
            TelemetryService.recordMetric "ChunkCount" (int64 chunks.Length)
            
            // Start chunk parsing measurement
            let chunkParsingMetric = TelemetryService.startMeasuring "ChunkParsing"
            
            // Create a configuration
            let config = {
                ChunkSize = chunkSize
                UseParallelParsing = true
                UseCache = true
                CollectTelemetry = true
            }
            
            // Parse the code incrementally
            let result = IncrementalParser.parseStringIncrementally code Parser.parseInternal config
            
            // Stop chunk parsing measurement
            TelemetryService.stopMeasuring chunkParsingMetric
            
            // Start chunk combining measurement
            let chunkCombiningMetric = TelemetryService.startMeasuring "ChunkCombining"
            
            // Combining happens inside the incremental parser, so we're just simulating the timing here
            System.Threading.Thread.Sleep(10)
            
            // Stop chunk combining measurement
            TelemetryService.stopMeasuring chunkCombiningMetric
            
            // Record memory usage
            TelemetryService.recordMetric "PeakMemoryUsage" (TelemetryService.getCurrentMemoryUsage())
            
            // Record cached chunk count (simulated)
            TelemetryService.recordMetric "CachedChunkCount" (int64 (chunks.Length / 2))
            
            // Stop overall parsing measurement
            TelemetryService.stopMeasuring overallMetric
            
            // Return the result
            result
