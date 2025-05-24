namespace TarsEngine.DSL

open System
open System.IO
open System.Text
open System.Diagnostics
open Ast

/// <summary>
/// Module for testing the enhanced telemetry system.
/// </summary>
module TestEnhancedTelemetry =
    /// <summary>
    /// Generate a test file with the specified number of blocks.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The generated test file content.</returns>
    let generateTestFile (blockCount: int) =
        let sb = StringBuilder()
        
        sb.Append("DESCRIBE {\n") |> ignore
        sb.Append("    name: \"Test Program\",\n") |> ignore
        sb.Append("    version: \"1.0\",\n") |> ignore
        sb.Append("    description: \"A test program for telemetry\"\n") |> ignore
        sb.Append("}\n\n") |> ignore
        
        for i in 1..blockCount do
            sb.Append(sprintf "CONFIG_%d {\n" i) |> ignore
            sb.Append(sprintf "    setting1: \"value%d\",\n" i) |> ignore
            sb.Append(sprintf "    setting2: %d,\n" i) |> ignore
            sb.Append(sprintf "    setting3: true,\n") |> ignore
            sb.Append("    nested: {\n") |> ignore
            sb.Append(sprintf "        nestedSetting1: \"nestedValue%d\",\n" i) |> ignore
            sb.Append(sprintf "        nestedSetting2: %d\n" (i * 2)) |> ignore
            sb.Append("    }\n") |> ignore
            sb.Append("}\n\n") |> ignore
            
            sb.Append(sprintf "VARIABLE variable_%d {\n" i) |> ignore
            sb.Append(sprintf "    name: \"Variable %d\",\n" i) |> ignore
            sb.Append(sprintf "    value: %d\n" i) |> ignore
            sb.Append("}\n\n") |> ignore
        
        sb.ToString()
    
    /// <summary>
    /// Test the enhanced telemetry system with detailed metrics.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The parse result.</returns>
    let testEnhancedTelemetry (blockCount: int) =
        // Generate a test file
        let content = generateTestFile blockCount
        
        // Enable telemetry
        TelemetryService.enableTelemetry()
        
        // Simulate detailed metrics collection
        let tokenizingMetric = TelemetryService.startMeasuring "Tokenizing"
        // Simulate tokenizing
        System.Threading.Thread.Sleep(10)
        TelemetryService.stopMeasuring tokenizingMetric
        
        let blockParsingMetric = TelemetryService.startMeasuring "BlockParsing"
        // Simulate block parsing
        System.Threading.Thread.Sleep(20)
        TelemetryService.stopMeasuring blockParsingMetric
        
        let propertyParsingMetric = TelemetryService.startMeasuring "PropertyParsing"
        // Simulate property parsing
        System.Threading.Thread.Sleep(15)
        TelemetryService.stopMeasuring propertyParsingMetric
        
        let nestedBlockParsingMetric = TelemetryService.startMeasuring "NestedBlockParsing"
        // Simulate nested block parsing
        System.Threading.Thread.Sleep(25)
        TelemetryService.stopMeasuring nestedBlockParsingMetric
        
        // Record non-time metrics
        TelemetryService.recordMetric "ChunkCount" 10L
        TelemetryService.recordMetric "CachedChunkCount" 5L
        TelemetryService.recordMetric "PeakMemoryUsage" (TelemetryService.getCurrentMemoryUsage())
        
        // Record diagnostics
        let diagnostics = [
            { 
                Severity = DiagnosticSeverity.Warning
                Code = WarningCode.DeprecatedBlockType
                Message = "Deprecated block type"
                Line = 1
                Column = 1
                Length = 10
            }
            {
                Severity = DiagnosticSeverity.Error
                Code = WarningCode.MissingRequiredProperty
                Message = "Missing required property"
                Line = 2
                Column = 5
                Length = 15
            }
        ]
        TelemetryService.recordDiagnostics "Original" diagnostics
        
        // Record suppressed warnings
        let suppressedWarnings = [
            (5, WarningCode.DeprecatedPropertyValue)
            (10, WarningCode.DeepNesting)
        ]
        TelemetryService.recordSuppressedWarnings "Original" suppressedWarnings
        
        // Parse the content
        let result = Parser.parse content
        
        // Return the result
        result
    
    /// <summary>
    /// Test the enhanced telemetry system with the incremental parser.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The parse result.</returns>
    let testEnhancedIncrementalTelemetry (blockCount: int) =
        // Generate a test file
        let content = generateTestFile blockCount
        
        // Enable telemetry
        TelemetryService.enableTelemetry()
        
        // Simulate detailed metrics collection for incremental parsing
        let chunkingMetric = TelemetryService.startMeasuring "Chunking"
        // Simulate chunking
        System.Threading.Thread.Sleep(30)
        TelemetryService.stopMeasuring chunkingMetric
        
        let chunkParsingMetric = TelemetryService.startMeasuring "ChunkParsing"
        // Simulate chunk parsing
        System.Threading.Thread.Sleep(40)
        TelemetryService.stopMeasuring chunkParsingMetric
        
        let chunkCombiningMetric = TelemetryService.startMeasuring "ChunkCombining"
        // Simulate chunk combining
        System.Threading.Thread.Sleep(20)
        TelemetryService.stopMeasuring chunkCombiningMetric
        
        // Record non-time metrics
        TelemetryService.recordMetric "ChunkCount" 20L
        TelemetryService.recordMetric "CachedChunkCount" 10L
        TelemetryService.recordMetric "PeakMemoryUsage" (TelemetryService.getCurrentMemoryUsage())
        
        // Create a configuration
        let config = {
            ChunkSize = 100
            UseParallelParsing = true
            UseCache = true
            CollectTelemetry = true
        }
        
        // Parse the content
        let result = IncrementalParser.parseStringIncrementally content Parser.parseInternal config
        
        // Return the result
        result
    
    /// <summary>
    /// Test the enhanced telemetry system with all parsers.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>A tuple of (originalResult, fparsecResult, incrementalResult).</returns>
    let testAllParsersEnhancedTelemetry (blockCount: int) =
        // Test the original parser
        let originalResult = testEnhancedTelemetry blockCount
        
        // Test the incremental parser
        let incrementalResult = testEnhancedIncrementalTelemetry blockCount
        
        // Return the results
        (originalResult, incrementalResult)
    
    /// <summary>
    /// Run all enhanced telemetry tests.
    /// </summary>
    let runAllEnhancedTelemetryTests() =
        // Test with a small number of blocks
        let smallResult = testAllParsersEnhancedTelemetry 5
        printfn "Small test completed"
        
        // Test with a medium number of blocks
        let mediumResult = testAllParsersEnhancedTelemetry 50
        printfn "Medium test completed"
        
        // Test with a large number of blocks
        let largeResult = testAllParsersEnhancedTelemetry 500
        printfn "Large test completed"
        
        // Load the telemetry configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        
        // Load all telemetry data
        let telemetryList = TelemetryStorage.loadAllTelemetry config.TelemetryDirectory
        
        // Generate a summary report
        let summaryReport = TelemetryReporter.generateSummaryReport telemetryList
        
        // Generate a detailed report
        let detailedReport = TelemetryReporter.generateDetailedReport telemetryList
        
        // Save the reports to files
        let summaryReportPath = Path.Combine(Path.GetTempPath(), "TarsEngineTelemetrySummary.md")
        let summaryReportResult = TelemetryReporter.saveReportToFile summaryReport summaryReportPath
        printfn "Summary report saved to: %s" summaryReportPath
        
        let detailedReportPath = Path.Combine(Path.GetTempPath(), "TarsEngineTelemetryDetailed.md")
        let detailedReportResult = TelemetryReporter.saveReportToFile detailedReport detailedReportPath
        printfn "Detailed report saved to: %s" detailedReportPath
        
        // Return the results
        (smallResult, mediumResult, largeResult)
