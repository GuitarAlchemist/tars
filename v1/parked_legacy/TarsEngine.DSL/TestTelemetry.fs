namespace TarsEngine.DSL

open System
open System.IO
open System.Text
open Ast

/// <summary>
/// Module for testing the telemetry system.
/// </summary>
module TestTelemetry =
    /// <summary>
    /// Generate a test file.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The generated test content.</returns>
    let generateTestFile (blockCount: int) =
        let sb = StringBuilder()
        
        // Add a header
        sb.Append("// This is a test file for the telemetry system\n") |> ignore
        sb.Append("// It contains a number of blocks to test telemetry collection\n\n") |> ignore
        
        // Add blocks
        for i in 1 .. blockCount do
            sb.Append(sprintf "CONFIG config_%d {\n" i) |> ignore
            sb.Append(sprintf "    name: \"Config %d\",\n" i) |> ignore
            sb.Append(sprintf "    version: \"1.0\",\n" i) |> ignore
            sb.Append(sprintf "    description: \"This is config %d\"\n" i) |> ignore
            sb.Append("}\n\n") |> ignore
            
            sb.Append(sprintf "PROMPT prompt_%d {\n" i) |> ignore
            sb.Append(sprintf "    content: \"This is prompt %d\",\n" i) |> ignore
            sb.Append(sprintf "    content_type: \"text\"\n" i) |> ignore
            sb.Append("}\n\n") |> ignore
            
            sb.Append(sprintf "AGENT agent_%d {\n" i) |> ignore
            sb.Append(sprintf "    name: \"Agent %d\",\n" i) |> ignore
            sb.Append(sprintf "    agent_type: \"standard\",\n" i) |> ignore
            sb.Append(sprintf "    description: \"This is agent %d\",\n" i) |> ignore
            sb.Append("\n") |> ignore
            sb.Append(sprintf "    TASK task_%d {\n" i) |> ignore
            sb.Append(sprintf "        name: \"Task %d\",\n" i) |> ignore
            sb.Append(sprintf "        description: \"This is task %d\",\n" i) |> ignore
            sb.Append("\n") |> ignore
            sb.Append(sprintf "        ACTION action_%d {\n" i) |> ignore
            sb.Append(sprintf "            name: \"Action %d\",\n" i) |> ignore
            sb.Append(sprintf "            description: \"This is action %d\"\n" i) |> ignore
            sb.Append("        }\n") |> ignore
            sb.Append("    }\n") |> ignore
            sb.Append("}\n\n") |> ignore
            
            sb.Append(sprintf "VARIABLE variable_%d {\n" i) |> ignore
            sb.Append(sprintf "    name: \"Variable %d\",\n" i) |> ignore
            sb.Append(sprintf "    value: %d\n" i) |> ignore
            sb.Append("}\n\n") |> ignore
        
        sb.ToString()
    
    /// <summary>
    /// Test the telemetry system with the original parser.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The parse result.</returns>
    let testOriginalParserTelemetry (blockCount: int) =
        // Generate a test file
        let content = generateTestFile blockCount
        
        // Enable telemetry
        TelemetryService.enableTelemetry()
        
        // Parse the content
        let result = Parser.parse content
        
        // Return the result
        result
    
    /// <summary>
    /// Test the telemetry system with the FParsec parser.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The parse result.</returns>
    let testFParsecParserTelemetry (blockCount: int) =
        // Generate a test file
        let content = generateTestFile blockCount
        
        // Enable telemetry
        TelemetryService.enableTelemetry()
        
        // Parse the content
        let result = FParsecParser.parse content
        
        // Return the result
        result
    
    /// <summary>
    /// Test the telemetry system with the incremental parser.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>The parse result.</returns>
    let testIncrementalParserTelemetry (blockCount: int) =
        // Generate a test file
        let content = generateTestFile blockCount
        
        // Enable telemetry
        TelemetryService.enableTelemetry()
        
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
    /// Test the telemetry system with all parsers.
    /// </summary>
    /// <param name="blockCount">The number of blocks to generate.</param>
    /// <returns>A tuple of (originalResult, fparsecResult, incrementalResult).</returns>
    let testAllParsersTelemetry (blockCount: int) =
        // Generate a test file
        let content = generateTestFile blockCount
        
        // Enable telemetry
        TelemetryService.enableTelemetry()
        
        // Parse the content with the original parser
        let originalResult = Parser.parse content
        
        // Parse the content with the FParsec parser
        let fparsecResult = FParsecParser.parse content
        
        // Create a configuration for the incremental parser
        let config = {
            ChunkSize = 100
            UseParallelParsing = true
            UseCache = true
            CollectTelemetry = true
        }
        
        // Parse the content with the incremental parser
        let incrementalResult = IncrementalParser.parseStringIncrementally content Parser.parseInternal config
        
        // Return the results
        (originalResult, fparsecResult, incrementalResult)
    
    /// <summary>
    /// Test the telemetry configuration.
    /// </summary>
    let testTelemetryConfiguration() =
        // Load the default configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        
        // Print the configuration
        printfn "Telemetry configuration:"
        printfn "  Enabled: %b" config.Enabled
        printfn "  Anonymize: %b" config.Anonymize
        printfn "  TelemetryDirectory: %s" config.TelemetryDirectory
        printfn "  CollectUsageTelemetry: %b" config.CollectUsageTelemetry
        printfn "  CollectPerformanceTelemetry: %b" config.CollectPerformanceTelemetry
        printfn "  CollectErrorWarningTelemetry: %b" config.CollectErrorWarningTelemetry
        
        // Enable telemetry
        let enableResult = TelemetryConfiguration.enableTelemetry TelemetryConfiguration.defaultConfigFilePath
        printfn "  Enable telemetry result: %b" enableResult
        
        // Disable telemetry
        let disableResult = TelemetryConfiguration.disableTelemetry TelemetryConfiguration.defaultConfigFilePath
        printfn "  Disable telemetry result: %b" disableResult
        
        // Enable anonymization
        let enableAnonymizationResult = TelemetryConfiguration.enableAnonymization TelemetryConfiguration.defaultConfigFilePath
        printfn "  Enable anonymization result: %b" enableAnonymizationResult
        
        // Disable anonymization
        let disableAnonymizationResult = TelemetryConfiguration.disableAnonymization TelemetryConfiguration.defaultConfigFilePath
        printfn "  Disable anonymization result: %b" disableAnonymizationResult
        
        // Set telemetry directory
        let setDirectoryResult = TelemetryConfiguration.setTelemetryDirectory (Path.Combine(Path.GetTempPath(), "TarsEngineTelemetry")) TelemetryConfiguration.defaultConfigFilePath
        printfn "  Set telemetry directory result: %b" setDirectoryResult
        
        // Configure telemetry collection
        let configureResult = TelemetryConfiguration.configureTelemetryCollection true true true TelemetryConfiguration.defaultConfigFilePath
        printfn "  Configure telemetry collection result: %b" configureResult
        
        // Load the updated configuration
        let updatedConfig = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        
        // Print the updated configuration
        printfn "Updated telemetry configuration:"
        printfn "  Enabled: %b" updatedConfig.Enabled
        printfn "  Anonymize: %b" updatedConfig.Anonymize
        printfn "  TelemetryDirectory: %s" updatedConfig.TelemetryDirectory
        printfn "  CollectUsageTelemetry: %b" updatedConfig.CollectUsageTelemetry
        printfn "  CollectPerformanceTelemetry: %b" updatedConfig.CollectPerformanceTelemetry
        printfn "  CollectErrorWarningTelemetry: %b" updatedConfig.CollectErrorWarningTelemetry
    
    /// <summary>
    /// Test the telemetry storage.
    /// </summary>
    let testTelemetryStorage() =
        // Load the telemetry configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        
        // Load all telemetry data
        let telemetryList = TelemetryStorage.loadAllTelemetry config.TelemetryDirectory
        
        // Print the telemetry data
        printfn "Telemetry data:"
        printfn "  Count: %d" telemetryList.Length
        
        // Aggregate the telemetry data
        let summary = TelemetryStorage.aggregateTelemetry telemetryList
        
        // Print the summary
        printfn "Telemetry summary:"
        printfn "  TelemetryCount: %d" summary.TelemetryCount
        printfn "  ParserTypes: %A" summary.ParserTypes
        printfn "  TotalParseTimeMs: %d" summary.TotalParseTimeMs
        printfn "  AverageParseTimeMs: %d" summary.AverageParseTimeMs
        printfn "  TotalFileSize: %d" summary.TotalFileSize
        printfn "  TotalLineCount: %d" summary.TotalLineCount
        printfn "  TotalBlockCount: %d" summary.TotalBlockCount
        printfn "  TotalPropertyCount: %d" summary.TotalPropertyCount
        printfn "  TotalNestedBlockCount: %d" summary.TotalNestedBlockCount
        printfn "  ErrorCount: %d" summary.ErrorCount
        printfn "  WarningCount: %d" summary.WarningCount
        printfn "  InfoCount: %d" summary.InfoCount
        printfn "  HintCount: %d" summary.HintCount
        printfn "  SuppressedWarningCount: %d" summary.SuppressedWarningCount
        
        // Export the telemetry data
        let exportPath = Path.Combine(Path.GetTempPath(), "TarsEngineTelemetry.json")
        let exportResult = TelemetryStorage.exportTelemetry telemetryList exportPath
        printfn "  Export telemetry result: %b" exportResult
        printfn "  Export path: %s" exportPath
    
    /// <summary>
    /// Test the telemetry reporter.
    /// </summary>
    let testTelemetryReporter() =
        // Load the telemetry configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        
        // Load all telemetry data
        let telemetryList = TelemetryStorage.loadAllTelemetry config.TelemetryDirectory
        
        // Generate a summary report
        let summaryReport = TelemetryReporter.generateSummaryReport telemetryList
        
        // Print the summary report
        printfn "Summary report:"
        printfn "%s" summaryReport
        
        // Generate a detailed report
        let detailedReport = TelemetryReporter.generateDetailedReport telemetryList
        
        // Print the detailed report
        printfn "Detailed report:"
        printfn "%s" detailedReport
        
        // Save the reports to files
        let summaryReportPath = Path.Combine(Path.GetTempPath(), "TarsEngineTelemetrySummary.md")
        let summaryReportResult = TelemetryReporter.saveReportToFile summaryReport summaryReportPath
        printfn "  Save summary report result: %b" summaryReportResult
        printfn "  Summary report path: %s" summaryReportPath
        
        let detailedReportPath = Path.Combine(Path.GetTempPath(), "TarsEngineTelemetryDetailed.md")
        let detailedReportResult = TelemetryReporter.saveReportToFile detailedReport detailedReportPath
        printfn "  Save detailed report result: %b" detailedReportResult
        printfn "  Detailed report path: %s" detailedReportPath
    
    /// <summary>
    /// Test the telemetry privacy.
    /// </summary>
    let testTelemetryPrivacy() =
        // Load the telemetry configuration
        let config = TelemetryConfiguration.loadConfig TelemetryConfiguration.defaultConfigFilePath
        
        // Load all telemetry data
        let telemetryList = TelemetryStorage.loadAllTelemetry config.TelemetryDirectory
        
        // Anonymize the telemetry data
        let anonymizedTelemetryList = telemetryList |> List.map TelemetryPrivacy.anonymizeTelemetry
        
        // Print the anonymized telemetry data
        printfn "Anonymized telemetry data:"
        printfn "  Count: %d" anonymizedTelemetryList.Length
        
        // Purge the telemetry data
        let purgeCount = TelemetryPrivacy.purgeTelemetry (Path.Combine(Path.GetTempPath(), "TarsEngineTelemetryTest"))
        printfn "  Purge count: %d" purgeCount
