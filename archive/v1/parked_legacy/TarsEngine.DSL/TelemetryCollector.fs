namespace TarsEngine.DSL

open System
open System.Diagnostics
open System.Runtime.InteropServices
open Ast

/// <summary>
/// Module for collecting telemetry data from the TARS DSL parser.
/// </summary>
module TelemetryCollector =
    /// <summary>
    /// Get the current parser version.
    /// </summary>
    /// <returns>The current parser version.</returns>
    let getParserVersion() =
        // In a real implementation, this would be read from assembly metadata
        "1.0.0"
    
    /// <summary>
    /// Get the current operating system.
    /// </summary>
    /// <returns>The current operating system.</returns>
    let getOperatingSystem() =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            "Windows"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            "Linux"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            "macOS"
        else
            "Unknown"
    
    /// <summary>
    /// Get the current .NET runtime version.
    /// </summary>
    /// <returns>The current .NET runtime version.</returns>
    let getRuntimeVersion() =
        Environment.Version.ToString()
    
    /// <summary>
    /// Count the number of lines in a string.
    /// </summary>
    /// <param name="content">The string to count lines in.</param>
    /// <returns>The number of lines in the string.</returns>
    let countLines (content: string) =
        if String.IsNullOrEmpty(content) then
            0
        else
            content.Split([|'\n'|], StringSplitOptions.None).Length
    
    /// <summary>
    /// Count the number of properties in a program.
    /// </summary>
    /// <param name="program">The program to count properties in.</param>
    /// <returns>The number of properties in the program.</returns>
    let rec countProperties (program: TarsProgram) =
        let mutable count = 0
        
        let rec countBlockProperties (block: TarsBlock) =
            count <- count + block.Properties.Count
            
            for nestedBlock in block.NestedBlocks do
                countBlockProperties nestedBlock
        
        for block in program.Blocks do
            countBlockProperties block
        
        count
    
    /// <summary>
    /// Count the number of nested blocks in a program.
    /// </summary>
    /// <param name="program">The program to count nested blocks in.</param>
    /// <returns>The number of nested blocks in the program.</returns>
    let rec countNestedBlocks (program: TarsProgram) =
        let mutable count = 0
        
        let rec countBlockNestedBlocks (block: TarsBlock) =
            count <- count + block.NestedBlocks.Length
            
            for nestedBlock in block.NestedBlocks do
                countBlockNestedBlocks nestedBlock
        
        for block in program.Blocks do
            countBlockNestedBlocks block
        
        count
    
    /// <summary>
    /// Collect parser usage telemetry.
    /// </summary>
    /// <param name="parserType">The parser type used.</param>
    /// <param name="content">The content being parsed.</param>
    /// <param name="program">The parsed program.</param>
    /// <param name="startTime">The time when parsing started.</param>
    /// <param name="endTime">The time when parsing completed.</param>
    /// <returns>The parser usage telemetry.</returns>
    let collectParserUsageTelemetry (parserType: string) (content: string) (program: TarsProgram) (startTime: DateTime) (endTime: DateTime) =
        {
            ParserType = parserType
            FileSizeBytes = int64 content.Length
            LineCount = countLines content
            BlockCount = program.Blocks.Length
            PropertyCount = countProperties program
            NestedBlockCount = countNestedBlocks program
            StartTimestamp = startTime
            EndTimestamp = endTime
            TotalParseTimeMs = int64 (endTime - startTime).TotalMilliseconds
        }
    
    /// <summary>
    /// Collect parsing performance telemetry.
    /// </summary>
    /// <param name="parserType">The parser type used.</param>
    /// <param name="content">The content being parsed.</param>
    /// <param name="totalParseTimeMs">The total parsing time in milliseconds.</param>
    /// <param name="tokenizingTimeMs">The time spent tokenizing in milliseconds.</param>
    /// <param name="blockParsingTimeMs">The time spent parsing blocks in milliseconds.</param>
    /// <param name="propertyParsingTimeMs">The time spent parsing properties in milliseconds.</param>
    /// <param name="nestedBlockParsingTimeMs">The time spent parsing nested blocks in milliseconds.</param>
    /// <param name="chunkingTimeMs">The time spent chunking in milliseconds (for incremental parsing).</param>
    /// <param name="chunkParsingTimeMs">The time spent parsing chunks in milliseconds (for incremental parsing).</param>
    /// <param name="chunkCombiningTimeMs">The time spent combining chunks in milliseconds (for incremental parsing).</param>
    /// <param name="chunkCount">The number of chunks (for incremental parsing).</param>
    /// <param name="cachedChunkCount">The number of cached chunks (for incremental parsing).</param>
    /// <param name="peakMemoryUsageBytes">The peak memory usage in bytes.</param>
    /// <returns>The parsing performance telemetry.</returns>
    let collectParsingPerformanceTelemetry 
        (parserType: string) 
        (content: string) 
        (totalParseTimeMs: int64) 
        (tokenizingTimeMs: int64 option) 
        (blockParsingTimeMs: int64 option) 
        (propertyParsingTimeMs: int64 option) 
        (nestedBlockParsingTimeMs: int64 option) 
        (chunkingTimeMs: int64 option) 
        (chunkParsingTimeMs: int64 option) 
        (chunkCombiningTimeMs: int64 option) 
        (chunkCount: int option) 
        (cachedChunkCount: int option) 
        (peakMemoryUsageBytes: int64 option) =
        {
            ParserType = parserType
            FileSizeBytes = int64 content.Length
            LineCount = countLines content
            TotalParseTimeMs = totalParseTimeMs
            TokenizingTimeMs = tokenizingTimeMs
            BlockParsingTimeMs = blockParsingTimeMs
            PropertyParsingTimeMs = propertyParsingTimeMs
            NestedBlockParsingTimeMs = nestedBlockParsingTimeMs
            ChunkingTimeMs = chunkingTimeMs
            ChunkParsingTimeMs = chunkParsingTimeMs
            ChunkCombiningTimeMs = chunkCombiningTimeMs
            ChunkCount = chunkCount
            CachedChunkCount = cachedChunkCount
            PeakMemoryUsageBytes = peakMemoryUsageBytes
        }
    
    /// <summary>
    /// Collect error and warning telemetry.
    /// </summary>
    /// <param name="parserType">The parser type used.</param>
    /// <param name="content">The content being parsed.</param>
    /// <param name="diagnostics">The diagnostics generated during parsing.</param>
    /// <param name="suppressedWarnings">The suppressed warnings.</param>
    /// <returns>The error and warning telemetry.</returns>
    let collectErrorWarningTelemetry (parserType: string) (content: string) (diagnostics: Diagnostic list) (suppressedWarnings: (int * WarningCode) list) =
        let errorCount = diagnostics |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Error) |> List.length
        let warningCount = diagnostics |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Warning) |> List.length
        let infoCount = diagnostics |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Information) |> List.length
        let hintCount = diagnostics |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Hint) |> List.length
        
        let errorCodes = 
            diagnostics 
            |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Error) 
            |> List.groupBy (fun d -> d.Code.ToString()) 
            |> List.map (fun (code, diagnostics) -> (code, List.length diagnostics)) 
            |> Map.ofList
        
        let warningCodes = 
            diagnostics 
            |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Warning) 
            |> List.groupBy (fun d -> d.Code.ToString()) 
            |> List.map (fun (code, diagnostics) -> (code, List.length diagnostics)) 
            |> Map.ofList
        
        let suppressedWarningCodes = 
            suppressedWarnings 
            |> List.groupBy (fun (_, code) -> code.ToString()) 
            |> List.map (fun (code, warnings) -> (code, List.length warnings)) 
            |> Map.ofList
        
        {
            ParserType = parserType
            FileSizeBytes = int64 content.Length
            LineCount = countLines content
            ErrorCount = errorCount
            WarningCount = warningCount
            InfoCount = infoCount
            HintCount = hintCount
            ErrorCodes = errorCodes
            WarningCodes = warningCodes
            SuppressedWarningCount = suppressedWarnings.Length
            SuppressedWarningCodes = suppressedWarningCodes
        }
    
    /// <summary>
    /// Collect telemetry data.
    /// </summary>
    /// <param name="parserType">The parser type used.</param>
    /// <param name="content">The content being parsed.</param>
    /// <param name="program">The parsed program.</param>
    /// <param name="startTime">The time when parsing started.</param>
    /// <param name="endTime">The time when parsing completed.</param>
    /// <param name="tokenizingTimeMs">The time spent tokenizing in milliseconds.</param>
    /// <param name="blockParsingTimeMs">The time spent parsing blocks in milliseconds.</param>
    /// <param name="propertyParsingTimeMs">The time spent parsing properties in milliseconds.</param>
    /// <param name="nestedBlockParsingTimeMs">The time spent parsing nested blocks in milliseconds.</param>
    /// <param name="chunkingTimeMs">The time spent chunking in milliseconds (for incremental parsing).</param>
    /// <param name="chunkParsingTimeMs">The time spent parsing chunks in milliseconds (for incremental parsing).</param>
    /// <param name="chunkCombiningTimeMs">The time spent combining chunks in milliseconds (for incremental parsing).</param>
    /// <param name="chunkCount">The number of chunks (for incremental parsing).</param>
    /// <param name="cachedChunkCount">The number of cached chunks (for incremental parsing).</param>
    /// <param name="peakMemoryUsageBytes">The peak memory usage in bytes.</param>
    /// <param name="diagnostics">The diagnostics generated during parsing.</param>
    /// <param name="suppressedWarnings">The suppressed warnings.</param>
    /// <returns>The telemetry data.</returns>
    let collectTelemetry 
        (parserType: string) 
        (content: string) 
        (program: TarsProgram) 
        (startTime: DateTime) 
        (endTime: DateTime) 
        (tokenizingTimeMs: int64 option) 
        (blockParsingTimeMs: int64 option) 
        (propertyParsingTimeMs: int64 option) 
        (nestedBlockParsingTimeMs: int64 option) 
        (chunkingTimeMs: int64 option) 
        (chunkParsingTimeMs: int64 option) 
        (chunkCombiningTimeMs: int64 option) 
        (chunkCount: int option) 
        (cachedChunkCount: int option) 
        (peakMemoryUsageBytes: int64 option) 
        (diagnostics: Diagnostic list option) 
        (suppressedWarnings: (int * WarningCode) list option) =
        let usageTelemetry = collectParserUsageTelemetry parserType content program startTime endTime
        
        let performanceTelemetry = 
            collectParsingPerformanceTelemetry 
                parserType 
                content 
                usageTelemetry.TotalParseTimeMs 
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
        
        let errorWarningTelemetry = 
            match diagnostics, suppressedWarnings with
            | Some diags, Some suppressed -> 
                Some (collectErrorWarningTelemetry parserType content diags suppressed)
            | _ -> 
                None
        
        {
            Id = Guid.NewGuid()
            Timestamp = DateTime.UtcNow
            ParserVersion = getParserVersion()
            OperatingSystem = getOperatingSystem()
            RuntimeVersion = getRuntimeVersion()
            UsageTelemetry = usageTelemetry
            PerformanceTelemetry = performanceTelemetry
            ErrorWarningTelemetry = errorWarningTelemetry
        }
