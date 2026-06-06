namespace TarsEngine.DSL

open System
open System.IO
open System.Text

/// <summary>
/// Module for generating reports from telemetry data.
/// </summary>
module TelemetryReporter =
    /// <summary>
    /// Generate a summary report from telemetry data.
    /// </summary>
    /// <param name="telemetryList">The list of telemetry data to generate a report from.</param>
    /// <returns>A string containing the summary report.</returns>
    let generateSummaryReport (telemetryList: TelemetryData list) =
        let sb = StringBuilder()
        
        sb.AppendLine("# TARS DSL Parser Telemetry Summary Report") |> ignore
        sb.AppendLine($"Generated: {DateTime.Now}") |> ignore
        sb.AppendLine() |> ignore
        
        let summary = TelemetryStorage.aggregateTelemetry telemetryList
        
        sb.AppendLine("## Overview") |> ignore
        sb.AppendLine($"- Total telemetry records: {summary.TelemetryCount}") |> ignore
        sb.AppendLine($"- Parser types used: {String.Join(", ", summary.ParserTypes)}") |> ignore
        sb.AppendLine($"- Total parse time: {summary.TotalParseTimeMs} ms") |> ignore
        sb.AppendLine($"- Average parse time: {summary.AverageParseTimeMs} ms") |> ignore
        sb.AppendLine($"- Total file size: {summary.TotalFileSize} bytes") |> ignore
        sb.AppendLine($"- Total line count: {summary.TotalLineCount}") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("## Parser Usage") |> ignore
        sb.AppendLine($"- Total blocks: {summary.TotalBlockCount}") |> ignore
        sb.AppendLine($"- Total properties: {summary.TotalPropertyCount}") |> ignore
        sb.AppendLine($"- Total nested blocks: {summary.TotalNestedBlockCount}") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("## Errors and Warnings") |> ignore
        sb.AppendLine($"- Total errors: {summary.ErrorCount}") |> ignore
        sb.AppendLine($"- Total warnings: {summary.WarningCount}") |> ignore
        sb.AppendLine($"- Total info messages: {summary.InfoCount}") |> ignore
        sb.AppendLine($"- Total hints: {summary.HintCount}") |> ignore
        sb.AppendLine($"- Total suppressed warnings: {summary.SuppressedWarningCount}") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("## Parser Type Breakdown") |> ignore
        
        let parserTypeGroups = 
            telemetryList 
            |> List.groupBy (fun t -> t.UsageTelemetry.ParserType)
        
        for (parserType, telemetry) in parserTypeGroups do
            let parserTypeSummary = TelemetryStorage.aggregateTelemetry telemetry
            
            sb.AppendLine($"### {parserType}") |> ignore
            sb.AppendLine($"- Count: {telemetry.Length}") |> ignore
            sb.AppendLine($"- Total parse time: {parserTypeSummary.TotalParseTimeMs} ms") |> ignore
            sb.AppendLine($"- Average parse time: {parserTypeSummary.AverageParseTimeMs} ms") |> ignore
            sb.AppendLine($"- Total blocks: {parserTypeSummary.TotalBlockCount}") |> ignore
            sb.AppendLine($"- Total errors: {parserTypeSummary.ErrorCount}") |> ignore
            sb.AppendLine($"- Total warnings: {parserTypeSummary.WarningCount}") |> ignore
            sb.AppendLine() |> ignore
        
        sb.ToString()
    
    /// <summary>
    /// Generate a detailed report from telemetry data.
    /// </summary>
    /// <param name="telemetryList">The list of telemetry data to generate a report from.</param>
    /// <returns>A string containing the detailed report.</returns>
    let generateDetailedReport (telemetryList: TelemetryData list) =
        let sb = StringBuilder()
        
        sb.AppendLine("# TARS DSL Parser Telemetry Detailed Report") |> ignore
        sb.AppendLine($"Generated: {DateTime.Now}") |> ignore
        sb.AppendLine() |> ignore
        
        let summary = TelemetryStorage.aggregateTelemetry telemetryList
        
        sb.AppendLine("## Overview") |> ignore
        sb.AppendLine($"- Total telemetry records: {summary.TelemetryCount}") |> ignore
        sb.AppendLine($"- Parser types used: {String.Join(", ", summary.ParserTypes)}") |> ignore
        sb.AppendLine($"- Total parse time: {summary.TotalParseTimeMs} ms") |> ignore
        sb.AppendLine($"- Average parse time: {summary.AverageParseTimeMs} ms") |> ignore
        sb.AppendLine($"- Total file size: {summary.TotalFileSize} bytes") |> ignore
        sb.AppendLine($"- Total line count: {summary.TotalLineCount}") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("## Parser Usage") |> ignore
        sb.AppendLine($"- Total blocks: {summary.TotalBlockCount}") |> ignore
        sb.AppendLine($"- Total properties: {summary.TotalPropertyCount}") |> ignore
        sb.AppendLine($"- Total nested blocks: {summary.TotalNestedBlockCount}") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("## Errors and Warnings") |> ignore
        sb.AppendLine($"- Total errors: {summary.ErrorCount}") |> ignore
        sb.AppendLine($"- Total warnings: {summary.WarningCount}") |> ignore
        sb.AppendLine($"- Total info messages: {summary.InfoCount}") |> ignore
        sb.AppendLine($"- Total hints: {summary.HintCount}") |> ignore
        sb.AppendLine($"- Total suppressed warnings: {summary.SuppressedWarningCount}") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("## Parser Type Breakdown") |> ignore
        
        let parserTypeGroups = 
            telemetryList 
            |> List.groupBy (fun t -> t.UsageTelemetry.ParserType)
        
        for (parserType, telemetry) in parserTypeGroups do
            let parserTypeSummary = TelemetryStorage.aggregateTelemetry telemetry
            
            sb.AppendLine($"### {parserType}") |> ignore
            sb.AppendLine($"- Count: {telemetry.Length}") |> ignore
            sb.AppendLine($"- Total parse time: {parserTypeSummary.TotalParseTimeMs} ms") |> ignore
            sb.AppendLine($"- Average parse time: {parserTypeSummary.AverageParseTimeMs} ms") |> ignore
            sb.AppendLine($"- Total blocks: {parserTypeSummary.TotalBlockCount}") |> ignore
            sb.AppendLine($"- Total errors: {parserTypeSummary.ErrorCount}") |> ignore
            sb.AppendLine($"- Total warnings: {parserTypeSummary.WarningCount}") |> ignore
            sb.AppendLine() |> ignore
            
            // Performance breakdown
            sb.AppendLine("#### Performance Breakdown") |> ignore
            
            let tokenizingTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.TokenizingTimeMs)
            
            let blockParsingTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.BlockParsingTimeMs)
            
            let propertyParsingTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.PropertyParsingTimeMs)
            
            let nestedBlockParsingTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.NestedBlockParsingTimeMs)
            
            let chunkingTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.ChunkingTimeMs)
            
            let chunkParsingTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.ChunkParsingTimeMs)
            
            let chunkCombiningTimes = 
                telemetry 
                |> List.choose (fun t -> t.PerformanceTelemetry.ChunkCombiningTimeMs)
            
            if tokenizingTimes.Length > 0 then
                let totalTokenizingTime = tokenizingTimes |> List.sum
                let averageTokenizingTime = totalTokenizingTime / int64 tokenizingTimes.Length
                sb.AppendLine($"- Tokenizing: {totalTokenizingTime} ms total, {averageTokenizingTime} ms average") |> ignore
            
            if blockParsingTimes.Length > 0 then
                let totalBlockParsingTime = blockParsingTimes |> List.sum
                let averageBlockParsingTime = totalBlockParsingTime / int64 blockParsingTimes.Length
                sb.AppendLine($"- Block parsing: {totalBlockParsingTime} ms total, {averageBlockParsingTime} ms average") |> ignore
            
            if propertyParsingTimes.Length > 0 then
                let totalPropertyParsingTime = propertyParsingTimes |> List.sum
                let averagePropertyParsingTime = totalPropertyParsingTime / int64 propertyParsingTimes.Length
                sb.AppendLine($"- Property parsing: {totalPropertyParsingTime} ms total, {averagePropertyParsingTime} ms average") |> ignore
            
            if nestedBlockParsingTimes.Length > 0 then
                let totalNestedBlockParsingTime = nestedBlockParsingTimes |> List.sum
                let averageNestedBlockParsingTime = totalNestedBlockParsingTime / int64 nestedBlockParsingTimes.Length
                sb.AppendLine($"- Nested block parsing: {totalNestedBlockParsingTime} ms total, {averageNestedBlockParsingTime} ms average") |> ignore
            
            if chunkingTimes.Length > 0 then
                let totalChunkingTime = chunkingTimes |> List.sum
                let averageChunkingTime = totalChunkingTime / int64 chunkingTimes.Length
                sb.AppendLine($"- Chunking: {totalChunkingTime} ms total, {averageChunkingTime} ms average") |> ignore
            
            if chunkParsingTimes.Length > 0 then
                let totalChunkParsingTime = chunkParsingTimes |> List.sum
                let averageChunkParsingTime = totalChunkParsingTime / int64 chunkParsingTimes.Length
                sb.AppendLine($"- Chunk parsing: {totalChunkParsingTime} ms total, {averageChunkParsingTime} ms average") |> ignore
            
            if chunkCombiningTimes.Length > 0 then
                let totalChunkCombiningTime = chunkCombiningTimes |> List.sum
                let averageChunkCombiningTime = totalChunkCombiningTime / int64 chunkCombiningTimes.Length
                sb.AppendLine($"- Chunk combining: {totalChunkCombiningTime} ms total, {averageChunkCombiningTime} ms average") |> ignore
            
            sb.AppendLine() |> ignore
            
            // Error and warning breakdown
            sb.AppendLine("#### Error and Warning Breakdown") |> ignore
            
            let errorWarningTelemetry = 
                telemetry 
                |> List.choose (fun t -> t.ErrorWarningTelemetry)
            
            if errorWarningTelemetry.Length > 0 then
                // Collect all error codes
                let allErrorCodes = 
                    errorWarningTelemetry 
                    |> List.collect (fun t -> t.ErrorCodes |> Map.toList) 
                    |> List.groupBy fst 
                    |> List.map (fun (code, counts) -> (code, counts |> List.sumBy snd))
                
                // Collect all warning codes
                let allWarningCodes = 
                    errorWarningTelemetry 
                    |> List.collect (fun t -> t.WarningCodes |> Map.toList) 
                    |> List.groupBy fst 
                    |> List.map (fun (code, counts) -> (code, counts |> List.sumBy snd))
                
                // Collect all suppressed warning codes
                let allSuppressedWarningCodes = 
                    errorWarningTelemetry 
                    |> List.collect (fun t -> t.SuppressedWarningCodes |> Map.toList) 
                    |> List.groupBy fst 
                    |> List.map (fun (code, counts) -> (code, counts |> List.sumBy snd))
                
                if allErrorCodes.Length > 0 then
                    sb.AppendLine("##### Error Codes") |> ignore
                    
                    for (code, count) in allErrorCodes |> List.sortByDescending snd do
                        sb.AppendLine($"- {code}: {count}") |> ignore
                    
                    sb.AppendLine() |> ignore
                
                if allWarningCodes.Length > 0 then
                    sb.AppendLine("##### Warning Codes") |> ignore
                    
                    for (code, count) in allWarningCodes |> List.sortByDescending snd do
                        sb.AppendLine($"- {code}: {count}") |> ignore
                    
                    sb.AppendLine() |> ignore
                
                if allSuppressedWarningCodes.Length > 0 then
                    sb.AppendLine("##### Suppressed Warning Codes") |> ignore
                    
                    for (code, count) in allSuppressedWarningCodes |> List.sortByDescending snd do
                        sb.AppendLine($"- {code}: {count}") |> ignore
                    
                    sb.AppendLine() |> ignore
            
            sb.AppendLine() |> ignore
        
        sb.ToString()
    
    /// <summary>
    /// Save a report to a file.
    /// </summary>
    /// <param name="report">The report to save.</param>
    /// <param name="filePath">The path to save the report to.</param>
    /// <returns>True if the report was saved successfully, false otherwise.</returns>
    let saveReportToFile (report: string) (filePath: string) =
        try
            File.WriteAllText(filePath, report)
            true
        with
        | _ -> false
