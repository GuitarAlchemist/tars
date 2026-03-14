namespace TarsEngine.DSL

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

/// <summary>
/// Module for storing telemetry data.
/// </summary>
module TelemetryStorage =
    /// <summary>
    /// The default directory for storing telemetry data.
    /// </summary>
    let defaultTelemetryDirectory = 
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TarsEngine",
            "Telemetry"
        )
    
    /// <summary>
    /// The maximum number of telemetry files to keep.
    /// </summary>
    let maxTelemetryFiles = 100
    
    /// <summary>
    /// JSON serializer options for telemetry data.
    /// </summary>
    let jsonOptions = 
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        options
    
    /// <summary>
    /// Ensure the telemetry directory exists.
    /// </summary>
    /// <param name="directory">The directory to ensure exists.</param>
    let ensureDirectoryExists (directory: string) =
        if not (Directory.Exists(directory)) then
            Directory.CreateDirectory(directory) |> ignore
    
    /// <summary>
    /// Get the path for a telemetry file.
    /// </summary>
    /// <param name="directory">The directory to store the telemetry file in.</param>
    /// <param name="id">The ID of the telemetry data.</param>
    /// <returns>The path for the telemetry file.</returns>
    let getTelemetryFilePath (directory: string) (id: Guid) =
        Path.Combine(directory, $"{id}.json")
    
    /// <summary>
    /// Store telemetry data to a file.
    /// </summary>
    /// <param name="directory">The directory to store the telemetry file in.</param>
    /// <param name="telemetry">The telemetry data to store.</param>
    /// <returns>The path to the stored telemetry file.</returns>
    let storeTelemetryToFile (directory: string) (telemetry: TelemetryData) =
        ensureDirectoryExists directory
        
        let filePath = getTelemetryFilePath directory telemetry.Id
        let json = JsonSerializer.Serialize(telemetry, jsonOptions)
        
        File.WriteAllText(filePath, json)
        
        // Clean up old telemetry files if there are too many
        let telemetryFiles = 
            Directory.GetFiles(directory, "*.json")
            |> Array.sortByDescending File.GetLastWriteTime
        
        if telemetryFiles.Length > maxTelemetryFiles then
            telemetryFiles
            |> Array.skip maxTelemetryFiles
            |> Array.iter (fun file -> 
                try
                    File.Delete(file)
                with
                | _ -> ()
            )
        
        filePath
    
    /// <summary>
    /// Load telemetry data from a file.
    /// </summary>
    /// <param name="filePath">The path to the telemetry file.</param>
    /// <returns>The loaded telemetry data, or None if the file could not be loaded.</returns>
    let loadTelemetryFromFile (filePath: string) =
        try
            if File.Exists(filePath) then
                let json = File.ReadAllText(filePath)
                Some (JsonSerializer.Deserialize<TelemetryData>(json, jsonOptions))
            else
                None
        with
        | _ -> None
    
    /// <summary>
    /// Load all telemetry data from a directory.
    /// </summary>
    /// <param name="directory">The directory to load telemetry data from.</param>
    /// <returns>A list of loaded telemetry data.</returns>
    let loadAllTelemetry (directory: string) =
        if Directory.Exists(directory) then
            Directory.GetFiles(directory, "*.json")
            |> Array.choose loadTelemetryFromFile
            |> Array.toList
        else
            []
    
    /// <summary>
    /// Aggregate telemetry data.
    /// </summary>
    /// <param name="telemetryList">The list of telemetry data to aggregate.</param>
    /// <returns>A summary of the aggregated telemetry data.</returns>
    let aggregateTelemetry (telemetryList: TelemetryData list) =
        let parserTypes = 
            telemetryList 
            |> List.map (fun t -> t.UsageTelemetry.ParserType) 
            |> List.distinct
        
        let totalParseTimeMs = 
            telemetryList 
            |> List.sumBy (fun t -> t.UsageTelemetry.TotalParseTimeMs)
        
        let totalFileSize = 
            telemetryList 
            |> List.sumBy (fun t -> t.UsageTelemetry.FileSizeBytes)
        
        let totalLineCount = 
            telemetryList 
            |> List.sumBy (fun t -> t.UsageTelemetry.LineCount)
        
        let totalBlockCount = 
            telemetryList 
            |> List.sumBy (fun t -> t.UsageTelemetry.BlockCount)
        
        let totalPropertyCount = 
            telemetryList 
            |> List.sumBy (fun t -> t.UsageTelemetry.PropertyCount)
        
        let totalNestedBlockCount = 
            telemetryList 
            |> List.sumBy (fun t -> t.UsageTelemetry.NestedBlockCount)
        
        let averageParseTimeMs = 
            if telemetryList.Length > 0 then
                totalParseTimeMs / int64 telemetryList.Length
            else
                0L
        
        let errorCount = 
            telemetryList 
            |> List.choose (fun t -> t.ErrorWarningTelemetry) 
            |> List.sumBy (fun t -> t.ErrorCount)
        
        let warningCount = 
            telemetryList 
            |> List.choose (fun t -> t.ErrorWarningTelemetry) 
            |> List.sumBy (fun t -> t.WarningCount)
        
        let infoCount = 
            telemetryList 
            |> List.choose (fun t -> t.ErrorWarningTelemetry) 
            |> List.sumBy (fun t -> t.InfoCount)
        
        let hintCount = 
            telemetryList 
            |> List.choose (fun t -> t.ErrorWarningTelemetry) 
            |> List.sumBy (fun t -> t.HintCount)
        
        let suppressedWarningCount = 
            telemetryList 
            |> List.choose (fun t -> t.ErrorWarningTelemetry) 
            |> List.sumBy (fun t -> t.SuppressedWarningCount)
        
        {|
            TelemetryCount = telemetryList.Length
            ParserTypes = parserTypes
            TotalParseTimeMs = totalParseTimeMs
            AverageParseTimeMs = averageParseTimeMs
            TotalFileSize = totalFileSize
            TotalLineCount = totalLineCount
            TotalBlockCount = totalBlockCount
            TotalPropertyCount = totalPropertyCount
            TotalNestedBlockCount = totalNestedBlockCount
            ErrorCount = errorCount
            WarningCount = warningCount
            InfoCount = infoCount
            HintCount = hintCount
            SuppressedWarningCount = suppressedWarningCount
        |}
    
    /// <summary>
    /// Export telemetry data to a file.
    /// </summary>
    /// <param name="telemetryList">The list of telemetry data to export.</param>
    /// <param name="filePath">The path to export the telemetry data to.</param>
    /// <returns>True if the export was successful, false otherwise.</returns>
    let exportTelemetry (telemetryList: TelemetryData list) (filePath: string) =
        try
            let json = JsonSerializer.Serialize(telemetryList, jsonOptions)
            File.WriteAllText(filePath, json)
            true
        with
        | _ -> false
