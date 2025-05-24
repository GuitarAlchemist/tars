namespace TarsEngine.DSL

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

/// <summary>
/// Configuration for telemetry collection.
/// </summary>
type TelemetryConfig = {
    /// <summary>
    /// Whether telemetry collection is enabled.
    /// </summary>
    Enabled: bool
    
    /// <summary>
    /// Whether to anonymize telemetry data.
    /// </summary>
    Anonymize: bool
    
    /// <summary>
    /// The directory to store telemetry data in.
    /// </summary>
    TelemetryDirectory: string
    
    /// <summary>
    /// Whether to collect parser usage telemetry.
    /// </summary>
    CollectUsageTelemetry: bool
    
    /// <summary>
    /// Whether to collect parsing performance telemetry.
    /// </summary>
    CollectPerformanceTelemetry: bool
    
    /// <summary>
    /// Whether to collect error and warning telemetry.
    /// </summary>
    CollectErrorWarningTelemetry: bool
}

/// <summary>
/// Module for configuring telemetry collection.
/// </summary>
module TelemetryConfiguration =
    /// <summary>
    /// The default configuration file path.
    /// </summary>
    let defaultConfigFilePath = 
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TarsEngine",
            "telemetry.json"
        )
    
    /// <summary>
    /// The default telemetry configuration.
    /// </summary>
    let defaultConfig = {
        Enabled = true
        Anonymize = true
        TelemetryDirectory = TelemetryStorage.defaultTelemetryDirectory
        CollectUsageTelemetry = true
        CollectPerformanceTelemetry = true
        CollectErrorWarningTelemetry = true
    }
    
    /// <summary>
    /// JSON serializer options for telemetry configuration.
    /// </summary>
    let jsonOptions = 
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        options
    
    /// <summary>
    /// Load telemetry configuration from a file.
    /// </summary>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>The loaded telemetry configuration, or the default configuration if the file could not be loaded.</returns>
    let loadConfig (filePath: string) =
        try
            if File.Exists(filePath) then
                let json = File.ReadAllText(filePath)
                JsonSerializer.Deserialize<TelemetryConfig>(json, jsonOptions)
            else
                defaultConfig
        with
        | _ -> defaultConfig
    
    /// <summary>
    /// Save telemetry configuration to a file.
    /// </summary>
    /// <param name="config">The telemetry configuration to save.</param>
    /// <param name="filePath">The path to save the configuration to.</param>
    /// <returns>True if the configuration was saved successfully, false otherwise.</returns>
    let saveConfig (config: TelemetryConfig) (filePath: string) =
        try
            let directory = Path.GetDirectoryName(filePath)
            
            if not (Directory.Exists(directory)) then
                Directory.CreateDirectory(directory) |> ignore
            
            let json = JsonSerializer.Serialize(config, jsonOptions)
            File.WriteAllText(filePath, json)
            true
        with
        | _ -> false
    
    /// <summary>
    /// Enable telemetry collection.
    /// </summary>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>True if telemetry collection was enabled successfully, false otherwise.</returns>
    let enableTelemetry (filePath: string) =
        let config = loadConfig filePath
        let updatedConfig = { config with Enabled = true }
        saveConfig updatedConfig filePath
    
    /// <summary>
    /// Disable telemetry collection.
    /// </summary>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>True if telemetry collection was disabled successfully, false otherwise.</returns>
    let disableTelemetry (filePath: string) =
        let config = loadConfig filePath
        let updatedConfig = { config with Enabled = false }
        saveConfig updatedConfig filePath
    
    /// <summary>
    /// Enable anonymization of telemetry data.
    /// </summary>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>True if anonymization was enabled successfully, false otherwise.</returns>
    let enableAnonymization (filePath: string) =
        let config = loadConfig filePath
        let updatedConfig = { config with Anonymize = true }
        saveConfig updatedConfig filePath
    
    /// <summary>
    /// Disable anonymization of telemetry data.
    /// </summary>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>True if anonymization was disabled successfully, false otherwise.</returns>
    let disableAnonymization (filePath: string) =
        let config = loadConfig filePath
        let updatedConfig = { config with Anonymize = false }
        saveConfig updatedConfig filePath
    
    /// <summary>
    /// Set the telemetry directory.
    /// </summary>
    /// <param name="directory">The directory to store telemetry data in.</param>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>True if the telemetry directory was set successfully, false otherwise.</returns>
    let setTelemetryDirectory (directory: string) (filePath: string) =
        let config = loadConfig filePath
        let updatedConfig = { config with TelemetryDirectory = directory }
        saveConfig updatedConfig filePath
    
    /// <summary>
    /// Configure telemetry collection.
    /// </summary>
    /// <param name="collectUsage">Whether to collect parser usage telemetry.</param>
    /// <param name="collectPerformance">Whether to collect parsing performance telemetry.</param>
    /// <param name="collectErrorWarning">Whether to collect error and warning telemetry.</param>
    /// <param name="filePath">The path to the configuration file.</param>
    /// <returns>True if telemetry collection was configured successfully, false otherwise.</returns>
    let configureTelemetryCollection (collectUsage: bool) (collectPerformance: bool) (collectErrorWarning: bool) (filePath: string) =
        let config = loadConfig filePath
        let updatedConfig = { 
            config with 
                CollectUsageTelemetry = collectUsage
                CollectPerformanceTelemetry = collectPerformance
                CollectErrorWarningTelemetry = collectErrorWarning
        }
        saveConfig updatedConfig filePath
