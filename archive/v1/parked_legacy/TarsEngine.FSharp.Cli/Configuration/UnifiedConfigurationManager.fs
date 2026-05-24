namespace TarsEngine.FSharp.Cli.Configuration

open System
open System.IO
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.UnifiedCore

/// Unified Configuration Management System - Centralized configuration for all TARS systems
module UnifiedConfigurationManager =
    
    /// Configuration value types
    type ConfigValue =
        | StringValue of string
        | IntValue of int
        | FloatValue of float
        | BoolValue of bool
        | ArrayValue of ConfigValue list
        | ObjectValue of Map<string, ConfigValue>
        | NullValue
    
    /// Configuration source types
    type ConfigSource =
        | FileSource of path: string
        | EnvironmentSource
        | CommandLineSource
        | DatabaseSource of connectionString: string
        | RemoteSource of url: string
        | InMemorySource
    
    /// Configuration change event
    type ConfigChangeEvent = {
        Key: string
        OldValue: ConfigValue option
        NewValue: ConfigValue option
        Source: ConfigSource
        Timestamp: DateTime
        CorrelationId: string
    }
    
    /// Configuration schema definition
    type ConfigSchema = {
        Key: string
        ValueType: Type
        Required: bool
        DefaultValue: ConfigValue option
        Validation: ConfigValue -> bool
        Description: string
        Category: string
    }
    
    /// Configuration environment
    type ConfigEnvironment = {
        Name: string
        Description: string
        IsActive: bool
        Priority: int
        Sources: ConfigSource list
        OverrideRules: Map<string, string>
    }
    
    /// Configuration snapshot for versioning
    type ConfigSnapshot = {
        SnapshotId: string
        Timestamp: DateTime
        Environment: string
        Configuration: Map<string, ConfigValue>
        Metadata: Map<string, obj>
    }
    
    /// Thread-safe unified configuration manager
    type UnifiedConfigurationManager(logger: ITarsLogger) =
        let configurations = ConcurrentDictionary<string, ConfigValue>()
        let schemas = ConcurrentDictionary<string, ConfigSchema>()
        let environments = ConcurrentDictionary<string, ConfigEnvironment>()
        let snapshots = ConcurrentDictionary<string, ConfigSnapshot>()
        let changeSubscribers = ConcurrentBag<ConfigChangeEvent -> unit>()
        let mutable currentEnvironment = "default"
        let mutable isInitialized = false
        
        /// Configuration file paths
        let configPaths = {|
            Main = ".tars/config/unified.json"
            Schemas = ".tars/config/schemas.json"
            Environments = ".tars/config/environments.json"
            Snapshots = ".tars/config/snapshots"
        |}
        
        /// Ensure configuration directories exist
        member private this.EnsureDirectories() =
            let dirs = [
                ".tars/config"
                ".tars/config/snapshots"
                ".tars/config/environments"
                ".tars/config/backups"
            ]
            
            for dir in dirs do
                Directory.CreateDirectory(dir) |> ignore
        
        /// Convert ConfigValue to JSON
        member private this.ConfigValueToJson(value: ConfigValue) =
            match value with
            | StringValue s -> JsonSerializer.Serialize(s)
            | IntValue i -> JsonSerializer.Serialize(i)
            | FloatValue f -> JsonSerializer.Serialize(f)
            | BoolValue b -> JsonSerializer.Serialize(b)
            | ArrayValue arr -> 
                let jsonArray = arr |> List.map this.ConfigValueToJson
                JsonSerializer.Serialize(jsonArray)
            | ObjectValue obj -> 
                let jsonObj = obj |> Map.map (fun _ v -> this.ConfigValueToJson(v))
                JsonSerializer.Serialize(jsonObj)
            | NullValue -> "null"
        
        /// Convert JSON to ConfigValue
        member private this.JsonToConfigValue(json: string) =
            try
                if json = "null" then NullValue
                elif json.StartsWith("\"") && json.EndsWith("\"") then
                    StringValue (JsonSerializer.Deserialize<string>(json))
                elif json = "true" || json = "false" then
                    BoolValue (JsonSerializer.Deserialize<bool>(json))
                elif json.Contains(".") then
                    FloatValue (JsonSerializer.Deserialize<float>(json))
                elif json.StartsWith("[") then
                    let arr = JsonSerializer.Deserialize<string[]>(json)
                    ArrayValue (arr |> Array.map (fun s -> this.JsonToConfigValue(s)) |> Array.toList)
                else
                    IntValue (JsonSerializer.Deserialize<int>(json))
            with
            | _ -> StringValue json
        
        /// Initialize configuration manager
        member this.InitializeAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, "Initializing Unified Configuration Manager")
                    
                    this.EnsureDirectories()
                    
                    // Load default environment
                    let defaultEnv = {
                        Name = "default"
                        Description = "Default TARS configuration environment"
                        IsActive = true
                        Priority = 1
                        Sources = [
                            FileSource configPaths.Main
                            EnvironmentSource
                        ]
                        OverrideRules = Map.empty
                    }
                    
                    environments.[defaultEnv.Name] <- defaultEnv
                    currentEnvironment <- defaultEnv.Name
                    
                    // Load existing configuration
                    let! loadResult = this.LoadConfigurationAsync(cancellationToken)
                    
                    // Register default schemas
                    this.RegisterDefaultSchemas()
                    
                    isInitialized <- true
                    logger.LogInformation(correlationId, "Configuration manager initialized successfully")
                    
                    return Success ((), Map [("environment", box currentEnvironment); ("configCount", box configurations.Count)])
                
                with
                | ex ->
                    let error = ExecutionError ("Configuration manager initialization failed", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Register default configuration schemas
        member private this.RegisterDefaultSchemas() =
            let defaultSchemas = [
                // Core system schemas
                {
                    Key = "tars.core.logLevel"
                    ValueType = typeof<string>
                    Required = false
                    DefaultValue = Some (StringValue "Information")
                    Validation = fun v -> match v with StringValue s -> List.contains s ["Debug"; "Information"; "Warning"; "Error"] | _ -> false
                    Description = "Global logging level for TARS system"
                    Category = "Core"
                }
                
                {
                    Key = "tars.core.maxConcurrency"
                    ValueType = typeof<int>
                    Required = false
                    DefaultValue = Some (IntValue 10)
                    Validation = fun v -> match v with IntValue i -> i > 0 && i <= 100 | _ -> false
                    Description = "Maximum concurrent operations"
                    Category = "Core"
                }
                
                // Agent system schemas
                {
                    Key = "tars.agents.maxAgents"
                    ValueType = typeof<int>
                    Required = false
                    DefaultValue = Some (IntValue 50)
                    Validation = fun v -> match v with IntValue i -> i > 0 && i <= 1000 | _ -> false
                    Description = "Maximum number of concurrent agents"
                    Category = "Agents"
                }
                
                {
                    Key = "tars.agents.healthCheckInterval"
                    ValueType = typeof<int>
                    Required = false
                    DefaultValue = Some (IntValue 60)
                    Validation = fun v -> match v with IntValue i -> i >= 10 && i <= 3600 | _ -> false
                    Description = "Agent health check interval in seconds"
                    Category = "Agents"
                }
                
                // CUDA system schemas
                {
                    Key = "tars.cuda.enabled"
                    ValueType = typeof<bool>
                    Required = false
                    DefaultValue = Some (BoolValue true)
                    Validation = fun _ -> true
                    Description = "Enable CUDA GPU acceleration"
                    Category = "CUDA"
                }
                
                {
                    Key = "tars.cuda.deviceId"
                    ValueType = typeof<int>
                    Required = false
                    DefaultValue = Some (IntValue 0)
                    Validation = fun v -> match v with IntValue i -> i >= 0 && i < 16 | _ -> false
                    Description = "CUDA device ID to use"
                    Category = "CUDA"
                }
                
                // Proof system schemas
                {
                    Key = "tars.proof.enabled"
                    ValueType = typeof<bool>
                    Required = false
                    DefaultValue = Some (BoolValue true)
                    Validation = fun _ -> true
                    Description = "Enable cryptographic proof generation"
                    Category = "Proof"
                }
                
                {
                    Key = "tars.proof.retentionDays"
                    ValueType = typeof<int>
                    Required = false
                    DefaultValue = Some (IntValue 30)
                    Validation = fun v -> match v with IntValue i -> i >= 1 && i <= 365 | _ -> false
                    Description = "Proof retention period in days"
                    Category = "Proof"
                }
            ]
            
            for schema in defaultSchemas do
                schemas.[schema.Key] <- schema
                
                // Set default values if not already configured
                if not (configurations.ContainsKey(schema.Key)) && schema.DefaultValue.IsSome then
                    configurations.[schema.Key] <- schema.DefaultValue.Value
        
        /// Load configuration from sources
        member this.LoadConfigurationAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    
                    // Load from main configuration file
                    if File.Exists(configPaths.Main) then
                        let! configJson = File.ReadAllTextAsync(configPaths.Main, cancellationToken)
                        let configDict = JsonSerializer.Deserialize<Dictionary<string, string>>(configJson)
                        
                        for kvp in configDict do
                            let configValue = this.JsonToConfigValue(kvp.Value)
                            configurations.[kvp.Key] <- configValue
                        
                        logger.LogInformation(correlationId, $"Loaded {configDict.Count} configuration values from file")
                    
                    // Load from environment variables (TARS_ prefix)
                    let envVars = 
                        Environment.GetEnvironmentVariables()
                        |> Seq.cast<System.Collections.DictionaryEntry>
                        |> Seq.filter (fun entry -> entry.Key.ToString().StartsWith("TARS_"))
                        |> Seq.map (fun entry -> 
                            let key = entry.Key.ToString().Substring(5).Replace("_", ".").ToLower()
                            let value = StringValue (entry.Value.ToString())
                            (key, value))
                    
                    for (key, value) in envVars do
                        configurations.[key] <- value
                    
                    let envCount = envVars |> Seq.length
                    if envCount > 0 then
                        logger.LogInformation(correlationId, $"Loaded {envCount} configuration values from environment")
                    
                    return Success ((), Map [("fileLoaded", box (File.Exists(configPaths.Main))); ("envCount", box envCount)])
                
                with
                | ex ->
                    let error = ExecutionError ("Configuration loading failed", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Save configuration to file
        member this.SaveConfigurationAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    
                    let configDict = 
                        configurations
                        |> Seq.map (fun kvp -> (kvp.Key, this.ConfigValueToJson(kvp.Value)))
                        |> dict
                    
                    let configJson = JsonSerializer.Serialize(configDict, JsonSerializerOptions(WriteIndented = true))
                    do! File.WriteAllTextAsync(configPaths.Main, configJson, cancellationToken)
                    
                    logger.LogInformation(correlationId, $"Saved {configurations.Count} configuration values to file")
                    
                    return Success ((), Map [("configCount", box configurations.Count)])
                
                with
                | ex ->
                    let error = ExecutionError ("Configuration saving failed", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Get configuration value
        member this.GetValue<'T>(key: string, defaultValue: 'T option) =
            try
                match configurations.TryGetValue(key) with
                | true, value ->
                    match value with
                    | StringValue s when typeof<'T> = typeof<string> -> Some (box s :?> 'T)
                    | IntValue i when typeof<'T> = typeof<int> -> Some (box i :?> 'T)
                    | FloatValue f when typeof<'T> = typeof<float> -> Some (box f :?> 'T)
                    | BoolValue b when typeof<'T> = typeof<bool> -> Some (box b :?> 'T)
                    | _ -> defaultValue
                | false, _ -> defaultValue
            with
            | _ -> defaultValue
        
        /// Set configuration value
        member this.SetValueAsync(key: string, value: 'T, correlationId: string option) =
            task {
                try
                    let corrId = correlationId |> Option.defaultValue (generateCorrelationId())
                    
                    // Validate against schema if exists
                    let validationError =
                        match schemas.TryGetValue(key) with
                        | true, schema ->
                            let configValue =
                                match box value with
                                | :? string as s -> StringValue s
                                | :? int as i -> IntValue i
                                | :? float as f -> FloatValue f
                                | :? bool as b -> BoolValue b
                                | _ -> StringValue (value.ToString())

                            if not (schema.Validation configValue) then
                                let keyStr = if isNull key then "null" else key
                                let valueStr = if isNull (box value) then "null" else (box value).ToString()
                                let error = ValidationError ($"Value for key '{keyStr}' failed validation", Map [("key", keyStr); ("value", valueStr)])
                                Some (Failure (error, corrId))
                            else
                                None
                        | false, _ -> None

                    // Check if validation failed
                    match validationError with
                    | Some (Failure (error, corrId)) ->
                        return Failure (error, corrId)
                    | _ ->
                        // Continue with execution
                        let oldValue = configurations.TryGetValue(key) |> function | true, v -> Some v | false, _ -> None

                        let newConfigValue =
                            match box value with
                            | :? string as s -> StringValue s
                            | :? int as i -> IntValue i
                            | :? float as f -> FloatValue f
                            | :? bool as b -> BoolValue b
                            | _ -> StringValue (value.ToString())

                        configurations.[key] <- newConfigValue

                        // Notify subscribers
                        let changeEvent = {
                            Key = key
                            OldValue = oldValue
                            NewValue = Some newConfigValue
                            Source = InMemorySource
                            Timestamp = DateTime.UtcNow
                            CorrelationId = corrId
                        }

                        for subscriber in changeSubscribers do
                            try
                                subscriber changeEvent
                            with
                            | _ -> ()

                        let keyStr = if isNull key then "null" else key
                        let valueStr = if isNull (box value) then "null" else (box value).ToString()
                        logger.LogInformation(corrId, $"Configuration value set: {keyStr} = {valueStr}")

                        return Success ((), Map [("key", box keyStr); ("value", box valueStr)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to set configuration value '{key}'", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Subscribe to configuration changes
        member this.SubscribeToChanges(callback: ConfigChangeEvent -> unit) =
            changeSubscribers.Add(callback)
        
        /// Get all configuration values
        member this.GetAllValues() =
            configurations
            |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
            |> Map.ofSeq
        
        /// Get configuration by category
        member this.GetValuesByCategory(category: string) =
            schemas.Values
            |> Seq.filter (fun schema -> schema.Category = category)
            |> Seq.choose (fun schema -> 
                match configurations.TryGetValue(schema.Key) with
                | true, value -> Some (schema.Key, value)
                | false, _ -> None)
            |> Map.ofSeq
        
        /// Create configuration snapshot
        member this.CreateSnapshotAsync(name: string, cancellationToken: CancellationToken) =
            task {
                try
                    let snapshotId = $"{name}_{DateTime.UtcNow:yyyyMMdd_HHmmss}"
                    let snapshot = {
                        SnapshotId = snapshotId
                        Timestamp = DateTime.UtcNow
                        Environment = currentEnvironment
                        Configuration = this.GetAllValues()
                        Metadata = Map [
                            ("creator", box "UnifiedConfigurationManager")
                            ("configCount", box configurations.Count)
                            ("schemaCount", box schemas.Count)
                        ]
                    }
                    
                    snapshots.[snapshotId] <- snapshot
                    
                    // Save snapshot to file
                    let snapshotPath = Path.Combine(configPaths.Snapshots, $"{snapshotId}.json")
                    let snapshotJson = JsonSerializer.Serialize(snapshot, JsonSerializerOptions(WriteIndented = true))
                    do! File.WriteAllTextAsync(snapshotPath, snapshotJson, cancellationToken)
                    
                    logger.LogInformation(generateCorrelationId(), $"Configuration snapshot created: {snapshotId}")
                    
                    return Success (snapshot, Map [("snapshotId", box snapshotId)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to create configuration snapshot '{name}'", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Get configuration statistics
        member this.GetStatistics() =
            Map [
                ("totalConfigurations", box configurations.Count)
                ("totalSchemas", box schemas.Count)
                ("totalEnvironments", box environments.Count)
                ("totalSnapshots", box snapshots.Count)
                ("currentEnvironment", box currentEnvironment)
                ("isInitialized", box isInitialized)
                ("lastUpdate", box DateTime.UtcNow)
            ]
        
        interface IDisposable with
            member this.Dispose() =
                configurations.Clear()
                schemas.Clear()
                environments.Clear()
                snapshots.Clear()
    
    /// Configuration extensions for unified operations
    module ConfigurationExtensions =
        
        /// Get string configuration value
        let getString (manager: UnifiedConfigurationManager) (key: string) (defaultValue: string) =
            manager.GetValue<string>(key, Some defaultValue) |> Option.defaultValue defaultValue
        
        /// Get integer configuration value
        let getInt (manager: UnifiedConfigurationManager) (key: string) (defaultValue: int) =
            manager.GetValue<int>(key, Some defaultValue) |> Option.defaultValue defaultValue
        
        /// Get boolean configuration value
        let getBool (manager: UnifiedConfigurationManager) (key: string) (defaultValue: bool) =
            manager.GetValue<bool>(key, Some defaultValue) |> Option.defaultValue defaultValue
        
        /// Get float configuration value
        let getFloat (manager: UnifiedConfigurationManager) (key: string) (defaultValue: float) =
            manager.GetValue<float>(key, Some defaultValue) |> Option.defaultValue defaultValue
    
    /// Create unified configuration manager
    let createConfigurationManager (logger: ITarsLogger) =
        new UnifiedConfigurationManager(logger)
