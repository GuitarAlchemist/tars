namespace TarsEngine.FSharp.WindowsService.Core

open System
open System.IO
open Microsoft.Extensions.Configuration
open Microsoft.Extensions.Logging

/// <summary>
/// Service configuration types and management
/// </summary>
module ServiceConfiguration =
    
    /// Service configuration settings
    type ServiceConfig = {
        ServiceName: string
        DisplayName: string
        Description: string
        StartType: ServiceStartType
        LogLevel: LogLevel
        WorkingDirectory: string
        MaxConcurrentTasks: int
        HealthCheckInterval: TimeSpan
        TaskTimeout: TimeSpan
        EnableAutoRecovery: bool
        RecoveryAttempts: int
        RecoveryDelay: TimeSpan
    }
    
    /// Service start type
    and ServiceStartType =
        | Automatic
        | Manual
        | Disabled
    
    /// Agent configuration
    type AgentConfig = {
        Name: string
        Type: string
        Enabled: bool
        MaxInstances: int
        StartupDelay: TimeSpan
        HealthCheckInterval: TimeSpan
        RestartOnFailure: bool
        Configuration: Map<string, obj>
    }
    
    /// Task configuration
    type TaskConfig = {
        MaxConcurrentTasks: int
        DefaultTimeout: TimeSpan
        RetryAttempts: int
        RetryDelay: TimeSpan
        PriorityLevels: int
        QueueCapacity: int
    }
    
    /// Monitoring configuration
    type MonitoringConfig = {
        EnableHealthChecks: bool
        HealthCheckInterval: TimeSpan
        EnablePerformanceCounters: bool
        PerformanceInterval: TimeSpan
        EnableDiagnostics: bool
        DiagnosticsInterval: TimeSpan
        AlertThresholds: Map<string, float>
        LogRetentionDays: int
    }
    
    /// Complete service configuration
    type TarsServiceConfiguration = {
        Service: ServiceConfig
        Agents: AgentConfig list
        Tasks: TaskConfig
        Monitoring: MonitoringConfig
    }

/// <summary>
/// Configuration loader and manager
/// </summary>
type ConfigurationManager(configPath: string, logger: ILogger<ConfigurationManager>) =
    
    let mutable currentConfig: ServiceConfiguration.TarsServiceConfiguration option = None
    
    /// Load configuration from file
    member this.LoadConfiguration() =
        try
            logger.LogInformation($"Loading configuration from: {configPath}")

            let builder = ConfigurationBuilder()
            let directory = Path.GetDirectoryName(configPath)
            let fileName = Path.GetFileName(configPath)
            builder.SetBasePath(directory) |> ignore
            builder.AddJsonFile(fileName, optional = false, reloadOnChange = true) |> ignore
            
            let configuration = builder.Build()
            
            // Parse service configuration
            let serviceConfig = {
                ServiceName = configuration.["Service:ServiceName"] |> Option.ofObj |> Option.defaultValue "TarsService"
                DisplayName = configuration.["Service:DisplayName"] |> Option.ofObj |> Option.defaultValue "TARS Autonomous Service"
                Description = configuration.["Service:Description"] |> Option.ofObj |> Option.defaultValue "TARS Autonomous Development Service"
                StartType = 
                    match configuration.["Service:StartType"] |> Option.ofObj |> Option.defaultValue "Automatic" with
                    | "Manual" -> ServiceConfiguration.Manual
                    | "Disabled" -> ServiceConfiguration.Disabled
                    | _ -> ServiceConfiguration.Automatic
                LogLevel = 
                    match configuration.["Service:LogLevel"] |> Option.ofObj |> Option.defaultValue "Information" with
                    | "Debug" -> LogLevel.Debug
                    | "Warning" -> LogLevel.Warning
                    | "Error" -> LogLevel.Error
                    | "Critical" -> LogLevel.Critical
                    | _ -> LogLevel.Information
                WorkingDirectory = configuration.["Service:WorkingDirectory"] |> Option.ofObj |> Option.defaultValue (Directory.GetCurrentDirectory())
                MaxConcurrentTasks = configuration.["Service:MaxConcurrentTasks"] |> Option.ofObj |> Option.map int |> Option.defaultValue 10
                HealthCheckInterval = configuration.["Service:HealthCheckInterval"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(5.0))
                TaskTimeout = configuration.["Service:TaskTimeout"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(30.0))
                EnableAutoRecovery = configuration.["Service:EnableAutoRecovery"] |> Option.ofObj |> Option.map bool.Parse |> Option.defaultValue true
                RecoveryAttempts = configuration.["Service:RecoveryAttempts"] |> Option.ofObj |> Option.map int |> Option.defaultValue 3
                RecoveryDelay = configuration.["Service:RecoveryDelay"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(1.0))
            }
            
            // Parse agent configurations
            let agentConfigs = 
                configuration.GetSection("Agents").GetChildren()
                |> Seq.map (fun section ->
                    {
                        Name = section.["Name"] |> Option.ofObj |> Option.defaultValue section.Key
                        Type = section.["Type"] |> Option.ofObj |> Option.defaultValue "Generic"
                        Enabled = section.["Enabled"] |> Option.ofObj |> Option.map bool.Parse |> Option.defaultValue true
                        MaxInstances = section.["MaxInstances"] |> Option.ofObj |> Option.map int |> Option.defaultValue 1
                        StartupDelay = section.["StartupDelay"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue TimeSpan.Zero
                        HealthCheckInterval = section.["HealthCheckInterval"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(5.0))
                        RestartOnFailure = section.["RestartOnFailure"] |> Option.ofObj |> Option.map bool.Parse |> Option.defaultValue true
                        Configuration = 
                            section.GetSection("Configuration").GetChildren()
                            |> Seq.map (fun kvp -> (kvp.Key, kvp.Value :> obj))
                            |> Map.ofSeq
                    })
                |> List.ofSeq
            
            // Parse task configuration
            let taskConfig = {
                MaxConcurrentTasks = configuration.["Tasks:MaxConcurrentTasks"] |> Option.ofObj |> Option.map int |> Option.defaultValue 10
                DefaultTimeout = configuration.["Tasks:DefaultTimeout"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(30.0))
                RetryAttempts = configuration.["Tasks:RetryAttempts"] |> Option.ofObj |> Option.map int |> Option.defaultValue 3
                RetryDelay = configuration.["Tasks:RetryDelay"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromSeconds(30.0))
                PriorityLevels = configuration.["Tasks:PriorityLevels"] |> Option.ofObj |> Option.map int |> Option.defaultValue 5
                QueueCapacity = configuration.["Tasks:QueueCapacity"] |> Option.ofObj |> Option.map int |> Option.defaultValue 1000
            }
            
            // Parse monitoring configuration
            let monitoringConfig = {
                EnableHealthChecks = configuration.["Monitoring:EnableHealthChecks"] |> Option.ofObj |> Option.map bool.Parse |> Option.defaultValue true
                HealthCheckInterval = configuration.["Monitoring:HealthCheckInterval"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(5.0))
                EnablePerformanceCounters = configuration.["Monitoring:EnablePerformanceCounters"] |> Option.ofObj |> Option.map bool.Parse |> Option.defaultValue true
                PerformanceInterval = configuration.["Monitoring:PerformanceInterval"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(1.0))
                EnableDiagnostics = configuration.["Monitoring:EnableDiagnostics"] |> Option.ofObj |> Option.map bool.Parse |> Option.defaultValue true
                DiagnosticsInterval = configuration.["Monitoring:DiagnosticsInterval"] |> Option.ofObj |> Option.map TimeSpan.Parse |> Option.defaultValue (TimeSpan.FromMinutes(10.0))
                AlertThresholds = 
                    configuration.GetSection("Monitoring:AlertThresholds").GetChildren()
                    |> Seq.map (fun kvp -> (kvp.Key, float kvp.Value))
                    |> Map.ofSeq
                LogRetentionDays = configuration.["Monitoring:LogRetentionDays"] |> Option.ofObj |> Option.map int |> Option.defaultValue 30
            }
            
            let config = {
                Service = serviceConfig
                Agents = agentConfigs
                Tasks = taskConfig
                Monitoring = monitoringConfig
            }
            
            currentConfig <- Some config
            logger.LogInformation("Configuration loaded successfully")
            Ok config
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to load configuration from: {configPath}")
            Error $"Configuration load failed: {ex.Message}"
    
    /// Get current configuration
    member this.GetConfiguration() =
        match currentConfig with
        | Some config -> Ok config
        | None -> Error "Configuration not loaded"
    
    /// Reload configuration
    member this.ReloadConfiguration() =
        this.LoadConfiguration()
    
    /// Validate configuration
    member this.ValidateConfiguration(config: ServiceConfiguration.TarsServiceConfiguration) =
        let errors = ResizeArray<string>()
        
        // Validate service configuration
        if String.IsNullOrWhiteSpace(config.Service.ServiceName) then
            errors.Add("Service name cannot be empty")
        
        if config.Service.MaxConcurrentTasks <= 0 then
            errors.Add("MaxConcurrentTasks must be greater than 0")
        
        if config.Service.HealthCheckInterval <= TimeSpan.Zero then
            errors.Add("HealthCheckInterval must be greater than 0")
        
        // Validate agent configurations
        for agent in config.Agents do
            if String.IsNullOrWhiteSpace(agent.Name) then
                errors.Add($"Agent name cannot be empty")
            
            if agent.MaxInstances <= 0 then
                errors.Add($"Agent {agent.Name}: MaxInstances must be greater than 0")
        
        // Validate task configuration
        if config.Tasks.MaxConcurrentTasks <= 0 then
            errors.Add("Tasks.MaxConcurrentTasks must be greater than 0")
        
        if config.Tasks.QueueCapacity <= 0 then
            errors.Add("Tasks.QueueCapacity must be greater than 0")
        
        if errors.Count = 0 then
            Ok ()
        else
            Error (String.Join("; ", errors))

/// <summary>
/// Default configuration factory
/// </summary>
module DefaultConfiguration =
    
    /// Create default service configuration
    let createDefaultServiceConfig() = {
        ServiceName = "TarsService"
        DisplayName = "TARS Autonomous Service"
        Description = "TARS Autonomous Development and Requirements Management Service"
        StartType = ServiceConfiguration.Automatic
        LogLevel = LogLevel.Information
        WorkingDirectory = Directory.GetCurrentDirectory()
        MaxConcurrentTasks = Environment.ProcessorCount * 2
        HealthCheckInterval = TimeSpan.FromMinutes(5.0)
        TaskTimeout = TimeSpan.FromMinutes(30.0)
        EnableAutoRecovery = true
        RecoveryAttempts = 3
        RecoveryDelay = TimeSpan.FromMinutes(1.0)
    }
    
    /// Create default agent configurations
    let createDefaultAgentConfigs() = [
        {
            Name = "RequirementsAgent"
            Type = "Requirements"
            Enabled = true
            MaxInstances = 1
            StartupDelay = TimeSpan.FromSeconds(10.0)
            HealthCheckInterval = TimeSpan.FromMinutes(5.0)
            RestartOnFailure = true
            Configuration = Map.empty
        }
        {
            Name = "AnalyticsAgent"
            Type = "Analytics"
            Enabled = true
            MaxInstances = 1
            StartupDelay = TimeSpan.FromSeconds(15.0)
            HealthCheckInterval = TimeSpan.FromMinutes(5.0)
            RestartOnFailure = true
            Configuration = Map.empty
        }
    ]
    
    /// Create default task configuration
    let createDefaultTaskConfig() = {
        MaxConcurrentTasks = Environment.ProcessorCount * 2
        DefaultTimeout = TimeSpan.FromMinutes(30.0)
        RetryAttempts = 3
        RetryDelay = TimeSpan.FromSeconds(30.0)
        PriorityLevels = 5
        QueueCapacity = 1000
    }
    
    /// Create default monitoring configuration
    let createDefaultMonitoringConfig() = {
        EnableHealthChecks = true
        HealthCheckInterval = TimeSpan.FromMinutes(5.0)
        EnablePerformanceCounters = true
        PerformanceInterval = TimeSpan.FromMinutes(1.0)
        EnableDiagnostics = true
        DiagnosticsInterval = TimeSpan.FromMinutes(10.0)
        AlertThresholds = Map.ofList [
            ("CpuUsage", 80.0)
            ("MemoryUsage", 85.0)
            ("DiskUsage", 90.0)
            ("TaskQueueSize", 500.0)
        ]
        LogRetentionDays = 30
    }
    
    /// Create complete default configuration
    let createDefaultConfiguration() = {
        Service = createDefaultServiceConfig()
        Agents = createDefaultAgentConfigs()
        Tasks = createDefaultTaskConfig()
        Monitoring = createDefaultMonitoringConfig()
    }
