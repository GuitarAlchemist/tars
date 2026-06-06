namespace TarsEngine.FSharp.WindowsService.Core

open System
open Microsoft.Extensions.Configuration
open Microsoft.Extensions.Logging

/// <summary>
/// Simple service configuration for TARS Windows Service
/// </summary>
type SimpleServiceConfiguration() =
    
    /// Service name
    member val ServiceName = "TarsService" with get, set
    
    /// Service display name
    member val DisplayName = "TARS Autonomous Development Platform" with get, set
    
    /// Service description
    member val Description = "Autonomous development platform with multi-agent orchestration" with get, set
    
    /// Maximum concurrent tasks
    member val MaxConcurrentTasks = 100 with get, set
    
    /// Health check interval in seconds
    member val HealthCheckIntervalSeconds = 30 with get, set
    
    /// Task timeout in minutes
    member val TaskTimeoutMinutes = 60 with get, set
    
    /// Enable auto recovery
    member val EnableAutoRecovery = true with get, set
    
    /// Recovery attempts
    member val RecoveryAttempts = 3 with get, set
    
    /// Recovery delay in seconds
    member val RecoveryDelaySeconds = 30 with get, set
    
    /// Load configuration from IConfiguration
    member this.LoadFromConfiguration(configuration: IConfiguration) =
        try
            // Service settings
            let serviceSection = configuration.GetSection("service")
            if serviceSection.Exists() then
                this.ServiceName <- serviceSection.["name"] |> Option.ofObj |> Option.defaultValue this.ServiceName
                this.DisplayName <- serviceSection.["displayName"] |> Option.ofObj |> Option.defaultValue this.DisplayName
                this.Description <- serviceSection.["description"] |> Option.ofObj |> Option.defaultValue this.Description
                
                // Parse numeric values safely
                match serviceSection.["maxConcurrentTasks"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> this.MaxConcurrentTasks <- Int32.Parse(value)
                | _ -> ()
                
                match serviceSection.["healthCheckIntervalSeconds"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> this.HealthCheckIntervalSeconds <- Int32.Parse(value)
                | _ -> ()
                
                match serviceSection.["taskTimeoutMinutes"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> this.TaskTimeoutMinutes <- Int32.Parse(value)
                | _ -> ()
                
                match serviceSection.["autoRestart"] |> Option.ofObj with
                | Some value when Boolean.TryParse(value) |> fst -> this.EnableAutoRecovery <- Boolean.Parse(value)
                | _ -> ()
                
                match serviceSection.["maxRestartAttempts"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> this.RecoveryAttempts <- Int32.Parse(value)
                | _ -> ()
                
                match serviceSection.["restartDelay"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> this.RecoveryDelaySeconds <- Int32.Parse(value)
                | _ -> ()
            
        with
        | ex ->
            // Log error but continue with defaults
            Console.WriteLine($"Warning: Failed to load configuration: {ex.Message}")
    
    /// Get service information
    member this.GetServiceInfo() =
        {|
            ServiceName = this.ServiceName
            DisplayName = this.DisplayName
            Description = this.Description
            MaxConcurrentTasks = this.MaxConcurrentTasks
            HealthCheckInterval = TimeSpan.FromSeconds(float this.HealthCheckIntervalSeconds)
            TaskTimeout = TimeSpan.FromMinutes(float this.TaskTimeoutMinutes)
            EnableAutoRecovery = this.EnableAutoRecovery
            RecoveryAttempts = this.RecoveryAttempts
            RecoveryDelay = TimeSpan.FromSeconds(float this.RecoveryDelaySeconds)
        |}
    
    /// Validate configuration
    member this.Validate() =
        let errors = ResizeArray<string>()
        
        if String.IsNullOrWhiteSpace(this.ServiceName) then
            errors.Add("Service name cannot be empty")
        
        if String.IsNullOrWhiteSpace(this.DisplayName) then
            errors.Add("Display name cannot be empty")
        
        if this.MaxConcurrentTasks <= 0 then
            errors.Add("MaxConcurrentTasks must be greater than 0")
        
        if this.HealthCheckIntervalSeconds <= 0 then
            errors.Add("HealthCheckIntervalSeconds must be greater than 0")
        
        if this.TaskTimeoutMinutes <= 0 then
            errors.Add("TaskTimeoutMinutes must be greater than 0")
        
        if this.RecoveryAttempts < 0 then
            errors.Add("RecoveryAttempts cannot be negative")
        
        if this.RecoveryDelaySeconds < 0 then
            errors.Add("RecoveryDelaySeconds cannot be negative")
        
        if errors.Count = 0 then
            Ok "Configuration is valid"
        else
            Error (String.Join("; ", errors))
    
    /// Create default configuration
    static member CreateDefault() =
        let config = SimpleServiceConfiguration()
        config.ServiceName <- "TarsService"
        config.DisplayName <- "TARS Autonomous Development Platform"
        config.Description <- "Autonomous development platform with multi-agent orchestration, semantic coordination, and continuous improvement capabilities"
        config.MaxConcurrentTasks <- 100
        config.HealthCheckIntervalSeconds <- 30
        config.TaskTimeoutMinutes <- 60
        config.EnableAutoRecovery <- true
        config.RecoveryAttempts <- 3
        config.RecoveryDelaySeconds <- 30
        config
