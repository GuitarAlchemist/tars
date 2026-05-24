namespace TarsEngine.FSharp.WindowsService.Core

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration
open TarsEngine.FSharp.WindowsService.Agents
open TarsEngine.FSharp.WindowsService.Tasks
open TarsEngine.FSharp.WindowsService.Monitoring

/// <summary>
/// Main TARS Windows Service implementation
/// Provides autonomous operation capabilities with agent orchestration
/// </summary>
type TarsService(
    logger: ILogger<TarsService>,
    serviceProvider: IServiceProvider,
    configManager: ConfigurationManager) =
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable agentManager: AgentManager option = None
    let mutable taskExecutor: TaskExecutor option = None
    let mutable healthMonitor: HealthMonitor option = None
    
    /// Service startup
    member private this.StartupAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("TARS Service starting up...")
            
            // Load configuration
            let! configResult = 
                match configManager.LoadConfiguration() with
                | Ok config -> 
                    logger.LogInformation("Configuration loaded successfully")
                    Task.FromResult(Ok config)
                | Error error -> 
                    logger.LogError($"Failed to load configuration: {error}")
                    Task.FromResult(Error error)
            
            match configResult with
            | Ok config ->
                // Validate configuration
                match configManager.ValidateConfiguration(config) with
                | Ok () ->
                    logger.LogInformation("Configuration validation passed")
                    
                    // Initialize components
                    do! this.InitializeComponentsAsync(config, cancellationToken)
                    
                    // Start components
                    do! this.StartComponentsAsync(cancellationToken)
                    
                    isRunning <- true
                    logger.LogInformation("TARS Service started successfully")
                    
                | Error validationError ->
                    logger.LogError($"Configuration validation failed: {validationError}")
                    raise (InvalidOperationException($"Invalid configuration: {validationError}"))
            
            | Error configError ->
                logger.LogError($"Configuration load failed: {configError}")
                raise (InvalidOperationException($"Configuration load failed: {configError}"))
                
        with
        | ex ->
            logger.LogCritical(ex, "TARS Service startup failed")
            raise
    }
    
    /// Initialize service components
    member private this.InitializeComponentsAsync(config: TarsServiceConfiguration, cancellationToken: CancellationToken) = task {
        logger.LogInformation("Initializing TARS Service components...")
        
        // Initialize Agent Manager
        let agentRegistry = serviceProvider.GetRequiredService<AgentRegistry>()
        let agentCommunication = serviceProvider.GetRequiredService<AgentCommunication>()
        let agentHost = serviceProvider.GetRequiredService<AgentHost>()
        
        let agentMgr = AgentManager(logger.CreateLogger<AgentManager>(), agentRegistry, agentCommunication, agentHost)
        agentManager <- Some agentMgr
        
        // Initialize Task Executor
        let taskQueue = serviceProvider.GetRequiredService<TaskQueue>()
        let taskScheduler = serviceProvider.GetRequiredService<TaskScheduler>()
        let taskMonitor = serviceProvider.GetRequiredService<TaskMonitor>()
        
        let taskExec = TaskExecutor(logger.CreateLogger<TaskExecutor>(), taskQueue, taskScheduler, taskMonitor)
        taskExecutor <- Some taskExec
        
        // Initialize Health Monitor
        let performanceCollector = serviceProvider.GetRequiredService<PerformanceCollector>()
        let alertManager = serviceProvider.GetRequiredService<AlertManager>()
        let diagnosticsCollector = serviceProvider.GetRequiredService<DiagnosticsCollector>()
        
        let healthMon = HealthMonitor(logger.CreateLogger<HealthMonitor>(), performanceCollector, alertManager, diagnosticsCollector)
        healthMonitor <- Some healthMon
        
        // Configure components with loaded configuration
        do! agentMgr.ConfigureAsync(config.Agents, cancellationToken)
        do! taskExec.ConfigureAsync(config.Tasks, cancellationToken)
        do! healthMon.ConfigureAsync(config.Monitoring, cancellationToken)
        
        logger.LogInformation("TARS Service components initialized successfully")
    }
    
    /// Start service components
    member private this.StartComponentsAsync(cancellationToken: CancellationToken) = task {
        logger.LogInformation("Starting TARS Service components...")
        
        // Start Health Monitor first
        match healthMonitor with
        | Some monitor -> 
            do! monitor.StartAsync(cancellationToken)
            logger.LogInformation("Health Monitor started")
        | None -> 
            logger.LogWarning("Health Monitor not initialized")
        
        // Start Task Executor
        match taskExecutor with
        | Some executor -> 
            do! executor.StartAsync(cancellationToken)
            logger.LogInformation("Task Executor started")
        | None -> 
            logger.LogWarning("Task Executor not initialized")
        
        // Start Agent Manager last
        match agentManager with
        | Some manager -> 
            do! manager.StartAsync(cancellationToken)
            logger.LogInformation("Agent Manager started")
        | None -> 
            logger.LogWarning("Agent Manager not initialized")
        
        logger.LogInformation("All TARS Service components started successfully")
    }
    
    /// Service shutdown
    member private this.ShutdownAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("TARS Service shutting down...")
            
            isRunning <- false
            
            // Stop components in reverse order
            match agentManager with
            | Some manager -> 
                do! manager.StopAsync(cancellationToken)
                logger.LogInformation("Agent Manager stopped")
            | None -> ()
            
            match taskExecutor with
            | Some executor -> 
                do! executor.StopAsync(cancellationToken)
                logger.LogInformation("Task Executor stopped")
            | None -> ()
            
            match healthMonitor with
            | Some monitor -> 
                do! monitor.StopAsync(cancellationToken)
                logger.LogInformation("Health Monitor stopped")
            | None -> ()
            
            logger.LogInformation("TARS Service shutdown completed")
            
        with
        | ex ->
            logger.LogError(ex, "Error during TARS Service shutdown")
    }
    
    /// Get service status
    member this.GetServiceStatus() =
        {|
            IsRunning = isRunning
            AgentManagerStatus = agentManager |> Option.map (fun am -> am.GetStatus()) |> Option.defaultValue "Not Initialized"
            TaskExecutorStatus = taskExecutor |> Option.map (fun te -> te.GetStatus()) |> Option.defaultValue "Not Initialized"
            HealthMonitorStatus = healthMonitor |> Option.map (fun hm -> hm.GetStatus()) |> Option.defaultValue "Not Initialized"
            Uptime = 
                match cancellationTokenSource with
                | Some cts when not cts.Token.IsCancellationRequested -> DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC")
                | _ -> "Not Running"
        |}
    
    /// Get service metrics
    member this.GetServiceMetrics() =
        let agentMetrics = agentManager |> Option.map (fun am -> am.GetMetrics()) |> Option.defaultValue Map.empty
        let taskMetrics = taskExecutor |> Option.map (fun te -> te.GetMetrics()) |> Option.defaultValue Map.empty
        let healthMetrics = healthMonitor |> Option.map (fun hm -> hm.GetMetrics()) |> Option.defaultValue Map.empty
        
        Map.empty
        |> Map.fold (fun acc k v -> Map.add $"Agent.{k}" v acc) agentMetrics
        |> Map.fold (fun acc k v -> Map.add $"Task.{k}" v acc) taskMetrics
        |> Map.fold (fun acc k v -> Map.add $"Health.{k}" v acc) healthMetrics
    
    /// Reload configuration
    member this.ReloadConfigurationAsync() = task {
        try
            logger.LogInformation("Reloading TARS Service configuration...")
            
            match configManager.ReloadConfiguration() with
            | Ok config ->
                // Reconfigure components
                match agentManager with
                | Some manager -> do! manager.ReconfigureAsync(config.Agents, CancellationToken.None)
                | None -> ()
                
                match taskExecutor with
                | Some executor -> do! executor.ReconfigureAsync(config.Tasks, CancellationToken.None)
                | None -> ()
                
                match healthMonitor with
                | Some monitor -> do! monitor.ReconfigureAsync(config.Monitoring, CancellationToken.None)
                | None -> ()
                
                logger.LogInformation("Configuration reloaded successfully")
                return Ok ()
            
            | Error error ->
                logger.LogError($"Failed to reload configuration: {error}")
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, "Error during configuration reload")
            return Error ex.Message
    }
    
    interface IHostedService with
        member this.StartAsync(cancellationToken: CancellationToken) = task {
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            do! this.StartupAsync(cancellationToken)
        }
        
        member this.StopAsync(cancellationToken: CancellationToken) = task {
            match cancellationTokenSource with
            | Some cts -> 
                cts.Cancel()
                do! this.ShutdownAsync(cancellationToken)
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
        }

/// <summary>
/// Service health status
/// </summary>
type ServiceHealthStatus = {
    IsHealthy: bool
    Status: string
    LastCheck: DateTime
    Issues: string list
    Metrics: Map<string, obj>
}

/// <summary>
/// Service performance metrics
/// </summary>
type ServicePerformanceMetrics = {
    CpuUsage: float
    MemoryUsage: float
    DiskUsage: float
    NetworkUsage: float
    ActiveTasks: int
    QueuedTasks: int
    ActiveAgents: int
    TotalAgents: int
    Uptime: TimeSpan
    RequestsPerSecond: float
    AverageResponseTime: TimeSpan
    ErrorRate: float
}
