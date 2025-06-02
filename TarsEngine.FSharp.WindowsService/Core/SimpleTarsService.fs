namespace TarsEngine.FSharp.WindowsService.Core

open System
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Configuration

/// <summary>
/// Simple TARS Windows Service implementation
/// Autonomous development platform running as a Windows service
/// </summary>
type SimpleTarsService(logger: ILogger<SimpleTarsService>, configuration: IConfiguration) =
    inherit BackgroundService()
    
    let mutable serviceConfig: SimpleServiceConfiguration option = None
    let mutable isRunning = false
    let mutable taskCount = 0
    let mutable lastHealthCheck = DateTime.UtcNow
    
    /// Initialize the service
    member private this.InitializeAsync() = task {
        try
            logger.LogInformation("🚀 Initializing TARS Service...")
            
            // Load configuration
            let config = SimpleServiceConfiguration.CreateDefault()
            config.LoadFromConfiguration(configuration)
            serviceConfig <- Some config
            
            // Validate configuration
            match config.Validate() with
            | Ok _ ->
                logger.LogInformation("✅ Configuration validated successfully")
                let info = config.GetServiceInfo()
                logger.LogInformation($"   Service: {info.ServiceName}")
                logger.LogInformation($"   Display Name: {info.DisplayName}")
                logger.LogInformation($"   Max Tasks: {info.MaxConcurrentTasks}")
                logger.LogInformation($"   Health Check: {info.HealthCheckInterval}")
                
            | Error errors ->
                logger.LogError($"❌ Configuration validation failed: {errors}")
                raise (InvalidOperationException($"Invalid configuration: {errors}"))
            
            logger.LogInformation("✅ TARS Service initialized successfully")
            
        with
        | ex ->
            logger.LogError(ex, "❌ Failed to initialize TARS Service")
            raise
    }
    
    /// Main service execution loop
    override this.ExecuteAsync(stoppingToken: CancellationToken) = task {
        try
            // Initialize the service
            do! this.InitializeAsync()
            
            isRunning <- true
            logger.LogInformation("🎯 TARS Service started and running")
            logger.LogInformation("📊 Status: Active")
            logger.LogInformation("🔧 Mode: Windows Service")
            logger.LogInformation("🤖 Autonomous Platform: Ready")
            
            // Main service loop
            while not stoppingToken.IsCancellationRequested && isRunning do
                try
                    // Perform health check
                    do! this.PerformHealthCheckAsync()
                    
                    // Simulate some work (in a real implementation, this would manage agents and tasks)
                    do! this.ProcessTasksAsync()
                    
                    // Wait before next iteration
                    let healthCheckInterval = 
                        match serviceConfig with
                        | Some config -> TimeSpan.FromSeconds(float config.HealthCheckIntervalSeconds)
                        | None -> TimeSpan.FromSeconds(30.0)
                    
                    do! Task.Delay(healthCheckInterval, stoppingToken)
                    
                with
                | :? OperationCanceledException ->
                    logger.LogInformation("🛑 Service execution cancelled")
                    () // Exit the loop
                | ex ->
                    logger.LogError(ex, "❌ Error in service execution loop")
                    
                    // Auto-recovery logic
                    match serviceConfig with
                    | Some config when config.EnableAutoRecovery ->
                        logger.LogInformation($"🔄 Auto-recovery enabled, waiting {config.RecoveryDelaySeconds} seconds...")
                        do! Task.Delay(TimeSpan.FromSeconds(float config.RecoveryDelaySeconds), stoppingToken)
                    | _ ->
                        logger.LogWarning("⚠️ Auto-recovery disabled, continuing...")
            
            logger.LogInformation("🛑 TARS Service execution completed")
            
        with
        | :? OperationCanceledException ->
            logger.LogInformation("🛑 TARS Service cancelled")
            isRunning <- false
        | ex ->
            logger.LogError(ex, "❌ TARS Service execution failed")
            isRunning <- false
            raise
    }
    
    /// Perform health check
    member private this.PerformHealthCheckAsync() = task {
        try
            let now = DateTime.UtcNow
            lastHealthCheck <- now
            
            // Basic health metrics
            let uptime = now - Process.GetCurrentProcess().StartTime
            let memoryUsage = GC.GetTotalMemory(false) / 1024L / 1024L // MB
            
            logger.LogDebug("💓 Health Check - Uptime: {uptime}, Memory: {memory}MB, Tasks: {tasks}", uptime, memoryUsage, taskCount)

            // Log periodic status
            if now.Minute % 5 = 0 && now.Second < 30 then
                logger.LogInformation("📊 TARS Status - Running: {running}, Tasks: {tasks}, Memory: {memory}MB", isRunning, taskCount, memoryUsage)
            
        with
        | ex ->
            logger.LogWarning(ex, "⚠️ Health check failed")
    }
    
    /// Process tasks (placeholder for actual task processing)
    member private this.ProcessTasksAsync() = task {
        try
            // Simulate task processing
            // In a real implementation, this would:
            // - Check for new tasks in the queue
            // - Assign tasks to available agents
            // - Monitor task progress
            // - Handle task completion and failures
            
            // For now, just simulate some activity
            if taskCount < 10 then
                taskCount <- taskCount + 1
                logger.LogDebug("🔧 Simulated task processing - Active tasks: {tasks}", taskCount)

            // Simulate task completion
            if taskCount > 0 && Random().Next(0, 10) < 3 then
                taskCount <- taskCount - 1
                logger.LogDebug("✅ Simulated task completion - Active tasks: {tasks}", taskCount)

        with
        | ex ->
            logger.LogWarning(ex, "⚠️ Task processing error")
    }
    
    /// Stop the service gracefully
    override this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("🛑 Stopping TARS Service...")
            
            isRunning <- false
            
            // Wait for current operations to complete
            let timeout = TimeSpan.FromSeconds(30.0)
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            while taskCount > 0 && stopwatch.Elapsed < timeout do
                logger.LogInformation("⏳ Waiting for {tasks} tasks to complete...", taskCount)
                do! Task.Delay(TimeSpan.FromSeconds(1.0), cancellationToken)

            if taskCount > 0 then
                logger.LogWarning("⚠️ Service stopped with {tasks} tasks still active", taskCount)
            else
                logger.LogInformation("✅ All tasks completed successfully")

            logger.LogInformation("✅ TARS Service stopped successfully")

        with
        | ex ->
            logger.LogError(ex, "❌ Error stopping TARS Service")
    }
    
    /// Get service status
    member this.GetStatus() =
        {|
            IsRunning = isRunning
            TaskCount = taskCount
            LastHealthCheck = lastHealthCheck
            Uptime = DateTime.UtcNow - Process.GetCurrentProcess().StartTime
            MemoryUsage = GC.GetTotalMemory(false) / 1024L / 1024L
            Configuration = serviceConfig |> Option.map (fun c -> c.GetServiceInfo())
        |}
    
    /// Get service information
    member this.GetServiceInfo() =
        match serviceConfig with
        | Some config -> config.GetServiceInfo()
        | None -> 
            {|
                ServiceName = "TarsService"
                DisplayName = "TARS Autonomous Development Platform"
                Description = "Service not initialized"
                MaxConcurrentTasks = 0
                HealthCheckInterval = TimeSpan.Zero
                TaskTimeout = TimeSpan.Zero
                EnableAutoRecovery = false
                RecoveryAttempts = 0
                RecoveryDelay = TimeSpan.Zero
            |}
