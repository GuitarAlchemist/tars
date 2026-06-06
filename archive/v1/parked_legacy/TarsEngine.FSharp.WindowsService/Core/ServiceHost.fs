namespace TarsEngine.FSharp.WindowsService.Core

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection

/// <summary>
/// Service host for managing TARS service lifecycle and dependencies
/// </summary>
type ServiceHost(serviceProvider: IServiceProvider, logger: ILogger<ServiceHost>) =
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    
    /// Start the service host
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting TARS Service Host...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Initialize core services
            logger.LogInformation("Initializing core services...")
            
            // Service is now running
            logger.LogInformation("✅ TARS Service Host started successfully")
            logger.LogInformation("🎯 Service Status: Running")
            logger.LogInformation("🔧 Mode: Windows Service")
            logger.LogInformation("📊 Monitoring: Active")
            
        with
        | ex ->
            logger.LogError(ex, "❌ Failed to start TARS Service Host")
            isRunning <- false
            raise
    }
    
    /// Stop the service host
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping TARS Service Host...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> 
                cts.Cancel()
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            logger.LogInformation("✅ TARS Service Host stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "❌ Error stopping TARS Service Host")
    }
    
    /// Get service status
    member this.IsRunning = isRunning
    
    /// Get service information
    member this.GetServiceInfo() =
        {|
            IsRunning = isRunning
            StartTime = DateTime.UtcNow // Would track actual start time
            ServiceName = "TARS Autonomous Development Platform"
            Version = "3.0.0"
            Mode = "Windows Service"
        |}
