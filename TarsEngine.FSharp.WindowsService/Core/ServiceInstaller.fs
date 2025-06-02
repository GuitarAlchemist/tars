namespace TarsEngine.FSharp.WindowsService.Core

open System
open System.Diagnostics
open System.IO
open System.ServiceProcess
open Microsoft.Extensions.Logging

/// <summary>
/// Service installer for TARS Windows Service
/// Handles installation, uninstallation, and service management
/// </summary>
type ServiceInstaller(logger: ILogger<ServiceInstaller>) =
    
    let serviceName = "TarsService"
    let serviceDisplayName = "TARS Autonomous Development Platform"
    let serviceDescription = "Autonomous development platform with multi-agent orchestration, semantic coordination, and continuous improvement capabilities."
    
    /// Install the TARS Windows Service
    member this.InstallService(executablePath: string) =
        try
            logger.LogInformation($"Installing TARS Windows Service...")
            logger.LogInformation($"Service Name: {serviceName}")
            logger.LogInformation($"Display Name: {serviceDisplayName}")
            logger.LogInformation($"Executable: {executablePath}")
            
            if not (File.Exists(executablePath)) then
                let error = $"Executable not found: {executablePath}"
                logger.LogError(error)
                Error error
            else
                // Use sc.exe to install the service
                let arguments = $"create \"{serviceName}\" binPath= \"\"{executablePath}\"\" DisplayName= \"{serviceDisplayName}\" start= auto"
                
                let processInfo = ProcessStartInfo(
                    FileName = "sc.exe",
                    Arguments = arguments,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                )
                
                use proc = Process.Start(processInfo)
                proc.WaitForExit()
                
                let output = proc.StandardOutput.ReadToEnd()
                let error = proc.StandardError.ReadToEnd()
                
                if proc.ExitCode = 0 then
                    logger.LogInformation("✅ Service installed successfully")
                    logger.LogInformation($"Output: {output}")
                    
                    // Set service description
                    this.SetServiceDescription() |> ignore
                    
                    Ok "Service installed successfully"
                else
                    let errorMsg = $"Failed to install service. Exit code: {proc.ExitCode}. Error: {error}"
                    logger.LogError(errorMsg)
                    Error errorMsg
                    
        with
        | ex ->
            let errorMsg = $"Exception installing service: {ex.Message}"
            logger.LogError(ex, errorMsg)
            Error errorMsg
    
    /// Uninstall the TARS Windows Service
    member this.UninstallService() =
        try
            logger.LogInformation($"Uninstalling TARS Windows Service...")
            
            // Stop the service first if it's running
            this.StopService() |> ignore
            
            // Use sc.exe to delete the service
            let arguments = $"delete \"{serviceName}\""
            
            let processInfo = ProcessStartInfo(
                FileName = "sc.exe",
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            )
            
            use proc = Process.Start(processInfo)
            proc.WaitForExit()
            
            let output = proc.StandardOutput.ReadToEnd()
            let error = proc.StandardError.ReadToEnd()
            
            if proc.ExitCode = 0 then
                logger.LogInformation("✅ Service uninstalled successfully")
                logger.LogInformation($"Output: {output}")
                Ok "Service uninstalled successfully"
            else
                let errorMsg = $"Failed to uninstall service. Exit code: {proc.ExitCode}. Error: {error}"
                logger.LogError(errorMsg)
                Error errorMsg
                
        with
        | ex ->
            let errorMsg = $"Exception uninstalling service: {ex.Message}"
            logger.LogError(ex, errorMsg)
            Error errorMsg
    
    /// Start the TARS Windows Service
    member this.StartService() =
        try
            logger.LogInformation($"Starting TARS Windows Service...")
            
            use serviceController = new ServiceController(serviceName)
            
            if serviceController.Status = ServiceControllerStatus.Running then
                logger.LogInformation("Service is already running")
                Ok "Service is already running"
            else
                serviceController.Start()
                serviceController.WaitForStatus(ServiceControllerStatus.Running, TimeSpan.FromSeconds(30.0))
                
                logger.LogInformation("✅ Service started successfully")
                Ok "Service started successfully"
                
        with
        | ex ->
            let errorMsg = $"Exception starting service: {ex.Message}"
            logger.LogError(ex, errorMsg)
            Error errorMsg
    
    /// Stop the TARS Windows Service
    member this.StopService() =
        try
            logger.LogInformation($"Stopping TARS Windows Service...")
            
            use serviceController = new ServiceController(serviceName)
            
            if serviceController.Status = ServiceControllerStatus.Stopped then
                logger.LogInformation("Service is already stopped")
                Ok "Service is already stopped"
            else
                serviceController.Stop()
                serviceController.WaitForStatus(ServiceControllerStatus.Stopped, TimeSpan.FromSeconds(30.0))
                
                logger.LogInformation("✅ Service stopped successfully")
                Ok "Service stopped successfully"
                
        with
        | ex ->
            let errorMsg = $"Exception stopping service: {ex.Message}"
            logger.LogError(ex, errorMsg)
            Error errorMsg
    
    /// Get service status
    member this.GetServiceStatus() =
        try
            use serviceController = new ServiceController(serviceName)
            serviceController.Refresh()
            
            let status = {|
                Name = serviceName
                DisplayName = serviceDisplayName
                Status = serviceController.Status.ToString()
                StartType = serviceController.StartType.ToString()
                CanStop = serviceController.CanStop
                CanPauseAndContinue = serviceController.CanPauseAndContinue
            |}
            
            Ok status
            
        with
        | :? InvalidOperationException ->
            Error "Service is not installed"
        | ex ->
            Error $"Exception getting service status: {ex.Message}"
    
    /// Check if service is installed
    member this.IsServiceInstalled() =
        try
            use serviceController = new ServiceController(serviceName)
            serviceController.Refresh()
            true
        with
        | :? InvalidOperationException -> false
        | _ -> false
    
    /// Set service description
    member private this.SetServiceDescription() =
        try
            let arguments = $"description \"{serviceName}\" \"{serviceDescription}\""
            
            let processInfo = ProcessStartInfo(
                FileName = "sc.exe",
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            )
            
            use proc = Process.Start(processInfo)
            proc.WaitForExit()
            
            if proc.ExitCode = 0 then
                logger.LogInformation("✅ Service description set successfully")
                Ok "Description set"
            else
                let error = proc.StandardError.ReadToEnd()
                logger.LogWarning($"Failed to set service description: {error}")
                Error error
                
        with
        | ex ->
            logger.LogWarning(ex, "Failed to set service description")
            Error ex.Message
