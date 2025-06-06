namespace TarsEngine.FSharp.Core.Services

open System
open System.IO
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging

/// Platform detection and abstraction service
module PlatformService =
    
    /// Supported platforms
    type Platform =
        | Windows
        | Linux
        | MacOS
        | Docker
        | Kubernetes
        | Hyperlight
        | WASM
        | Unknown
    
    /// Platform-specific paths
    type PlatformPaths = {
        InstallPath: string
        DataPath: string
        LogPath: string
        ConfigPath: string
        TempPath: string
    }
    
    /// Service management configuration
    type ServiceConfig = {
        ServiceType: string
        ServiceName: string
        DisplayName: string
        Description: string
        StartType: string
        Account: string
        Dependencies: string list
    }
    
    /// Platform capabilities
    type PlatformCapabilities = {
        SupportsServices: bool
        SupportsContainers: bool
        SupportsIsolation: bool
        SupportsGPU: bool
        SupportsNetworking: bool
        MaxMemoryMB: int option
        MaxCpuCores: int option
    }
    
    /// Detect current platform
    let detectPlatform () =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            // Check if running in Docker
            if Environment.GetEnvironmentVariable("DOTNET_RUNNING_IN_CONTAINER") = "true" then
                Docker
            else
                Windows
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            // Check for various Linux environments
            if Environment.GetEnvironmentVariable("KUBERNETES_SERVICE_HOST") <> null then
                Kubernetes
            elif Environment.GetEnvironmentVariable("DOTNET_RUNNING_IN_CONTAINER") = "true" then
                Docker
            elif File.Exists("/proc/version") then
                let version = File.ReadAllText("/proc/version")
                if version.Contains("Microsoft") then
                    Linux // WSL
                else
                    Linux
            else
                Linux
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            MacOS
        else
            Unknown
    
    /// Get platform-specific paths
    let getPlatformPaths platform =
        match platform with
        | Windows ->
            {
                InstallPath = @"C:\Program Files\TARS"
                DataPath = @"C:\ProgramData\TARS"
                LogPath = @"C:\ProgramData\TARS\Logs"
                ConfigPath = @"C:\ProgramData\TARS\Config"
                TempPath = @"C:\ProgramData\TARS\Temp"
            }
        | Linux ->
            {
                InstallPath = "/opt/tars"
                DataPath = "/var/lib/tars"
                LogPath = "/var/log/tars"
                ConfigPath = "/etc/tars"
                TempPath = "/tmp/tars"
            }
        | MacOS ->
            {
                InstallPath = "/usr/local/opt/tars"
                DataPath = "/usr/local/var/lib/tars"
                LogPath = "/usr/local/var/log/tars"
                ConfigPath = "/usr/local/etc/tars"
                TempPath = "/tmp/tars"
            }
        | Docker ->
            {
                InstallPath = "/app"
                DataPath = "/app/data"
                LogPath = "/app/logs"
                ConfigPath = "/app/config"
                TempPath = "/tmp"
            }
        | Kubernetes ->
            {
                InstallPath = "/app"
                DataPath = "/app/data"
                LogPath = "/app/logs"
                ConfigPath = "/app/config"
                TempPath = "/tmp"
            }
        | Hyperlight ->
            {
                InstallPath = "/app"
                DataPath = "/app/data"
                LogPath = "/app/logs"
                ConfigPath = "/app/config"
                TempPath = "/tmp"
            }
        | WASM ->
            {
                InstallPath = "/app"
                DataPath = "/app/data"
                LogPath = "/app/logs"
                ConfigPath = "/app/config"
                TempPath = "/tmp"
            }
        | Unknown ->
            {
                InstallPath = "./tars"
                DataPath = "./tars/data"
                LogPath = "./tars/logs"
                ConfigPath = "./tars/config"
                TempPath = "./tars/temp"
            }
    
    /// Get service configuration for platform
    let getServiceConfig platform =
        match platform with
        | Windows ->
            {
                ServiceType = "WindowsService"
                ServiceName = "TarsAgentService"
                DisplayName = "TARS Agent Service"
                Description = "TARS Autonomous Reasoning System Agent Service"
                StartType = "Automatic"
                Account = "LocalSystem"
                Dependencies = []
            }
        | Linux ->
            {
                ServiceType = "systemd"
                ServiceName = "tars-agent"
                DisplayName = "TARS Agent Service"
                Description = "TARS Autonomous Reasoning System Agent Service"
                StartType = "enabled"
                Account = "tars"
                Dependencies = ["network.target"]
            }
        | MacOS ->
            {
                ServiceType = "launchd"
                ServiceName = "com.tars.agent"
                DisplayName = "TARS Agent Service"
                Description = "TARS Autonomous Reasoning System Agent Service"
                StartType = "auto"
                Account = "tars"
                Dependencies = []
            }
        | Docker ->
            {
                ServiceType = "container"
                ServiceName = "tars-agent"
                DisplayName = "TARS Agent Container"
                Description = "TARS Agent running in Docker container"
                StartType = "always"
                Account = "tars"
                Dependencies = []
            }
        | Kubernetes ->
            {
                ServiceType = "deployment"
                ServiceName = "tars-agent-deployment"
                DisplayName = "TARS Agent Deployment"
                Description = "TARS Agent running in Kubernetes"
                StartType = "always"
                Account = "tars"
                Dependencies = []
            }
        | _ ->
            {
                ServiceType = "process"
                ServiceName = "tars-agent"
                DisplayName = "TARS Agent Process"
                Description = "TARS Agent running as process"
                StartType = "manual"
                Account = "current"
                Dependencies = []
            }
    
    /// Get platform capabilities
    let getPlatformCapabilities platform =
        match platform with
        | Windows ->
            {
                SupportsServices = true
                SupportsContainers = true
                SupportsIsolation = true
                SupportsGPU = true
                SupportsNetworking = true
                MaxMemoryMB = Some 32768
                MaxCpuCores = Some 16
            }
        | Linux ->
            {
                SupportsServices = true
                SupportsContainers = true
                SupportsIsolation = true
                SupportsGPU = true
                SupportsNetworking = true
                MaxMemoryMB = Some 65536
                MaxCpuCores = Some 32
            }
        | MacOS ->
            {
                SupportsServices = true
                SupportsContainers = true
                SupportsIsolation = false
                SupportsGPU = false
                SupportsNetworking = true
                MaxMemoryMB = Some 16384
                MaxCpuCores = Some 8
            }
        | Docker ->
            {
                SupportsServices = false
                SupportsContainers = true
                SupportsIsolation = true
                SupportsGPU = true
                SupportsNetworking = true
                MaxMemoryMB = Some 8192
                MaxCpuCores = Some 4
            }
        | Kubernetes ->
            {
                SupportsServices = false
                SupportsContainers = true
                SupportsIsolation = true
                SupportsGPU = true
                SupportsNetworking = true
                MaxMemoryMB = Some 16384
                MaxCpuCores = Some 8
            }
        | Hyperlight ->
            {
                SupportsServices = false
                SupportsContainers = false
                SupportsIsolation = true
                SupportsGPU = false
                SupportsNetworking = true
                MaxMemoryMB = Some 64
                MaxCpuCores = Some 1
            }
        | WASM ->
            {
                SupportsServices = false
                SupportsContainers = false
                SupportsIsolation = true
                SupportsGPU = false
                SupportsNetworking = false
                MaxMemoryMB = Some 32
                MaxCpuCores = Some 1
            }
        | Unknown ->
            {
                SupportsServices = false
                SupportsContainers = false
                SupportsIsolation = false
                SupportsGPU = false
                SupportsNetworking = true
                MaxMemoryMB = Some 1024
                MaxCpuCores = Some 1
            }
    
    /// Ensure platform directories exist
    let ensurePlatformDirectories platform (logger: ILogger) =
        let paths = getPlatformPaths platform
        let directories = [
            paths.DataPath
            paths.LogPath
            paths.ConfigPath
            paths.TempPath
        ]
        
        for directory in directories do
            try
                if not (Directory.Exists(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                    logger.LogDebug($"Created directory: {directory}")
            with
            | ex ->
                logger.LogWarning(ex, $"Failed to create directory: {directory}")
    
    /// Get environment-specific configuration path
    let getConfigurationPath platform configFile =
        let paths = getPlatformPaths platform
        Path.Combine(paths.ConfigPath, configFile)
    
    /// Platform-specific service installer
    type IPlatformServiceInstaller =
        abstract member InstallService: ServiceConfig -> Result<unit, string>
        abstract member UninstallService: string -> Result<unit, string>
        abstract member StartService: string -> Result<unit, string>
        abstract member StopService: string -> Result<unit, string>
        abstract member GetServiceStatus: string -> Result<string, string>
    
    /// Create platform-specific service installer
    let createServiceInstaller platform (logger: ILogger) =
        match platform with
        | Windows ->
            // Windows Service installer implementation would go here
            { new IPlatformServiceInstaller with
                member _.InstallService config = Ok ()
                member _.UninstallService name = Ok ()
                member _.StartService name = Ok ()
                member _.StopService name = Ok ()
                member _.GetServiceStatus name = Ok "Running"
            }
        | Linux ->
            // systemd service installer implementation would go here
            { new IPlatformServiceInstaller with
                member _.InstallService config = Ok ()
                member _.UninstallService name = Ok ()
                member _.StartService name = Ok ()
                member _.StopService name = Ok ()
                member _.GetServiceStatus name = Ok "active"
            }
        | MacOS ->
            // launchd service installer implementation would go here
            { new IPlatformServiceInstaller with
                member _.InstallService config = Ok ()
                member _.UninstallService name = Ok ()
                member _.StartService name = Ok ()
                member _.StopService name = Ok ()
                member _.GetServiceStatus name = Ok "running"
            }
        | _ ->
            // Default implementation for other platforms
            { new IPlatformServiceInstaller with
                member _.InstallService config = Error "Service installation not supported on this platform"
                member _.UninstallService name = Error "Service uninstallation not supported on this platform"
                member _.StartService name = Error "Service start not supported on this platform"
                member _.StopService name = Error "Service stop not supported on this platform"
                member _.GetServiceStatus name = Error "Service status not supported on this platform"
            }
