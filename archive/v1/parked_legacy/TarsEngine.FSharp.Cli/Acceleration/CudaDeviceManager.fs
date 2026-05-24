namespace TarsEngine.FSharp.Cli.Acceleration

open System
open System.Runtime.InteropServices
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Acceleration.CudaTypes
open TarsEngine.FSharp.Cli.Acceleration.CudaInterop

/// CUDA Device Manager - Handles CUDA device detection and management
module CudaDeviceManager =
    
    /// Detect and initialize CUDA devices
    let detectAndInitializeDevices (logger: ITarsLogger) (correlationId: string) =
        try
            logger.LogInformation(correlationId, "🔍 Detecting CUDA devices")
            
            // Check for CUDA devices
            let deviceCount = 
                try
                    CudaInterop.tars_cuda_device_count()
                with
                | _ -> 0 // Fallback if CUDA not available
            
            if deviceCount > 0 then
                logger.LogInformation(correlationId, $"✅ Found {deviceCount} CUDA device(s)")
                
                // Get device information
                let devices = ResizeArray<CudaDeviceInfo>()
                
                for deviceId in 0 .. deviceCount - 1 do
                    try
                        let deviceInfo = getDeviceInfoInternal deviceId logger correlationId
                        match deviceInfo with
                        | Some info ->
                            devices.Add(info)
                            logger.LogInformation(correlationId, 
                                $"📊 Device {deviceId}: {info.Name} ({info.TotalMemory / 1024L / 1024L / 1024L} GB)")
                        | None ->
                            logger.LogWarning(correlationId, 
                                $"⚠️ Failed to get info for device {deviceId}")
                    with
                    | ex ->
                        let error = ExecutionError ($"Error getting device {deviceId} info", Some ex)
                        logger.LogError(correlationId, error, ex)
                
                Success (devices |> Seq.toList, Map [("deviceCount", box deviceCount)])
            else
                logger.LogWarning(correlationId, "⚠️ No CUDA devices found")
                Success ([], Map [("deviceCount", box 0)])
        
        with
        | ex ->
            let error = ExecutionError ("CUDA device detection failed", Some ex)
            logger.LogError(correlationId, error, ex)
            Failure (error, correlationId)
    
    /// Get information for a specific CUDA device (internal function)
    let getDeviceInfoInternal (deviceId: int) (logger: ITarsLogger) (correlationId: string) : CudaDeviceInfo option =
        try
            let nameBuffer = Marshal.AllocHGlobal(256)
            let mutable totalMemory = 0L
            let mutable computeCapability = 0.0f

            let result = CudaInterop.tars_cuda_get_device_info(
                deviceId, nameBuffer, 256, &totalMemory, &computeCapability)
            
            if result = CudaError.Success then
                let name = Marshal.PtrToStringAnsi(nameBuffer)
                Marshal.FreeHGlobal(nameBuffer)
                
                let deviceInfo = {
                    DeviceId = deviceId
                    Name = name
                    TotalMemory = totalMemory
                    ComputeCapability = float computeCapability
                    MultiprocessorCount = 0 // Would need additional API call
                    MaxThreadsPerBlock = 1024 // Standard default
                    IsAvailable = true
                }
                
                Some deviceInfo
            else
                Marshal.FreeHGlobal(nameBuffer)
                logger.LogWarning(correlationId, 
                    $"⚠️ Failed to get info for device {deviceId}: {result}")
                None
        
        with
        | ex ->
            logger.LogError(correlationId,
                ExecutionError ($"Error getting device {deviceId} info", Some ex), ex)
            None

    /// Public function to get device info
    let getDeviceInfo (deviceId: int) (logger: ITarsLogger) (correlationId: string) : CudaDeviceInfo option =
        getDeviceInfoInternal deviceId logger correlationId
    
    /// Initialize a specific CUDA device
    let initializeDevice (deviceId: int) (logger: ITarsLogger) (correlationId: string) =
        try
            logger.LogInformation(correlationId, $"🚀 Initializing CUDA device {deviceId}")
            
            let initResult = CudaInterop.tars_cuda_init(deviceId)
            
            if initResult = CudaError.Success then
                logger.LogInformation(correlationId, 
                    $"✅ CUDA device {deviceId} initialized successfully")
                Success (deviceId, Map [("deviceId", box deviceId)])
            else
                let error = ConfigurationError (
                    $"Failed to initialize CUDA device {deviceId}: {initResult}", 
                    "CUDA")
                Failure (error, correlationId)
        
        with
        | ex ->
            let error = ExecutionError ($"CUDA device {deviceId} initialization failed", Some ex)
            logger.LogError(correlationId, error, ex)
            Failure (error, correlationId)
    
    /// Cleanup CUDA resources
    let cleanupCuda (logger: ITarsLogger) (correlationId: string) =
        try
            logger.LogInformation(correlationId, "🧹 Cleaning up CUDA resources")
            
            let result = CudaInterop.tars_cuda_cleanup()
            logger.LogInformation(correlationId, $"✅ CUDA cleanup result: {result}")
            
            Success ((), Map [("cleanupResult", box result)])
        
        with
        | ex ->
            let error = ExecutionError ("CUDA cleanup failed", Some ex)
            logger.LogError(correlationId, error, ex)
            Failure (error, correlationId)
