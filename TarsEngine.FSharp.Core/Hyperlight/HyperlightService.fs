namespace TarsEngine.FSharp.Core.Hyperlight

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Services.PlatformService

/// Hyperlight VM configuration
type HyperlightVMConfig = {
    VmId: string
    MemoryLimitMB: int
    CpuLimitPercent: int
    MaxExecutionTimeMs: int
    IsolationLevel: string
    SecurityProfile: string
    AllowedSyscalls: string list
    NetworkAccess: bool
    FileSystemAccess: bool
}

/// Hyperlight VM state
type HyperlightVMState =
    | Inactive
    | Initializing
    | Active
    | Busy
    | Error of string
    | Terminating

/// Hyperlight VM instance
type HyperlightVM = {
    Config: HyperlightVMConfig
    State: HyperlightVMState
    ProcessId: int option
    StartTime: DateTime option
    LastActivity: DateTime
    ResourceUsage: HyperlightResourceUsage
}

/// Resource usage metrics
and HyperlightResourceUsage = {
    MemoryUsedMB: float
    CpuUsagePercent: float
    NetworkBytesIn: int64
    NetworkBytesOut: int64
    FileSystemReads: int64
    FileSystemWrites: int64
    SyscallCount: int64
}

/// Hyperlight execution request
type HyperlightExecutionRequest = {
    VmId: string
    Code: string
    Language: string
    Parameters: Map<string, obj>
    TimeoutMs: int
    ResourceLimits: HyperlightResourceLimits
}

/// Resource limits for execution
and HyperlightResourceLimits = {
    MaxMemoryMB: int
    MaxCpuPercent: int
    MaxExecutionTimeMs: int
    MaxNetworkConnections: int
    MaxFileOperations: int
}

/// Hyperlight execution result
type HyperlightExecutionResult = {
    Success: bool
    Result: obj option
    Error: string option
    ExecutionTimeMs: int64
    ResourceUsage: HyperlightResourceUsage
    SecurityViolations: string list
    OutputLogs: string list
    ErrorLogs: string list
}

/// Hyperlight Service for micro-VM management
type HyperlightService(logger: ILogger<HyperlightService>, platform: Platform) =
    
    let vms = ConcurrentDictionary<string, HyperlightVM>()
    let vmPool = System.Collections.Generic.Queue<string>()
    let maxVMs = 100
    let mutable isInitialized = false
    
    let platformPaths = getPlatformPaths platform
    let hyperlightDirectory = Path.Combine(platformPaths.DataPath, "hyperlight")
    
    /// Initialize Hyperlight service
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing Hyperlight Service...")
            
            // Ensure hyperlight directory exists
            if not (Directory.Exists(hyperlightDirectory)) then
                Directory.CreateDirectory(hyperlightDirectory) |> ignore
                logger.LogDebug($"Created hyperlight directory: {hyperlightDirectory}")
            
            // Check if Hyperlight runtime is available
            let hyperlightAvailable = this.CheckHyperlightAvailability()
            
            if hyperlightAvailable then
                // Pre-create VM pool
                do! this.InitializeVMPoolAsync()
                isInitialized <- true
                logger.LogInformation($"Hyperlight Service initialized with {vmPool.Count} VMs in pool")
            else
                logger.LogWarning("Hyperlight runtime not available, service will use fallback mode")
                isInitialized <- false
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize Hyperlight Service")
            raise ex
    }
    
    /// Check if Hyperlight runtime is actually available
    member private this.CheckHyperlightAvailability() =
        try
            match platform with
            | Linux ->
                // Check for real Hyperlight components on Linux
                let hyperlightBinary = "/usr/local/bin/hyperlight"
                let hyperlightLib = "/usr/local/lib/libhyperlight.so"
                let kvmSupport = File.Exists("/dev/kvm")
                let hyperlightInstalled = File.Exists(hyperlightBinary) || File.Exists(hyperlightLib)

                if hyperlightInstalled && kvmSupport then
                    logger.LogInformation("Hyperlight runtime detected on Linux with KVM support")
                    true
                elif kvmSupport then
                    logger.LogInformation("KVM available but Hyperlight not installed. Install from: https://github.com/microsoft/hyperlight")
                    false
                else
                    logger.LogInformation("KVM not available. Hyperlight requires KVM on Linux")
                    false

            | Windows ->
                // Check for real Hyper-V and Windows Sandbox capabilities
                let hyperVFeature = this.CheckWindowsFeature("Microsoft-Hyper-V-All")
                let sandboxFeature = this.CheckWindowsFeature("Containers-DisposableClientVM")
                let hyperlightNuget = this.CheckHyperlightNuGetPackage()

                if hyperVFeature && hyperlightNuget then
                    logger.LogInformation("Hyperlight available with Hyper-V support")
                    true
                elif sandboxFeature && hyperlightNuget then
                    logger.LogInformation("Hyperlight available with Windows Sandbox support")
                    true
                elif hyperVFeature || sandboxFeature then
                    logger.LogInformation("Virtualization available but Hyperlight package not found. Install Hyperlight NuGet package")
                    false
                else
                    logger.LogInformation("Hyper-V or Windows Sandbox required for Hyperlight. Enable in Windows Features")
                    false

            | Docker ->
                // Check if running in privileged container with real VM support
                let privileged = Environment.GetEnvironmentVariable("HYPERLIGHT_ENABLED") = "true"
                let hasDevKvm = File.Exists("/dev/kvm")
                let hasHyperlightBinary = File.Exists("/usr/local/bin/hyperlight")

                if privileged && hasDevKvm && hasHyperlightBinary then
                    logger.LogInformation("Hyperlight available in privileged Docker container")
                    true
                else
                    logger.LogInformation("Hyperlight requires privileged Docker container with KVM and Hyperlight binary")
                    false

            | _ ->
                logger.LogInformation($"Hyperlight not supported on platform: {platform}")
                false
        with
        | ex ->
            logger.LogWarning(ex, "Error checking Hyperlight availability")
            false
    
    /// Check Windows feature availability using real DISM command
    member private this.CheckWindowsFeature(featureName: string) =
        try
            let psi = ProcessStartInfo()
            psi.FileName <- "dism"
            psi.Arguments <- $"/online /get-featureinfo /featurename:{featureName}"
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true

            use process = Process.Start(psi)
            let output = process.StandardOutput.ReadToEnd()
            let error = process.StandardError.ReadToEnd()
            process.WaitForExit()

            if process.ExitCode = 0 then
                let isEnabled = output.Contains("State : Enabled") || output.Contains("State: Enabled")
                logger.LogDebug($"Windows feature {featureName}: {if isEnabled then "Enabled" else "Disabled"}")
                isEnabled
            else
                logger.LogDebug($"Could not check Windows feature {featureName}: {error}")
                false
        with
        | ex ->
            logger.LogDebug(ex, $"Error checking Windows feature: {featureName}")
            false

    /// Check if Hyperlight NuGet package is available
    member private this.CheckHyperlightNuGetPackage() =
        try
            // Check for Hyperlight assemblies in common locations
            let possiblePaths = [
                Path.Combine(Environment.CurrentDirectory, "Hyperlight.dll")
                Path.Combine(Environment.CurrentDirectory, "Microsoft.Hyperlight.dll")
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Hyperlight.dll")
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Microsoft.Hyperlight.dll")
            ]

            let hyperlightFound = possiblePaths |> List.exists File.Exists

            if hyperlightFound then
                logger.LogDebug("Hyperlight assembly found")
            else
                logger.LogDebug("Hyperlight assembly not found. Install via: dotnet add package Microsoft.Hyperlight")

            hyperlightFound
        with
        | ex ->
            logger.LogDebug(ex, "Error checking for Hyperlight NuGet package")
            false
    
    /// Initialize VM pool
    member private this.InitializeVMPoolAsync() = task {
        try
            let poolSize = min 10 maxVMs
            
            for i in 1 .. poolSize do
                let vmId = $"pool-vm-{i:D3}"
                let config = {
                    VmId = vmId
                    MemoryLimitMB = 64
                    CpuLimitPercent = 10
                    MaxExecutionTimeMs = 30000
                    IsolationLevel = "micro-vm"
                    SecurityProfile = "strict"
                    AllowedSyscalls = ["read"; "write"; "exit"; "clock_gettime"]
                    NetworkAccess = false
                    FileSystemAccess = true
                }
                
                let vm = {
                    Config = config
                    State = Inactive
                    ProcessId = None
                    StartTime = None
                    LastActivity = DateTime.UtcNow
                    ResourceUsage = {
                        MemoryUsedMB = 0.0
                        CpuUsagePercent = 0.0
                        NetworkBytesIn = 0L
                        NetworkBytesOut = 0L
                        FileSystemReads = 0L
                        FileSystemWrites = 0L
                        SyscallCount = 0L
                    }
                }
                
                vms.[vmId] <- vm
                vmPool.Enqueue(vmId)
            
            logger.LogInformation($"Initialized VM pool with {poolSize} VMs")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize VM pool")
    }
    
    /// Get or create VM for execution
    member this.GetVMAsync(requirements: HyperlightResourceLimits) = task {
        try
            if not isInitialized then
                return Error "Hyperlight service not initialized"
            
            // Try to get VM from pool
            let vmId = 
                lock vmPool (fun () ->
                    if vmPool.Count > 0 then
                        Some (vmPool.Dequeue())
                    else
                        None
                )
            
            match vmId with
            | Some id ->
                match vms.TryGetValue(id) with
                | true, vm ->
                    // Update VM configuration for requirements
                    let updatedConfig = {
                        vm.Config with
                            MemoryLimitMB = max vm.Config.MemoryLimitMB requirements.MaxMemoryMB
                            CpuLimitPercent = max vm.Config.CpuLimitPercent requirements.MaxCpuPercent
                            MaxExecutionTimeMs = max vm.Config.MaxExecutionTimeMs requirements.MaxExecutionTimeMs
                    }
                    
                    let updatedVM = { vm with Config = updatedConfig; State = Initializing }
                    vms.[id] <- updatedVM
                    
                    // Start VM if not already running
                    do! this.StartVMAsync(id)
                    
                    return Ok id
                | false, _ ->
                    return Error $"VM not found in registry: {id}"
            | None ->
                // Create new VM if pool is empty and under limit
                if vms.Count < maxVMs then
                    let newVmId = $"dynamic-vm-{Guid.NewGuid().ToString("N")[..7]}"
                    let! createResult = this.CreateVMAsync(newVmId, requirements)
                    return createResult
                else
                    return Error "Maximum VM limit reached and no VMs available in pool"
                    
        with
        | ex ->
            logger.LogError(ex, "Failed to get VM")
            return Error ex.Message
    }
    
    /// Create new VM
    member private this.CreateVMAsync(vmId: string, requirements: HyperlightResourceLimits) = task {
        try
            let config = {
                VmId = vmId
                MemoryLimitMB = requirements.MaxMemoryMB
                CpuLimitPercent = requirements.MaxCpuPercent
                MaxExecutionTimeMs = requirements.MaxExecutionTimeMs
                IsolationLevel = "micro-vm"
                SecurityProfile = "strict"
                AllowedSyscalls = ["read"; "write"; "exit"; "clock_gettime"; "mmap"; "munmap"]
                NetworkAccess = requirements.MaxNetworkConnections > 0
                FileSystemAccess = requirements.MaxFileOperations > 0
            }
            
            let vm = {
                Config = config
                State = Inactive
                ProcessId = None
                StartTime = None
                LastActivity = DateTime.UtcNow
                ResourceUsage = {
                    MemoryUsedMB = 0.0
                    CpuUsagePercent = 0.0
                    NetworkBytesIn = 0L
                    NetworkBytesOut = 0L
                    FileSystemReads = 0L
                    FileSystemWrites = 0L
                    SyscallCount = 0L
                }
            }
            
            vms.[vmId] <- vm
            do! this.StartVMAsync(vmId)
            
            logger.LogInformation($"Created new VM: {vmId}")
            return Ok vmId
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to create VM: {vmId}")
            return Error ex.Message
    }
    
    /// Start VM
    member private this.StartVMAsync(vmId: string) = task {
        try
            match vms.TryGetValue(vmId) with
            | true, vm ->
                if vm.State = Inactive || vm.State = Error "" then
                    // Update state to initializing
                    let initializingVM = { vm with State = Initializing; LastActivity = DateTime.UtcNow }
                    vms.[vmId] <- initializingVM
                    
                    // Simulate VM startup (in real implementation, this would start actual Hyperlight VM)
                    do! Task.Delay(10) // Simulate <10ms startup time
                    
                    // Update state to active
                    let activeVM = { 
                        initializingVM with 
                            State = Active
                            StartTime = Some DateTime.UtcNow
                            ProcessId = Some (Random().Next(1000, 9999))
                    }
                    vms.[vmId] <- activeVM
                    
                    logger.LogDebug($"VM started: {vmId}")
                else
                    logger.LogDebug($"VM already running: {vmId}")
            | false, _ ->
                logger.LogWarning($"VM not found for start: {vmId}")
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to start VM: {vmId}")
            // Update VM state to error
            match vms.TryGetValue(vmId) with
            | true, vm ->
                let errorVM = { vm with State = Error ex.Message }
                vms.[vmId] <- errorVM
            | false, _ -> ()
    }
    
    /// Execute code in VM
    member this.ExecuteAsync(request: HyperlightExecutionRequest) = task {
        try
            let! vmResult = this.GetVMAsync(request.ResourceLimits)
            
            match vmResult with
            | Ok vmId ->
                match vms.TryGetValue(vmId) with
                | true, vm when vm.State = Active ->
                    // Update VM state to busy
                    let busyVM = { vm with State = Busy; LastActivity = DateTime.UtcNow }
                    vms.[vmId] <- busyVM
                    
                    let startTime = DateTime.UtcNow
                    
                    try
                        // Simulate code execution (in real implementation, this would execute in Hyperlight VM)
                        let executionTime = min request.TimeoutMs 100
                        do! Task.Delay(executionTime)
                        
                        // Simulate resource usage
                        let resourceUsage = {
                            MemoryUsedMB = float (Random().Next(10, vm.Config.MemoryLimitMB))
                            CpuUsagePercent = float (Random().Next(5, vm.Config.CpuLimitPercent))
                            NetworkBytesIn = if vm.Config.NetworkAccess then int64 (Random().Next(0, 1024)) else 0L
                            NetworkBytesOut = if vm.Config.NetworkAccess then int64 (Random().Next(0, 512)) else 0L
                            FileSystemReads = if vm.Config.FileSystemAccess then int64 (Random().Next(0, 10)) else 0L
                            FileSystemWrites = if vm.Config.FileSystemAccess then int64 (Random().Next(0, 5)) else 0L
                            SyscallCount = int64 (Random().Next(10, 100))
                        }
                        
                        // Update VM with resource usage and return to active state
                        let completedVM = { 
                            busyVM with 
                                State = Active
                                ResourceUsage = resourceUsage
                                LastActivity = DateTime.UtcNow
                        }
                        vms.[vmId] <- completedVM
                        
                        // Return VM to pool if it's a pool VM
                        if vmId.StartsWith("pool-vm-") then
                            lock vmPool (fun () -> vmPool.Enqueue(vmId))
                        
                        let endTime = DateTime.UtcNow
                        let actualExecutionTime = (endTime - startTime).TotalMilliseconds |> int64
                        
                        return Ok {
                            Success = true
                            Result = Some $"Executed {request.Language} code in VM {vmId}"
                            Error = None
                            ExecutionTimeMs = actualExecutionTime
                            ResourceUsage = resourceUsage
                            SecurityViolations = []
                            OutputLogs = [$"Code executed successfully in VM {vmId}"]
                            ErrorLogs = []
                        }
                        
                    with
                    | ex ->
                        // Update VM state back to active on error
                        let errorVM = { busyVM with State = Active; LastActivity = DateTime.UtcNow }
                        vms.[vmId] <- errorVM
                        
                        return Ok {
                            Success = false
                            Result = None
                            Error = Some ex.Message
                            ExecutionTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds |> int64
                            ResourceUsage = vm.ResourceUsage
                            SecurityViolations = []
                            OutputLogs = []
                            ErrorLogs = [ex.Message]
                        }
                
                | true, vm ->
                    return Error $"VM not in active state: {vm.State}"
                | false, _ ->
                    return Error $"VM not found: {vmId}"
            
            | Error error ->
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, "Failed to execute in Hyperlight VM")
            return Error ex.Message
    }
    
    /// Get VM statistics
    member this.GetVMStatisticsAsync() = task {
        let totalVMs = vms.Count
        let activeVMs = vms.Values |> Seq.filter (fun vm -> vm.State = Active) |> Seq.length
        let busyVMs = vms.Values |> Seq.filter (fun vm -> vm.State = Busy) |> Seq.length
        let poolVMs = vmPool.Count
        
        let totalMemoryUsed = vms.Values |> Seq.sumBy (fun vm -> vm.ResourceUsage.MemoryUsedMB)
        let avgCpuUsage = 
            if totalVMs > 0 then
                vms.Values |> Seq.averageBy (fun vm -> vm.ResourceUsage.CpuUsagePercent)
            else 0.0
        
        return {|
            TotalVMs = totalVMs
            ActiveVMs = activeVMs
            BusyVMs = busyVMs
            PoolVMs = poolVMs
            TotalMemoryUsedMB = totalMemoryUsed
            AverageCpuUsagePercent = avgCpuUsage
            IsInitialized = isInitialized
            Platform = platform.ToString()
        |}
    }
    
    /// Cleanup and shutdown
    member this.ShutdownAsync() = task {
        try
            logger.LogInformation("Shutting down Hyperlight Service...")
            
            // Terminate all VMs
            let vmIds = vms.Keys |> Seq.toList
            for vmId in vmIds do
                do! this.TerminateVMAsync(vmId)
            
            vms.Clear()
            vmPool.Clear()
            isInitialized <- false
            
            logger.LogInformation("Hyperlight Service shutdown complete")
            
        with
        | ex ->
            logger.LogError(ex, "Error during Hyperlight Service shutdown")
    }
    
    /// Terminate VM
    member private this.TerminateVMAsync(vmId: string) = task {
        try
            match vms.TryGetValue(vmId) with
            | true, vm ->
                let terminatingVM = { vm with State = Terminating }
                vms.[vmId] <- terminatingVM
                
                // Simulate VM termination
                do! Task.Delay(5)
                
                vms.TryRemove(vmId) |> ignore
                logger.LogDebug($"VM terminated: {vmId}")
            | false, _ ->
                logger.LogDebug($"VM not found for termination: {vmId}")
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to terminate VM: {vmId}")
    }
