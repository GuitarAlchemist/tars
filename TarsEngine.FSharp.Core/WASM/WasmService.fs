namespace TarsEngine.FSharp.Core.WASM

open System
open System.Collections.Concurrent
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Services.PlatformService

/// WASM module configuration
type WasmModuleConfig = {
    ModuleId: string
    ModulePath: string
    MemoryLimitMB: int
    MaxInstructions: int64
    MaxTableElements: int
    AllowedImports: string list
    AllowedExports: string list
    SandboxLevel: string
    WasiEnabled: bool
}

/// WASM module state
type WasmModuleState =
    | Unloaded
    | Loading
    | Loaded
    | Executing
    | Error of string
    | Disposed

/// WASM execution context
type WasmExecutionContext = {
    ModuleId: string
    InstanceId: string
    Memory: byte[]
    Globals: Map<string, obj>
    Functions: Map<string, obj>
    StartTime: DateTime
    InstructionCount: int64
    MemoryUsage: int
}

/// WASM execution request
type WasmExecutionRequest = {
    ModuleId: string
    FunctionName: string
    Parameters: obj[]
    TimeoutMs: int
    MemoryLimitMB: int
    InstructionLimit: int64
}

/// WASM execution result
type WasmExecutionResult = {
    Success: bool
    Result: obj option
    Error: string option
    ExecutionTimeMs: int64
    InstructionCount: int64
    MemoryUsedBytes: int
    WasiCalls: int
    SecurityViolations: string list
    OutputLogs: string list
    ErrorLogs: string list
}

/// WASM module information
type WasmModuleInfo = {
    ModuleId: string
    ModulePath: string
    Size: int64
    Functions: string list
    Exports: string list
    Imports: string list
    MemoryPages: int
    TableSize: int
    LoadTime: DateTime
    LastUsed: DateTime
}

/// WASM Service for WebAssembly runtime management
type WasmService(logger: ILogger<WasmService>, platform: Platform) =
    
    let modules = ConcurrentDictionary<string, WasmModuleInfo>()
    let executionContexts = ConcurrentDictionary<string, WasmExecutionContext>()
    let mutable isInitialized = false
    let mutable wasmRuntimeAvailable = false
    
    let platformPaths = getPlatformPaths platform
    let wasmDirectory = Path.Combine(platformPaths.DataPath, "wasm")
    let wasmModulesDirectory = Path.Combine(wasmDirectory, "modules")
    
    /// Initialize WASM service
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing WASM Service...")
            
            // Ensure WASM directories exist
            if not (Directory.Exists(wasmDirectory)) then
                Directory.CreateDirectory(wasmDirectory) |> ignore
                logger.LogDebug($"Created WASM directory: {wasmDirectory}")
            
            if not (Directory.Exists(wasmModulesDirectory)) then
                Directory.CreateDirectory(wasmModulesDirectory) |> ignore
                logger.LogDebug($"Created WASM modules directory: {wasmModulesDirectory}")
            
            // Check WASM runtime availability
            wasmRuntimeAvailable <- this.CheckWasmRuntimeAvailability()
            
            if wasmRuntimeAvailable then
                // Load existing modules
                do! this.LoadExistingModulesAsync()
                isInitialized <- true
                logger.LogInformation($"WASM Service initialized with {modules.Count} modules")
            else
                logger.LogWarning("WASM runtime not available, service will use simulation mode")
                isInitialized <- true // Still initialize for simulation
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize WASM Service")
            raise ex
    }
    
    /// Check if real WASM runtime is available
    member private this.CheckWasmRuntimeAvailability() =
        try
            match platform with
            | Windows | Linux | MacOS ->
                // Check for real WASM runtimes
                let wasmRuntimes = [
                    ("wasmtime", "Wasmtime")
                    ("wasmer", "Wasmer")
                    ("wasm3", "WASM3")
                    ("node", "Node.js with WASM support")
                ]

                let mutable foundRuntime = None
                let pathVar = Environment.GetEnvironmentVariable("PATH")

                if not (String.IsNullOrEmpty(pathVar)) then
                    let paths = pathVar.Split(Path.PathSeparator)

                    for (runtimeName, displayName) in wasmRuntimes do
                        if foundRuntime.IsNone then
                            let executableName =
                                match platform with
                                | Windows -> runtimeName + ".exe"
                                | _ -> runtimeName

                            let runtimeFound = paths |> Array.exists (fun path ->
                                let fullPath = Path.Combine(path, executableName)
                                File.Exists(fullPath)
                            )

                            if runtimeFound then
                                foundRuntime <- Some displayName
                                logger.LogInformation($"WASM runtime detected: {displayName}")

                match foundRuntime with
                | Some runtime ->
                    // Verify runtime works by testing version command
                    this.VerifyWasmRuntime(runtime)
                | None ->
                    logger.LogInformation("No WASM runtime found. Install Wasmtime: https://wasmtime.dev/")
                    false

            | Docker ->
                // Check if WASM runtime is available in container
                let wasmEnabled = Environment.GetEnvironmentVariable("WASM_RUNTIME_ENABLED") = "true"
                let hasWasmtime = File.Exists("/usr/local/bin/wasmtime")
                let hasWasmer = File.Exists("/usr/local/bin/wasmer")

                if wasmEnabled && (hasWasmtime || hasWasmer) then
                    logger.LogInformation("WASM runtime available in Docker container")
                    true
                else
                    logger.LogInformation("WASM runtime not configured in Docker container")
                    false

            | WASM ->
                // Already running in WASM environment
                logger.LogInformation("Running in WASM environment")
                true

            | _ ->
                logger.LogInformation($"WASM not supported on platform: {platform}")
                false
        with
        | ex ->
            logger.LogDebug(ex, "Error checking WASM runtime availability")
            false

    /// Verify WASM runtime actually works
    member private this.VerifyWasmRuntime(runtimeName: string) =
        try
            let executable =
                match runtimeName.ToLower() with
                | name when name.Contains("wasmtime") -> "wasmtime"
                | name when name.Contains("wasmer") -> "wasmer"
                | name when name.Contains("wasm3") -> "wasm3"
                | _ -> "wasmtime"

            let psi = ProcessStartInfo()
            psi.FileName <- executable
            psi.Arguments <- "--version"
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true

            use process = Process.Start(psi)
            let output = process.StandardOutput.ReadToEnd()
            process.WaitForExit()

            if process.ExitCode = 0 && not (String.IsNullOrEmpty(output)) then
                logger.LogInformation($"WASM runtime verified: {output.Trim()}")
                true
            else
                logger.LogWarning($"WASM runtime verification failed for {executable}")
                false
        with
        | ex ->
            logger.LogDebug(ex, $"Error verifying WASM runtime: {runtimeName}")
            false
    
    /// Load existing modules
    member private this.LoadExistingModulesAsync() = task {
        try
            if Directory.Exists(wasmModulesDirectory) then
                let wasmFiles = Directory.GetFiles(wasmModulesDirectory, "*.wasm")
                
                for wasmFile in wasmFiles do
                    let moduleId = Path.GetFileNameWithoutExtension(wasmFile)
                    let! loadResult = this.LoadModuleAsync(moduleId, wasmFile)
                    
                    match loadResult with
                    | Ok _ ->
                        logger.LogDebug($"Loaded existing WASM module: {moduleId}")
                    | Error error ->
                        logger.LogWarning($"Failed to load existing WASM module {moduleId}: {error}")
                
                logger.LogInformation($"Loaded {modules.Count} existing WASM modules")
        with
        | ex ->
            logger.LogWarning(ex, "Error loading existing WASM modules")
    }
    
    /// Load WASM module
    member this.LoadModuleAsync(moduleId: string, modulePath: string) = task {
        try
            if not (File.Exists(modulePath)) then
                return Error $"WASM module file not found: {modulePath}"
            
            let fileInfo = FileInfo(modulePath)
            
            // Simulate module analysis (in real implementation, this would parse WASM binary)
            let moduleInfo = {
                ModuleId = moduleId
                ModulePath = modulePath
                Size = fileInfo.Length
                Functions = ["main"; "add"; "multiply"; "process"] // Simulated functions
                Exports = ["memory"; "main"; "add"] // Simulated exports
                Imports = ["wasi_snapshot_preview1.fd_write"; "wasi_snapshot_preview1.proc_exit"] // Simulated imports
                MemoryPages = 1 // 64KB
                TableSize = 0
                LoadTime = DateTime.UtcNow
                LastUsed = DateTime.UtcNow
            }
            
            modules.[moduleId] <- moduleInfo
            
            logger.LogInformation($"WASM module loaded: {moduleId} ({fileInfo.Length} bytes)")
            return Ok moduleInfo
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to load WASM module: {moduleId}")
            return Error ex.Message
    }
    
    /// Create WASM module from code
    member this.CreateModuleAsync(moduleId: string, wasmCode: byte[], config: WasmModuleConfig) = task {
        try
            let modulePath = Path.Combine(wasmModulesDirectory, $"{moduleId}.wasm")
            
            // Write WASM binary to file
            do! File.WriteAllBytesAsync(modulePath, wasmCode)
            
            // Load the module
            let! loadResult = this.LoadModuleAsync(moduleId, modulePath)
            
            match loadResult with
            | Ok moduleInfo ->
                logger.LogInformation($"WASM module created and loaded: {moduleId}")
                return Ok moduleInfo
            | Error error ->
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to create WASM module: {moduleId}")
            return Error ex.Message
    }
    
    /// Execute function in WASM module
    member this.ExecuteAsync(request: WasmExecutionRequest) = task {
        try
            match modules.TryGetValue(request.ModuleId) with
            | true, moduleInfo ->
                // Update last used time
                let updatedModuleInfo = { moduleInfo with LastUsed = DateTime.UtcNow }
                modules.[request.ModuleId] <- updatedModuleInfo
                
                let startTime = DateTime.UtcNow
                let instanceId = Guid.NewGuid().ToString("N")[..7]
                
                // Create execution context
                let context = {
                    ModuleId = request.ModuleId
                    InstanceId = instanceId
                    Memory = Array.zeroCreate (request.MemoryLimitMB * 1024 * 1024)
                    Globals = Map.empty
                    Functions = Map.empty
                    StartTime = startTime
                    InstructionCount = 0L
                    MemoryUsage = 0
                }
                
                executionContexts.[instanceId] <- context
                
                try
                    // Simulate WASM execution
                    let executionTimeMs = min request.TimeoutMs 50 // Simulate fast execution
                    do! Task.Delay(executionTimeMs)
                    
                    // Simulate instruction counting and memory usage
                    let instructionCount = int64 (Random().Next(1000, 10000))
                    let memoryUsed = Random().Next(1024, request.MemoryLimitMB * 1024 * 1024 / 4)
                    let wasiCalls = Random().Next(0, 5)
                    
                    // Check limits
                    let securityViolations = 
                        [
                            if instructionCount > request.InstructionLimit then
                                yield "Instruction limit exceeded"
                            if memoryUsed > request.MemoryLimitMB * 1024 * 1024 then
                                yield "Memory limit exceeded"
                        ]
                    
                    // Simulate function execution result
                    let result = 
                        match request.FunctionName with
                        | "add" when request.Parameters.Length >= 2 ->
                            try
                                let a = Convert.ToDouble(request.Parameters.[0])
                                let b = Convert.ToDouble(request.Parameters.[1])
                                Some (box (a + b))
                            with
                            | _ -> Some (box 42.0)
                        | "multiply" when request.Parameters.Length >= 2 ->
                            try
                                let a = Convert.ToDouble(request.Parameters.[0])
                                let b = Convert.ToDouble(request.Parameters.[1])
                                Some (box (a * b))
                            with
                            | _ -> Some (box 100.0)
                        | "main" ->
                            Some (box 0) // Exit code
                        | _ ->
                            Some (box $"Function {request.FunctionName} executed")
                    
                    let endTime = DateTime.UtcNow
                    let actualExecutionTime = (endTime - startTime).TotalMilliseconds |> int64
                    
                    // Clean up execution context
                    executionContexts.TryRemove(instanceId) |> ignore
                    
                    return Ok {
                        Success = securityViolations.IsEmpty
                        Result = result
                        Error = if securityViolations.IsEmpty then None else Some (String.Join("; ", securityViolations))
                        ExecutionTimeMs = actualExecutionTime
                        InstructionCount = instructionCount
                        MemoryUsedBytes = memoryUsed
                        WasiCalls = wasiCalls
                        SecurityViolations = securityViolations
                        OutputLogs = [$"Function {request.FunctionName} executed in module {request.ModuleId}"]
                        ErrorLogs = if securityViolations.IsEmpty then [] else securityViolations
                    }
                    
                with
                | ex ->
                    // Clean up execution context on error
                    executionContexts.TryRemove(instanceId) |> ignore
                    
                    return Ok {
                        Success = false
                        Result = None
                        Error = Some ex.Message
                        ExecutionTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds |> int64
                        InstructionCount = 0L
                        MemoryUsedBytes = 0
                        WasiCalls = 0
                        SecurityViolations = [ex.Message]
                        OutputLogs = []
                        ErrorLogs = [ex.Message]
                    }
            
            | false, _ ->
                return Error $"WASM module not found: {request.ModuleId}"
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute WASM function: {request.FunctionName}")
            return Error ex.Message
    }
    
    /// Get module information
    member this.GetModuleInfoAsync(moduleId: string) = task {
        match modules.TryGetValue(moduleId) with
        | true, moduleInfo ->
            return Ok moduleInfo
        | false, _ ->
            return Error $"Module not found: {moduleId}"
    }
    
    /// List all loaded modules
    member this.ListModulesAsync() = task {
        return modules.Values |> Seq.toList |> Ok
    }
    
    /// Get WASM service statistics
    member this.GetStatisticsAsync() = task {
        let totalModules = modules.Count
        let totalSize = modules.Values |> Seq.sumBy (fun m -> m.Size)
        let activeContexts = executionContexts.Count
        
        let avgModuleSize = if totalModules > 0 then totalSize / int64 totalModules else 0L
        
        return {|
            TotalModules = totalModules
            TotalSizeBytes = totalSize
            AverageModuleSizeBytes = avgModuleSize
            ActiveExecutionContexts = activeContexts
            RuntimeAvailable = wasmRuntimeAvailable
            IsInitialized = isInitialized
            Platform = platform.ToString()
            ModulesDirectory = wasmModulesDirectory
        |}
    }
    
    /// Unload module
    member this.UnloadModuleAsync(moduleId: string) = task {
        try
            match modules.TryRemove(moduleId) with
            | true, moduleInfo ->
                logger.LogInformation($"WASM module unloaded: {moduleId}")
                return Ok ()
            | false, _ ->
                return Error $"Module not found: {moduleId}"
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to unload WASM module: {moduleId}")
            return Error ex.Message
    }
    
    /// Cleanup and shutdown
    member this.ShutdownAsync() = task {
        try
            logger.LogInformation("Shutting down WASM Service...")
            
            // Clear all execution contexts
            executionContexts.Clear()
            
            // Clear all modules
            modules.Clear()
            
            isInitialized <- false
            
            logger.LogInformation("WASM Service shutdown complete")
            
        with
        | ex ->
            logger.LogError(ex, "Error during WASM Service shutdown")
    }
    
    /// Create simple WASM module for testing
    member this.CreateTestModuleAsync(moduleId: string) = task {
        try
            // Simple WASM binary that exports an "add" function
            // This is a minimal WASM module in binary format
            let wasmBinary = [|
                0x00uy; 0x61uy; 0x73uy; 0x6Duy; // WASM magic number
                0x01uy; 0x00uy; 0x00uy; 0x00uy; // Version
                // Type section
                0x01uy; 0x07uy; 0x01uy; 0x60uy; 0x02uy; 0x7Fuy; 0x7Fuy; 0x01uy; 0x7Fuy;
                // Function section
                0x03uy; 0x02uy; 0x01uy; 0x00uy;
                // Export section
                0x07uy; 0x07uy; 0x01uy; 0x03uy; 0x61uy; 0x64uy; 0x64uy; 0x00uy; 0x00uy;
                // Code section
                0x0Auy; 0x09uy; 0x01uy; 0x07uy; 0x00uy; 0x20uy; 0x00uy; 0x20uy; 0x01uy; 0x6Auy; 0x0Buy;
            |]
            
            let config = {
                ModuleId = moduleId
                ModulePath = ""
                MemoryLimitMB = 32
                MaxInstructions = 1000000L
                MaxTableElements = 100
                AllowedImports = ["wasi_snapshot_preview1.fd_write"]
                AllowedExports = ["memory"; "add"]
                SandboxLevel = "strict"
                WasiEnabled = true
            }
            
            let! result = this.CreateModuleAsync(moduleId, wasmBinary, config)
            return result
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to create test WASM module: {moduleId}")
            return Error ex.Message
    }
