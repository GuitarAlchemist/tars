namespace TarsEngine.FSharp.Notebooks.Execution

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Kernel management for Jupyter notebook execution
/// </summary>

/// Kernel status
type KernelStatus = 
    | Starting
    | Ready
    | Busy
    | Idle
    | Stopping
    | Dead

/// Kernel connection info
type KernelConnection = {
    KernelId: string
    ProcessId: int option
    Port: int
    Status: KernelStatus
    LastActivity: DateTime
    WorkingDirectory: string
}

/// Execution request
type ExecutionRequest = {
    RequestId: string
    Code: string
    Silent: bool
    StoreHistory: bool
    UserExpressions: Map<string, string>
    AllowStdin: bool
}

/// Execution result
type ExecutionResult = {
    RequestId: string
    Success: bool
    ExecutionCount: int option
    Output: string
    Error: string option
    Data: Map<string, obj>
    Metadata: Map<string, obj>
}

/// Kernel manager interface
type IKernelManager =
    /// Start a new kernel
    abstract member StartKernelAsync: SupportedKernel -> Async<KernelConnection>
    
    /// Stop a kernel
    abstract member StopKernelAsync: string -> Async<bool>
    
    /// Execute code in a kernel
    abstract member ExecuteAsync: string -> ExecutionRequest -> Async<ExecutionResult>
    
    /// Get kernel status
    abstract member GetKernelStatus: string -> KernelStatus option
    
    /// List all active kernels
    abstract member ListKernels: unit -> KernelConnection list
    
    /// Restart a kernel
    abstract member RestartKernelAsync: string -> Async<KernelConnection option>

/// Basic kernel manager implementation
type KernelManager(logger: ILogger<KernelManager>) =
    
    let activeKernels = Dictionary<string, KernelConnection>()
    let mutable nextPort = 8888
    
    interface IKernelManager with
        member _.StartKernelAsync(kernelSpec: SupportedKernel) = async {
            try
                let kernelId = Guid.NewGuid().ToString()
                let port = nextPort
                nextPort <- nextPort + 1
                
                logger.LogInformation("Starting kernel {KernelId} on port {Port}", kernelId, port)
                
                let processId = this.StartKernelProcess(kernelSpec, port)
                
                let connection = {
                    KernelId = kernelId
                    ProcessId = processId
                    Port = port
                    Status = Starting
                    LastActivity = DateTime.UtcNow
                    WorkingDirectory = Environment.CurrentDirectory
                }
                
                activeKernels.[kernelId] <- connection
                
                // Wait a bit for kernel to start
                do! Async.Sleep(2000)
                
                let updatedConnection = { connection with Status = Ready }
                activeKernels.[kernelId] <- updatedConnection
                
                logger.LogInformation("Kernel {KernelId} started successfully", kernelId)
                return updatedConnection
                
            with
            | ex ->
                logger.LogError(ex, "Failed to start kernel")
                return failwith $"Failed to start kernel: {ex.Message}"
        }
        
        member _.StopKernelAsync(kernelId: string) = async {
            try
                match activeKernels.TryGetValue(kernelId) with
                | true, connection ->
                    logger.LogInformation("Stopping kernel {KernelId}", kernelId)
                    
                    match connection.ProcessId with
                    | Some processId ->
                        try
                            let process = Process.GetProcessById(processId)
                            if not process.HasExited then
                                process.Kill()
                                process.WaitForExit(5000) |> ignore
                        with
                        | _ -> () // Process might already be dead
                    | None -> ()
                    
                    activeKernels.Remove(kernelId) |> ignore
                    logger.LogInformation("Kernel {KernelId} stopped", kernelId)
                    return true
                | false, _ ->
                    logger.LogWarning("Kernel {KernelId} not found", kernelId)
                    return false
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to stop kernel {KernelId}", kernelId)
                return false
        }
        
        member _.ExecuteAsync(kernelId: string, request: ExecutionRequest) = async {
            try
                match activeKernels.TryGetValue(kernelId) with
                | true, connection ->
                    logger.LogInformation("Executing code in kernel {KernelId}", kernelId)
                    
                    // Update kernel status
                    let busyConnection = { connection with Status = Busy; LastActivity = DateTime.UtcNow }
                    activeKernels.[kernelId] <- busyConnection
                    
                    // Simulate execution (in real implementation, this would communicate with actual kernel)
                    let! result = this.SimulateExecution(request)
                    
                    // Update kernel status back to idle
                    let idleConnection = { busyConnection with Status = Idle; LastActivity = DateTime.UtcNow }
                    activeKernels.[kernelId] <- idleConnection
                    
                    return result
                    
                | false, _ ->
                    return {
                        RequestId = request.RequestId
                        Success = false
                        ExecutionCount = None
                        Output = ""
                        Error = Some $"Kernel {kernelId} not found"
                        Data = Map.empty
                        Metadata = Map.empty
                    }
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to execute code in kernel {KernelId}", kernelId)
                return {
                    RequestId = request.RequestId
                    Success = false
                    ExecutionCount = None
                    Output = ""
                    Error = Some ex.Message
                    Data = Map.empty
                    Metadata = Map.empty
                }
        }
        
        member _.GetKernelStatus(kernelId: string) =
            match activeKernels.TryGetValue(kernelId) with
            | true, connection -> Some connection.Status
            | false, _ -> None
        
        member _.ListKernels() =
            activeKernels.Values |> List.ofSeq
        
        member _.RestartKernelAsync(kernelId: string) = async {
            try
                match activeKernels.TryGetValue(kernelId) with
                | true, connection ->
                    logger.LogInformation("Restarting kernel {KernelId}", kernelId)
                    
                    // Stop the kernel
                    let! stopped = (this :> IKernelManager).StopKernelAsync(kernelId)
                    
                    if stopped then
                        // Start a new kernel (we'd need to store the original kernel spec)
                        // For now, just return None as we don't have the original spec
                        return None
                    else
                        return None
                        
                | false, _ ->
                    logger.LogWarning("Kernel {KernelId} not found for restart", kernelId)
                    return None
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to restart kernel {KernelId}", kernelId)
                return None
        }
    
    /// Start kernel process
    member private _.StartKernelProcess(kernelSpec: SupportedKernel, port: int) : int option =
        try
            match kernelSpec with
            | Python pythonConfig ->
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "python"
                startInfo.Arguments <- $"-m ipykernel_launcher -f connection_file.json --port={port}"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                Some process.Id
                
            | FSharp fsharpConfig ->
                // F# kernel would require .NET Interactive
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- $"interactive --port {port}"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                Some process.Id
                
            | CSharp csharpConfig ->
                // C# kernel would also use .NET Interactive
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- $"interactive --port {port}"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                Some process.Id
                
            | JavaScript jsConfig ->
                // JavaScript kernel would require IJavaScript or similar
                None // Not implemented
                
            | SQL sqlConfig ->
                // SQL kernel would require specialized SQL kernel
                None // Not implemented
                
            | R rConfig ->
                // R kernel would require IRkernel
                None // Not implemented
                
        with
        | ex ->
            logger.LogError(ex, "Failed to start kernel process")
            None
    
    /// Simulate code execution (placeholder)
    member private _.SimulateExecution(request: ExecutionRequest) : Async<ExecutionResult> = async {
        // Simulate some processing time
        do! Async.Sleep(100)
        
        // Simple simulation based on code content
        let output = 
            if request.Code.Contains("print") then
                "Simulated output from print statement"
            elif request.Code.Contains("=") then
                "Variable assignment completed"
            elif request.Code.Contains("import") then
                "Module imported successfully"
            else
                "Code executed successfully"
        
        return {
            RequestId = request.RequestId
            Success = true
            ExecutionCount = Some 1
            Output = output
            Error = None
            Data = Map.empty
            Metadata = Map.empty
        }
    }

/// Kernel utilities
module KernelUtils =
    
    /// Create execution request
    let createExecutionRequest code =
        {
            RequestId = Guid.NewGuid().ToString()
            Code = code
            Silent = false
            StoreHistory = true
            UserExpressions = Map.empty
            AllowStdin = false
        }
    
    /// Create silent execution request
    let createSilentExecutionRequest code =
        {
            RequestId = Guid.NewGuid().ToString()
            Code = code
            Silent = true
            StoreHistory = false
            UserExpressions = Map.empty
            AllowStdin = false
        }
    
    /// Check if kernel is available
    let isKernelAvailable (kernelSpec: SupportedKernel) : bool =
        match kernelSpec with
        | Python _ ->
            try
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "python"
                startInfo.Arguments <- "--version"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                startInfo.RedirectStandardOutput <- true
                
                use process = Process.Start(startInfo)
                process.WaitForExit()
                process.ExitCode = 0
            with
            | _ -> false
            
        | FSharp _ ->
            try
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- "--version"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                startInfo.RedirectStandardOutput <- true
                
                use process = Process.Start(startInfo)
                process.WaitForExit()
                process.ExitCode = 0
            with
            | _ -> false
            
        | CSharp _ ->
            try
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- "--version"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                startInfo.RedirectStandardOutput <- true
                
                use process = Process.Start(startInfo)
                process.WaitForExit()
                process.ExitCode = 0
            with
            | _ -> false
            
        | JavaScript _ ->
            try
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "node"
                startInfo.Arguments <- "--version"
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true
                startInfo.RedirectStandardOutput <- true
                
                use process = Process.Start(startInfo)
                process.WaitForExit()
                process.ExitCode = 0
            with
            | _ -> false
            
        | SQL _ -> false // Would need to check for SQL kernel
        | R _ -> false   // Would need to check for R kernel
    
    /// Get kernel display name
    let getKernelDisplayName (kernelSpec: SupportedKernel) : string =
        match kernelSpec with
        | Python config -> $"Python {config.Version}"
        | FSharp config -> $"F# (.NET {config.DotNetVersion})"
        | CSharp config -> $"C# (.NET {config.DotNetVersion})"
        | JavaScript config -> $"JavaScript (Node {config.NodeVersion})"
        | SQL config -> $"SQL ({config.DatabaseType})"
        | R config -> $"R {config.Version}"
    
    /// Format execution result
    let formatExecutionResult (result: ExecutionResult) : string =
        let sb = System.Text.StringBuilder()
        
        if result.Success then
            sb.AppendLine($"✅ Execution successful (Request: {result.RequestId})") |> ignore
            if not (String.IsNullOrEmpty(result.Output)) then
                sb.AppendLine($"Output: {result.Output}") |> ignore
        else
            sb.AppendLine($"❌ Execution failed (Request: {result.RequestId})") |> ignore
            match result.Error with
            | Some error -> sb.AppendLine($"Error: {error}") |> ignore
            | None -> ()
        
        sb.ToString()
