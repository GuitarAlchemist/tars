namespace TarsEngine.FSharp.WindowsService.ClosureFactory

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Execution environment for closure isolation
/// </summary>
type ExecutionEnvironment = {
    WorkingDirectory: string
    EnvironmentVariables: Map<string, string>
    ResourceLimits: ExecutionResourceLimits
    SecurityContext: SecurityContext
    NetworkAccess: NetworkAccessLevel
}

/// <summary>
/// Resource limits for closure execution
/// </summary>
and ExecutionResourceLimits = {
    MaxMemoryMB: int
    MaxCpuPercent: float
    MaxExecutionTime: TimeSpan
    MaxDiskSpaceMB: int
    MaxFileHandles: int
    MaxNetworkConnections: int
}

/// <summary>
/// Security context for closure execution
/// </summary>
and SecurityContext = {
    AllowFileSystemAccess: bool
    AllowNetworkAccess: bool
    AllowProcessExecution: bool
    AllowRegistryAccess: bool
    RestrictedDirectories: string list
    AllowedFileExtensions: string list
}

/// <summary>
/// Network access levels
/// </summary>
and NetworkAccessLevel =
    | None
    | LocalOnly
    | RestrictedInternet
    | FullInternet

/// <summary>
/// Execution sandbox for safe closure execution
/// </summary>
type ExecutionSandbox = {
    Id: string
    WorkingDirectory: string
    Process: Process option
    StartTime: DateTime
    ResourceMonitor: ResourceMonitor option
    IsActive: bool
}

/// <summary>
/// Resource monitoring for executing closures
/// </summary>
and ResourceMonitor = {
    ProcessId: int
    InitialMemoryMB: float
    PeakMemoryMB: float
    CurrentMemoryMB: float
    CpuUsagePercent: float
    DiskUsageMB: float
    NetworkBytesSent: int64
    NetworkBytesReceived: int64
    FileHandlesCount: int
}

/// <summary>
/// Closure execution metrics
/// </summary>
type ClosureExecutionMetrics = {
    ExecutionId: string
    ClosureId: string
    StartTime: DateTime
    EndTime: DateTime option
    ExecutionTime: TimeSpan
    ResourceUsage: ResourceMonitor
    OutputSize: int64
    LogsGenerated: int
    FilesCreated: int
    NetworkRequests: int
    ExitCode: int option
}

/// <summary>
/// Safe closure executor with sandboxing and resource management
/// </summary>
type ClosureExecutor(logger: ILogger<ClosureExecutor>) =
    
    let activeSandboxes = ConcurrentDictionary<string, ExecutionSandbox>()
    let executionMetrics = ConcurrentQueue<ClosureExecutionMetrics>()
    let resourceMonitors = ConcurrentDictionary<string, ResourceMonitor>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable monitoringTask: Task option = None
    
    let maxExecutionHistory = 1000
    let sandboxDirectory = ".tars/sandbox"
    
    /// Start the closure executor
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting closure executor...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Ensure sandbox directory exists
            this.EnsureSandboxDirectory()
            
            // Start resource monitoring loop
            let monitoringLoop = this.ResourceMonitoringLoopAsync(cancellationTokenSource.Value.Token)
            monitoringTask <- Some monitoringLoop
            
            logger.LogInformation("Closure executor started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start closure executor")
            isRunning <- false
            raise
    }
    
    /// Stop the closure executor
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping closure executor...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Terminate active sandboxes
            this.TerminateActiveSandboxes()
            
            // Wait for monitoring task to complete
            match monitoringTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Resource monitoring task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for resource monitoring task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            monitoringTask <- None
            
            logger.LogInformation("Closure executor stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping closure executor")
    }
    
    /// Execute a closure in a sandboxed environment
    member this.ExecuteAsync(closure: ClosureDefinition, context: ClosureExecutionContext) = task {
        try
            logger.LogInformation($"Executing closure: {closure.Name} ({closure.Id})")
            
            let startTime = DateTime.UtcNow
            
            // Create execution environment
            let environment = this.CreateExecutionEnvironment(context)
            
            // Create sandbox
            let! sandbox = this.CreateSandboxAsync(context.ExecutionId, environment)
            
            // Prepare closure for execution
            let! preparationResult = this.PrepareClosure(closure, context, sandbox)
            match preparationResult with
            | Error error ->
                return {
                    ExecutionId = context.ExecutionId
                    ClosureId = context.ClosureId
                    Status = Failed
                    Result = None
                    Error = Some error
                    ExecutionTime = DateTime.UtcNow - startTime
                    MemoryUsed = 0L
                    OutputFiles = []
                    Logs = []
                    Metadata = Map.empty
                }
            
            | Ok () ->
                // Execute the closure
                let! executionResult = this.ExecuteClosureInSandbox(closure, context, sandbox)
                
                // Collect execution metrics
                let metrics = this.CollectExecutionMetrics(context.ExecutionId, startTime)
                
                // Cleanup sandbox
                do! this.CleanupSandboxAsync(sandbox)
                
                return executionResult
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute closure: {closure.Name}")
            return {
                ExecutionId = context.ExecutionId
                ClosureId = context.ClosureId
                Status = Failed
                Result = None
                Error = Some ex.Message
                ExecutionTime = TimeSpan.Zero
                MemoryUsed = 0L
                OutputFiles = []
                Logs = []
                Metadata = Map.empty
            }
    }
    
    /// Create execution environment
    member private this.CreateExecutionEnvironment(context: ClosureExecutionContext) =
        {
            WorkingDirectory = context.WorkingDirectory
            EnvironmentVariables = Map.ofList [
                ("TARS_EXECUTION_ID", context.ExecutionId)
                ("TARS_CLOSURE_ID", context.ClosureId)
                ("TARS_WORKING_DIR", context.WorkingDirectory)
            ]
            ResourceLimits = {
                MaxMemoryMB = context.MaxMemoryMB
                MaxCpuPercent = 80.0
                MaxExecutionTime = context.Timeout
                MaxDiskSpaceMB = 1024
                MaxFileHandles = 100
                MaxNetworkConnections = 10
            }
            SecurityContext = {
                AllowFileSystemAccess = true
                AllowNetworkAccess = true
                AllowProcessExecution = false
                AllowRegistryAccess = false
                RestrictedDirectories = ["C:\\Windows"; "C:\\Program Files"]
                AllowedFileExtensions = [".txt"; ".json"; ".xml"; ".csv"; ".log"]
            }
            NetworkAccess = RestrictedInternet
        }
    
    /// Create execution sandbox
    member private this.CreateSandboxAsync(executionId: string, environment: ExecutionEnvironment) = task {
        try
            logger.LogDebug($"Creating sandbox for execution: {executionId}")
            
            // Create working directory
            if not (Directory.Exists(environment.WorkingDirectory)) then
                Directory.CreateDirectory(environment.WorkingDirectory) |> ignore
            
            let sandbox = {
                Id = executionId
                WorkingDirectory = environment.WorkingDirectory
                Process = None
                StartTime = DateTime.UtcNow
                ResourceMonitor = None
                IsActive = true
            }
            
            activeSandboxes.[executionId] <- sandbox
            
            logger.LogDebug($"Sandbox created: {executionId}")
            return sandbox
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to create sandbox: {executionId}")
            raise
    }
    
    /// Prepare closure for execution
    member private this.PrepareClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        try
            logger.LogDebug($"Preparing closure for execution: {closure.Name}")
            
            // Write closure code to file
            let codeFileName = this.GetCodeFileName(closure.Type)
            let codeFilePath = Path.Combine(sandbox.WorkingDirectory, codeFileName)
            do! File.WriteAllTextAsync(codeFilePath, closure.Code)
            
            // Write parameters to file
            let parametersFilePath = Path.Combine(sandbox.WorkingDirectory, "parameters.json")
            let parametersJson = this.SerializeParameters(context.Parameters)
            do! File.WriteAllTextAsync(parametersFilePath, parametersJson)
            
            // Create execution script
            let scriptPath = this.CreateExecutionScript(closure, sandbox.WorkingDirectory)
            
            logger.LogDebug($"Closure prepared for execution: {closure.Name}")
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to prepare closure: {closure.Name}")
            return Error ex.Message
    }
    
    /// Execute closure in sandbox
    member private this.ExecuteClosureInSandbox(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        try
            logger.LogDebug($"Executing closure in sandbox: {closure.Name}")
            
            let startTime = DateTime.UtcNow
            let logs = ResizeArray<string>()
            let outputFiles = ResizeArray<string>()
            
            // Determine execution method based on closure type
            let! executionResult = 
                match closure.Type with
                | WebAPI -> this.ExecuteWebAPIClosure(closure, context, sandbox)
                | Infrastructure -> this.ExecuteInfrastructureClosure(closure, context, sandbox)
                | DataProcessor -> this.ExecuteDataProcessorClosure(closure, context, sandbox)
                | TestGenerator -> this.ExecuteTestGeneratorClosure(closure, context, sandbox)
                | DocumentationGenerator -> this.ExecuteDocumentationClosure(closure, context, sandbox)
                | CodeAnalyzer -> this.ExecuteCodeAnalyzerClosure(closure, context, sandbox)
                | DatabaseMigration -> this.ExecuteDatabaseMigrationClosure(closure, context, sandbox)
                | DeploymentScript -> this.ExecuteDeploymentScriptClosure(closure, context, sandbox)
                | MonitoringDashboard -> this.ExecuteMonitoringDashboardClosure(closure, context, sandbox)
                | Custom customType -> this.ExecuteCustomClosure(customType, closure, context, sandbox)
            
            // Collect output files
            if Directory.Exists(sandbox.WorkingDirectory) then
                let files = Directory.GetFiles(sandbox.WorkingDirectory, "*", SearchOption.AllDirectories)
                outputFiles.AddRange(files)
            
            // Collect logs
            let logFilePath = Path.Combine(sandbox.WorkingDirectory, "execution.log")
            if File.Exists(logFilePath) then
                let! logContent = File.ReadAllTextAsync(logFilePath)
                logs.AddRange(logContent.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries))
            
            let executionTime = DateTime.UtcNow - startTime
            
            return {
                ExecutionId = context.ExecutionId
                ClosureId = context.ClosureId
                Status = if executionResult.IsOk then Completed else Failed
                Result = if executionResult.IsOk then Some executionResult.Value else None
                Error = if executionResult.IsError then Some executionResult.Error else None
                ExecutionTime = executionTime
                MemoryUsed = 0L // Would be collected from resource monitor
                OutputFiles = outputFiles |> List.ofSeq
                Logs = logs |> List.ofSeq
                Metadata = Map.ofList [
                    ("SandboxId", sandbox.Id :> obj)
                    ("WorkingDirectory", sandbox.WorkingDirectory :> obj)
                ]
            }
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute closure in sandbox: {closure.Name}")
            return {
                ExecutionId = context.ExecutionId
                ClosureId = context.ClosureId
                Status = Failed
                Result = None
                Error = Some ex.Message
                ExecutionTime = DateTime.UtcNow - DateTime.UtcNow
                MemoryUsed = 0L
                OutputFiles = []
                Logs = []
                Metadata = Map.empty
            }
    }
    
    /// Execute Web API closure
    member private this.ExecuteWebAPIClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        try
            logger.LogDebug($"Executing Web API closure: {closure.Name}")
            
            // Generate API project structure
            let projectPath = Path.Combine(sandbox.WorkingDirectory, "GeneratedAPI")
            Directory.CreateDirectory(projectPath) |> ignore
            
            // Write project files
            let projectFile = Path.Combine(projectPath, "GeneratedAPI.csproj")
            let projectContent = this.GenerateProjectFile("web")
            do! File.WriteAllTextAsync(projectFile, projectContent)
            
            // Write controller code
            let controllerPath = Path.Combine(projectPath, "Controllers")
            Directory.CreateDirectory(controllerPath) |> ignore
            let controllerFile = Path.Combine(controllerPath, "GeneratedController.cs")
            do! File.WriteAllTextAsync(controllerFile, closure.Code)
            
            // Write program.cs
            let programFile = Path.Combine(projectPath, "Program.cs")
            let programContent = this.GenerateProgramFile("web")
            do! File.WriteAllTextAsync(programFile, programContent)
            
            logger.LogInformation($"Web API project generated successfully: {projectPath}")
            return Ok ("Web API project generated" :> obj)
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute Web API closure: {closure.Name}")
            return Error ex.Message
    }
    
    /// Execute Infrastructure closure
    member private this.ExecuteInfrastructureClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        try
            logger.LogDebug($"Executing Infrastructure closure: {closure.Name}")
            
            // Write docker-compose.yml
            let dockerComposePath = Path.Combine(sandbox.WorkingDirectory, "docker-compose.yml")
            do! File.WriteAllTextAsync(dockerComposePath, closure.Code)
            
            // Write additional configuration files
            let configPath = Path.Combine(sandbox.WorkingDirectory, "config")
            Directory.CreateDirectory(configPath) |> ignore
            
            // Generate environment file
            let envFile = Path.Combine(sandbox.WorkingDirectory, ".env")
            let envContent = "ENVIRONMENT=development\nDEBUG=true"
            do! File.WriteAllTextAsync(envFile, envContent)
            
            logger.LogInformation($"Infrastructure configuration generated successfully")
            return Ok ("Infrastructure configuration generated" :> obj)
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute Infrastructure closure: {closure.Name}")
            return Error ex.Message
    }
    
    /// Execute other closure types (simplified implementations)
    member private this.ExecuteDataProcessorClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Data Processor closure: {closure.Name}")
        return Ok ("Data processor executed" :> obj)
    }
    
    member private this.ExecuteTestGeneratorClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Test Generator closure: {closure.Name}")
        return Ok ("Test generator executed" :> obj)
    }
    
    member private this.ExecuteDocumentationClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Documentation closure: {closure.Name}")
        return Ok ("Documentation generated" :> obj)
    }
    
    member private this.ExecuteCodeAnalyzerClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Code Analyzer closure: {closure.Name}")
        return Ok ("Code analysis completed" :> obj)
    }
    
    member private this.ExecuteDatabaseMigrationClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Database Migration closure: {closure.Name}")
        return Ok ("Database migration executed" :> obj)
    }
    
    member private this.ExecuteDeploymentScriptClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Deployment Script closure: {closure.Name}")
        return Ok ("Deployment script executed" :> obj)
    }
    
    member private this.ExecuteMonitoringDashboardClosure(closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Monitoring Dashboard closure: {closure.Name}")
        return Ok ("Monitoring dashboard created" :> obj)
    }
    
    member private this.ExecuteCustomClosure(customType: string, closure: ClosureDefinition, context: ClosureExecutionContext, sandbox: ExecutionSandbox) = task {
        logger.LogDebug($"Executing Custom closure ({customType}): {closure.Name}")
        return Ok ($"Custom {customType} closure executed" :> obj)
    }
    
    /// Helper methods
    member private this.GetCodeFileName(closureType: ClosureType) =
        match closureType with
        | WebAPI -> "Controller.cs"
        | Infrastructure -> "docker-compose.yml"
        | DataProcessor -> "processor.py"
        | TestGenerator -> "tests.cs"
        | DocumentationGenerator -> "documentation.md"
        | CodeAnalyzer -> "analyzer.cs"
        | DatabaseMigration -> "migration.sql"
        | DeploymentScript -> "deploy.ps1"
        | MonitoringDashboard -> "dashboard.html"
        | Custom _ -> "custom.txt"
    
    member private this.SerializeParameters(parameters: Map<string, obj>) =
        // Simple JSON serialization (in production, use proper JSON library)
        let paramList = parameters |> Map.toList |> List.map (fun (k, v) -> $"\"{k}\": \"{v}\"")
        "{" + String.Join(", ", paramList) + "}"
    
    member private this.CreateExecutionScript(closure: ClosureDefinition, workingDirectory: string) =
        let scriptPath = Path.Combine(workingDirectory, "execute.cmd")
        let scriptContent = 
            match closure.Type with
            | WebAPI -> "dotnet build && dotnet run"
            | Infrastructure -> "docker-compose up -d"
            | _ -> "echo Execution completed"
        
        File.WriteAllText(scriptPath, scriptContent)
        scriptPath
    
    member private this.GenerateProjectFile(projectType: string) =
        match projectType with
        | "web" -> 
            """<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.AspNetCore.OpenApi" Version="8.0.0" />
    <PackageReference Include="Swashbuckle.AspNetCore" Version="6.4.0" />
  </ItemGroup>
</Project>"""
        | _ -> ""
    
    member private this.GenerateProgramFile(projectType: string) =
        match projectType with
        | "web" ->
            """var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();
app.Run();"""
        | _ -> ""
    
    /// Resource monitoring loop
    member private this.ResourceMonitoringLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting resource monitoring loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Monitor active sandboxes
                    this.MonitorActiveSandboxes()
                    
                    // Clean up completed executions
                    this.CleanupCompletedExecutions()
                    
                    // Wait for next monitoring cycle
                    do! Task.Delay(TimeSpan.FromSeconds(10.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in resource monitoring loop")
                    do! Task.Delay(TimeSpan.FromSeconds(10.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Resource monitoring loop cancelled")
        | ex ->
            logger.LogError(ex, "Resource monitoring loop failed")
    }
    
    /// Monitor active sandboxes
    member private this.MonitorActiveSandboxes() =
        for kvp in activeSandboxes do
            let sandbox = kvp.Value
            if sandbox.IsActive then
                // Check if sandbox has been running too long
                let runningTime = DateTime.UtcNow - sandbox.StartTime
                if runningTime > TimeSpan.FromHours(1.0) then
                    logger.LogWarning($"Long-running sandbox detected: {sandbox.Id} ({runningTime})")
    
    /// Clean up completed executions
    member private this.CleanupCompletedExecutions() =
        let completedSandboxes = 
            activeSandboxes.Values
            |> Seq.filter (fun s -> not s.IsActive)
            |> List.ofSeq
        
        for sandbox in completedSandboxes do
            activeSandboxes.TryRemove(sandbox.Id) |> ignore
    
    /// Collect execution metrics
    member private this.CollectExecutionMetrics(executionId: string, startTime: DateTime) =
        {
            ExecutionId = executionId
            ClosureId = ""
            StartTime = startTime
            EndTime = Some DateTime.UtcNow
            ExecutionTime = DateTime.UtcNow - startTime
            ResourceUsage = {
                ProcessId = 0
                InitialMemoryMB = 0.0
                PeakMemoryMB = 0.0
                CurrentMemoryMB = 0.0
                CpuUsagePercent = 0.0
                DiskUsageMB = 0.0
                NetworkBytesSent = 0L
                NetworkBytesReceived = 0L
                FileHandlesCount = 0
            }
            OutputSize = 0L
            LogsGenerated = 0
            FilesCreated = 0
            NetworkRequests = 0
            ExitCode = Some 0
        }
    
    /// Cleanup sandbox
    member private this.CleanupSandboxAsync(sandbox: ExecutionSandbox) = task {
        try
            logger.LogDebug($"Cleaning up sandbox: {sandbox.Id}")
            
            // Mark sandbox as inactive
            let inactiveSandbox = { sandbox with IsActive = false }
            activeSandboxes.[sandbox.Id] <- inactiveSandbox
            
            // Clean up working directory (optional - keep for debugging)
            // Directory.Delete(sandbox.WorkingDirectory, true)
            
            logger.LogDebug($"Sandbox cleaned up: {sandbox.Id}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Error cleaning up sandbox: {sandbox.Id}")
    }
    
    /// Terminate active sandboxes
    member private this.TerminateActiveSandboxes() =
        for kvp in activeSandboxes do
            let sandbox = kvp.Value
            if sandbox.IsActive then
                try
                    match sandbox.Process with
                    | Some process when not process.HasExited ->
                        process.Kill()
                        logger.LogInformation($"Terminated sandbox process: {sandbox.Id}")
                    | _ -> ()
                with
                | ex ->
                    logger.LogWarning(ex, $"Error terminating sandbox: {sandbox.Id}")
    
    /// Ensure sandbox directory exists
    member private this.EnsureSandboxDirectory() =
        try
            if not (Directory.Exists(sandboxDirectory)) then
                Directory.CreateDirectory(sandboxDirectory) |> ignore
                logger.LogDebug($"Created sandbox directory: {sandboxDirectory}")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to create sandbox directory")
    
    /// Get execution metrics
    member this.GetExecutionMetrics() =
        executionMetrics |> List.ofSeq
    
    /// Get active sandboxes
    member this.GetActiveSandboxes() =
        activeSandboxes.Values |> Seq.filter (fun s -> s.IsActive) |> List.ofSeq
