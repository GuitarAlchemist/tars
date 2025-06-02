namespace TarsEngine.FSharp.WindowsService.ClosureFactory

open System
open System.Collections.Concurrent
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Closure execution context
/// </summary>
type ClosureExecutionContext = {
    ClosureId: string
    ExecutionId: string
    Parameters: Map<string, obj>
    WorkingDirectory: string
    Timeout: TimeSpan
    MaxMemoryMB: int
    CancellationToken: CancellationToken
    Logger: ILogger
    Metadata: Map<string, obj>
}

/// <summary>
/// Closure execution result
/// </summary>
type ClosureExecutionResult = {
    ExecutionId: string
    ClosureId: string
    Status: ClosureExecutionStatus
    Result: obj option
    Error: string option
    ExecutionTime: TimeSpan
    MemoryUsed: int64
    OutputFiles: string list
    Logs: string list
    Metadata: Map<string, obj>
}

/// <summary>
/// Closure execution status
/// </summary>
and ClosureExecutionStatus =
    | Pending
    | Running
    | Completed
    | Failed
    | Cancelled
    | Timeout

/// <summary>
/// Closure definition
/// </summary>
type ClosureDefinition = {
    Id: string
    Name: string
    Description: string
    Type: ClosureType
    Version: string
    Author: string
    CreatedAt: DateTime
    UpdatedAt: DateTime
    Parameters: ClosureParameter list
    Dependencies: string list
    Template: string
    Code: string
    Configuration: Map<string, obj>
    Tags: string list
    IsActive: bool
}

/// <summary>
/// Closure types
/// </summary>
and ClosureType =
    | WebAPI
    | Infrastructure
    | DataProcessor
    | TestGenerator
    | DocumentationGenerator
    | CodeAnalyzer
    | DatabaseMigration
    | DeploymentScript
    | MonitoringDashboard
    | Custom of string

/// <summary>
/// Closure parameter definition
/// </summary>
and ClosureParameter = {
    Name: string
    Type: ParameterType
    Description: string
    Required: bool
    DefaultValue: obj option
    Validation: string option
}

/// <summary>
/// Parameter types
/// </summary>
and ParameterType =
    | String
    | Integer
    | Boolean
    | Float
    | Array
    | Object
    | File
    | Directory

/// <summary>
/// Closure factory statistics
/// </summary>
type ClosureFactoryStatistics = {
    TotalClosures: int
    ActiveClosures: int
    TotalExecutions: int64
    SuccessfulExecutions: int64
    FailedExecutions: int64
    AverageExecutionTimeMs: float
    ClosuresByType: Map<ClosureType, int>
    ExecutionsByType: Map<ClosureType, int64>
    MemoryUsageStats: MemoryUsageStatistics
}

/// <summary>
/// Memory usage statistics
/// </summary>
and MemoryUsageStatistics = {
    AverageMemoryUsageMB: float
    PeakMemoryUsageMB: int64
    TotalMemoryAllocatedMB: int64
}

/// <summary>
/// Dynamic closure factory for autonomous capability expansion
/// </summary>
type ClosureFactory(logger: ILogger<ClosureFactory>, closureRegistry: ClosureRegistry, closureExecutor: ClosureExecutor) =
    
    let activeExecutions = ConcurrentDictionary<string, ClosureExecutionResult>()
    let executionHistory = ConcurrentQueue<ClosureExecutionResult>()
    let factoryStatistics = ConcurrentDictionary<string, int64>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable monitoringTask: Task option = None
    
    let maxExecutionHistory = 10000
    let closuresDirectory = ".tars/closures"
    let outputDirectory = ".tars/output"
    
    /// Start the closure factory
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting closure factory...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Ensure directories exist
            this.EnsureDirectories()
            
            // Load built-in closure templates
            do! this.LoadBuiltInClosuresAsync()
            
            // Start monitoring loop
            let monitoringLoop = this.MonitoringLoopAsync(cancellationTokenSource.Value.Token)
            monitoringTask <- Some monitoringLoop
            
            logger.LogInformation("Closure factory started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start closure factory")
            isRunning <- false
            raise
    }
    
    /// Stop the closure factory
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping closure factory...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for monitoring task to complete
            match monitoringTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Closure factory monitoring task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for closure factory monitoring task to complete")
            | None -> ()
            
            // Cancel active executions
            this.CancelActiveExecutions()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            monitoringTask <- None
            
            logger.LogInformation("Closure factory stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping closure factory")
    }
    
    /// Create a new closure
    member this.CreateClosureAsync(name: string, closureType: ClosureType, template: string, parameters: ClosureParameter list) = task {
        try
            logger.LogInformation($"Creating closure: {name} of type {closureType}")
            
            let closureDefinition = {
                Id = Guid.NewGuid().ToString()
                Name = name
                Description = $"Auto-generated {closureType} closure"
                Type = closureType
                Version = "1.0.0"
                Author = "TARS Closure Factory"
                CreatedAt = DateTime.UtcNow
                UpdatedAt = DateTime.UtcNow
                Parameters = parameters
                Dependencies = []
                Template = template
                Code = this.GenerateCodeFromTemplate(closureType, template, parameters)
                Configuration = Map.empty
                Tags = [closureType.ToString().ToLower()]
                IsActive = true
            }
            
            // Register the closure
            let! registrationResult = closureRegistry.RegisterClosureAsync(closureDefinition)
            match registrationResult with
            | Ok () ->
                // Save closure to file system
                do! this.SaveClosureToFileAsync(closureDefinition)
                
                this.UpdateStatistics("TotalClosures", 1L)
                this.UpdateStatistics($"ClosuresCreated_{closureType}", 1L)
                
                logger.LogInformation($"Closure created successfully: {name} ({closureDefinition.Id})")
                return Ok closureDefinition.Id
            
            | Error error ->
                logger.LogError($"Failed to register closure: {error}")
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to create closure: {name}")
            return Error ex.Message
    }
    
    /// Execute a closure
    member this.ExecuteClosureAsync(closureId: string, parameters: Map<string, obj>, timeout: TimeSpan option) = task {
        try
            logger.LogInformation($"Executing closure: {closureId}")
            
            // Get closure definition
            match! closureRegistry.GetClosureAsync(closureId) with
            | Some closureDefinition ->
                let executionId = Guid.NewGuid().ToString()
                let executionTimeout = timeout |> Option.defaultValue (TimeSpan.FromMinutes(30.0))
                
                // Create execution context
                let context = {
                    ClosureId = closureId
                    ExecutionId = executionId
                    Parameters = parameters
                    WorkingDirectory = Path.Combine(outputDirectory, executionId)
                    Timeout = executionTimeout
                    MaxMemoryMB = 512
                    CancellationToken = cancellationTokenSource.Value.Token
                    Logger = logger.CreateLogger($"Closure.{closureDefinition.Name}")
                    Metadata = Map.empty
                }
                
                // Initialize execution result
                let initialResult = {
                    ExecutionId = executionId
                    ClosureId = closureId
                    Status = Pending
                    Result = None
                    Error = None
                    ExecutionTime = TimeSpan.Zero
                    MemoryUsed = 0L
                    OutputFiles = []
                    Logs = []
                    Metadata = Map.empty
                }
                
                activeExecutions.[executionId] <- initialResult
                
                // Execute the closure
                let! executionResult = closureExecutor.ExecuteAsync(closureDefinition, context)
                
                // Update active executions
                activeExecutions.[executionId] <- executionResult
                
                // Move to history if completed
                if executionResult.Status = Completed || executionResult.Status = Failed || executionResult.Status = Cancelled then
                    activeExecutions.TryRemove(executionId) |> ignore
                    executionHistory.Enqueue(executionResult)
                    
                    // Keep history size manageable
                    while executionHistory.Count > maxExecutionHistory do
                        executionHistory.TryDequeue() |> ignore
                
                // Update statistics
                this.UpdateExecutionStatistics(executionResult)
                
                logger.LogInformation($"Closure execution completed: {closureId} ({executionId}) - {executionResult.Status}")
                return Ok executionResult
            
            | None ->
                let error = $"Closure not found: {closureId}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute closure: {closureId}")
            return Error ex.Message
    }
    
    /// Create closure from natural language description
    member this.CreateClosureFromDescriptionAsync(description: string, closureType: ClosureType option) = task {
        try
            logger.LogInformation($"Creating closure from description: {description}")
            
            // Analyze description to determine closure type and parameters
            let analyzedType = closureType |> Option.defaultValue (this.AnalyzeClosureType(description))
            let parameters = this.ExtractParametersFromDescription(description)
            let template = this.GenerateTemplateFromDescription(description, analyzedType)
            let name = this.GenerateNameFromDescription(description)
            
            // Create the closure
            return! this.CreateClosureAsync(name, analyzedType, template, parameters)
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to create closure from description: {description}")
            return Error ex.Message
    }
    
    /// Generate code from template
    member private this.GenerateCodeFromTemplate(closureType: ClosureType, template: string, parameters: ClosureParameter list) =
        match closureType with
        | WebAPI ->
            this.GenerateWebAPICode(template, parameters)
        | Infrastructure ->
            this.GenerateInfrastructureCode(template, parameters)
        | DataProcessor ->
            this.GenerateDataProcessorCode(template, parameters)
        | TestGenerator ->
            this.GenerateTestCode(template, parameters)
        | DocumentationGenerator ->
            this.GenerateDocumentationCode(template, parameters)
        | CodeAnalyzer ->
            this.GenerateCodeAnalyzerCode(template, parameters)
        | DatabaseMigration ->
            this.GenerateDatabaseMigrationCode(template, parameters)
        | DeploymentScript ->
            this.GenerateDeploymentScriptCode(template, parameters)
        | MonitoringDashboard ->
            this.GenerateMonitoringDashboardCode(template, parameters)
        | Custom customType ->
            this.GenerateCustomCode(customType, template, parameters)
    
    /// Generate Web API code
    member private this.GenerateWebAPICode(template: string, parameters: ClosureParameter list) =
        let endpoints = parameters |> List.filter (fun p -> p.Name.Contains("endpoint")) |> List.length
        let entityName = parameters |> List.tryFind (fun p -> p.Name = "entity") |> Option.map (fun p -> p.DefaultValue.ToString()) |> Option.defaultValue "Item"
        
        $"""
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace GeneratedAPI.Controllers
{{
    [ApiController]
    [Route("api/[controller]")]
    public class {entityName}Controller : ControllerBase
    {{
        private readonly ILogger<{entityName}Controller> _logger;
        
        public {entityName}Controller(ILogger<{entityName}Controller> logger)
        {{
            _logger = logger;
        }}
        
        [HttpGet]
        public async Task<ActionResult<IEnumerable<{entityName}>>> Get()
        {{
            _logger.LogInformation("Getting all {entityName.ToLower()}s");
            // Implementation would be generated based on parameters
            return Ok(new List<{entityName}>());
        }}
        
        [HttpGet("{{id}}")]
        public async Task<ActionResult<{entityName}>> Get(int id)
        {{
            _logger.LogInformation($"Getting {entityName.ToLower()} with id {{id}}");
            // Implementation would be generated based on parameters
            return Ok(new {entityName}());
        }}
        
        [HttpPost]
        public async Task<ActionResult<{entityName}>> Post([FromBody] {entityName} {entityName.ToLower()})
        {{
            _logger.LogInformation("Creating new {entityName.ToLower()}");
            // Implementation would be generated based on parameters
            return CreatedAtAction(nameof(Get), new {{ id = 1 }}, {entityName.ToLower()});
        }}
        
        [HttpPut("{{id}}")]
        public async Task<IActionResult> Put(int id, [FromBody] {entityName} {entityName.ToLower()})
        {{
            _logger.LogInformation($"Updating {entityName.ToLower()} with id {{id}}");
            // Implementation would be generated based on parameters
            return NoContent();
        }}
        
        [HttpDelete("{{id}}")]
        public async Task<IActionResult> Delete(int id)
        {{
            _logger.LogInformation($"Deleting {entityName.ToLower()} with id {{id}}");
            // Implementation would be generated based on parameters
            return NoContent();
        }}
    }}
}}

public class {entityName}
{{
    public int Id {{ get; set; }}
    public string Name {{ get; set; }} = string.Empty;
    public DateTime CreatedAt {{ get; set; }} = DateTime.UtcNow;
    public DateTime UpdatedAt {{ get; set; }} = DateTime.UtcNow;
}}
"""
    
    /// Generate Infrastructure code
    member private this.GenerateInfrastructureCode(template: string, parameters: ClosureParameter list) =
        let services = parameters |> List.filter (fun p -> p.Name.Contains("service")) |> List.map (fun p -> p.DefaultValue.ToString())
        let serviceList = services |> String.concat "\n  - "
        
        $"""
version: '3.8'
services:
  {serviceList}
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  mongodb:
    image: mongo:6
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped
    
  postgresql:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: tars
      POSTGRES_USER: tars
      POSTGRES_PASSWORD: password
    volumes:
      - postgresql_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  mongodb_data:
  postgresql_data:
"""
    
    /// Generate other closure types (simplified for demo)
    member private this.GenerateDataProcessorCode(template: string, parameters: ClosureParameter list) =
        "// Data processor code would be generated based on parameters"
    
    member private this.GenerateTestCode(template: string, parameters: ClosureParameter list) =
        "// Test code would be generated based on parameters"
    
    member private this.GenerateDocumentationCode(template: string, parameters: ClosureParameter list) =
        "// Documentation code would be generated based on parameters"
    
    member private this.GenerateCodeAnalyzerCode(template: string, parameters: ClosureParameter list) =
        "// Code analyzer code would be generated based on parameters"
    
    member private this.GenerateDatabaseMigrationCode(template: string, parameters: ClosureParameter list) =
        "// Database migration code would be generated based on parameters"
    
    member private this.GenerateDeploymentScriptCode(template: string, parameters: ClosureParameter list) =
        "// Deployment script code would be generated based on parameters"
    
    member private this.GenerateMonitoringDashboardCode(template: string, parameters: ClosureParameter list) =
        "// Monitoring dashboard code would be generated based on parameters"
    
    member private this.GenerateCustomCode(customType: string, template: string, parameters: ClosureParameter list) =
        $"// Custom {customType} code would be generated based on parameters"
    
    /// Analyze closure type from description
    member private this.AnalyzeClosureType(description: string) =
        let lowerDesc = description.ToLower()
        if lowerDesc.Contains("api") || lowerDesc.Contains("endpoint") || lowerDesc.Contains("rest") then WebAPI
        elif lowerDesc.Contains("docker") || lowerDesc.Contains("infrastructure") || lowerDesc.Contains("deploy") then Infrastructure
        elif lowerDesc.Contains("data") || lowerDesc.Contains("process") || lowerDesc.Contains("transform") then DataProcessor
        elif lowerDesc.Contains("test") || lowerDesc.Contains("unit") || lowerDesc.Contains("integration") then TestGenerator
        elif lowerDesc.Contains("document") || lowerDesc.Contains("doc") || lowerDesc.Contains("readme") then DocumentationGenerator
        elif lowerDesc.Contains("analyze") || lowerDesc.Contains("code") || lowerDesc.Contains("quality") then CodeAnalyzer
        elif lowerDesc.Contains("database") || lowerDesc.Contains("migration") || lowerDesc.Contains("schema") then DatabaseMigration
        elif lowerDesc.Contains("monitor") || lowerDesc.Contains("dashboard") || lowerDesc.Contains("metrics") then MonitoringDashboard
        else Custom "Unknown"
    
    /// Extract parameters from description
    member private this.ExtractParametersFromDescription(description: string) =
        // Simple parameter extraction (in production, this would use NLP)
        [
            { Name = "description"; Type = String; Description = "Original description"; Required = true; DefaultValue = Some (description :> obj); Validation = None }
        ]
    
    /// Generate template from description
    member private this.GenerateTemplateFromDescription(description: string, closureType: ClosureType) =
        $"Template for {closureType} based on: {description}"
    
    /// Generate name from description
    member private this.GenerateNameFromDescription(description: string) =
        let words = description.Split([|' '; '.'; ','; ';'|], StringSplitOptions.RemoveEmptyEntries)
        let name = words |> Array.take (min 3 words.Length) |> String.concat ""
        $"Generated{name}"
    
    /// Ensure required directories exist
    member private this.EnsureDirectories() =
        try
            if not (Directory.Exists(closuresDirectory)) then
                Directory.CreateDirectory(closuresDirectory) |> ignore
                logger.LogDebug($"Created closures directory: {closuresDirectory}")
            
            if not (Directory.Exists(outputDirectory)) then
                Directory.CreateDirectory(outputDirectory) |> ignore
                logger.LogDebug($"Created output directory: {outputDirectory}")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to create directories")
    
    /// Load built-in closure templates
    member private this.LoadBuiltInClosuresAsync() = task {
        try
            logger.LogInformation("Loading built-in closure templates...")
            
            // Load built-in templates (simplified for demo)
            let builtInClosures = [
                ("WebAPI Template", WebAPI, "Standard REST API template")
                ("Infrastructure Template", Infrastructure, "Docker infrastructure template")
                ("Data Processor Template", DataProcessor, "Data processing template")
            ]
            
            for (name, closureType, description) in builtInClosures do
                let! result = this.CreateClosureAsync(name, closureType, description, [])
                match result with
                | Ok closureId ->
                    logger.LogDebug($"Loaded built-in closure: {name} ({closureId})")
                | Error error ->
                    logger.LogWarning($"Failed to load built-in closure {name}: {error}")
            
            logger.LogInformation("Built-in closure templates loaded")
            
        with
        | ex ->
            logger.LogWarning(ex, "Error loading built-in closure templates")
    }
    
    /// Save closure to file system
    member private this.SaveClosureToFileAsync(closure: ClosureDefinition) = task {
        try
            let fileName = $"{closure.Name}_{closure.Id}.json"
            let filePath = Path.Combine(closuresDirectory, fileName)
            
            // In a real implementation, we'd serialize the closure definition to JSON
            let content = $"Closure: {closure.Name}\nType: {closure.Type}\nCreated: {closure.CreatedAt}"
            do! File.WriteAllTextAsync(filePath, content)
            
            logger.LogDebug($"Closure saved to file: {fileName}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to save closure to file: {closure.Name}")
    }
    
    /// Monitoring loop
    member private this.MonitoringLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting closure factory monitoring loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Monitor active executions
                    this.MonitorActiveExecutions()
                    
                    // Clean up old executions
                    this.CleanupOldExecutions()
                    
                    // Update statistics
                    this.UpdateFactoryStatistics()
                    
                    // Wait for next monitoring cycle
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in closure factory monitoring loop")
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Closure factory monitoring loop cancelled")
        | ex ->
            logger.LogError(ex, "Closure factory monitoring loop failed")
    }
    
    /// Monitor active executions
    member private this.MonitorActiveExecutions() =
        let longRunningExecutions = 
            activeExecutions.Values
            |> Seq.filter (fun exec -> DateTime.UtcNow - exec.Metadata.GetValueOrDefault("StartTime", DateTime.UtcNow :> obj) :?> DateTime > TimeSpan.FromMinutes(30.0))
            |> List.ofSeq
        
        for execution in longRunningExecutions do
            logger.LogWarning($"Long-running execution detected: {execution.ExecutionId} ({execution.ClosureId})")
    
    /// Clean up old executions
    member private this.CleanupOldExecutions() =
        while executionHistory.Count > maxExecutionHistory do
            executionHistory.TryDequeue() |> ignore
    
    /// Cancel active executions
    member private this.CancelActiveExecutions() =
        for kvp in activeExecutions do
            let execution = kvp.Value
            if execution.Status = Running || execution.Status = Pending then
                let cancelledExecution = { execution with Status = Cancelled }
                activeExecutions.[kvp.Key] <- cancelledExecution
                logger.LogInformation($"Cancelled execution: {execution.ExecutionId}")
    
    /// Update execution statistics
    member private this.UpdateExecutionStatistics(result: ClosureExecutionResult) =
        this.UpdateStatistics("TotalExecutions", 1L)
        
        match result.Status with
        | Completed ->
            this.UpdateStatistics("SuccessfulExecutions", 1L)
        | Failed ->
            this.UpdateStatistics("FailedExecutions", 1L)
        | _ -> ()
        
        // Update execution time statistics
        let executionTimeMs = result.ExecutionTime.TotalMilliseconds
        this.UpdateStatistics("TotalExecutionTimeMs", int64 executionTimeMs)
    
    /// Update factory statistics
    member private this.UpdateFactoryStatistics() =
        let activeCount = activeExecutions.Count
        factoryStatistics.["ActiveExecutions"] <- int64 activeCount
    
    /// Update statistics helper
    member private this.UpdateStatistics(key: string, increment: int64) =
        factoryStatistics.AddOrUpdate(key, increment, fun _ current -> current + increment) |> ignore
    
    /// Get closure factory statistics
    member this.GetStatistics() =
        let totalClosures = closureRegistry.GetTotalClosures()
        let activeClosures = closureRegistry.GetActiveClosures()
        let totalExecutions = factoryStatistics.GetOrAdd("TotalExecutions", 0L)
        let successfulExecutions = factoryStatistics.GetOrAdd("SuccessfulExecutions", 0L)
        let failedExecutions = factoryStatistics.GetOrAdd("FailedExecutions", 0L)
        let totalExecutionTime = factoryStatistics.GetOrAdd("TotalExecutionTimeMs", 0L)
        
        let averageExecutionTime = 
            if totalExecutions > 0L then
                float totalExecutionTime / float totalExecutions
            else 0.0
        
        {
            TotalClosures = totalClosures
            ActiveClosures = activeClosures
            TotalExecutions = totalExecutions
            SuccessfulExecutions = successfulExecutions
            FailedExecutions = failedExecutions
            AverageExecutionTimeMs = averageExecutionTime
            ClosuresByType = Map.empty // Would be populated from registry
            ExecutionsByType = Map.empty // Would be calculated from history
            MemoryUsageStats = {
                AverageMemoryUsageMB = 0.0
                PeakMemoryUsageMB = 0L
                TotalMemoryAllocatedMB = 0L
            }
        }
    
    /// Get active executions
    member this.GetActiveExecutions() =
        activeExecutions.Values |> List.ofSeq
    
    /// Get execution history
    member this.GetExecutionHistory(count: int) =
        executionHistory 
        |> Seq.take (min count executionHistory.Count)
        |> List.ofSeq
    
    /// Get execution by ID
    member this.GetExecution(executionId: string) =
        match activeExecutions.TryGetValue(executionId) with
        | true, execution -> Some execution
        | false, _ ->
            executionHistory
            |> Seq.tryFind (fun exec -> exec.ExecutionId = executionId)
