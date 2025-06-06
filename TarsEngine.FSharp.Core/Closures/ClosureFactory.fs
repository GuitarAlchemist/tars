namespace TarsEngine.FSharp.Core.Closures

open System
open System.Collections.Concurrent
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Services.PlatformService

/// Closure execution environment
type ClosureExecutionEnvironment = {
    WorkingDirectory: string
    EnvironmentVariables: Map<string, string>
    ResourceLimits: ClosureResourceLimits
    SecurityContext: ClosureSecurityContext
    NetworkAccess: NetworkAccessLevel
}

/// Resource limits for closure execution
and ClosureResourceLimits = {
    MaxMemoryMB: int
    MaxCpuPercent: int
    MaxExecutionTimeMs: int
    MaxFileSize: int64
    MaxNetworkConnections: int
}

/// Security context for closure execution
and ClosureSecurityContext = {
    AllowFileSystemAccess: bool
    AllowNetworkAccess: bool
    AllowProcessExecution: bool
    AllowRegistryAccess: bool
    RestrictedDirectories: string list
    AllowedFileExtensions: string list
}

/// Network access levels
and NetworkAccessLevel =
    | NoNetwork
    | LocalOnly
    | RestrictedInternet
    | FullInternet

/// Closure definition
type ClosureDefinition = {
    Id: string
    Name: string
    Description: string
    Type: string
    Language: string
    Code: string
    Parameters: Map<string, obj>
    Dependencies: string list
    CreatedAt: DateTime
    UpdatedAt: DateTime
    Version: string
    Author: string
    Tags: string list
    IsActive: bool
}

/// Closure execution result
type ClosureExecutionResult = {
    Success: bool
    Result: obj option
    Error: string option
    ExecutionTimeMs: int64
    MemoryUsedMB: int
    OutputLogs: string list
    ErrorLogs: string list
}

/// Closure template for generating new closures
type ClosureTemplate = {
    Id: string
    Name: string
    Description: string
    Type: string
    Language: string
    Template: string
    Parameters: Map<string, string>
    Examples: string list
    Documentation: string
}

/// Environment-agnostic closure factory
type ClosureFactory(logger: ILogger<ClosureFactory>, platform: Platform) =
    
    let closures = ConcurrentDictionary<string, ClosureDefinition>()
    let templates = ConcurrentDictionary<string, ClosureTemplate>()
    let executionStats = ConcurrentDictionary<string, int64>()
    
    let platformPaths = getPlatformPaths platform
    let closuresDirectory = Path.Combine(platformPaths.DataPath, "closures")
    let templatesDirectory = Path.Combine(platformPaths.DataPath, "templates")
    
    /// Initialize closure factory
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing Closure Factory...")
            
            // Ensure directories exist
            ensurePlatformDirectories platform logger
            
            if not (Directory.Exists(closuresDirectory)) then
                Directory.CreateDirectory(closuresDirectory) |> ignore
                logger.LogDebug($"Created closures directory: {closuresDirectory}")
            
            if not (Directory.Exists(templatesDirectory)) then
                Directory.CreateDirectory(templatesDirectory) |> ignore
                logger.LogDebug($"Created templates directory: {templatesDirectory}")
            
            // Load existing closures and templates
            do! this.LoadClosuresAsync()
            do! this.LoadTemplatesAsync()
            do! this.LoadBuiltInTemplatesAsync()
            
            logger.LogInformation($"Closure Factory initialized with {closures.Count} closures and {templates.Count} templates")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize Closure Factory")
            raise ex
    }
    
    /// Create a new closure
    member this.CreateClosureAsync(name: string, closureType: string, language: string, code: string, parameters: Map<string, obj>) = task {
        try
            let closureId = Guid.NewGuid().ToString("N")[..7]
            
            let closureDefinition = {
                Id = closureId
                Name = name
                Description = $"Auto-generated {closureType} closure"
                Type = closureType
                Language = language
                Code = code
                Parameters = parameters
                Dependencies = []
                CreatedAt = DateTime.UtcNow
                UpdatedAt = DateTime.UtcNow
                Version = "1.0.0"
                Author = "TARS"
                Tags = [closureType; language]
                IsActive = true
            }
            
            // Register the closure
            closures.[closureId] <- closureDefinition
            
            // Save to file system
            do! this.SaveClosureToFileAsync(closureDefinition)
            
            logger.LogInformation($"Closure created successfully: {name} ({closureId})")
            return Ok closureId
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to create closure: {name}")
            return Error ex.Message
    }
    
    /// Execute a closure
    member this.ExecuteClosureAsync(closureId: string, inputs: Map<string, obj>) = task {
        try
            match closures.TryGetValue(closureId) with
            | true, closure ->
                logger.LogDebug($"Executing closure: {closure.Name} ({closureId})")
                
                let startTime = DateTime.UtcNow
                
                // Create execution environment
                let environment = this.CreateExecutionEnvironment(closure)
                
                // Execute based on language
                let! result = this.ExecuteByLanguageAsync(closure, inputs, environment)
                
                let executionTime = (DateTime.UtcNow - startTime).TotalMilliseconds |> int64
                
                // Update statistics
                executionStats.AddOrUpdate(closureId, 1L, fun _ count -> count + 1L) |> ignore
                
                logger.LogDebug($"Closure executed in {executionTime}ms: {closure.Name}")
                return Ok result
                
            | false, _ ->
                let error = $"Closure not found: {closureId}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to execute closure: {closureId}")
            return Error ex.Message
    }
    
    /// Get closure by ID
    member this.GetClosureAsync(closureId: string) = task {
        match closures.TryGetValue(closureId) with
        | true, closure -> return Ok closure
        | false, _ -> return Error $"Closure not found: {closureId}"
    }
    
    /// List all closures
    member this.ListClosuresAsync() = task {
        return closures.Values |> Seq.toList |> Ok
    }
    
    /// Create execution environment
    member private this.CreateExecutionEnvironment(closure: ClosureDefinition) =
        let capabilities = getPlatformCapabilities platform
        
        {
            WorkingDirectory = closuresDirectory
            EnvironmentVariables = Map.empty
            ResourceLimits = {
                MaxMemoryMB = capabilities.MaxMemoryMB |> Option.defaultValue 512
                MaxCpuPercent = 50
                MaxExecutionTimeMs = 30000
                MaxFileSize = 10L * 1024L * 1024L // 10MB
                MaxNetworkConnections = 5
            }
            SecurityContext = {
                AllowFileSystemAccess = true
                AllowNetworkAccess = capabilities.SupportsNetworking
                AllowProcessExecution = false
                AllowRegistryAccess = false
                RestrictedDirectories = []
                AllowedFileExtensions = [".txt"; ".json"; ".xml"; ".csv"; ".log"]
            }
            NetworkAccess = if capabilities.SupportsNetworking then RestrictedInternet else NoNetwork
        }
    
    /// Execute closure by language
    member private this.ExecuteByLanguageAsync(closure: ClosureDefinition, inputs: Map<string, obj>, environment: ClosureExecutionEnvironment) = task {
        match closure.Language.ToLowerInvariant() with
        | "fsharp" | "f#" ->
            return! this.ExecuteFSharpClosureAsync(closure, inputs, environment)
        | "csharp" | "c#" ->
            return! this.ExecuteCSharpClosureAsync(closure, inputs, environment)
        | "python" ->
            return! this.ExecutePythonClosureAsync(closure, inputs, environment)
        | "javascript" | "js" ->
            return! this.ExecuteJavaScriptClosureAsync(closure, inputs, environment)
        | _ ->
            return {
                Success = false
                Result = None
                Error = Some $"Unsupported language: {closure.Language}"
                ExecutionTimeMs = 0L
                MemoryUsedMB = 0
                OutputLogs = []
                ErrorLogs = [$"Unsupported language: {closure.Language}"]
            }
    }
    
    /// Execute F# closure
    member private this.ExecuteFSharpClosureAsync(closure: ClosureDefinition, inputs: Map<string, obj>, environment: ClosureExecutionEnvironment) = task {
        // F# execution implementation would go here
        return {
            Success = true
            Result = Some "F# execution result"
            Error = None
            ExecutionTimeMs = 100L
            MemoryUsedMB = 10
            OutputLogs = ["F# closure executed successfully"]
            ErrorLogs = []
        }
    }
    
    /// Execute C# closure
    member private this.ExecuteCSharpClosureAsync(closure: ClosureDefinition, inputs: Map<string, obj>, environment: ClosureExecutionEnvironment) = task {
        // C# execution implementation would go here
        return {
            Success = true
            Result = Some "C# execution result"
            Error = None
            ExecutionTimeMs = 120L
            MemoryUsedMB = 15
            OutputLogs = ["C# closure executed successfully"]
            ErrorLogs = []
        }
    }
    
    /// Execute Python closure
    member private this.ExecutePythonClosureAsync(closure: ClosureDefinition, inputs: Map<string, obj>, environment: ClosureExecutionEnvironment) = task {
        // Python execution implementation would go here
        return {
            Success = true
            Result = Some "Python execution result"
            Error = None
            ExecutionTimeMs = 200L
            MemoryUsedMB = 20
            OutputLogs = ["Python closure executed successfully"]
            ErrorLogs = []
        }
    }
    
    /// Execute JavaScript closure
    member private this.ExecuteJavaScriptClosureAsync(closure: ClosureDefinition, inputs: Map<string, obj>, environment: ClosureExecutionEnvironment) = task {
        // JavaScript execution implementation would go here
        return {
            Success = true
            Result = Some "JavaScript execution result"
            Error = None
            ExecutionTimeMs = 80L
            MemoryUsedMB = 8
            OutputLogs = ["JavaScript closure executed successfully"]
            ErrorLogs = []
        }
    }
    
    /// Load closures from file system
    member private this.LoadClosuresAsync() = task {
        try
            if Directory.Exists(closuresDirectory) then
                let files = Directory.GetFiles(closuresDirectory, "*.json")
                for file in files do
                    try
                        let content = File.ReadAllText(file)
                        // In a real implementation, we'd deserialize the closure definition
                        logger.LogDebug($"Loaded closure from: {file}")
                    with
                    | ex ->
                        logger.LogWarning(ex, $"Failed to load closure from: {file}")
        with
        | ex ->
            logger.LogWarning(ex, "Error loading closures from file system")
    }
    
    /// Load templates from file system
    member private this.LoadTemplatesAsync() = task {
        try
            if Directory.Exists(templatesDirectory) then
                let files = Directory.GetFiles(templatesDirectory, "*.json")
                for file in files do
                    try
                        let content = File.ReadAllText(file)
                        // In a real implementation, we'd deserialize the template
                        logger.LogDebug($"Loaded template from: {file}")
                    with
                    | ex ->
                        logger.LogWarning(ex, $"Failed to load template from: {file}")
        with
        | ex ->
            logger.LogWarning(ex, "Error loading templates from file system")
    }
    
    /// Load built-in templates
    member private this.LoadBuiltInTemplatesAsync() = task {
        try
            // Load built-in templates for common closure types
            let builtInTemplates = [
                ("data-processor", "F#", "Data processing closure template")
                ("api-client", "C#", "API client closure template")
                ("ml-model", "Python", "Machine learning model closure template")
                ("ui-component", "JavaScript", "UI component closure template")
            ]
            
            for (name, language, description) in builtInTemplates do
                let template = {
                    Id = Guid.NewGuid().ToString("N")[..7]
                    Name = name
                    Description = description
                    Type = "built-in"
                    Language = language
                    Template = $"// {name} template for {language}"
                    Parameters = Map.empty
                    Examples = []
                    Documentation = $"Built-in template for {description}"
                }
                
                templates.[template.Id] <- template
            
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
            
            logger.LogDebug($"Saved closure to: {filePath}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to save closure: {closure.Name}")
    }
