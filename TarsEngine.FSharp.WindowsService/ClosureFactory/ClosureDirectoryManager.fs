namespace TarsEngine.FSharp.WindowsService.ClosureFactory

open System
open System.IO
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Text.Json
open YamlDotNet.Serialization

/// <summary>
/// Closure definition from YAML file
/// </summary>
type ClosureDefinitionFile = {
    Name: string
    Version: string
    Description: string
    Author: string
    Category: string
    Type: string
    Language: string
    Template: string
    Parameters: Map<string, ParameterDefinition>
    Dependencies: string list
    Examples: ExampleDefinition list
    Metadata: Map<string, obj>
}

/// <summary>
/// Parameter definition for closures
/// </summary>
and ParameterDefinition = {
    Name: string
    Type: string
    Required: bool
    DefaultValue: obj option
    Description: string
    Validation: ValidationRule list
}

/// <summary>
/// Validation rule for parameters
/// </summary>
and ValidationRule = {
    Type: string
    Value: obj
    Message: string
}

/// <summary>
/// Example definition for closures
/// </summary>
and ExampleDefinition = {
    Name: string
    Description: string
    Parameters: Map<string, obj>
    ExpectedOutput: string option
}

/// <summary>
/// Closure directory structure
/// </summary>
type ClosureDirectoryStructure = {
    RootPath: string
    TemplatesPath: string
    DefinitionsPath: string
    ScriptsPath: string
    ConfigsPath: string
    ExamplesPath: string
    MarketplacePath: string
}

/// <summary>
/// File change event
/// </summary>
type FileChangeEvent = {
    FilePath: string
    ChangeType: WatcherChangeTypes
    Timestamp: DateTime
    ClosureName: string option
}

/// <summary>
/// Closure validation result
/// </summary>
type ClosureValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
    Suggestions: string list
    ValidationTime: TimeSpan
}

/// <summary>
/// Closure directory manager for extensible closure factory
/// </summary>
type ClosureDirectoryManager(logger: ILogger<ClosureDirectoryManager>) =
    
    let loadedClosures = ConcurrentDictionary<string, ClosureDefinitionFile>()
    let fileWatchers = ConcurrentDictionary<string, FileSystemWatcher>()
    let changeEvents = ConcurrentQueue<FileChangeEvent>()
    let validationCache = ConcurrentDictionary<string, ClosureValidationResult>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable processingTask: Task option = None
    
    let directoryStructure = {
        RootPath = ".tars/closures"
        TemplatesPath = ".tars/closures/templates"
        DefinitionsPath = ".tars/closures/definitions"
        ScriptsPath = ".tars/closures/scripts"
        ConfigsPath = ".tars/closures/configs"
        ExamplesPath = ".tars/closures/examples"
        MarketplacePath = ".tars/closures/marketplace"
    }
    
    let yamlDeserializer = DeserializerBuilder().Build()
    let yamlSerializer = SerializerBuilder().Build()
    
    /// Start the closure directory manager
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting closure directory manager...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Ensure directory structure exists
            this.EnsureDirectoryStructure()
            
            // Load existing closures
            do! this.LoadExistingClosuresAsync()
            
            // Setup file watchers
            this.SetupFileWatchers()
            
            // Start change processing loop
            let processingLoop = this.ChangeProcessingLoopAsync(cancellationTokenSource.Value.Token)
            processingTask <- Some processingLoop
            
            logger.LogInformation($"Closure directory manager started. Loaded {loadedClosures.Count} closures.")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start closure directory manager")
            isRunning <- false
            raise
    }
    
    /// Stop the closure directory manager
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping closure directory manager...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Dispose file watchers
            for kvp in fileWatchers do
                try
                    kvp.Value.Dispose()
                with
                | ex -> logger.LogWarning(ex, $"Error disposing file watcher: {kvp.Key}")
            
            fileWatchers.Clear()
            
            // Wait for processing task to complete
            match processingTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Closure directory manager processing task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for closure directory manager processing task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            processingTask <- None
            
            logger.LogInformation("Closure directory manager stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping closure directory manager")
    }
    
    /// Get all loaded closures
    member this.GetLoadedClosures() =
        loadedClosures.Values |> List.ofSeq
    
    /// Get closure by name
    member this.GetClosure(name: string) =
        match loadedClosures.TryGetValue(name) with
        | true, closure -> Some closure
        | false, _ -> None
    
    /// Validate closure definition
    member this.ValidateClosureAsync(closureDefinition: ClosureDefinitionFile) = task {
        try
            let startTime = DateTime.UtcNow
            let errors = ResizeArray<string>()
            let warnings = ResizeArray<string>()
            let suggestions = ResizeArray<string>()
            
            // Basic validation
            if String.IsNullOrWhiteSpace(closureDefinition.Name) then
                errors.Add("Closure name is required")
            
            if String.IsNullOrWhiteSpace(closureDefinition.Version) then
                errors.Add("Closure version is required")
            
            if String.IsNullOrWhiteSpace(closureDefinition.Type) then
                errors.Add("Closure type is required")
            
            if String.IsNullOrWhiteSpace(closureDefinition.Template) then
                errors.Add("Closure template is required")
            
            // Template validation
            let templatePath = Path.Combine(directoryStructure.TemplatesPath, closureDefinition.Template)
            if not (File.Exists(templatePath)) then
                errors.Add($"Template file not found: {closureDefinition.Template}")
            
            // Parameter validation
            for kvp in closureDefinition.Parameters do
                let param = kvp.Value
                if String.IsNullOrWhiteSpace(param.Name) then
                    errors.Add($"Parameter name is required")
                
                if String.IsNullOrWhiteSpace(param.Type) then
                    errors.Add($"Parameter type is required for: {param.Name}")
            
            // Dependency validation
            for dependency in closureDefinition.Dependencies do
                if not (loadedClosures.ContainsKey(dependency)) then
                    warnings.Add($"Dependency not found: {dependency}")
            
            // Suggestions
            if closureDefinition.Examples.IsEmpty then
                suggestions.Add("Consider adding usage examples")
            
            if String.IsNullOrWhiteSpace(closureDefinition.Author) then
                suggestions.Add("Consider adding author information")
            
            let validationTime = DateTime.UtcNow - startTime
            
            let result = {
                IsValid = errors.Count = 0
                Errors = errors |> List.ofSeq
                Warnings = warnings |> List.ofSeq
                Suggestions = suggestions |> List.ofSeq
                ValidationTime = validationTime
            }
            
            // Cache validation result
            validationCache.[closureDefinition.Name] <- result
            
            return result
            
        with
        | ex ->
            logger.LogError(ex, $"Error validating closure: {closureDefinition.Name}")
            return {
                IsValid = false
                Errors = [ex.Message]
                Warnings = []
                Suggestions = []
                ValidationTime = TimeSpan.Zero
            }
    }
    
    /// Load closure from file
    member this.LoadClosureFromFileAsync(filePath: string) = task {
        try
            logger.LogDebug($"Loading closure from file: {filePath}")
            
            let! content = File.ReadAllTextAsync(filePath)
            let closureDefinition = yamlDeserializer.Deserialize<ClosureDefinitionFile>(content)
            
            // Validate closure
            let! validationResult = this.ValidateClosureAsync(closureDefinition)
            
            if validationResult.IsValid then
                loadedClosures.[closureDefinition.Name] <- closureDefinition
                logger.LogInformation($"Loaded closure: {closureDefinition.Name} v{closureDefinition.Version}")
                return Ok closureDefinition
            else
                let errors = String.Join("; ", validationResult.Errors)
                logger.LogWarning($"Invalid closure definition in {filePath}: {errors}")
                return Error $"Invalid closure: {errors}"
            
        with
        | ex ->
            logger.LogError(ex, $"Error loading closure from file: {filePath}")
            return Error ex.Message
    }
    
    /// Save closure to file
    member this.SaveClosureToFileAsync(closureDefinition: ClosureDefinitionFile, filePath: string) = task {
        try
            logger.LogDebug($"Saving closure to file: {filePath}")
            
            // Validate before saving
            let! validationResult = this.ValidateClosureAsync(closureDefinition)
            
            if not validationResult.IsValid then
                let errors = String.Join("; ", validationResult.Errors)
                return Error $"Cannot save invalid closure: {errors}"
            
            let yamlContent = yamlSerializer.Serialize(closureDefinition)
            do! File.WriteAllTextAsync(filePath, yamlContent)
            
            logger.LogInformation($"Saved closure: {closureDefinition.Name} to {filePath}")
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Error saving closure to file: {filePath}")
            return Error ex.Message
    }
    
    /// Create example closure definitions
    member this.CreateExampleClosuresAsync() = task {
        try
            logger.LogInformation("Creating example closure definitions...")
            
            // Web API closure example
            let webApiClosure = {
                Name = "WebAPIGenerator"
                Version = "1.0.0"
                Description = "Generates a complete REST API with CRUD operations"
                Author = "TARS System"
                Category = "WebDevelopment"
                Type = "WebAPI"
                Language = "CSharp"
                Template = "webapi.template.cs"
                Parameters = Map.ofList [
                    ("EntityName", {
                        Name = "EntityName"
                        Type = "string"
                        Required = true
                        DefaultValue = None
                        Description = "Name of the entity for CRUD operations"
                        Validation = []
                    })
                    ("DatabaseType", {
                        Name = "DatabaseType"
                        Type = "string"
                        Required = false
                        DefaultValue = Some ("SqlServer" :> obj)
                        Description = "Type of database to use"
                        Validation = []
                    })
                ]
                Dependencies = []
                Examples = [
                    {
                        Name = "User Management API"
                        Description = "Create a user management API"
                        Parameters = Map.ofList [("EntityName", "User" :> obj); ("DatabaseType", "SqlServer" :> obj)]
                        ExpectedOutput = Some "Complete ASP.NET Core Web API project"
                    }
                ]
                Metadata = Map.ofList [("CreatedBy", "TARS" :> obj); ("CreatedAt", DateTime.UtcNow :> obj)]
            }
            
            // Infrastructure closure example
            let infrastructureClosure = {
                Name = "DockerInfrastructure"
                Version = "1.0.0"
                Description = "Creates Docker-based infrastructure with multiple services"
                Author = "TARS System"
                Category = "Infrastructure"
                Type = "Infrastructure"
                Language = "Docker"
                Template = "docker-infrastructure.template.yml"
                Parameters = Map.ofList [
                    ("Services", {
                        Name = "Services"
                        Type = "array"
                        Required = true
                        DefaultValue = None
                        Description = "List of services to include"
                        Validation = []
                    })
                    ("Environment", {
                        Name = "Environment"
                        Type = "string"
                        Required = false
                        DefaultValue = Some ("development" :> obj)
                        Description = "Target environment"
                        Validation = []
                    })
                ]
                Dependencies = []
                Examples = [
                    {
                        Name = "Microservices Stack"
                        Description = "Create a complete microservices infrastructure"
                        Parameters = Map.ofList [("Services", ["api"; "database"; "redis"; "nginx"] :> obj)]
                        ExpectedOutput = Some "Docker Compose configuration with all services"
                    }
                ]
                Metadata = Map.ofList [("CreatedBy", "TARS" :> obj); ("CreatedAt", DateTime.UtcNow :> obj)]
            }
            
            // Save example closures
            let webApiPath = Path.Combine(directoryStructure.DefinitionsPath, "webapi-generator.closure.yaml")
            let! webApiResult = this.SaveClosureToFileAsync(webApiClosure, webApiPath)
            
            let infraPath = Path.Combine(directoryStructure.DefinitionsPath, "docker-infrastructure.closure.yaml")
            let! infraResult = this.SaveClosureToFileAsync(infrastructureClosure, infraPath)
            
            match webApiResult, infraResult with
            | Ok (), Ok () ->
                logger.LogInformation("Example closures created successfully")
                return Ok ()
            | Error error, _ | _, Error error ->
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, "Error creating example closures")
            return Error ex.Message
    }
    
    /// Ensure directory structure exists
    member private this.EnsureDirectoryStructure() =
        let directories = [
            directoryStructure.RootPath
            directoryStructure.TemplatesPath
            directoryStructure.DefinitionsPath
            directoryStructure.ScriptsPath
            directoryStructure.ConfigsPath
            directoryStructure.ExamplesPath
            directoryStructure.MarketplacePath
        ]
        
        for directory in directories do
            if not (Directory.Exists(directory)) then
                Directory.CreateDirectory(directory) |> ignore
                logger.LogDebug($"Created directory: {directory}")
    
    /// Load existing closures from directory
    member private this.LoadExistingClosuresAsync() = task {
        try
            let definitionFiles = Directory.GetFiles(directoryStructure.DefinitionsPath, "*.closure.yaml")
            
            for filePath in definitionFiles do
                let! result = this.LoadClosureFromFileAsync(filePath)
                match result with
                | Ok _ -> ()
                | Error error -> logger.LogWarning($"Failed to load closure from {filePath}: {error}")
            
            logger.LogInformation($"Loaded {loadedClosures.Count} closures from directory")
            
        with
        | ex ->
            logger.LogError(ex, "Error loading existing closures")
    }
    
    /// Setup file system watchers
    member private this.SetupFileWatchers() =
        let watchDirectories = [
            directoryStructure.DefinitionsPath
            directoryStructure.TemplatesPath
            directoryStructure.ConfigsPath
        ]
        
        for directory in watchDirectories do
            try
                let watcher = new FileSystemWatcher(directory)
                watcher.Filter <- "*.*"
                watcher.IncludeSubdirectories <- true
                watcher.NotifyFilter <- NotifyFilters.CreationTime ||| NotifyFilters.LastWrite ||| NotifyFilters.FileName
                
                watcher.Created.Add(fun args -> this.OnFileChanged(args))
                watcher.Changed.Add(fun args -> this.OnFileChanged(args))
                watcher.Deleted.Add(fun args -> this.OnFileChanged(args))
                watcher.Renamed.Add(fun args -> this.OnFileRenamed(args))
                
                watcher.EnableRaisingEvents <- true
                fileWatchers.[directory] <- watcher
                
                logger.LogDebug($"Setup file watcher for: {directory}")
                
            with
            | ex ->
                logger.LogWarning(ex, $"Failed to setup file watcher for: {directory}")
    
    /// Handle file change events
    member private this.OnFileChanged(args: FileSystemEventArgs) =
        let changeEvent = {
            FilePath = args.FullPath
            ChangeType = args.ChangeType
            Timestamp = DateTime.UtcNow
            ClosureName = this.ExtractClosureNameFromPath(args.FullPath)
        }
        
        changeEvents.Enqueue(changeEvent)
        logger.LogDebug($"File change detected: {args.ChangeType} - {args.FullPath}")
    
    /// Handle file rename events
    member private this.OnFileRenamed(args: RenamedEventArgs) =
        let changeEvent = {
            FilePath = args.FullPath
            ChangeType = WatcherChangeTypes.Renamed
            Timestamp = DateTime.UtcNow
            ClosureName = this.ExtractClosureNameFromPath(args.FullPath)
        }
        
        changeEvents.Enqueue(changeEvent)
        logger.LogDebug($"File renamed: {args.OldFullPath} -> {args.FullPath}")
    
    /// Extract closure name from file path
    member private this.ExtractClosureNameFromPath(filePath: string) =
        try
            let fileName = Path.GetFileNameWithoutExtension(filePath)
            if fileName.EndsWith(".closure") then
                Some (fileName.Substring(0, fileName.Length - 8)) // Remove ".closure"
            else
                None
        with
        | _ -> None
    
    /// Change processing loop
    member private this.ChangeProcessingLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting closure directory change processing loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Process pending change events
                    while changeEvents.TryDequeue() |> fst do
                        let (success, changeEvent) = changeEvents.TryDequeue()
                        if success then
                            do! this.ProcessChangeEventAsync(changeEvent)
                    
                    // Wait before next processing cycle
                    do! Task.Delay(TimeSpan.FromSeconds(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in closure directory change processing loop")
                    do! Task.Delay(TimeSpan.FromSeconds(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Closure directory change processing loop cancelled")
        | ex ->
            logger.LogError(ex, "Closure directory change processing loop failed")
    }
    
    /// Process individual change event
    member private this.ProcessChangeEventAsync(changeEvent: FileChangeEvent) = task {
        try
            match changeEvent.ChangeType with
            | WatcherChangeTypes.Created | WatcherChangeTypes.Changed ->
                if changeEvent.FilePath.EndsWith(".closure.yaml") then
                    // Reload closure definition
                    let! result = this.LoadClosureFromFileAsync(changeEvent.FilePath)
                    match result with
                    | Ok closure ->
                        logger.LogInformation($"Reloaded closure: {closure.Name}")
                    | Error error ->
                        logger.LogWarning($"Failed to reload closure from {changeEvent.FilePath}: {error}")
            
            | WatcherChangeTypes.Deleted ->
                match changeEvent.ClosureName with
                | Some closureName ->
                    loadedClosures.TryRemove(closureName) |> ignore
                    validationCache.TryRemove(closureName) |> ignore
                    logger.LogInformation($"Removed closure: {closureName}")
                | None -> ()
            
            | _ -> ()
            
        with
        | ex ->
            logger.LogWarning(ex, $"Error processing change event for: {changeEvent.FilePath}")
    }
    
    /// Get directory structure
    member this.GetDirectoryStructure() = directoryStructure
    
    /// Get validation result for closure
    member this.GetValidationResult(closureName: string) =
        match validationCache.TryGetValue(closureName) with
        | true, result -> Some result
        | false, _ -> None
