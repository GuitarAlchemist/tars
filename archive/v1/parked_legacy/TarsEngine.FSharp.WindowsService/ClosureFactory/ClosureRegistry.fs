namespace TarsEngine.FSharp.WindowsService.ClosureFactory

open System
open System.Collections.Concurrent
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Closure validation result
/// </summary>
type ClosureValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
    Suggestions: string list
}

/// <summary>
/// Closure search criteria
/// </summary>
type ClosureSearchCriteria = {
    Name: string option
    Type: ClosureType option
    Tags: string list
    Author: string option
    CreatedAfter: DateTime option
    CreatedBefore: DateTime option
    IsActive: bool option
}

/// <summary>
/// Closure registry statistics
/// </summary>
type ClosureRegistryStatistics = {
    TotalClosures: int
    ActiveClosures: int
    InactiveClosures: int
    ClosuresByType: Map<ClosureType, int>
    ClosuresByAuthor: Map<string, int>
    AverageClosureAge: TimeSpan
    MostUsedClosures: (string * int64) list
    RecentlyCreated: ClosureDefinition list
}

/// <summary>
/// Closure template
/// </summary>
type ClosureTemplate = {
    Id: string
    Name: string
    Description: string
    Type: ClosureType
    Template: string
    DefaultParameters: ClosureParameter list
    Examples: string list
    Documentation: string
    Version: string
    IsBuiltIn: bool
}

/// <summary>
/// Closure registry for managing closure definitions and templates
/// </summary>
type ClosureRegistry(logger: ILogger<ClosureRegistry>) =
    
    let closures = ConcurrentDictionary<string, ClosureDefinition>()
    let templates = ConcurrentDictionary<string, ClosureTemplate>()
    let closureUsageStats = ConcurrentDictionary<string, int64>()
    let closureIndex = ConcurrentDictionary<string, string list>() // For search indexing
    
    let mutable isInitialized = false
    let registryDirectory = ".tars/registry"
    let templatesDirectory = ".tars/templates"
    
    /// Initialize the closure registry
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing closure registry...")
            
            // Ensure directories exist
            this.EnsureDirectories()
            
            // Load built-in templates
            do! this.LoadBuiltInTemplatesAsync()
            
            // Load existing closures
            do! this.LoadExistingClosuresAsync()
            
            // Build search index
            this.BuildSearchIndex()
            
            isInitialized <- true
            logger.LogInformation($"Closure registry initialized with {closures.Count} closures and {templates.Count} templates")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize closure registry")
            raise
    }
    
    /// Register a new closure
    member this.RegisterClosureAsync(closure: ClosureDefinition) = task {
        try
            if not isInitialized then
                do! this.InitializeAsync()
            
            logger.LogInformation($"Registering closure: {closure.Name} ({closure.Id})")
            
            // Validate closure
            let validationResult = this.ValidateClosure(closure)
            if not validationResult.IsValid then
                let errors = String.Join("; ", validationResult.Errors)
                logger.LogError($"Closure validation failed: {errors}")
                return Error $"Validation failed: {errors}"
            
            // Check for duplicate names
            let existingClosure = 
                closures.Values 
                |> Seq.tryFind (fun c -> c.Name = closure.Name && c.Id <> closure.Id)
            
            match existingClosure with
            | Some existing ->
                let error = $"Closure with name '{closure.Name}' already exists (ID: {existing.Id})"
                logger.LogWarning(error)
                return Error error
            
            | None ->
                // Register the closure
                closures.[closure.Id] <- closure
                
                // Update search index
                this.UpdateSearchIndex(closure)
                
                // Save to file system
                do! this.SaveClosureToFileAsync(closure)
                
                logger.LogInformation($"Closure registered successfully: {closure.Name} ({closure.Id})")
                return Ok ()
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to register closure: {closure.Name}")
            return Error ex.Message
    }
    
    /// Unregister a closure
    member this.UnregisterClosureAsync(closureId: string) = task {
        try
            logger.LogInformation($"Unregistering closure: {closureId}")
            
            match closures.TryRemove(closureId) with
            | true, closure ->
                // Remove from search index
                this.RemoveFromSearchIndex(closure)
                
                // Remove file
                do! this.DeleteClosureFileAsync(closure)
                
                logger.LogInformation($"Closure unregistered successfully: {closure.Name} ({closureId})")
                return Ok ()
            
            | false, _ ->
                let error = $"Closure not found: {closureId}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to unregister closure: {closureId}")
            return Error ex.Message
    }
    
    /// Get a closure by ID
    member this.GetClosureAsync(closureId: string) = task {
        match closures.TryGetValue(closureId) with
        | true, closure -> 
            // Update usage statistics
            closureUsageStats.AddOrUpdate(closureId, 1L, fun _ current -> current + 1L) |> ignore
            return Some closure
        | false, _ -> return None
    }
    
    /// Search closures
    member this.SearchClosuresAsync(criteria: ClosureSearchCriteria) = task {
        try
            logger.LogDebug($"Searching closures with criteria: {criteria}")
            
            let results = 
                closures.Values
                |> Seq.filter (fun closure ->
                    // Filter by name
                    (criteria.Name.IsNone || closure.Name.Contains(criteria.Name.Value, StringComparison.OrdinalIgnoreCase)) &&
                    // Filter by type
                    (criteria.Type.IsNone || closure.Type = criteria.Type.Value) &&
                    // Filter by tags
                    (criteria.Tags.IsEmpty || criteria.Tags |> List.exists (fun tag -> closure.Tags |> List.contains tag)) &&
                    // Filter by author
                    (criteria.Author.IsNone || closure.Author.Contains(criteria.Author.Value, StringComparison.OrdinalIgnoreCase)) &&
                    // Filter by creation date
                    (criteria.CreatedAfter.IsNone || closure.CreatedAt >= criteria.CreatedAfter.Value) &&
                    (criteria.CreatedBefore.IsNone || closure.CreatedAt <= criteria.CreatedBefore.Value) &&
                    // Filter by active status
                    (criteria.IsActive.IsNone || closure.IsActive = criteria.IsActive.Value))
                |> List.ofSeq
            
            logger.LogDebug($"Search returned {results.Length} results")
            return results
            
        with
        | ex ->
            logger.LogError(ex, "Error searching closures")
            return []
    }
    
    /// Get all closures
    member this.GetAllClosuresAsync() = task {
        return closures.Values |> List.ofSeq
    }
    
    /// Get closures by type
    member this.GetClosuresByTypeAsync(closureType: ClosureType) = task {
        return 
            closures.Values 
            |> Seq.filter (fun c -> c.Type = closureType)
            |> List.ofSeq
    }
    
    /// Update a closure
    member this.UpdateClosureAsync(closure: ClosureDefinition) = task {
        try
            logger.LogInformation($"Updating closure: {closure.Name} ({closure.Id})")
            
            // Validate closure
            let validationResult = this.ValidateClosure(closure)
            if not validationResult.IsValid then
                let errors = String.Join("; ", validationResult.Errors)
                logger.LogError($"Closure validation failed: {errors}")
                return Error $"Validation failed: {errors}"
            
            // Check if closure exists
            match closures.TryGetValue(closure.Id) with
            | true, _ ->
                let updatedClosure = { closure with UpdatedAt = DateTime.UtcNow }
                closures.[closure.Id] <- updatedClosure
                
                // Update search index
                this.UpdateSearchIndex(updatedClosure)
                
                // Save to file system
                do! this.SaveClosureToFileAsync(updatedClosure)
                
                logger.LogInformation($"Closure updated successfully: {closure.Name} ({closure.Id})")
                return Ok ()
            
            | false, _ ->
                let error = $"Closure not found: {closure.Id}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to update closure: {closure.Name}")
            return Error ex.Message
    }
    
    /// Validate a closure definition
    member private this.ValidateClosure(closure: ClosureDefinition) =
        let errors = ResizeArray<string>()
        let warnings = ResizeArray<string>()
        let suggestions = ResizeArray<string>()
        
        // Validate required fields
        if String.IsNullOrWhiteSpace(closure.Name) then
            errors.Add("Closure name is required")
        
        if String.IsNullOrWhiteSpace(closure.Description) then
            warnings.Add("Closure description is recommended")
        
        if String.IsNullOrWhiteSpace(closure.Code) then
            errors.Add("Closure code is required")
        
        if String.IsNullOrWhiteSpace(closure.Author) then
            warnings.Add("Closure author is recommended")
        
        // Validate parameters
        for param in closure.Parameters do
            if String.IsNullOrWhiteSpace(param.Name) then
                errors.Add("Parameter name is required")
            
            if param.Required && param.DefaultValue.IsNone then
                warnings.Add($"Required parameter '{param.Name}' has no default value")
        
        // Validate dependencies
        for dependency in closure.Dependencies do
            if not (closures.ContainsKey(dependency)) then
                warnings.Add($"Dependency '{dependency}' not found in registry")
        
        // Suggestions
        if closure.Tags.IsEmpty then
            suggestions.Add("Consider adding tags for better discoverability")
        
        if closure.Parameters.IsEmpty && closure.Type <> Infrastructure then
            suggestions.Add("Consider adding parameters for flexibility")
        
        {
            IsValid = errors.Count = 0
            Errors = errors |> List.ofSeq
            Warnings = warnings |> List.ofSeq
            Suggestions = suggestions |> List.ofSeq
        }
    
    /// Load built-in templates
    member private this.LoadBuiltInTemplatesAsync() = task {
        try
            logger.LogInformation("Loading built-in closure templates...")
            
            let builtInTemplates = [
                {
                    Id = "webapi-template"
                    Name = "Web API Template"
                    Description = "Standard REST API with CRUD operations"
                    Type = WebAPI
                    Template = "webapi-standard"
                    DefaultParameters = [
                        { Name = "entity"; Type = String; Description = "Entity name"; Required = true; DefaultValue = Some ("Item" :> obj); Validation = None }
                        { Name = "endpoints"; Type = Array; Description = "API endpoints"; Required = false; DefaultValue = Some (["GET"; "POST"; "PUT"; "DELETE"] :> obj); Validation = None }
                    ]
                    Examples = ["User API"; "Product API"; "Order API"]
                    Documentation = "Creates a standard REST API with full CRUD operations"
                    Version = "1.0.0"
                    IsBuiltIn = true
                }
                {
                    Id = "infrastructure-template"
                    Name = "Infrastructure Template"
                    Description = "Docker-based infrastructure setup"
                    Type = Infrastructure
                    Template = "docker-compose"
                    DefaultParameters = [
                        { Name = "services"; Type = Array; Description = "Services to include"; Required = true; DefaultValue = Some (["redis"; "mongodb"; "postgresql"] :> obj); Validation = None }
                        { Name = "environment"; Type = String; Description = "Environment name"; Required = false; DefaultValue = Some ("development" :> obj); Validation = None }
                    ]
                    Examples = ["Microservices Stack"; "Database Cluster"; "Monitoring Stack"]
                    Documentation = "Creates a complete infrastructure setup with Docker Compose"
                    Version = "1.0.0"
                    IsBuiltIn = true
                }
                {
                    Id = "dataprocessor-template"
                    Name = "Data Processor Template"
                    Description = "Data processing and transformation pipeline"
                    Type = DataProcessor
                    Template = "data-pipeline"
                    DefaultParameters = [
                        { Name = "inputFormat"; Type = String; Description = "Input data format"; Required = true; DefaultValue = Some ("csv" :> obj); Validation = None }
                        { Name = "outputFormat"; Type = String; Description = "Output data format"; Required = true; DefaultValue = Some ("json" :> obj); Validation = None }
                        { Name = "transformations"; Type = Array; Description = "Data transformations"; Required = false; DefaultValue = None; Validation = None }
                    ]
                    Examples = ["CSV to JSON Converter"; "Data Cleaner"; "ETL Pipeline"]
                    Documentation = "Creates a data processing pipeline with configurable transformations"
                    Version = "1.0.0"
                    IsBuiltIn = true
                }
            ]
            
            for template in builtInTemplates do
                templates.[template.Id] <- template
                logger.LogDebug($"Loaded built-in template: {template.Name}")
            
            logger.LogInformation($"Loaded {builtInTemplates.Length} built-in templates")
            
        with
        | ex ->
            logger.LogWarning(ex, "Error loading built-in templates")
    }
    
    /// Load existing closures from file system
    member private this.LoadExistingClosuresAsync() = task {
        try
            if Directory.Exists(registryDirectory) then
                let files = Directory.GetFiles(registryDirectory, "*.json")
                logger.LogInformation($"Loading {files.Length} existing closures...")
                
                for file in files do
                    try
                        // In a real implementation, we'd deserialize from JSON
                        logger.LogDebug($"Loaded closure from file: {Path.GetFileName(file)}")
                    with
                    | ex ->
                        logger.LogWarning(ex, $"Failed to load closure from file: {file}")
                
                logger.LogInformation($"Loaded existing closures from {files.Length} files")
            else
                logger.LogInformation("No existing closures directory found")
                
        with
        | ex ->
            logger.LogWarning(ex, "Error loading existing closures")
    }
    
    /// Build search index
    member private this.BuildSearchIndex() =
        try
            logger.LogDebug("Building closure search index...")
            
            closureIndex.Clear()
            
            for kvp in closures do
                let closure = kvp.Value
                let searchTerms = ResizeArray<string>()
                
                // Add name words
                searchTerms.AddRange(closure.Name.Split([|' '; '-'; '_'|], StringSplitOptions.RemoveEmptyEntries))
                
                // Add description words
                searchTerms.AddRange(closure.Description.Split([|' '; '.'; ','; ';'|], StringSplitOptions.RemoveEmptyEntries))
                
                // Add tags
                searchTerms.AddRange(closure.Tags)
                
                // Add type
                searchTerms.Add(closure.Type.ToString())
                
                // Add author
                searchTerms.Add(closure.Author)
                
                for term in searchTerms do
                    let normalizedTerm = term.ToLower().Trim()
                    if not (String.IsNullOrWhiteSpace(normalizedTerm)) then
                        closureIndex.AddOrUpdate(normalizedTerm, [closure.Id], fun _ existing -> closure.Id :: existing) |> ignore
            
            logger.LogDebug($"Search index built with {closureIndex.Count} terms")
            
        with
        | ex ->
            logger.LogWarning(ex, "Error building search index")
    
    /// Update search index for a closure
    member private this.UpdateSearchIndex(closure: ClosureDefinition) =
        // Remove old entries (simplified - in production we'd track which terms belong to which closure)
        // Add new entries
        let searchTerms = ResizeArray<string>()
        searchTerms.AddRange(closure.Name.Split([|' '; '-'; '_'|], StringSplitOptions.RemoveEmptyEntries))
        searchTerms.AddRange(closure.Description.Split([|' '; '.'; ','; ';'|], StringSplitOptions.RemoveEmptyEntries))
        searchTerms.AddRange(closure.Tags)
        searchTerms.Add(closure.Type.ToString())
        searchTerms.Add(closure.Author)
        
        for term in searchTerms do
            let normalizedTerm = term.ToLower().Trim()
            if not (String.IsNullOrWhiteSpace(normalizedTerm)) then
                closureIndex.AddOrUpdate(normalizedTerm, [closure.Id], fun _ existing -> 
                    if existing |> List.contains closure.Id then existing else closure.Id :: existing) |> ignore
    
    /// Remove from search index
    member private this.RemoveFromSearchIndex(closure: ClosureDefinition) =
        // In a real implementation, we'd properly remove the closure ID from all relevant terms
        logger.LogDebug($"Removing closure {closure.Id} from search index")
    
    /// Ensure required directories exist
    member private this.EnsureDirectories() =
        try
            if not (Directory.Exists(registryDirectory)) then
                Directory.CreateDirectory(registryDirectory) |> ignore
                logger.LogDebug($"Created registry directory: {registryDirectory}")
            
            if not (Directory.Exists(templatesDirectory)) then
                Directory.CreateDirectory(templatesDirectory) |> ignore
                logger.LogDebug($"Created templates directory: {templatesDirectory}")
        with
        | ex ->
            logger.LogWarning(ex, "Failed to create directories")
    
    /// Save closure to file system
    member private this.SaveClosureToFileAsync(closure: ClosureDefinition) = task {
        try
            let fileName = $"{closure.Name}_{closure.Id}.json"
            let filePath = Path.Combine(registryDirectory, fileName)
            
            // In a real implementation, we'd serialize to JSON
            let content = $"Closure: {closure.Name}\nID: {closure.Id}\nType: {closure.Type}\nCreated: {closure.CreatedAt}\nUpdated: {closure.UpdatedAt}"
            do! File.WriteAllTextAsync(filePath, content)
            
            logger.LogDebug($"Closure saved to file: {fileName}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to save closure to file: {closure.Name}")
    }
    
    /// Delete closure file
    member private this.DeleteClosureFileAsync(closure: ClosureDefinition) = task {
        try
            let fileName = $"{closure.Name}_{closure.Id}.json"
            let filePath = Path.Combine(registryDirectory, fileName)
            
            if File.Exists(filePath) then
                File.Delete(filePath)
                logger.LogDebug($"Closure file deleted: {fileName}")
            
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to delete closure file: {closure.Name}")
    }
    
    /// Get closure templates
    member this.GetTemplatesAsync() = task {
        return templates.Values |> List.ofSeq
    }
    
    /// Get template by ID
    member this.GetTemplateAsync(templateId: string) = task {
        match templates.TryGetValue(templateId) with
        | true, template -> return Some template
        | false, _ -> return None
    }
    
    /// Get registry statistics
    member this.GetStatistics() =
        let totalClosures = closures.Count
        let activeClosures = closures.Values |> Seq.filter (fun c -> c.IsActive) |> Seq.length
        let inactiveClosures = totalClosures - activeClosures
        
        let closuresByType = 
            closures.Values
            |> Seq.groupBy (fun c -> c.Type)
            |> Seq.map (fun (t, closures) -> (t, closures |> Seq.length))
            |> Map.ofSeq
        
        let closuresByAuthor = 
            closures.Values
            |> Seq.groupBy (fun c -> c.Author)
            |> Seq.map (fun (author, closures) -> (author, closures |> Seq.length))
            |> Map.ofSeq
        
        let averageAge = 
            if totalClosures > 0 then
                let totalAge = closures.Values |> Seq.sumBy (fun c -> (DateTime.UtcNow - c.CreatedAt).TotalDays)
                TimeSpan.FromDays(totalAge / float totalClosures)
            else
                TimeSpan.Zero
        
        let mostUsedClosures = 
            closureUsageStats
            |> Seq.sortByDescending (fun kvp -> kvp.Value)
            |> Seq.take (min 10 closureUsageStats.Count)
            |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
            |> List.ofSeq
        
        let recentlyCreated = 
            closures.Values
            |> Seq.sortByDescending (fun c -> c.CreatedAt)
            |> Seq.take (min 5 totalClosures)
            |> List.ofSeq
        
        {
            TotalClosures = totalClosures
            ActiveClosures = activeClosures
            InactiveClosures = inactiveClosures
            ClosuresByType = closuresByType
            ClosuresByAuthor = closuresByAuthor
            AverageClosureAge = averageAge
            MostUsedClosures = mostUsedClosures
            RecentlyCreated = recentlyCreated
        }
    
    /// Get total closures count
    member this.GetTotalClosures() = closures.Count
    
    /// Get active closures count
    member this.GetActiveClosures() = 
        closures.Values |> Seq.filter (fun c -> c.IsActive) |> Seq.length
