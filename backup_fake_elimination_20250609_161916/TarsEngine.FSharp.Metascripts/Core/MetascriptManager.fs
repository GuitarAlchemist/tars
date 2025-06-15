namespace TarsEngine.FSharp.Metascripts.Core

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascripts.Core

/// <summary>
/// Manager for metascript operations.
/// </summary>
type MetascriptManager(registry: MetascriptRegistry, logger: ILogger<MetascriptManager>) =
    
    /// <summary>
    /// Loads a metascript from file.
    /// </summary>
    member _.LoadMetascriptAsync(filePath: string) =
        task {
            try
                logger.LogInformation(sprintf "Loading metascript from: %s" filePath)
                
                if File.Exists(filePath) then
                    let content = File.ReadAllText(filePath)
                    let name = Path.GetFileNameWithoutExtension(filePath)
                    
                    let metadata = MetascriptHelpers.createMetadata name (sprintf "Loaded from %s" filePath) "TARS"
                    let source = MetascriptHelpers.createMetascriptSource name content filePath metadata
                    
                    let registered = registry.RegisterMetascript(source)
                    logger.LogInformation(sprintf "Metascript loaded successfully: %s" name)
                    
                    return Ok registered
                else
                    let error = sprintf "File not found: %s" filePath
                    logger.LogError(error)
                    return Error error
            with
            | ex ->
                let error = sprintf "Error loading metascript: %s" ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Saves a metascript to file.
    /// </summary>
    member _.SaveMetascriptAsync(source: MetascriptSource, filePath: string) =
        task {
            try
                logger.LogInformation(sprintf "Saving metascript to: %s" filePath)
                
                let directory = Path.GetDirectoryName(filePath)
                if not (Directory.Exists(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                
                File.WriteAllText(filePath, source.Content)
                logger.LogInformation(sprintf "Metascript saved successfully: %s" filePath)
                
                return Ok ()
            with
            | ex ->
                let error = sprintf "Error saving metascript: %s" ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Validates a metascript.
    /// </summary>
    member _.ValidateMetascript(source: MetascriptSource) =
        try
            let errors = ResizeArray<string>()
            
            // Basic validation
            if String.IsNullOrWhiteSpace(source.Name) then
                errors.Add("Metascript name cannot be empty")
            
            if String.IsNullOrWhiteSpace(source.Content) then
                errors.Add("Metascript content cannot be empty")
            
            if source.Content.Length > 1000000 then // 1MB limit
                errors.Add("Metascript content is too large (max 1MB)")
            
            let validationErrors = errors |> Seq.toList
            let isValid = validationErrors.IsEmpty
            
            { source with IsValid = isValid; ValidationErrors = validationErrors }
        with
        | ex ->
            logger.LogError(ex, sprintf "Error validating metascript: %s" source.Name)
            { source with IsValid = false; ValidationErrors = [ex.Message] }
    
    /// <summary>
    /// Gets metascript execution history.
    /// </summary>
    member _.GetExecutionHistory(metascriptName: string) =
        // Simplified implementation - in real system would query database
        []
    
    /// <summary>
    /// Cleans up old metascript data.
    /// </summary>
    member _.CleanupAsync() =
        task {
            try
                logger.LogInformation("Starting metascript cleanup")
                // Simplified cleanup - in real system would clean old execution results, etc.
                logger.LogInformation("Metascript cleanup completed")
                return Ok ()
            with
            | ex ->
                let error = sprintf "Error during cleanup: %s" ex.Message
                logger.LogError(ex, error)
                return Error error
        }
