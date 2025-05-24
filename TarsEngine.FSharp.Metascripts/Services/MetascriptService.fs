namespace TarsEngine.FSharp.Metascripts.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascripts.Core
open TarsEngine.FSharp.Metascripts.Discovery

/// <summary>
/// Implementation of metascript services.
/// </summary>
type MetascriptService(
    registry: MetascriptRegistry, 
    manager: MetascriptManager, 
    discovery: MetascriptDiscovery,
    logger: ILogger<MetascriptService>) =
    
    interface IMetascriptService with
        member _.DiscoverMetascriptsAsync(directory: string) =
            task {
                try
                    logger.LogInformation(sprintf "Starting metascript discovery in: %s" directory)
                    let! result = discovery.DiscoverMetascriptsAsync(directory, true)
                    match result with
                    | Ok metascripts ->
                        logger.LogInformation(sprintf "Discovery completed. Found %d metascripts" metascripts.Length)
                        return Ok metascripts
                    | Error error ->
                        logger.LogError(sprintf "Discovery failed: %s" error)
                        return Error error
                with
                | ex ->
                    let error = sprintf "Error during discovery: %s" ex.Message
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.ListMetascriptsAsync() =
            task {
                try
                    logger.LogInformation("Listing all registered metascripts")
                    let metascripts = registry.GetAllMetascripts()
                    logger.LogInformation(sprintf "Found %d registered metascripts" metascripts.Length)
                    return Ok metascripts
                with
                | ex ->
                    let error = sprintf "Error listing metascripts: %s" ex.Message
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.GetMetascriptAsync(name: string) =
            task {
                try
                    logger.LogDebug(sprintf "Getting metascript: %s" name)
                    let metascript = registry.GetMetascript(name)
                    return Ok metascript
                with
                | ex ->
                    let error = sprintf "Error getting metascript %s: %s" name ex.Message
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.ExecuteMetascriptAsync(name: string) =
            task {
                try
                    logger.LogInformation(sprintf "Executing metascript: %s" name)
                    
                    match registry.GetMetascript(name) with
                    | Some registered ->
                        let startTime = DateTime.UtcNow
                        
                        // Update usage statistics
                        registry.UpdateUsage(name) |> ignore
                        
                        // Simulate metascript execution
                        do! Task.Delay(500) // Simulate processing time
                        
                        let endTime = DateTime.UtcNow
                        let executionTime = endTime - startTime
                        
                        let result = {
                            Id = Guid.NewGuid().ToString()
                            MetascriptId = registered.Source.Id
                            Status = Completed
                            Output = sprintf "Metascript '%s' executed successfully\nContent: %d characters\nExecution completed at: %s" 
                                        name registered.Source.Content.Length (endTime.ToString("yyyy-MM-dd HH:mm:ss"))
                            Error = None
                            Variables = Map.empty
                            ExecutionTime = executionTime
                            StartTime = startTime
                            EndTime = Some endTime
                        }
                        
                        logger.LogInformation(sprintf "Metascript execution completed: %s (took %dms)" 
                            name (int executionTime.TotalMilliseconds))
                        
                        return Ok result
                    | None ->
                        let error = sprintf "Metascript not found: %s" name
                        logger.LogWarning(error)
                        return Error error
                with
                | ex ->
                    let error = sprintf "Error executing metascript %s: %s" name ex.Message
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.GetStatisticsAsync() =
            task {
                try
                    logger.LogDebug("Getting metascript statistics")
                    let stats = registry.GetStatistics()
                    return Ok stats
                with
                | ex ->
                    let error = sprintf "Error getting statistics: %s" ex.Message
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.ValidateMetascriptAsync(source: MetascriptSource) =
            task {
                try
                    logger.LogDebug(sprintf "Validating metascript: %s" source.Name)
                    let validatedSource = manager.ValidateMetascript(source)
                    return Ok validatedSource
                with
                | ex ->
                    let error = sprintf "Error validating metascript %s: %s" source.Name ex.Message
                    logger.LogError(ex, error)
                    return Error error
            }
