namespace TarsEngine.FSharp.Metascripts.Core

open System
open System.Collections.Generic
open TarsEngine.FSharp.Metascripts.Core

/// <summary>
/// Registry for managing metascripts.
/// </summary>
type MetascriptRegistry() =
    let mutable metascripts = Map.empty<string, RegisteredMetascript>
    
    /// <summary>
    /// Registers a metascript.
    /// </summary>
    member _.RegisterMetascript(source: MetascriptSource) =
        let registered = {
            Source = source
            RegistrationTime = DateTime.UtcNow
            UsageCount = 0
            LastUsed = None
            IsActive = true
        }
        metascripts <- Map.add source.Name registered metascripts
        registered
    
    /// <summary>
    /// Gets a registered metascript by name.
    /// </summary>
    member _.GetMetascript(name: string) =
        Map.tryFind name metascripts
    
    /// <summary>
    /// Gets all registered metascripts.
    /// </summary>
    member _.GetAllMetascripts() =
        metascripts |> Map.toList |> List.map snd
    
    /// <summary>
    /// Gets metascript statistics.
    /// </summary>
    member _.GetStatistics() =
        let allMetascripts = metascripts |> Map.toList |> List.map snd
        let totalExecutions = allMetascripts |> List.sumBy (fun m -> m.UsageCount)
        let successfulExecutions = totalExecutions // Simplified for now
        let averageTime = TimeSpan.FromSeconds(1.0) // Simplified
        
        {
            TotalMetascripts = allMetascripts.Length
            ExecutedToday = 0 // Simplified
            SuccessRate = if totalExecutions > 0 then float successfulExecutions / float totalExecutions else 0.0
            AverageExecutionTime = averageTime
            MostUsedCategory = Custom // Simplified
            LastExecuted = 
                allMetascripts 
                |> List.choose (fun m -> m.LastUsed)
                |> List.sortDescending
                |> List.tryHead
        }
    
    /// <summary>
    /// Updates usage statistics for a metascript.
    /// </summary>
    member _.UpdateUsage(name: string) =
        match Map.tryFind name metascripts with
        | Some registered ->
            let updated = { registered with 
                              UsageCount = registered.UsageCount + 1
                              LastUsed = Some DateTime.UtcNow }
            metascripts <- Map.add name updated metascripts
            Some updated
        | None -> None
    
    /// <summary>
    /// Removes a metascript from the registry.
    /// </summary>
    member _.UnregisterMetascript(name: string) =
        metascripts <- Map.remove name metascripts
    
    /// <summary>
    /// Clears all registered metascripts.
    /// </summary>
    member _.Clear() =
        metascripts <- Map.empty
