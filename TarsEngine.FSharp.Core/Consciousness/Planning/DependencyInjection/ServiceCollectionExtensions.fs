namespace TarsEngine.FSharp.Core.Consciousness.Planning.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Consciousness.Planning.Services

/// <summary>
/// Extension methods for IServiceCollection to register Planning services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.Consciousness.Planning services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpPlanning (services: IServiceCollection) =
        // Register planning services
        services.AddSingleton<IPlanningService, ExecutionPlanner>() |> ignore
        
        // Return the service collection
        services
