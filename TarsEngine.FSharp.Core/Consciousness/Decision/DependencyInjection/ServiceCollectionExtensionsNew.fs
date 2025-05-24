namespace TarsEngine.FSharp.Core.Consciousness.Decision.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Consciousness.Decision.Services

/// <summary>
/// Extension methods for IServiceCollection to register Decision services.
/// </summary>
module ServiceCollectionExtensionsNew =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.Consciousness.Decision services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpDecision (services: IServiceCollection) =
        // Register decision services
        services.AddSingleton<IDecisionService, DecisionServiceComplete>() |> ignore
        
        // Return the service collection
        services
