namespace TarsEngine.FSharp.Core.Consciousness.DependencyInjection

open System
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Services

/// <summary>
/// Extension methods for IServiceCollection to register TarsEngine.FSharp.Core.Consciousness services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.Consciousness services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpConsciousness (services: IServiceCollection) =
        // Register core services
        services.AddSingleton<ConsciousnessCore>() |> ignore
        services.AddSingleton<PureConsciousnessCore>() |> ignore
        
        // Register consciousness services
        services.AddSingleton<IConsciousnessService, ConsciousnessService>() |> ignore
        
        // Return the service collection
        services
    
    /// <summary>
    /// Extension method for IServiceCollection to add TarsEngine.FSharp.Core.Consciousness services.
    /// </summary>
    type IServiceCollection with
        /// <summary>
        /// Adds TarsEngine.FSharp.Core.Consciousness services to the service collection.
        /// </summary>
        /// <returns>The service collection.</returns>
        member this.AddTarsEngineFSharpConsciousness() =
            addTarsEngineFSharpConsciousness this
