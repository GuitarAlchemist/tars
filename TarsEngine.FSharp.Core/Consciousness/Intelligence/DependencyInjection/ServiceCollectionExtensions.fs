namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

/// <summary>
/// Extension methods for IServiceCollection to register Intelligence services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.Consciousness.Intelligence services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpIntelligence (services: IServiceCollection) =
        // Register creative thinking services
        services.AddSingleton<ICreativeThinking, CreativeThinking>() |> ignore
        
        // Register intuitive reasoning services
        services.AddSingleton<IIntuitiveReasoning, IntuitiveReasoning>() |> ignore
        
        // Register spontaneous thought services
        services.AddSingleton<ISpontaneousThought, SpontaneousThought>() |> ignore
        
        // Register curiosity drive services
        services.AddSingleton<ICuriosityDrive, CuriosityDrive>() |> ignore
        
        // Return the service collection
        services
