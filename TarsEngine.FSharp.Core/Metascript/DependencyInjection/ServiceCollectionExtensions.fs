namespace TarsEngine.FSharp.Core.Metascript.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Metascript.Services

/// <summary>
/// Extension methods for IServiceCollection to register Metascript services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.Metascript services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpMetascript (services: IServiceCollection) =
        // Register metascript services
        services.AddSingleton<IMetascriptService, MetascriptService>() |> ignore
        services.AddSingleton<IMetascriptExecutor, MetascriptExecutor>() |> ignore
        
        // Return the service collection
        services
