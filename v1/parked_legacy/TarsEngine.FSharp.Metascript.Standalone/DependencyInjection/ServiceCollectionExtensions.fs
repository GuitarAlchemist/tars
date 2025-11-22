namespace TarsEngine.FSharp.Metascript.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Metascript.Services
open TarsEngine.FSharp.Metascript.BlockHandlers

/// <summary>
/// Extension methods for IServiceCollection to register metascript services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Metascript services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpMetascript (services: IServiceCollection) =
        // Register block handlers
        services.AddSingleton<IBlockHandler, FSharpBlockHandler>() |> ignore
        services.AddSingleton<IBlockHandler, CommandBlockHandler>() |> ignore
        services.AddSingleton<IBlockHandler, TextBlockHandler>() |> ignore
        
        // Register registry
        services.AddSingleton<BlockHandlerRegistry>() |> ignore
        
        // Register executor and service
        services.AddSingleton<IMetascriptExecutor, MetascriptExecutor>() |> ignore
        services.AddSingleton<IMetascriptService, MetascriptService>() |> ignore
        
        // Return the service collection
        services
