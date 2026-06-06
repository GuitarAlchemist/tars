namespace TarsEngine.FSharp.ML.DependencyInjection

open System
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.ML.Core
open TarsEngine.FSharp.ML.Services

/// <summary>
/// Extension methods for IServiceCollection to register ML services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.ML services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <param name="configureOptions">Optional function to configure ML framework options.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpML (services: IServiceCollection) (configureOptions: Func<MLFrameworkOptions, MLFrameworkOptions> option) =
        // Configure options
        let options = 
            match configureOptions with
            | Some configure -> configure.Invoke(MLFrameworkOptionsDefaults.defaultOptions)
            | None -> MLFrameworkOptionsDefaults.defaultOptions
        
        // Register MLFramework
        services.AddSingleton<MLFramework>(fun sp -> 
            let logger = sp.GetRequiredService<ILogger<MLFramework>>()
            new MLFramework(logger, options)
        ) |> ignore
        
        // Register MLService
        services.AddSingleton<IMLService, MLService>() |> ignore
        
        // Return the service collection
        services
    
    /// <summary>
    /// Adds TarsEngine.FSharp.ML services to the service collection with default options.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpMLWithDefaultOptions (services: IServiceCollection) =
        addTarsEngineFSharpML services None
