namespace TarsEngine.FSharp.Core.CodeAnalysis.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.CodeAnalysis.Services

/// <summary>
/// Extension methods for IServiceCollection to register code analysis services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.CodeAnalysis services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpCodeAnalysis (services: IServiceCollection) =
        // Register CodeAnalysisService
        services.AddSingleton<ICodeAnalysisService, CodeAnalysisService>() |> ignore
        
        // Return the service collection
        services
