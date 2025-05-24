namespace TarsEngine.FSharp.Core.DependencyInjection

open System
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Compilation
open TarsEngine.Services.Compilation

/// <summary>
/// Extension methods for IServiceCollection to register TarsEngine.FSharp.Core services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpCore (services: IServiceCollection) =
        // Register the F# compiler
        services.AddSingleton<FSharpCompiler>() |> ignore
        
        // Register the F# compiler adapter
        services.AddSingleton<IFSharpCompiler, FSharpCompilerAdapter>() |> ignore
        
        // Return the service collection
        services
    
    /// <summary>
    /// Extension method for IServiceCollection to add TarsEngine.FSharp.Core services.
    /// </summary>
    type IServiceCollection with
        /// <summary>
        /// Adds TarsEngine.FSharp.Core services to the service collection.
        /// </summary>
        /// <returns>The service collection.</returns>
        member this.AddTarsEngineFSharpCore() =
            addTarsEngineFSharpCore this
