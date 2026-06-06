namespace TarsEngine.FSharp.Core.Api

open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Core.Metascript.Services

/// TARS API Registry for dependency injection
module TarsApiRegistry =
    
    let mutable private serviceProvider: System.IServiceProvider option = None
    
    /// Initialize the service provider
    let Initialize(provider: System.IServiceProvider) =
        serviceProvider <- Some provider
    
    /// Get API service
    let GetApi<'T>() =
        match serviceProvider with
        | Some provider -> provider.GetService<'T>()
        | None -> failwith "Service provider not initialized"
    
    /// Get TARS API
    let GetTarsApi() = GetApi<ITarsApi>()
    
    /// Configure services
    let ConfigureServices(services: IServiceCollection) =
        services
            .AddSingleton<ITarsApi, TarsApiService>()
            .AddSingleton<MetascriptExecutor>()
            .AddLogging(fun builder -> 
                builder.AddConsole() |> ignore
                builder.SetMinimumLevel(LogLevel.Information) |> ignore
            )
        |> ignore
