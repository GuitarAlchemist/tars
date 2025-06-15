namespace TarsEngine.FSharp.Core.DependencyInjection

open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.Api.TarsApiRegistry

/// Service collection extensions for TARS
module ServiceCollectionExtensions =
    
    type IServiceCollection with
        member this.AddTarsCore() =
            ConfigureServices(this)
            this
