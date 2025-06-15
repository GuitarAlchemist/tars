namespace TarsEngine.FSharp.Main.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Service for executing metascripts
/// </summary>
type MetascriptService(logger: ILogger<MetascriptService>) =
    /// <summary>
    /// Initializes a new instance of the MetascriptService class
    /// </summary>
    /// <param name="logger">The logger.</param>
    new(logger) = MetascriptService(logger)
    
    interface IMetascriptService with
        /// <inheritdoc/>
        member this.ExecuteMetascriptAsync(metascript) =
            task {
                try
                    logger.LogInformation("Executing metascript with REAL execution")

                    // REAL METASCRIPT EXECUTION - NO MORE FAKE CODE!
                    // Use the real MetascriptService from TarsEngine.FSharp.Metascripts
                    let realService = TarsEngine.FSharp.Metascripts.Services.MetascriptService(logger)
                    let! result = realService.ExecuteMetascriptAsync(metascript)

                    return result.Output :> obj
                with
                | ex ->
                    logger.LogError(ex, "Error executing metascript")
                    raise ex
            }
