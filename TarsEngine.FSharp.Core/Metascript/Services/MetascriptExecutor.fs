namespace TarsEngine.FSharp.Core.Metascript.Services

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Core.Metascript.Services

/// <summary>
/// Implementation of the IMetascriptExecutor interface.
/// </summary>
type MetascriptExecutor(logger: ILogger<MetascriptExecutor>, metascriptService: IMetascriptService) =
    /// <summary>
    /// Executes a metascript asynchronously.
    /// </summary>
    /// <param name="metascriptPath">The path to the metascript file.</param>
    /// <param name="parameters">Optional parameters to pass to the metascript.</param>
    /// <returns>The result of the metascript execution.</returns>
    member _.ExecuteMetascriptAsync(metascriptPath: string, ?parameters: obj) =
        Task.Run(fun () ->
            try
                logger.LogInformation("Executing metascript: {MetascriptPath}", metascriptPath)

                // Check if the file exists
                if not (File.Exists(metascriptPath)) then
                    logger.LogError("Metascript file not found: {MetascriptPath}", metascriptPath)
                    {
                        Success = false
                        Output = None
                        ErrorMessage = Some $"Metascript file not found: {metascriptPath}"
                    }
                else
                    // Read the metascript content
                    let metascriptContent = File.ReadAllText(metascriptPath)

                    // REAL METASCRIPT EXECUTION - NO SIMULATION
                    logger.LogInformation("Executing metascript file with REAL F# execution: {MetascriptPath}", metascriptPath)

                    // Use the MetascriptService for real execution
                    let metascriptService = MetascriptService(logger :?> ILogger<MetascriptService>)
                    let executionResult = (metascriptService.ExecuteMetascriptAsync(metascriptContent)).Result

                    // Return real execution result
                    {
                        Success = true
                        Output = Some (executionResult.ToString())
                        ErrorMessage = None
                    }
            with
            | ex ->
                logger.LogError(ex, "Error executing metascript: {MetascriptPath}", metascriptPath)
                {
                    Success = false
                    Output = None
                    ErrorMessage = Some ex.Message
                }
        )
    
    interface IMetascriptExecutor with
        member this.ExecuteMetascriptAsync(metascriptPath, ?parameters) =
            this.ExecuteMetascriptAsync(metascriptPath, ?parameters = parameters)
