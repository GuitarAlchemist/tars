namespace TarsEngine.FSharp.Core.Metascript.Services

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript

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
        task {
            try
                logger.LogInformation("Executing metascript: {MetascriptPath}", metascriptPath)
                
                // Check if the file exists
                if not (File.Exists(metascriptPath)) then
                    logger.LogError("Metascript file not found: {MetascriptPath}", metascriptPath)
                    return {
                        Success = false
                        Output = None
                        ErrorMessage = Some $"Metascript file not found: {metascriptPath}"
                    }
                
                // Read the metascript content
                let! metascriptContent = File.ReadAllTextAsync(metascriptPath)
                
                // Convert parameters to JSON if provided
                let parametersJson = 
                    match parameters with
                    | Some p -> JsonSerializer.Serialize(p)
                    | None -> String.Empty
                
                // Execute the metascript
                let! result = metascriptService.ExecuteMetascriptAsync(metascriptContent)
                
                // Convert the result to a MetascriptExecutionResult
                return {
                    Success = true
                    Output = Some (result.ToString())
                    ErrorMessage = None
                }
            with
            | ex ->
                logger.LogError(ex, "Error executing metascript: {MetascriptPath}", metascriptPath)
                return {
                    Success = false
                    Output = None
                    ErrorMessage = Some ex.Message
                }
        }
    
    interface IMetascriptExecutor with
        member this.ExecuteMetascriptAsync(metascriptPath, ?parameters) = 
            this.ExecuteMetascriptAsync(metascriptPath, ?parameters = parameters)
