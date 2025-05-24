namespace TarsEngine.FSharp.Main.Metascripts

open System.Threading.Tasks

/// <summary>
/// Interface for executing metascripts.
/// </summary>
type IMetascriptExecutor =
    /// <summary>
    /// Executes a metascript asynchronously.
    /// </summary>
    /// <param name="metascriptPath">The path to the metascript file.</param>
    /// <param name="parameters">Optional parameters to pass to the metascript.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the execution result.</returns>
    abstract member ExecuteMetascriptAsync : metascriptPath:string * ?parameters:obj -> Task<MetascriptExecutionResult>
