namespace TarsEngine.FSharp.Core.Metascript.Services

open System.Threading.Tasks
open TarsEngine.FSharp.Core.Metascript

/// <summary>
/// Interface for executing metascripts.
/// </summary>
type IMetascriptExecutor =
    /// <summary>
    /// Executes a metascript asynchronously.
    /// </summary>
    /// <param name="metascriptPath">The path to the metascript file.</param>
    /// <param name="parameters">Optional parameters to pass to the metascript.</param>
    /// <returns>The result of the metascript execution.</returns>
    abstract member ExecuteMetascriptAsync : metascriptPath:string * ?parameters:obj -> Task<MetascriptExecutionResult>
