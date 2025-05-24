namespace TarsEngine.FSharp.Main.Services

open System.Threading.Tasks

/// <summary>
/// Service for executing metascripts
/// </summary>
type IMetascriptService =
    /// <summary>
    /// Executes a metascript asynchronously.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the execution result.</returns>
    abstract member ExecuteMetascriptAsync : metascript:string -> Task<obj>
