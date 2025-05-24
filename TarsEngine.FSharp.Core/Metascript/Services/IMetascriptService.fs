namespace TarsEngine.FSharp.Core.Metascript.Services

open System.Threading.Tasks

/// <summary>
/// Interface for the metascript service
/// </summary>
type IMetascriptService =
    /// <summary>
    /// Executes a metascript
    /// </summary>
    /// <param name="metascript">Metascript to execute</param>
    /// <returns>Result of the metascript execution</returns>
    abstract member ExecuteMetascriptAsync : metascript:string -> Task<obj>
