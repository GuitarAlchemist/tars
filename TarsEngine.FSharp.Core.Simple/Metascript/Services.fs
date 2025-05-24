namespace TarsEngine.FSharp.Core.Simple.Metascript.Services

open System.Threading.Tasks
open TarsEngine.FSharp.Core.Simple.Metascript

/// <summary>
/// Interface for metascript execution services.
/// </summary>
type IMetascriptExecutor =
    /// <summary>
    /// Executes a metascript.
    /// </summary>
    abstract member ExecuteAsync: metascript: Metascript * context: MetascriptContext -> Task<MetascriptExecutionResult>
    
    /// <summary>
    /// Creates a new execution context.
    /// </summary>
    abstract member CreateContextAsync: workingDirectory: string * ?variables: Map<string, MetascriptVariable> -> Task<MetascriptContext>
