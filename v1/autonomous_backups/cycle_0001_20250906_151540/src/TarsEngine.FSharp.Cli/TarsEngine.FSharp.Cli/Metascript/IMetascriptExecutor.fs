namespace TarsEngine.FSharp.Cli.Metascript

open System.Threading.Tasks
open System.Collections.Generic

/// <summary>
/// Variable in a metascript execution result.
/// </summary>
type MetascriptVariable = {
    /// <summary>
    /// The name of the variable.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the variable.
    /// </summary>
    Value: obj
    
    /// <summary>
    /// The type of the variable.
    /// </summary>
    Type: System.Type
    
    /// <summary>
    /// Whether the variable is read-only.
    /// </summary>
    IsReadOnly: bool
    
    /// <summary>
    /// Metadata for the variable.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Result of a metascript execution.
/// </summary>
type MetascriptExecutionResult = {
    /// <summary>
    /// Whether the execution was successful.
    /// </summary>
    Success: bool
    
    /// <summary>
    /// The error message, if any.
    /// </summary>
    ErrorMessage: string
    
    /// <summary>
    /// The output of the execution.
    /// </summary>
    Output: string
    
    /// <summary>
    /// The variables from the execution.
    /// </summary>
    Variables: Map<string, MetascriptVariable>
}

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
    abstract member ExecuteMetascriptAsync : metascriptPath: string * parameters: obj -> Task<MetascriptExecutionResult>
