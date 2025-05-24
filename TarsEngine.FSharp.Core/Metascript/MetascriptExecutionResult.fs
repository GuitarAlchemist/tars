namespace TarsEngine.FSharp.Core.Metascript

/// <summary>
/// Represents the result of executing a metascript.
/// </summary>
type MetascriptExecutionResult = {
    /// <summary>
    /// Gets or sets a value indicating whether the execution was successful.
    /// </summary>
    Success: bool
    
    /// <summary>
    /// Gets or sets the output of the execution.
    /// </summary>
    Output: string option
    
    /// <summary>
    /// Gets or sets the error message if the execution failed.
    /// </summary>
    ErrorMessage: string option
}
