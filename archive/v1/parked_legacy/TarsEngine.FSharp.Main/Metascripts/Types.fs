namespace TarsEngine.FSharp.Main.Metascripts

/// <summary>
/// Metascript types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Metascript type.
    /// </summary>
    type Metascript = {
        /// <summary>
        /// The name of the metascript.
        /// </summary>
        Name: string
        
        /// <summary>
        /// The code of the metascript.
        /// </summary>
        Code: string
        
        /// <summary>
        /// The language of the metascript.
        /// </summary>
        Language: string
    }
    
    /// <summary>
    /// Metascript execution result.
    /// </summary>
    type MetascriptExecutionResult = {
        /// <summary>
        /// Whether the execution was successful.
        /// </summary>
        Success: bool
        
        /// <summary>
        /// The output of the execution.
        /// </summary>
        Output: string
        
        /// <summary>
        /// The errors of the execution.
        /// </summary>
        Errors: string list
    }
