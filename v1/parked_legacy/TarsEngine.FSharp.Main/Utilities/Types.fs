namespace TarsEngine.FSharp.Main.Utilities

/// <summary>
/// Utility types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Utility type.
    /// </summary>
    type Utility = {
        /// <summary>
        /// The name of the utility.
        /// </summary>
        Name: string
        
        /// <summary>
        /// The description of the utility.
        /// </summary>
        Description: string
        
        /// <summary>
        /// The capabilities of the utility.
        /// </summary>
        Capabilities: string list
    }
    
    /// <summary>
    /// Utility execution result.
    /// </summary>
    type UtilityExecutionResult = {
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
