namespace TarsEngine.FSharp.Main.SelfImprovement

/// <summary>
/// Self-improvement types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Self-improvement type.
    /// </summary>
    type SelfImprovement = {
        /// <summary>
        /// The name of the self-improvement.
        /// </summary>
        Name: string
        
        /// <summary>
        /// The description of the self-improvement.
        /// </summary>
        Description: string
        
        /// <summary>
        /// The capabilities of the self-improvement.
        /// </summary>
        Capabilities: string list
    }
    
    /// <summary>
    /// Self-improvement execution result.
    /// </summary>
    type SelfImprovementExecutionResult = {
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
