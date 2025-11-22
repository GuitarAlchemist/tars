namespace TarsEngine.FSharp.Main.Intelligence

/// <summary>
/// Intelligence types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Intelligence type.
    /// </summary>
    type Intelligence = {
        /// <summary>
        /// The name of the intelligence.
        /// </summary>
        Name: string
        
        /// <summary>
        /// The description of the intelligence.
        /// </summary>
        Description: string
        
        /// <summary>
        /// The capabilities of the intelligence.
        /// </summary>
        Capabilities: string list
    }
    
    /// <summary>
    /// Intelligence execution result.
    /// </summary>
    type IntelligenceExecutionResult = {
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
