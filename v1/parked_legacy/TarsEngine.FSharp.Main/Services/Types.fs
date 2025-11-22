namespace TarsEngine.FSharp.Main.Services

/// <summary>
/// Service types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Service type.
    /// </summary>
    type Service = {
        /// <summary>
        /// The name of the service.
        /// </summary>
        Name: string
        
        /// <summary>
        /// The description of the service.
        /// </summary>
        Description: string
        
        /// <summary>
        /// The capabilities of the service.
        /// </summary>
        Capabilities: string list
    }
    
    /// <summary>
    /// Service execution result.
    /// </summary>
    type ServiceExecutionResult = {
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
