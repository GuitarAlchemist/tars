namespace TarsEngine.FSharp.Metascript

open System
open System.Collections.Generic

/// <summary>
/// Represents the status of a metascript execution.
/// </summary>
type MetascriptExecutionStatus =
    | Success
    | Failure
    | Partial
    | Cancelled
    | Timeout
    | NotExecuted

/// <summary>
/// Represents the result of executing a metascript block.
/// </summary>
type MetascriptBlockExecutionResult = {
    /// <summary>
    /// The block that was executed.
    /// </summary>
    Block: MetascriptBlock
    
    /// <summary>
    /// The output of the block execution.
    /// </summary>
    Output: string
    
    /// <summary>
    /// The error message, if any.
    /// </summary>
    Error: string option
    
    /// <summary>
    /// The status of the block execution.
    /// </summary>
    Status: MetascriptExecutionStatus
    
    /// <summary>
    /// The execution time in milliseconds.
    /// </summary>
    ExecutionTimeMs: float
    
    /// <summary>
    /// The return value, if any.
    /// </summary>
    ReturnValue: obj option
    
    /// <summary>
    /// The variables created or modified by the block.
    /// </summary>
    Variables: Map<string, MetascriptVariable>
    
    /// <summary>
    /// Additional metadata for the block execution result.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Represents the result of executing a metascript.
/// </summary>
type MetascriptExecutionResult = {
    /// <summary>
    /// The metascript that was executed.
    /// </summary>
    Metascript: Metascript
    
    /// <summary>
    /// The block execution results.
    /// </summary>
    BlockResults: MetascriptBlockExecutionResult list
    
    /// <summary>
    /// The overall status of the metascript execution.
    /// </summary>
    Status: MetascriptExecutionStatus
    
    /// <summary>
    /// The overall output of the metascript execution.
    /// </summary>
    Output: string
    
    /// <summary>
    /// The overall error message, if any.
    /// </summary>
    Error: string option
    
    /// <summary>
    /// The execution time in milliseconds.
    /// </summary>
    ExecutionTimeMs: float
    
    /// <summary>
    /// The return value, if any.
    /// </summary>
    ReturnValue: obj option
    
    /// <summary>
    /// The variables created or modified by the metascript.
    /// </summary>
    Variables: Map<string, MetascriptVariable>
    
    /// <summary>
    /// The execution context after execution.
    /// </summary>
    Context: MetascriptContext option
    
    /// <summary>
    /// Additional metadata for the metascript execution result.
    /// </summary>
    Metadata: Map<string, string>
}
