namespace TarsEngine.FSharp.Core.Working.Metascript

open System
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Represents a variable in a metascript.
/// </summary>
type MetascriptVariable = {
    Name: string
    Value: obj
    Type: Type
    IsReadOnly: bool
    Metadata: Metadata
}

/// <summary>
/// Represents the status of metascript execution.
/// </summary>
type MetascriptExecutionStatus =
    | Success
    | Failed
    | Cancelled

/// <summary>
/// Represents the result of metascript execution.
/// </summary>
type MetascriptExecutionResult = {
    Status: MetascriptExecutionStatus
    Output: string
    Error: string option
    Variables: Map<string, MetascriptVariable>
    ExecutionTime: TimeSpan
}

/// <summary>
/// Module for working with metascript variables.
/// </summary>
module MetascriptVariable =
    
    /// <summary>
    /// Creates a new metascript variable.
    /// </summary>
    let create name value =
        {
            Name = name
            Value = value
            Type = if isNull value then typeof<obj> else value.GetType()
            IsReadOnly = false
            Metadata = Map.empty
        }
