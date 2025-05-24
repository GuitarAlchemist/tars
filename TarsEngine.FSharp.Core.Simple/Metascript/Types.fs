namespace TarsEngine.FSharp.Core.Simple.Metascript

open System
open System.Collections.Generic
open TarsEngine.FSharp.Core.Simple.Types

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
/// Represents a metascript context.
/// </summary>
type MetascriptContext = {
    WorkingDirectory: string
    Variables: Map<string, MetascriptVariable>
    Metadata: Metadata
}

/// <summary>
/// Represents a metascript block.
/// </summary>
type MetascriptBlock =
    | ConfigBlock of Map<string, obj>
    | FSharpBlock of string
    | TextBlock of string
    | CommandBlock of string

/// <summary>
/// Represents a parsed metascript.
/// </summary>
type Metascript = {
    Id: Id
    Name: string
    FilePath: string option
    Blocks: MetascriptBlock list
    Metadata: Metadata
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
    
    /// <summary>
    /// Creates a read-only metascript variable.
    /// </summary>
    let createReadOnly name value =
        { create name value with IsReadOnly = true }

/// <summary>
/// Module for working with metascript contexts.
/// </summary>
module MetascriptContext =
    
    /// <summary>
    /// Creates a new metascript context.
    /// </summary>
    let create workingDirectory =
        {
            WorkingDirectory = workingDirectory
            Variables = Map.empty
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Adds a variable to the context.
    /// </summary>
    let addVariable name value context =
        let variable = MetascriptVariable.create name value
        { context with Variables = context.Variables.Add(name, variable) }
    
    /// <summary>
    /// Gets a variable from the context.
    /// </summary>
    let getVariable name context =
        context.Variables.TryFind(name)
