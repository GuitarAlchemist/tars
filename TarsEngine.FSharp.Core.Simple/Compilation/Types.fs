namespace TarsEngine.FSharp.Core.Simple.Compilation

open System
open TarsEngine.FSharp.Core.Simple.Types

/// <summary>
/// Represents compilation options.
/// </summary>
type CompilationOptions = {
    OutputPath: string option
    Target: CompilationTarget
    Optimize: bool
    Debug: bool
    References: string list
}

/// <summary>
/// Represents compilation target.
/// </summary>
and CompilationTarget =
    | Library
    | Executable
    | Module

/// <summary>
/// Represents compilation result.
/// </summary>
type CompilationResult = {
    Success: bool
    OutputPath: string option
    Errors: string list
    Warnings: string list
    CompilationTime: TimeSpan
}

/// <summary>
/// Module for working with compilation options.
/// </summary>
module CompilationOptions =
    
    /// <summary>
    /// Creates default compilation options.
    /// </summary>
    let createDefault() = {
        OutputPath = None
        Target = Library
        Optimize = false
        Debug = true
        References = []
    }
    
    /// <summary>
    /// Sets the output path.
    /// </summary>
    let withOutputPath path options =
        { options with OutputPath = Some path }
    
    /// <summary>
    /// Sets the target.
    /// </summary>
    let withTarget target options =
        { options with Target = target }
    
    /// <summary>
    /// Enables optimization.
    /// </summary>
    let withOptimization enabled options =
        { options with Optimize = enabled }
    
    /// <summary>
    /// Enables debug information.
    /// </summary>
    let withDebug enabled options =
        { options with Debug = enabled }
    
    /// <summary>
    /// Adds references.
    /// </summary>
    let withReferences references options =
        { options with References = options.References @ references }
