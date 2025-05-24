namespace TarsEngine.FSharp.Main.Core

/// <summary>
/// Core types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Result type for operations that can fail.
    /// </summary>
    type Result<'T, 'E> =
        | Ok of 'T
        | Error of 'E
    
    /// <summary>
    /// Option type for values that may not exist.
    /// </summary>
    type Option<'T> =
        | Some of 'T
        | None
