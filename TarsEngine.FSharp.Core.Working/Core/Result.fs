namespace TarsEngine.FSharp.Core.Working

/// <summary>
/// Result type extensions and utilities.
/// </summary>
module Result =
    
    /// <summary>
    /// Maps a Result<'T, 'TError> to a Result<'U, 'TError> by applying a function to the value.
    /// </summary>
    let map f result =
        match result with
        | Ok x -> Ok (f x)
        | Error e -> Error e
    
    /// <summary>
    /// Maps a Result<'T, 'TError> to a Result<'T, 'UError> by applying a function to the error.
    /// </summary>
    let mapError f result =
        match result with
        | Ok x -> Ok x
        | Error e -> Error (f e)
    
    /// <summary>
    /// Binds a Result<'T, 'TError> to a function that returns Result<'U, 'TError>.
    /// </summary>
    let bind f result =
        match result with
        | Ok x -> f x
        | Error e -> Error e
    
    /// <summary>
    /// Returns the value if Ok, otherwise returns the default value.
    /// </summary>
    let defaultValue defaultVal result =
        match result with
        | Ok x -> x
        | Error _ -> defaultVal
