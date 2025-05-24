namespace TarsEngine.FSharp.Core.Simple

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
    
    /// <summary>
    /// Returns the value if Ok, otherwise throws an exception.
    /// </summary>
    let get result =
        match result with
        | Ok x -> x
        | Error e -> failwith (sprintf "Result was Error: %A" e)
    
    /// <summary>
    /// Converts an Option to a Result.
    /// </summary>
    let ofOption error option =
        match option with
        | Some x -> Ok x
        | None -> Error error
    
    /// <summary>
    /// Converts a Result to an Option.
    /// </summary>
    let toOption result =
        match result with
        | Ok x -> Some x
        | Error _ -> None
