namespace TarsEngine.FSharp.ML.Core

/// <summary>
/// Utility functions for working with Result types.
/// </summary>
module Result =
    /// <summary>
    /// Maps a function over the Ok value of a Result.
    /// </summary>
    /// <param name="f">The function to apply to the Ok value.</param>
    /// <param name="result">The Result to map over.</param>
    /// <returns>A new Result with the function applied to the Ok value.</returns>
    let map f result =
        match result with
        | Ok x -> Ok (f x)
        | Error e -> Error e
    
    /// <summary>
    /// Binds a function over the Ok value of a Result.
    /// </summary>
    /// <param name="f">The function to apply to the Ok value.</param>
    /// <param name="result">The Result to bind over.</param>
    /// <returns>The Result of applying the function to the Ok value.</returns>
    let bind f result =
        match result with
        | Ok x -> f x
        | Error e -> Error e
    
    /// <summary>
    /// Applies a function to the Error value of a Result.
    /// </summary>
    /// <param name="f">The function to apply to the Error value.</param>
    /// <param name="result">The Result to map over.</param>
    /// <returns>A new Result with the function applied to the Error value.</returns>
    let mapError f result =
        match result with
        | Ok x -> Ok x
        | Error e -> Error (f e)
    
    /// <summary>
    /// Converts an Option to a Result with the given error if None.
    /// </summary>
    /// <param name="error">The error to use if the Option is None.</param>
    /// <param name="option">The Option to convert.</param>
    /// <returns>A Result with the Option value or the given error.</returns>
    let ofOption error option =
        match option with
        | Some x -> Ok x
        | None -> Error error
    
    /// <summary>
    /// Converts a Result to an Option, discarding the error.
    /// </summary>
    /// <param name="result">The Result to convert.</param>
    /// <returns>An Option with the Ok value or None.</returns>
    let toOption result =
        match result with
        | Ok x -> Some x
        | Error _ -> None
    
    /// <summary>
    /// Returns the Ok value or the given default value if Error.
    /// </summary>
    /// <param name="defaultValue">The default value to use if Error.</param>
    /// <param name="result">The Result to get the value from.</param>
    /// <returns>The Ok value or the default value.</returns>
    let defaultValue defaultValue result =
        match result with
        | Ok x -> x
        | Error _ -> defaultValue
    
    /// <summary>
    /// Returns the Ok value or calls the given function with the Error value.
    /// </summary>
    /// <param name="f">The function to call with the Error value.</param>
    /// <param name="result">The Result to get the value from.</param>
    /// <returns>The Ok value or the result of calling the function with the Error value.</returns>
    let defaultWith f result =
        match result with
        | Ok x -> x
        | Error e -> f e
