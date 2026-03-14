namespace TarsEngine.FSharp.ML.Core

/// <summary>
/// Utility functions for working with Option types.
/// </summary>
module Option =
    /// <summary>
    /// Maps a function over an Option value.
    /// </summary>
    /// <param name="f">The function to apply to the Option value.</param>
    /// <param name="option">The Option to map over.</param>
    /// <returns>A new Option with the function applied to the value.</returns>
    let map f option =
        match option with
        | Some x -> Some (f x)
        | None -> None
    
    /// <summary>
    /// Binds a function over an Option value.
    /// </summary>
    /// <param name="f">The function to apply to the Option value.</param>
    /// <param name="option">The Option to bind over.</param>
    /// <returns>The Option result of applying the function to the value.</returns>
    let bind f option =
        match option with
        | Some x -> f x
        | None -> None
    
    /// <summary>
    /// Returns the Option value or the given default value if None.
    /// </summary>
    /// <param name="defaultValue">The default value to use if None.</param>
    /// <param name="option">The Option to get the value from.</param>
    /// <returns>The Option value or the default value.</returns>
    let defaultValue defaultValue option =
        match option with
        | Some x -> x
        | None -> defaultValue
    
    /// <summary>
    /// Returns the Option value or calls the given function if None.
    /// </summary>
    /// <param name="f">The function to call if None.</param>
    /// <param name="option">The Option to get the value from.</param>
    /// <returns>The Option value or the result of calling the function.</returns>
    let defaultWith f option =
        match option with
        | Some x -> x
        | None -> f ()
    
    /// <summary>
    /// Returns true if the Option is Some.
    /// </summary>
    /// <param name="option">The Option to check.</param>
    /// <returns>True if the Option is Some, false otherwise.</returns>
    let isSome option =
        match option with
        | Some _ -> true
        | None -> false
    
    /// <summary>
    /// Returns true if the Option is None.
    /// </summary>
    /// <param name="option">The Option to check.</param>
    /// <returns>True if the Option is None, false otherwise.</returns>
    let isNone option =
        match option with
        | Some _ -> false
        | None -> true
