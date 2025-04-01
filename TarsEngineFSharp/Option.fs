namespace TarsEngineFSharp

/// <summary>
/// Module containing helper functions for Option
/// </summary>
module Option =
    /// <summary>
    /// Create a Some option
    /// </summary>
    /// <param name="value">The value</param>
    /// <returns>A Some option</returns>
    let some value = Some value

    /// <summary>
    /// Create a None option
    /// </summary>
    /// <returns>A None option</returns>
    let none = None

    /// <summary>
    /// Apply a function to the value of an option
    /// </summary>
    /// <param name="f">The function to apply</param>
    /// <param name="option">The option</param>
    /// <returns>A new option with the function applied to the value</returns>
    let map f option =
        match option with
        | Some value -> Some (f value)
        | None -> None

    /// <summary>
    /// Apply a function that returns an option to the value of an option
    /// </summary>
    /// <param name="f">The function to apply</param>
    /// <param name="option">The option</param>
    /// <returns>A new option with the function applied to the value</returns>
    let bind f option =
        match option with
        | Some value -> f value
        | None -> None

    /// <summary>
    /// Get the value of an option or a default value if the option is None
    /// </summary>
    /// <param name="defaultValue">The default value</param>
    /// <param name="option">The option</param>
    /// <returns>The value of the option or the default value</returns>
    let defaultValue defaultValue option =
        match option with
        | Some value -> value
        | None -> defaultValue

    /// <summary>
    /// Get the value of an option or apply a function to get a default value if the option is None
    /// </summary>
    /// <param name="defaultFn">The function to get a default value</param>
    /// <param name="option">The option</param>
    /// <returns>The value of the option or the result of the default function</returns>
    let defaultWith defaultFn option =
        match option with
        | Some value -> value
        | None -> defaultFn()

    /// <summary>
    /// Apply a function to the value of an option if it is Some
    /// </summary>
    /// <param name="f">The function to apply</param>
    /// <param name="option">The option</param>
    /// <returns>The option</returns>
    let iter f option =
        match option with
        | Some value -> f value
        | None -> ()
        option

    /// <summary>
    /// Apply a function to both the Some and None cases of an option
    /// </summary>
    /// <param name="onSome">The function to apply to the Some case</param>
    /// <param name="onNone">The function to apply to the None case</param>
    /// <param name="option">The option</param>
    /// <returns>The result of applying the appropriate function</returns>
    let either onSome onNone option =
        match option with
        | Some value -> onSome value
        | None -> onNone()

    /// <summary>
    /// Try to execute a function and return an option
    /// </summary>
    /// <param name="f">The function to execute</param>
    /// <returns>A Some option with the function result or None if an exception is thrown</returns>
    let tryWith f =
        try
            Some (f())
        with
        | _ -> None
