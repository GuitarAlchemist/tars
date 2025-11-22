namespace TarsEngine.FSharp.Main.Monads

open System

/// <summary>
/// Module containing functions for working with Option types
/// </summary>
module Option =
    /// <summary>
    /// Gets the value if present, or returns the default value if not
    /// </summary>
    let valueOrDefault defaultValue option =
        match option with
        | Some value -> value
        | None -> defaultValue

    /// <summary>
    /// Gets the value if present, or returns the result of the specified function if not
    /// </summary>
    let valueOr defaultValueProvider option =
        match option with
        | Some value -> value
        | None -> defaultValueProvider()

    /// <summary>
    /// Applies a function to the value if present, or returns None if not
    /// </summary>
    let map mapper option =
        match option with
        | Some value -> Some (mapper value)
        | None -> None

    /// <summary>
    /// Applies a function that returns an Option to the value if present, or returns None if not
    /// </summary>
    let bind binder option =
        match option with
        | Some value -> binder value
        | None -> None

    /// <summary>
    /// Applies one of two functions depending on whether the option has a value
    /// </summary>
    let match' some none option =
        match option with
        | Some value -> some value
        | None -> none()

    /// <summary>
    /// Performs an action if the option has a value
    /// </summary>
    let ifSome action option =
        match option with
        | Some value -> action value
        | None -> ()
        option

    /// <summary>
    /// Performs an action if the option has no value
    /// </summary>
    let ifNone action option =
        match option with
        | Some _ -> ()
        | None -> action()
        option

    /// <summary>
    /// Converts a nullable reference to an Option
    /// </summary>
    let ofObj (value: 'T) =
        if isNull (box value) then None else Some value

    /// <summary>
    /// Converts a nullable value to an Option
    /// </summary>
    let ofNullable (value: Nullable<'T>) =
        if value.HasValue then Some value.Value else None
