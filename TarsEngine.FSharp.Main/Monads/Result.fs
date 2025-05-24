namespace TarsEngine.FSharp.Main.Monads

open System

/// <summary>
/// Module containing functions for working with Result types
/// </summary>
module Result =
    /// <summary>
    /// Creates a successful result with the given value
    /// </summary>
    let success value = Ok value

    /// <summary>
    /// Creates a failed result with the given error
    /// </summary>
    let error err = Error err

    /// <summary>
    /// Returns true if the result is successful
    /// </summary>
    let isOk result =
        match result with
        | Ok _ -> true
        | Error _ -> false

    /// <summary>
    /// Returns true if the result is a failure
    /// </summary>
    let isError result =
        match result with
        | Ok _ -> false
        | Error _ -> true

    /// <summary>
    /// Gets the value if the result is successful, or throws an exception if it's a failure
    /// </summary>
    let getValue result =
        match result with
        | Ok value -> value
        | Error err -> raise (InvalidOperationException($"Cannot get value from Error: {err}"))

    /// <summary>
    /// Gets the error if the result is a failure, or throws an exception if it's successful
    /// </summary>
    let getError result =
        match result with
        | Ok value -> raise (InvalidOperationException($"Cannot get error from Ok: {value}"))
        | Error err -> err

    /// <summary>
    /// Gets the value if the result is successful, or returns the default value if it's a failure
    /// </summary>
    let valueOrDefault defaultValue result =
        match result with
        | Ok value -> value
        | Error _ -> defaultValue

    /// <summary>
    /// Gets the value if the result is successful, or returns the result of the specified function if it's a failure
    /// </summary>
    let valueOr defaultValueProvider result =
        match result with
        | Ok value -> value
        | Error err -> defaultValueProvider err

    /// <summary>
    /// Applies a function to the value if the result is successful, or returns the error if it's a failure
    /// </summary>
    let map mapper result =
        match result with
        | Ok value -> Ok (mapper value)
        | Error err -> Error err

    /// <summary>
    /// Applies a function to the error if the result is a failure, or returns the value if it's successful
    /// </summary>
    let mapError mapper result =
        match result with
        | Ok value -> Ok value
        | Error err -> Error (mapper err)

    /// <summary>
    /// Applies a function that returns a Result to the value if the result is successful, or returns the error if it's a failure
    /// </summary>
    let bind binder result =
        match result with
        | Ok value -> binder value
        | Error err -> Error err

    /// <summary>
    /// Applies one of two functions depending on whether the result is successful or a failure
    /// </summary>
    let match' okFunc errorFunc result =
        match result with
        | Ok value -> okFunc value
        | Error err -> errorFunc err

    /// <summary>
    /// Performs an action if the result is successful
    /// </summary>
    let ifOk action result =
        match result with
        | Ok value -> action value
        | Error _ -> ()
        result

    /// <summary>
    /// Performs an action if the result is a failure
    /// </summary>
    let ifError action result =
        match result with
        | Ok _ -> ()
        | Error err -> action err
        result

    /// <summary>
    /// Tries to execute a function and returns a successful result with the return value if it succeeds, or a failed result with the exception if it throws
    /// </summary>
    let tryFunc (func: unit -> 'T) : Result<'T, exn> =
        try
            Ok (func())
        with
        | ex -> Error ex

    /// <summary>
    /// Converts an Option to a Result, using the provided error value for None
    /// </summary>
    let ofOption errorValue option =
        match option with
        | Some value -> Ok value
        | None -> Error errorValue

    /// <summary>
    /// Converts a Result to an Option, discarding the error
    /// </summary>
    let toOption result =
        match result with
        | Ok value -> Some value
        | Error _ -> None
