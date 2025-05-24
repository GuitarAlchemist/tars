namespace TarsEngine.FSharp.Core

/// Module containing Result type extensions and utilities
module Result =
    /// Maps a Result<'T, 'TError> to a Result<'U, 'TError> by applying a function to the value
    let map f result =
        match result with
        | Ok x -> Ok (f x)
        | Error e -> Error e

    /// Maps a Result<'T, 'TError> to a Result<'T, 'UError> by applying a function to the error
    let mapError f result =
        match result with
        | Ok x -> Ok x
        | Error e -> Error (f e)

    /// Binds a Result<'T, 'TError> to a Result<'U, 'TError> by applying a function that returns a Result
    let bind f result =
        match result with
        | Ok x -> f x
        | Error e -> Error e

    /// Applies a function to a Result<'T, 'TError> and returns a new Result<'U, 'TError>
    let apply fResult xResult =
        match fResult, xResult with
        | Ok f, Ok x -> Ok (f x)
        | Error e, _ -> Error e
        | _, Error e -> Error e

    /// Returns the value if the result is Ok, otherwise returns the default value
    let defaultValue defaultValue result =
        match result with
        | Ok x -> x
        | Error _ -> defaultValue

    /// Returns the value if the result is Ok, otherwise returns the result of the default function
    let defaultWith defaultFunc result =
        match result with
        | Ok x -> x
        | Error e -> defaultFunc e

    /// Returns true if the result is Ok
    let isOk result =
        match result with
        | Ok _ -> true
        | Error _ -> false

    /// Returns true if the result is Error
    let isError result =
        match result with
        | Ok _ -> false
        | Error _ -> true

    /// Converts an Option<'T> to a Result<'T, 'TError> using the provided error if None
    let ofOption error option =
        match option with
        | Some x -> Ok x
        | None -> Error error

    /// Converts a Result<'T, 'TError> to an Option<'T>
    let toOption result =
        match result with
        | Ok x -> Some x
        | Error _ -> None

    /// Combines two results into a tuple if both are Ok
    let zip result1 result2 =
        match result1, result2 with
        | Ok x, Ok y -> Ok (x, y)
        | Error e, _ -> Error e
        | _, Error e -> Error e

    /// Combines a list of results into a single result with a list of values
    let sequence results =
        let folder state result =
            match state, result with
            | Ok acc, Ok x -> Ok (x :: acc)
            | Error e, _ -> Error e
            | _, Error e -> Error e

        results
        |> List.fold folder (Ok [])
        |> map List.rev

    /// Maps a function over a list and collects the results into a single result with a list of values
    let traverse f list =
        let folder state x =
            match state with
            | Ok acc -> 
                match f x with
                | Ok y -> Ok (y :: acc)
                | Error e -> Error e
            | Error e -> Error e

        list
        |> List.fold folder (Ok [])
        |> map List.rev
