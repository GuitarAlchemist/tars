namespace TarsEngine.FSharp.Core

/// Module containing Option type extensions and utilities
module Option =
    /// Maps an Option<'T> to an Option<'U> by applying a function to the value
    let map f option =
        match option with
        | Some x -> Some (f x)
        | None -> None

    /// Binds an Option<'T> to an Option<'U> by applying a function that returns an Option
    let bind f option =
        match option with
        | Some x -> f x
        | None -> None

    /// Applies a function to an Option<'T> and returns a new Option<'U>
    let apply fOption xOption =
        match fOption, xOption with
        | Some f, Some x -> Some (f x)
        | _ -> None

    /// Returns the value if the option is Some, otherwise returns the default value
    let defaultValue defaultValue option =
        match option with
        | Some x -> x
        | None -> defaultValue

    /// Returns the value if the option is Some, otherwise returns the result of the default function
    let defaultWith defaultFunc option =
        match option with
        | Some x -> x
        | None -> defaultFunc()

    /// Returns true if the option is Some
    let isSome option =
        match option with
        | Some _ -> true
        | None -> false

    /// Returns true if the option is None
    let isNone option =
        match option with
        | Some _ -> false
        | None -> true

    /// Converts a Result<'T, 'TError> to an Option<'T>
    let ofResult result =
        match result with
        | Ok x -> Some x
        | Error _ -> None

    /// Converts an Option<'T> to a Result<'T, 'TError> using the provided error if None
    let toResult error option =
        match option with
        | Some x -> Ok x
        | None -> Error error

    /// Combines two options into a tuple if both are Some
    let zip option1 option2 =
        match option1, option2 with
        | Some x, Some y -> Some (x, y)
        | _ -> None

    /// Combines a list of options into a single option with a list of values
    let sequence options =
        let folder state option =
            match state, option with
            | Some acc, Some x -> Some (x :: acc)
            | _ -> None

        options
        |> List.fold folder (Some [])
        |> map List.rev

    /// Maps a function over a list and collects the options into a single option with a list of values
    let traverse f list =
        let folder state x =
            match state with
            | Some acc -> 
                match f x with
                | Some y -> Some (y :: acc)
                | None -> None
            | None -> None

        list
        |> List.fold folder (Some [])
        |> map List.rev
