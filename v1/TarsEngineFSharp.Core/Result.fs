namespace TarsEngineFSharp.Core

/// Represents a Result type that can be either Success or Failure
type Result<'TSuccess, 'TFailure> =
    | Success of 'TSuccess
    | Failure of 'TFailure

/// Core Result operations
module Result =
    /// Maps a function over the success value
    let map f result =
        match result with
        | Success x -> Success (f x)
        | Failure err -> Failure err

    /// Maps a function over the failure value
    let mapError f result =
        match result with
        | Success x -> Success x
        | Failure err -> Failure (f err)

    /// Binds a function over the success value
    let bind f result =
        match result with
        | Success x -> f x
        | Failure err -> Failure err

    /// Converts an Option to a Result with a default error
    let ofOption error opt =
        match opt with
        | Some x -> Success x
        | None -> Failure error

    /// Gets the success value or a default
    let defaultValue value result =
        match result with
        | Success x -> x
        | Failure _ -> value

    /// Converts the Result to an Option
    let toOption result =
        match result with
        | Success x -> Some x
        | Failure _ -> None