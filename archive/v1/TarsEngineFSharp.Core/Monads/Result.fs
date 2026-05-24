namespace TarsEngineFSharp.Core.Monads

/// <summary>
/// Represents a result that is either a success with a value or a failure with an error.
/// </summary>
type Result<'TSuccess, 'TFailure> =
    | Success of 'TSuccess
    | Failure of 'TFailure

/// <summary>
/// Contains operations for the Result type.
/// </summary>
module Result =
    /// <summary>
    /// Creates a success result with the given value.
    /// </summary>
    let success value = Success value

    /// <summary>
    /// Creates a failure result with the given error.
    /// </summary>
    let failure error = Failure error

    /// <summary>
    /// Returns true if the result is a success.
    /// </summary>
    let isSuccess = function
        | Success _ -> true
        | Failure _ -> false

    /// <summary>
    /// Returns true if the result is a failure.
    /// </summary>
    let isFailure = function
        | Success _ -> false
        | Failure _ -> true

    /// <summary>
    /// Returns the success value or raises an exception if the result is a failure.
    /// </summary>
    let getValue = function
        | Success value -> value
        | Failure _ -> failwith "Cannot get value from a failure result"

    /// <summary>
    /// Returns the error or raises an exception if the result is a success.
    /// </summary>
    let getError = function
        | Success _ -> failwith "Cannot get error from a success result"
        | Failure error -> error

    /// <summary>
    /// Returns the success value or the provided default value if the result is a failure.
    /// </summary>
    let getValueOrDefault defaultValue = function
        | Success value -> value
        | Failure _ -> defaultValue

    /// <summary>
    /// Returns the success value or computes a default value if the result is a failure.
    /// </summary>
    let getValueOrElse defaultFn = function
        | Success value -> value
        | Failure error -> defaultFn error

    /// <summary>
    /// Applies a function to the success value if the result is a success, otherwise returns the failure.
    /// </summary>
    let map f = function
        | Success value -> Success (f value)
        | Failure error -> Failure error

    /// <summary>
    /// Applies a function to the error if the result is a failure, otherwise returns the success.
    /// </summary>
    let mapError f = function
        | Success value -> Success value
        | Failure error -> Failure (f error)

    /// <summary>
    /// Applies a function that returns a result to the success value if the result is a success,
    /// otherwise returns the failure.
    /// </summary>
    let bind f = function
        | Success value -> f value
        | Failure error -> Failure error

    /// <summary>
    /// Applies a function to the success value and a function to the error.
    /// </summary>
    let either onSuccess onFailure = function
        | Success value -> onSuccess value
        | Failure error -> onFailure error

    /// <summary>
    /// Converts an Option to a Result, using the provided error if the option is None.
    /// </summary>
    let ofOption error = function
        | Some value -> Success value
        | None -> Failure error

    /// <summary>
    /// Converts a Result to an Option, discarding the error.
    /// </summary>
    let toOption = function
        | Success value -> Some value
        | Failure _ -> None

    /// <summary>
    /// Combines two results, returning a tuple of the success values if both are successful,
    /// or the first failure if either fails.
    /// </summary>
    let zip result1 result2 =
        match result1, result2 with
        | Success value1, Success value2 -> Success (value1, value2)
        | Failure error, _ -> Failure error
        | _, Failure error -> Failure error

    /// <summary>
    /// Combines a list of results into a single result with a list of values.
    /// </summary>
    let sequence results =
        let folder state result =
            match state, result with
            | Success values, Success value -> Success (value :: values)
            | Failure error, _ -> Failure error
            | _, Failure error -> Failure error

        let initialState = Success []

        results
        |> List.fold folder initialState
        |> map List.rev

    /// <summary>
    /// Applies a function that returns a result to each element in the list and collects the results.
    /// </summary>
    let traverse f list =
        let folder state element =
            match state with
            | Success values ->
                match f element with
                | Success value -> Success (value :: values)
                | Failure error -> Failure error
            | Failure error -> Failure error

        let initialState = Success []

        list
        |> List.fold folder initialState
        |> map List.rev

    /// <summary>
    /// Converts a Result to a C# style result with pattern matching.
    /// </summary>
    let toCSharp result =
        match result with
        | Success value -> struct (true, value, Unchecked.defaultof<_>)
        | Failure error -> struct (false, Unchecked.defaultof<_>, error)

/// <summary>
/// Contains extension methods for working with the Result type.
/// </summary>
[<AutoOpen>]
module ResultExtensions =
    /// <summary>
    /// Provides a fluent syntax for working with results.
    /// </summary>
    type ResultBuilder() =
        member _.Return(value) = Result.success value
        member _.ReturnFrom(result) = result
        member _.Bind(result, f) = Result.bind f result
        member _.Zero() = Result.success ()
        member _.Delay(f) = f
        member _.Run(f) = f()

        member _.TryWith(body, handler) =
            try
                body()
            with
            | ex -> handler ex

        member _.TryFinally(body, compensation) =
            try
                body()
            finally
                compensation()

        member _.Using(disposable: #System.IDisposable, body) =
            using disposable body

        member this.While(guard, body) =
            if not (guard()) then
                Result.success ()
            else
                body() |> Result.bind (fun () -> this.While(guard, body))

        member _.For(sequence, body) =
            sequence
            |> Seq.fold (fun state element ->
                state |> Result.bind (fun () -> body element)
            ) (Result.success ())

        member _.Combine(result, f) =
            result |> Result.bind (fun () -> f())

        member _.MergeSources(result1, result2) =
            Result.zip result1 result2

    /// <summary>
    /// Creates a computation expression for working with results.
    /// </summary>
    let result = ResultBuilder()
