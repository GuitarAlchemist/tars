namespace TarsEngine.FSharp.Core.Unified

/// <summary>
/// Module containing Result type utilities for error handling.
/// </summary>
module Result =
    
    /// <summary>
    /// Maps a function over the success value of a Result.
    /// </summary>
    let map f result =
        match result with
        | Ok value -> Ok (f value)
        | Error error -> Error error
    
    /// <summary>
    /// Maps a function over the error value of a Result.
    /// </summary>
    let mapError f result =
        match result with
        | Ok value -> Ok value
        | Error error -> Error (f error)
    
    /// <summary>
    /// Binds a function that returns a Result over the success value.
    /// </summary>
    let bind f result =
        match result with
        | Ok value -> f value
        | Error error -> Error error
    
    /// <summary>
    /// Returns the success value or a default value if error.
    /// </summary>
    let defaultValue defaultVal result =
        match result with
        | Ok value -> value
        | Error _ -> defaultVal
    
    /// <summary>
    /// Returns the success value or the result of a function if error.
    /// </summary>
    let defaultWith f result =
        match result with
        | Ok value -> value
        | Error error -> f error
    
    /// <summary>
    /// Converts a Result to an Option, losing error information.
    /// </summary>
    let toOption result =
        match result with
        | Ok value -> Some value
        | Error _ -> None
    
    /// <summary>
    /// Converts an Option to a Result with a provided error.
    /// </summary>
    let ofOption error option =
        match option with
        | Some value -> Ok value
        | None -> Error error
    
    /// <summary>
    /// Applies a function if both Results are successful.
    /// </summary>
    let apply fResult xResult =
        match fResult, xResult with
        | Ok f, Ok x -> Ok (f x)
        | Error e, _ -> Error e
        | _, Error e -> Error e
    
    /// <summary>
    /// Combines two Results, returning the first error if any.
    /// </summary>
    let combine result1 result2 =
        match result1, result2 with
        | Ok x, Ok y -> Ok (x, y)
        | Error e, _ -> Error e
        | _, Error e -> Error e
    
    /// <summary>
    /// Executes a function and wraps the result in a Result.
    /// </summary>
    let tryWith f =
        try
            Ok (f())
        with
        | ex -> Error ex
    
    /// <summary>
    /// Folds over a sequence of Results, collecting successes and errors.
    /// </summary>
    let partition results =
        let successes, errors =
            results
            |> List.fold (fun (succ, errs) result ->
                match result with
                | Ok value -> (value :: succ, errs)
                | Error error -> (succ, error :: errs)
            ) ([], [])
        (List.rev successes, List.rev errors)
    
    /// <summary>
    /// Traverses a list with a function that returns Results.
    /// </summary>
    let traverse f list =
        let folder head tail =
            match f head, tail with
            | Ok h, Ok t -> Ok (h :: t)
            | Error e, _ -> Error e
            | _, Error e -> Error e
        
        List.foldBack folder list (Ok [])
    
    /// <summary>
    /// Sequences a list of Results into a Result of list.
    /// </summary>
    let sequence results =
        traverse id results
