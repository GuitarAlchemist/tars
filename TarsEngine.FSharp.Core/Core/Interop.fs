namespace TarsEngine.FSharp.Core

/// Module containing interop functions for working with C# code
module Interop =
    open System.Threading.Tasks

    /// <summary>
    /// Executes a function and wraps the result in a Result.
    /// </summary>
    let tryExecute f =
        try
            Ok (f())
        with
        | ex -> Error ex

    /// <summary>
    /// Executes an async function and wraps the result in a Result.
    /// </summary>
    let tryExecuteAsync (f: unit -> Async<'T>) =
        async {
            try
                let! result = f()
                return Ok result
            with
            | ex -> return Error ex
        }

    /// <summary>
    /// Converts a C# Task to an F# Async.
    /// </summary>
    let ofTask (task: Task<'T>) =
        Async.AwaitTask task

    /// <summary>
    /// Converts an F# Async to a C# Task.
    /// </summary>
    let toTask (async: Async<'T>) =
        Async.StartAsTask async

    /// <summary>
    /// Converts an F# Result to a C# Task.
    /// </summary>
    let resultToTask<'TSuccess, 'TFailure when 'TFailure :> exn> (result: Result<'TSuccess, 'TFailure>) =
        match result with
        | Ok value -> Task.FromResult value
        | Error error -> Task.FromException<'TSuccess>(error)

    /// <summary>
    /// Converts an F# Async Result to a C# Task.
    /// </summary>
    let asyncResultToTask<'TSuccess, 'TFailure when 'TFailure :> exn> (asyncResult: Async<Result<'TSuccess, 'TFailure>>) =
        async {
            let! result = asyncResult
            return
                match result with
                | Ok value -> value
                | Error error -> raise error
        } |> toTask

    /// <summary>
    /// Converts an F# Option to a C# nullable value.
    /// </summary>
    let optionToNullable (option: 'T option) =
        match option with
        | Some value -> value
        | None -> Unchecked.defaultof<'T>

    /// <summary>
    /// Converts a C# nullable value to an F# Option.
    /// </summary>
    let nullableToOption (nullable: 'T when 'T: null) =
        if isNull (box nullable) then None else Some nullable
