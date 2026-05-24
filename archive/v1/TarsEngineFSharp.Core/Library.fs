namespace TarsEngineFSharp.Core

open TarsEngineFSharp.Core.Monads

/// <summary>
/// Core utility functions for the TARS engine.
/// </summary>
module Core =
    /// <summary>
    /// Executes a function and wraps the result in a Result.
    /// </summary>
    let tryExecute f =
        try
            Result.success (f())
        with
        | ex -> Result.failure ex

    /// <summary>
    /// Executes an async function and wraps the result in a Result.
    /// </summary>
    let tryExecuteAsync (f: unit -> Async<'T>) =
        async {
            try
                let! result = f()
                return Result.success result
            with
            | ex -> return Result.failure ex
        }

    /// <summary>
    /// Converts a C# Task to an F# Async.
    /// </summary>
    let ofTask (task: System.Threading.Tasks.Task<'T>) =
        Async.AwaitTask task

    /// <summary>
    /// Converts an F# Async to a C# Task.
    /// </summary>
    let toTask (async: Async<'T>) =
        Async.StartAsTask async

/// <summary>
/// Interop functions for working with C# code.
/// </summary>
module Interop =
    open System.Threading.Tasks

    /// <summary>
    /// Converts an F# Result to a C# Task.
    /// </summary>
    let resultToTask<'TSuccess, 'TFailure when 'TFailure :> exn> (result: Result<'TSuccess, 'TFailure>) =
        match result with
        | Success value -> Task.FromResult value
        | Failure error -> Task.FromException<'TSuccess>(error)

    /// <summary>
    /// Converts an F# Async Result to a C# Task.
    /// </summary>
    let asyncResultToTask<'TSuccess, 'TFailure when 'TFailure :> exn> (asyncResult: Async<Result<'TSuccess, 'TFailure>>) =
        async {
            let! result = asyncResult
            return
                match result with
                | Success value -> value
                | Failure error -> raise error
        } |> Core.toTask

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
