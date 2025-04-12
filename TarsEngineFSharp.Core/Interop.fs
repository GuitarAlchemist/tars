namespace TarsEngineFSharp.Core

/// Interop functions for working with C# code
module Interop =
    open System
    open System.Threading.Tasks

    /// Converts an F# Result to a C# Task
    let resultToTask<'TSuccess, 'TFailure when 'TFailure :> exn> (result: Result<'TSuccess, 'TFailure>) =
        match result with
        | Success value -> Task.FromResult value
        | Failure error -> Task.FromException<'TSuccess>(error)

    /// Converts an F# Async Result to a C# Task
    let asyncResultToTask<'TSuccess, 'TFailure when 'TFailure :> exn> (asyncResult: Async<Result<'TSuccess, 'TFailure>>) =
        async {
            let! result = asyncResult
            return
                match result with
                | Success value -> value
                | Failure error -> raise error
        } |> Async.StartAsTask

    /// Converts a C# Task to an F# Async Result
    let taskToAsyncResult<'TSuccess> (task: Task<'TSuccess>) =
        async {
            try
                let! result = task |> Async.AwaitTask
                return Success result
            with
            | ex -> return Failure ex
        }

    /// Creates a Result from a nullable value
    let ofNullable<'T when 'T : not struct> (value: 'T) =
        if isNull value then
            Failure (ArgumentNullException())
        else
            Success value