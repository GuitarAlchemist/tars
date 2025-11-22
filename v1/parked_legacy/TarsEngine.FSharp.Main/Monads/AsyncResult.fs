namespace TarsEngine.FSharp.Main.Monads

open System
open System.Threading.Tasks

/// <summary>
/// Module containing functions for working with async Result types
/// </summary>
module AsyncResult =
    /// <summary>
    /// Creates a successful async result with the given value
    /// </summary>
    let success value = async { return Ok value }

    /// <summary>
    /// Creates a failed async result with the given error
    /// </summary>
    let error err = async { return Error err }

    /// <summary>
    /// Converts a Result to an async Result
    /// </summary>
    let ofResult result = async { return result }

    /// <summary>
    /// Converts a Task to an async Result, capturing any exceptions as errors
    /// </summary>
    let ofTask (task: Task<'T>) =
        async {
            try
                let! result = Async.AwaitTask task
                return Ok result
            with
            | ex -> return Error ex
        }

    /// <summary>
    /// Converts a Task to an async Result with a specific error type, using the provided error mapping function
    /// </summary>
    let ofTaskWithError (errorMapper: exn -> 'TError) (task: Task<'T>) =
        async {
            try
                let! result = Async.AwaitTask task
                return Ok result
            with
            | ex -> return Error (errorMapper ex)
        }

    /// <summary>
    /// Applies a function to the value if the async result is successful, or returns the error if it's a failure
    /// </summary>
    let map mapper asyncResult =
        async {
            let! result = asyncResult
            return Result.map mapper result
        }

    /// <summary>
    /// Applies a function to the error if the async result is a failure, or returns the value if it's successful
    /// </summary>
    let mapError mapper asyncResult =
        async {
            let! result = asyncResult
            return Result.mapError mapper result
        }

    /// <summary>
    /// Applies a function that returns a Result to the value if the async result is successful, or returns the error if it's a failure
    /// </summary>
    let bind binder asyncResult =
        async {
            let! result = asyncResult
            return! 
                match result with
                | Ok value -> binder value
                | Error err -> async { return Error err }
        }

    /// <summary>
    /// Applies a function that returns an async Result to the value if the async result is successful, or returns the error if it's a failure
    /// </summary>
    let bindAsync binder asyncResult =
        async {
            let! result = asyncResult
            match result with
            | Ok value -> return! binder value
            | Error err -> return Error err
        }

    /// <summary>
    /// Applies one of two functions depending on whether the async result is successful or a failure
    /// </summary>
    let match' okFunc errorFunc asyncResult =
        async {
            let! result = asyncResult
            return Result.match' okFunc errorFunc result
        }

    /// <summary>
    /// Performs an action if the async result is successful
    /// </summary>
    let ifOk action asyncResult =
        async {
            let! result = asyncResult
            match result with
            | Ok value -> action value
            | Error _ -> ()
            return result
        }

    /// <summary>
    /// Performs an action if the async result is a failure
    /// </summary>
    let ifError action asyncResult =
        async {
            let! result = asyncResult
            match result with
            | Ok _ -> ()
            | Error err -> action err
            return result
        }

    /// <summary>
    /// Tries to execute an async function and returns a successful async result with the return value if it succeeds, or a failed async result with the exception if it throws
    /// </summary>
    let tryAsync (func: unit -> Async<'T>) : Async<Result<'T, exn>> =
        async {
            try
                let! result = func()
                return Ok result
            with
            | ex -> return Error ex
        }

    /// <summary>
    /// Converts an async Option to an async Result, using the provided error value for None
    /// </summary>
    let ofAsyncOption errorValue asyncOption =
        async {
            let! option = asyncOption
            return Result.ofOption errorValue option
        }
