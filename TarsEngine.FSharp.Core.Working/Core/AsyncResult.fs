namespace TarsEngine.FSharp.Core.Working

open System.Threading.Tasks

/// <summary>
/// Module containing AsyncResult type utilities for async error handling.
/// </summary>
module AsyncResult =
    
    /// <summary>
    /// Represents an asynchronous result.
    /// </summary>
    type AsyncResult<'T, 'TError> = Task<Result<'T, 'TError>>
    
    /// <summary>
    /// Creates a successful async result.
    /// </summary>
    let retn value : AsyncResult<'T, 'TError> =
        Task.FromResult(Ok value)
    
    /// <summary>
    /// Creates a failed async result.
    /// </summary>
    let fail error : AsyncResult<'T, 'TError> =
        Task.FromResult(Error error)
    
    /// <summary>
    /// Maps a function over the success value of an AsyncResult.
    /// </summary>
    let map f (asyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'U, 'TError> =
        task {
            let! result = asyncResult
            return Result.map f result
        }
    
    /// <summary>
    /// Maps a function over the error value of an AsyncResult.
    /// </summary>
    let mapError f (asyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'T, 'UError> =
        task {
            let! result = asyncResult
            return Result.mapError f result
        }
    
    /// <summary>
    /// Binds a function that returns an AsyncResult over the success value.
    /// </summary>
    let bind f (asyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'U, 'TError> =
        task {
            let! result = asyncResult
            match result with
            | Ok value -> return! f value
            | Error error -> return Error error
        }
    
    /// <summary>
    /// Converts a Task to an AsyncResult.
    /// </summary>
    let ofTask (task: Task<'T>) : AsyncResult<'T, exn> =
        task {
            try
                let! result = task
                return Ok result
            with
            | ex -> return Error ex
        }
    
    /// <summary>
    /// Converts an AsyncResult to a Task, using a default value for errors.
    /// </summary>
    let toTask defaultValue (asyncResult: AsyncResult<'T, 'TError>) : Task<'T> =
        task {
            let! result = asyncResult
            return Result.defaultValue defaultValue result
        }
