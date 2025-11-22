namespace TarsEngine.FSharp.ML.Core

open System.Threading.Tasks

/// <summary>
/// Utility functions for working with async Result types.
/// </summary>
module AsyncResult =
    /// <summary>
    /// Maps a function over the Ok value of an async Result.
    /// </summary>
    /// <param name="f">The function to apply to the Ok value.</param>
    /// <param name="asyncResult">The async Result to map over.</param>
    /// <returns>A new async Result with the function applied to the Ok value.</returns>
    let map f asyncResult =
        async {
            let! result = asyncResult
            return Result.map f result
        }
    
    /// <summary>
    /// Binds a function over the Ok value of an async Result.
    /// </summary>
    /// <param name="f">The function to apply to the Ok value.</param>
    /// <param name="asyncResult">The async Result to bind over.</param>
    /// <returns>The async Result of applying the function to the Ok value.</returns>
    let bind f asyncResult =
        async {
            let! result = asyncResult
            match result with
            | Ok x -> return! f x
            | Error e -> return Error e
        }
    
    /// <summary>
    /// Applies a function to the Error value of an async Result.
    /// </summary>
    /// <param name="f">The function to apply to the Error value.</param>
    /// <param name="asyncResult">The async Result to map over.</param>
    /// <returns>A new async Result with the function applied to the Error value.</returns>
    let mapError f asyncResult =
        async {
            let! result = asyncResult
            return Result.mapError f result
        }
    
    /// <summary>
    /// Converts a Task to an async Result, with any exception wrapped in Error.
    /// </summary>
    /// <param name="task">The Task to convert.</param>
    /// <returns>An async Result with the Task result or any exception.</returns>
    let ofTask (task: Task<'T>) =
        async {
            try
                let! result = Async.AwaitTask task
                return Ok result
            with
            | ex -> return Error ex
        }
    
    /// <summary>
    /// Converts a Task to an async Result, with any exception mapped by the given function.
    /// </summary>
    /// <param name="f">The function to map exceptions to error values.</param>
    /// <param name="task">The Task to convert.</param>
    /// <returns>An async Result with the Task result or the mapped exception.</returns>
    let ofTaskWithError f (task: Task<'T>) =
        async {
            try
                let! result = Async.AwaitTask task
                return Ok result
            with
            | ex -> return Error (f ex)
        }
    
    /// <summary>
    /// Converts a Task to an async Result, with any exception wrapped in Error.
    /// </summary>
    /// <param name="task">The Task to convert.</param>
    /// <returns>An async Result with unit or any exception.</returns>
    let ofUnitTask (task: Task) =
        async {
            try
                do! Async.AwaitTask task
                return Ok ()
            with
            | ex -> return Error ex
        }
    
    /// <summary>
    /// Converts a Task to an async Result, with any exception mapped by the given function.
    /// </summary>
    /// <param name="f">The function to map exceptions to error values.</param>
    /// <param name="task">The Task to convert.</param>
    /// <returns>An async Result with unit or the mapped exception.</returns>
    let ofUnitTaskWithError f (task: Task) =
        async {
            try
                do! Async.AwaitTask task
                return Ok ()
            with
            | ex -> return Error (f ex)
        }
