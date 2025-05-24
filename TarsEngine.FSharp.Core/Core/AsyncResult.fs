namespace TarsEngine.FSharp.Core

/// Module containing AsyncResult type and utilities
module AsyncResult =
    /// Type alias for Async<Result<'T, 'TError>>
    type AsyncResult<'T, 'TError> = Async<Result<'T, 'TError>>

    /// Creates an AsyncResult from a value
    let retn x : AsyncResult<'T, 'TError> = 
        async { return Ok x }

    /// Creates an AsyncResult from an error
    let returnError x : AsyncResult<'T, 'TError> = 
        async { return Error x }

    /// Maps a function over an AsyncResult
    let map f (asyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'U, 'TError> =
        async {
            let! result = asyncResult
            return Result.map f result
        }

    /// Maps a function over the error of an AsyncResult
    let mapError f (asyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'T, 'UError> =
        async {
            let! result = asyncResult
            return Result.mapError f result
        }

    /// Binds a function over an AsyncResult
    let bind (f: 'T -> AsyncResult<'U, 'TError>) (asyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'U, 'TError> =
        async {
            let! result = asyncResult
            match result with
            | Ok x -> return! f x
            | Error e -> return Error e
        }

    /// Applies a function wrapped in an AsyncResult to an AsyncResult
    let apply (fAsyncResult: AsyncResult<'T -> 'U, 'TError>) (xAsyncResult: AsyncResult<'T, 'TError>) : AsyncResult<'U, 'TError> =
        async {
            let! fResult = fAsyncResult
            let! xResult = xAsyncResult
            return Result.apply fResult xResult
        }

    /// Converts an Async<'T> to an AsyncResult<'T, 'TError>
    let ofAsync (x: Async<'T>) : AsyncResult<'T, 'TError> =
        async {
            let! result = x
            return Ok result
        }

    /// Converts a Result<'T, 'TError> to an AsyncResult<'T, 'TError>
    let ofResult (result: Result<'T, 'TError>) : AsyncResult<'T, 'TError> =
        async { return result }

    /// Converts an AsyncResult<'T, 'TError> to an Async<Option<'T>>
    let toAsyncOption (asyncResult: AsyncResult<'T, 'TError>) : Async<Option<'T>> =
        async {
            let! result = asyncResult
            return Result.toOption result
        }

    /// Combines two AsyncResults into a tuple if both are Ok
    let zip (asyncResult1: AsyncResult<'T, 'TError>) (asyncResult2: AsyncResult<'U, 'TError>) : AsyncResult<'T * 'U, 'TError> =
        async {
            let! result1 = asyncResult1
            let! result2 = asyncResult2
            return Result.zip result1 result2
        }

    /// Combines a list of AsyncResults into a single AsyncResult with a list of values
    let sequence (asyncResults: AsyncResult<'T, 'TError> list) : AsyncResult<'T list, 'TError> =
        let folder (state: AsyncResult<'T list, 'TError>) (asyncResult: AsyncResult<'T, 'TError>) =
            bind (fun stateList ->
                map (fun x -> x :: stateList) asyncResult) state

        let initialState = retn []
        let combined = List.fold folder initialState asyncResults
        map List.rev combined

    /// Maps a function over a list and collects the AsyncResults into a single AsyncResult with a list of values
    let traverse (f: 'T -> AsyncResult<'U, 'TError>) (list: 'T list) : AsyncResult<'U list, 'TError> =
        sequence (List.map f list)

    /// Runs an AsyncResult and waits for the result
    let run (asyncResult: AsyncResult<'T, 'TError>) : Result<'T, 'TError> =
        asyncResult |> Async.RunSynchronously

    /// Handles both success and error cases of an AsyncResult
    let either (successHandler: 'T -> 'U) (errorHandler: 'TError -> 'U) (asyncResult: AsyncResult<'T, 'TError>) : Async<'U> =
        async {
            let! result = asyncResult
            return match result with
                   | Ok x -> successHandler x
                   | Error e -> errorHandler e
        }
