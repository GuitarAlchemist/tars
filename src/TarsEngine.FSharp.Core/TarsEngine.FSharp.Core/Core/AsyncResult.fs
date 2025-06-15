namespace TarsEngine.FSharp.Core.Core

/// Async Result type for async error handling
type AsyncResult<'T, 'E> = Async<Result<'T, 'E>>

module AsyncResult =
    
    let map f (asyncResult: AsyncResult<'T, 'E>) : AsyncResult<'U, 'E> =
        async {
            let! result = asyncResult
            return Result.map f result
        }
    
    let bind f (asyncResult: AsyncResult<'T, 'E>) : AsyncResult<'U, 'E> =
        async {
            let! result = asyncResult
            match result with
            | Ok x -> return! f x
            | Error e -> return Error e
        }
    
    let ofResult result = async { return result }
    
    let ofAsync asyncValue = async {
        let! value = asyncValue
        return Ok value
    }
