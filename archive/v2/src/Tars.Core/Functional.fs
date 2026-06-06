// Lightweight functional helpers (kept minimal for v2)
namespace Tars.Core

open System

/// Combine Async and Result
type AsyncResult<'T, 'E> = Async<Result<'T, 'E>>

module AsyncResult =
    let retn x : AsyncResult<'T, 'E> = async { return Ok x }

    let map f (ar: AsyncResult<'T, 'E>) : AsyncResult<'U, 'E> =
        async {
            let! result = ar
            return Result.map f result
        }

    let bind (ar: AsyncResult<'T, 'E>) (f: 'T -> AsyncResult<'U, 'E>) : AsyncResult<'U, 'E> =
        async {
            let! result = ar

            match result with
            | Ok v -> return! f v
            | Error e -> return Error e
        }

    let ofResult r = async { return r }

    let ofAsync a =
        async {
            let! v = a
            return Ok v
        }

type AsyncResultBuilder() =
    member _.Return x = AsyncResult.retn x
    member _.Bind(m, f) = AsyncResult.bind m f
    member _.ReturnFrom m = m
    member _.Zero() = AsyncResult.retn ()
    member _.Delay(f) = async.Delay(f)
    member _.Run(f) = f

    member _.TryWith(m, h) =
        async {
            try
                return! m
            with ex ->
                return! h ex
        }

    member _.TryFinally(m, compensation) =
        async {
            try
                return! m
            finally
                compensation ()
        }

    member this.Using(res: #IDisposable, body) =
        let dispose =
            fun () ->
                if not (isNull (box res)) then
                    res.Dispose()

        this.TryFinally(body res, dispose)

    member this.While(guard, body) =
        if not (guard ()) then
            this.Zero()
        else
            this.Bind(body, fun () -> this.While(guard, body))

    member this.For(sequence: seq<_>, body) =
        this.Using(
            sequence.GetEnumerator(),
            fun enum -> this.While(enum.MoveNext, this.Delay(fun () -> body enum.Current))
        )

[<AutoOpen>]
module AsyncResultCE =
    let asyncResult = AsyncResultBuilder()
