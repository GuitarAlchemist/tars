module Tars.Tests.LocalModelPoolTests

open System
open System.Threading.Tasks
open System.Collections.Generic
open Xunit
open Tars.Llm
open Tars.Llm.Routing

// The pool's create function is injectable, so the LRU / eviction / disposal
// bookkeeping is testable with a fake disposable service — no real model load.

type private FakeModel() =
    member val Disposed = false with get, set

    interface ILlmService with
        member _.CompleteAsync(_) =
            Task.FromResult(
                { Text = ""
                  FinishReason = None
                  Usage = None
                  Raw = None }
            )

        member _.CompleteStreamAsync(_, _) =
            Task.FromResult(
                { Text = ""
                  FinishReason = None
                  Usage = None
                  Raw = None }
            )

        member _.EmbedAsync(_) = Task.FromResult(Array.empty<float32>)

        member _.RouteAsync(_) =
            Task.FromResult(
                ({ Backend = LlmBackend.Ollama "fake"
                   Endpoint = Uri "http://localhost"
                   ApiKey = None }
                : RoutedBackend)
            )

    interface IDisposable with
        member this.Dispose() = this.Disposed <- true

// create ignores cfg, so a null config is never dereferenced.
let private noCfg = Unchecked.defaultof<LlmServiceConfig>

[<Fact>]
let ``Acquire reuses the same service for a repeated model path`` () =
    let mutable created = 0

    let pool =
        new LocalModelPool(
            2,
            fun _ _ ->
                created <- created + 1
                new FakeModel() :> ILlmService
        )
        :> ILocalModelPool

    let a1 = pool.Acquire(noCfg, "modelA")
    let a2 = pool.Acquire(noCfg, "modelA")

    Assert.Same(a1, a2)
    Assert.Equal(1, created)

[<Fact>]
let ``Exceeding the bound evicts and disposes the least-recently-used model`` () =
    let fakes = Dictionary<string, FakeModel>()

    let create =
        fun _ path ->
            let f = new FakeModel()
            fakes.[path] <- f
            f :> ILlmService

    let pool = new LocalModelPool(1, create) :> ILocalModelPool

    pool.Acquire(noCfg, "A") |> ignore
    pool.Acquire(noCfg, "B") |> ignore // bound is 1 → A is evicted

    Assert.True(fakes.["A"].Disposed, "evicted model A should be disposed")
    Assert.False(fakes.["B"].Disposed, "resident model B should not be disposed")

[<Fact>]
let ``Dispose releases all resident models`` () =
    let fakes = Dictionary<string, FakeModel>()

    let create =
        fun _ path ->
            let f = new FakeModel()
            fakes.[path] <- f
            f :> ILlmService

    let pool = new LocalModelPool(3, create)
    (pool :> ILocalModelPool).Acquire(noCfg, "A") |> ignore
    (pool :> ILocalModelPool).Acquire(noCfg, "B") |> ignore

    (pool :> IDisposable).Dispose()

    Assert.True(fakes.["A"].Disposed)
    Assert.True(fakes.["B"].Disposed)
