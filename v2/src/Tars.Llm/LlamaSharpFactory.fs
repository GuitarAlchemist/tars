namespace Tars.Llm

open System
open System.Collections.Generic

/// A bounded pool of in-process local (LlamaSharp) model services. Acquiring a
/// model path returns a ready ILlmService; when the resident count exceeds the
/// bound the least-recently-used entry is evicted and its native handles are
/// disposed. Dispose() frees every resident model at shutdown.
///
/// Eviction assumes the evicted model is not concurrently in use — safe for the
/// default serial, single-resident-model usage. Raise the bound only when models
/// are used one-at-a-time per slot.
type ILocalModelPool =
    inherit IDisposable
    abstract member Acquire: cfg: LlmServiceConfig * modelPath: string -> ILlmService

/// Bounded LRU pool. `create` builds a service for a model path (its result must
/// be IDisposable for eviction to release handles); injecting it keeps the
/// eviction/LRU bookkeeping testable without loading a real model.
type LocalModelPool(maxResident: int, create: LlmServiceConfig -> string -> ILlmService) =
    let gate = obj ()
    // Most-recently-used at the end; least-recently-used (eviction target) at the front.
    let entries = LinkedList<string * ILlmService>()
    let index = Dictionary<string, LinkedListNode<string * ILlmService>>()
    let mutable disposed = false

    let disposeService (svc: ILlmService) =
        match box svc with
        | :? IDisposable as d -> d.Dispose()
        | _ -> ()

    interface ILocalModelPool with
        member _.Acquire(cfg, modelPath) =
            lock gate (fun () ->
                if disposed then
                    raise (ObjectDisposedException("LocalModelPool"))

                match index.TryGetValue modelPath with
                | true, node ->
                    // Cache hit — promote to most-recently-used.
                    entries.Remove node
                    entries.AddLast node |> ignore
                    snd node.Value
                | _ ->
                    let svc = create cfg modelPath
                    let node = entries.AddLast((modelPath, svc))
                    index.[modelPath] <- node

                    // Evict least-recently-used entries beyond the bound.
                    while entries.Count > max 1 maxResident do
                        let lru = entries.First
                        entries.RemoveFirst()
                        index.Remove(fst lru.Value) |> ignore
                        disposeService (snd lru.Value)

                    svc)

    interface IDisposable with
        member _.Dispose() =
            lock gate (fun () ->
                if not disposed then
                    disposed <- true

                    for (_, svc) in entries do
                        disposeService svc

                    entries.Clear()
                    index.Clear())

/// Process-wide shared local-model pool and the thin accessor the backend
/// resolver uses. Replaces the former unbounded, never-disposed cache.
module LlamaSharpFactory =

    let private poolSize =
        match Int32.TryParse(Environment.GetEnvironmentVariable("TARS_LLAMASHARP_POOL_SIZE")) with
        | true, n when n > 0 -> n
        | _ -> 1

    let private shared: ILocalModelPool =
        let pool =
            new LocalModelPool(poolSize, (fun cfg path -> new LlamaSharpService(cfg, path) :> ILlmService))
            :> ILocalModelPool
        // Best-effort release of native handles on graceful process exit.
        AppDomain.CurrentDomain.ProcessExit.Add(fun _ -> pool.Dispose())
        pool

    /// Acquire (load-or-reuse) the local model service for a model path.
    /// apiKey is unused by the local backend and kept only for call-site stability.
    let getService (cfg: LlmServiceConfig) (_apiKey: string option) (modelPath: string) : ILlmService =
        shared.Acquire(cfg, modelPath)
