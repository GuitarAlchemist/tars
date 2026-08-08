/// Hybrid Plan Storage - Combines Multiple Backends
/// "PostgreSQL for truth, Graphiti for time, ChromaDB for similarity"
namespace Tars.Knowledge

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Connectors
open Tars.Cortex

/// Storage backend selection
type PlanStorageBackend =
    | InMemory
    | PostgreSQL
    | Graphiti of url: string
    | ChromaDB of url: string
    | Hybrid of primary: PlanStorageBackend * secondary: PlanStorageBackend list

/// Coordinator that writes to multiple backends with eventual consistency
type HybridPlanStorage(primary: IPlanStore, ?secondaries: IPlanStore list) =
    let secondary = defaultArg secondaries []

    /// Write to all backends (fire and forget for secondaries)
    let writeToAll (operation: IPlanStore -> Task<Result<unit, string>>) =
        task {
            // Primary write (wait for result)
            let! primaryResult = operation primary

            // Secondary writes (fire and forget - eventual consistency)
            for store in secondary do
                Task.Run(fun () ->
                    task {
                        let! _ = operation store
                        () // Ignore secondary failures
                    })
                |> ignore

            return primaryResult
        }

    interface IPlanStore with
        member _.SavePlan(plan) =
            writeToAll (fun store -> store.SavePlan(plan))

        member _.UpdatePlan(plan) =
            writeToAll (fun store -> store.UpdatePlan(plan))

        member _.GetPlan(planId) =
            // Always read from primary (strong consistency)
            primary.GetPlan(planId)

        member _.GetPlansByStatus(status) =
            // Always read from primary
            primary.GetPlansByStatus(status)

        member _.AppendEvent(event) =
            writeToAll (fun store -> store.AppendEvent(event))

    /// Access to individual backends for specialized queries
    member _.Primary = primary
    member _.Secondaries = secondary

/// Factory for creating hybrid storage
module HybridPlanStorage =

    /// Create storage from backend specification
    let rec createStorage (backend: PlanStorageBackend) : IPlanStore =
        match backend with
        | InMemory -> InMemoryLedgerStorage() :> IPlanStore

        | PostgreSQL -> PostgresLedgerStorage.create () :> IPlanStore

        | Graphiti url -> GraphitiPlanStorage.create (url) :> IPlanStore

        | ChromaDB url -> ChromaPlanStorage.create (url) :> IPlanStore

        | Hybrid(primary, secondaryBackends) ->
            let primaryStorage = createStorage primary
            let secondaryStorages = secondaryBackends |> List.map createStorage
            HybridPlanStorage(primaryStorage, secondaryStorages) :> IPlanStore

    /// Create default hybrid (PostgreSQL + Graphiti + ChromaDB)
    let createDefault (?pgConnString: string, ?graphitiUrl: string, ?chromaUrl: string) : IPlanStore =

        let pg =
            match pgConnString with
            | Some conn -> PostgresLedgerStorage.createWithConnectionString (conn)
            | None -> PostgresLedgerStorage.create ()

        let secondaries =
            [ match graphitiUrl with
              | Some url -> yield GraphitiPlanStorage.create (url) :> IPlanStore
              | None -> ()

              match chromaUrl with
              | Some url -> yield ChromaPlanStorage.create (url) :> IPlanStore
              | None -> () ]

        HybridPlanStorage(pg :> IPlanStore, secondaries) :> IPlanStore

    /// Create for development (In-Memory only)
    let createDevelopment () : IPlanStore = InMemoryLedgerStorage() :> IPlanStore

    /// Create for production (Full hybrid stack)
    let createProduction (pgConnString: string) (graphitiUrl: string) (chromaUrl: string) : IPlanStore =

        let primary = PostgresLedgerStorage.createWithConnectionString (pgConnString)
        let graphiti = GraphitiPlanStorage.create (graphitiUrl)
        let chroma = ChromaPlanStorage.create (chromaUrl)

        HybridPlanStorage(primary :> IPlanStore, [ graphiti :> IPlanStore; chroma :> IPlanStore ]) :> IPlanStore

    /// Create with custom configuration
    let create (primary: IPlanStore) (secondaries: IPlanStore list) : IPlanStore =
        HybridPlanStorage(primary, secondaries) :> IPlanStore
