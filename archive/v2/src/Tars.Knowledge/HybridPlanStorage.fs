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
type HybridPlanStorage(primary: IPlanStorage, ?secondaries: IPlanStorage list) =
    let secondary = defaultArg secondaries []

    /// Write to all backends (fire and forget for secondaries)
    let writeToAll (operation: IPlanStorage -> Task<Result<unit, string>>) =
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

    interface IPlanStorage with
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
    let rec createStorage (backend: PlanStorageBackend) : IPlanStorage =
        match backend with
        | InMemory -> InMemoryLedgerStorage() :> IPlanStorage

        | PostgreSQL -> PostgresLedgerStorage.create () :> IPlanStorage

        | Graphiti url -> GraphitiPlanStorage.create (url) :> IPlanStorage

        | ChromaDB url -> ChromaPlanStorage.create (url) :> IPlanStorage

        | Hybrid(primary, secondaryBackends) ->
            let primaryStorage = createStorage primary
            let secondaryStorages = secondaryBackends |> List.map createStorage
            HybridPlanStorage(primaryStorage, secondaryStorages) :> IPlanStorage

    /// Create default hybrid (PostgreSQL + Graphiti + ChromaDB)
    let createDefault (?pgConnString: string, ?graphitiUrl: string, ?chromaUrl: string) : IPlanStorage =

        let pg =
            match pgConnString with
            | Some conn -> PostgresLedgerStorage.createWithConnectionString (conn)
            | None -> PostgresLedgerStorage.create ()

        let secondaries =
            [ match graphitiUrl with
              | Some url -> yield GraphitiPlanStorage.create (url) :> IPlanStorage
              | None -> ()

              match chromaUrl with
              | Some url -> yield ChromaPlanStorage.create (url) :> IPlanStorage
              | None -> () ]

        HybridPlanStorage(pg :> IPlanStorage, secondaries) :> IPlanStorage

    /// Create for development (In-Memory only)
    let createDevelopment () : IPlanStorage = InMemoryLedgerStorage() :> IPlanStorage

    /// Create for production (Full hybrid stack)
    let createProduction (pgConnString: string) (graphitiUrl: string) (chromaUrl: string) : IPlanStorage =

        let primary = PostgresLedgerStorage.createWithConnectionString (pgConnString)
        let graphiti = GraphitiPlanStorage.create (graphitiUrl)
        let chroma = ChromaPlanStorage.create (chromaUrl)

        HybridPlanStorage(primary :> IPlanStorage, [ graphiti :> IPlanStorage; chroma :> IPlanStorage ]) :> IPlanStorage

    /// Create with custom configuration
    let create (primary: IPlanStorage) (secondaries: IPlanStorage list) : IPlanStorage =
        HybridPlanStorage(primary, secondaries) :> IPlanStorage
