namespace Tars.Kernel

open System
open Tars.Core

/// Manages routing of logical agent names to specific instances
type AgentRouter() =
    let routes =
        System.Collections.Concurrent.ConcurrentDictionary<string, RoutingStrategy>()

    let random = Random()

    /// Register or update a route for a logical agent name
    member this.SetRoute(logicalName: string, strategy: RoutingStrategy) =
        routes.AddOrUpdate(logicalName, strategy, (fun _ _ -> strategy)) |> ignore

    /// Resolve a logical name to a specific AgentId based on the strategy
    member this.Resolve(logicalName: string) =
        match routes.TryGetValue(logicalName) with
        | true, strategy ->
            match strategy with
            | Pinned id -> Some id
            | Canary(primary, canary, weight) ->
                if random.NextDouble() < weight then
                    Some canary
                else
                    Some primary
            | RoundRobin ids ->
                if ids.IsEmpty then
                    None
                else
                    let idx = random.Next(ids.Length)
                    Some ids[idx]
        | false, _ -> None
