namespace Tars.Tools.Standard

open System
open System.IO
open Tars.Tools
open Tars.Core
open Tars.Core.TemporalKnowledgeGraph

/// <summary>
/// Professional tools for temporal knowledge analysis and evolution tracking.
/// These tools leverage the core TemporalKnowledgeGraph and TemporalReasoning modules.
/// </summary>
module TemporalTools =

    let mutable private globalGraphService: IGraphService option = None

    /// Internal helper to get or initialize the graph service
    let private getService () =
        match globalGraphService with
        | Some s -> s
        | None ->
            // Default to local storage in .tars directory
            let path = Path.Combine(Environment.CurrentDirectory, ".tars", "knowledge")
            let s = InternalGraphService(path) :> IGraphService
            globalGraphService <- Some s
            s

    /// Set a pre-initialized graph service (useful for CLI/Web integration)
    let setGraphService (s: IGraphService) =
        globalGraphService <- Some s

    [<TarsToolAttribute("temporal_trace_evolution", 
                        "Traces the lineage of an entity backwards through its evolutionary history. Input JSON: { \"entity_id\": \"canonical_id\" }")>]
    let temporalTraceEvolution (args: string) =
        task {
            try
                let entityId = ToolHelpers.tryParseStringArg args "entity_id" |> Option.defaultValue args
                if String.IsNullOrWhiteSpace entityId then 
                    return "Please provide an entity_id."
                else
                    let service = getService()
                    match service with
                    | :? InternalGraphService as internalService ->
                        let graph = internalService.Graph
                        let entityNode = graph.GetNode(entityId)
                        
                        match entityNode with
                        | Some node ->
                            let chain = TemporalReasoning.findEvolutionChain graph node.Entity
                            if List.isEmpty chain then
                                return $"Entity '{entityId}' has no known evolution history (it is an original state)."
                            else
                                let report = 
                                    chain 
                                    |> List.mapi (fun i (state, delta) -> 
                                        let id = TarsEntity.getId state
                                        $"{i+1}. {id} (Delta: {delta})")
                                    |> String.concat "\n"
                                return $"# Evolution Chain for {entityId}\n\nTracing backwards from current state:\n{report}"
                        | None -> return $"Entity '{entityId}' not found in the temporal graph."
                    | _ -> return "Temporal analysis not supported by the current graph service implementation."
            with ex ->
                return $"Error in temporal_trace_evolution: {ex.Message}"
        }

    [<TarsToolAttribute("temporal_get_history", 
                        "Returns all historical (invalidated or superseded) facts in the graph. Useful for forensic analysis. Input: { \"limit\": 20 }")>]
    let temporalGetHistory (args: string) =
        task {
            try
                let limit = ToolHelpers.tryParseIntArg args "limit" |> Option.defaultValue 20
                let service = getService()
                match service with
                | :? InternalGraphService as internalService ->
                    let graph = internalService.Graph
                    let historical = TemporalReasoning.getHistoricalFacts graph
                    
                    if List.isEmpty historical then
                        return "No historical facts found. The graph is currently in its initial valid state."
                    else
                        let displayed = historical |> List.truncate limit
                        let lines = displayed |> List.map (fun f -> $"- {f}") |> String.concat "\n"
                        let count = List.length historical
                        let footer = if count > limit then $"\n\n... and {count - limit} more." else ""
                        return $"# Historical Knowledge ({count} records)\n\n{lines}{footer}"
                | _ -> return "Historical analysis not supported by the current graph service implementation."
            with ex ->
                return $"Error in temporal_get_history: {ex.Message}"
        }

    [<TarsToolAttribute("temporal_detect_contradictions", 
                        "Analyzes the temporal graph for logic contradictions based on specific fact types. No input required.")>]
    let temporalDetectContradictions (_: string) =
        task {
            try
                let service = getService()
                match service with
                | :? InternalGraphService as internalService ->
                    let graph = internalService.Graph
                    let contradictions = TemporalReasoning.detectContradictions graph DateTime.UtcNow
                    
                    if List.isEmpty contradictions then
                        return "No active temporal contradictions detected."
                    else
                        let reports = 
                            contradictions 
                            |> List.map (fun (s, t, reason) ->
                                let reasonStr = reason |> Option.defaultValue "No reason provided"
                                $"- {TarsEntity.getId s} ⚡ {TarsEntity.getId t}: {reasonStr}")
                            |> String.concat "\n"
                        return $"# Temporal Contradictions Found ({List.length contradictions})\n\n{reports}"
                | _ -> return "Contradiction detection not supported by the current graph service implementation."
            with ex ->
                return $"Error in temporal_detect_contradictions: {ex.Message}"
        }
