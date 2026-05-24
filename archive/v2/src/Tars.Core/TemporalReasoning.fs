namespace Tars.Core

open System

/// <summary>
/// Temporal Reasoning module for analyzing the evolution and consistency of the Knowledge Graph.
/// </summary>
module TemporalReasoning =

    /// Trace the evolution chain of an entity backwards to its origins.
    /// Returns a list of (Entity, Delta) from newest to oldest.
    let findEvolutionChain (graph: TemporalKnowledgeGraph.TemporalGraph) (startEntity: TarsEntity) =
        let rec trace current acc =
            let facts = graph.GetOutgoingFacts(current)
            let evolutionStep = 
                facts |> List.tryFind (function 
                    | EvolvedFrom _ -> true 
                    | _ -> false)
            
            match evolutionStep with
            | Some (EvolvedFrom(_, target, delta)) ->
                trace target ((target, delta) :: acc)
            | _ -> acc
        
        trace startEntity []

    /// Get historical facts that were once valid but are now superseded or invalidated.
    let getHistoricalFacts (graph: TemporalKnowledgeGraph.TemporalGraph) =
        graph.GetAllEdges()
        |> List.filter (fun e -> 
            e.Validity.InvalidAt.IsSome || e.SupersededBy.IsSome)
        |> List.map (fun e -> e.Fact)

    /// Detect contradictions valid at a specific time.
    let detectContradictions (graph: TemporalKnowledgeGraph.TemporalGraph) (atTime: DateTime) =
        let facts = graph.GetSnapshot(atTime)
        facts |> List.choose (function
            | Contradicts(s, t, reason) -> Some (s, t, reason)
            | _ -> None)

    /// Calculate stability score for an entity (validity duration / total life duration).
    let calculateStability (graph: TemporalKnowledgeGraph.TemporalGraph) (entityId: string) =
        match graph.GetNode(entityId) with
        | Some node ->
            let totalEnd = 
                match node.Validity.InvalidAt with
                | Some d -> d
                | None -> DateTime.UtcNow
            
            let lifetime = (totalEnd - node.Validity.ValidFrom).TotalSeconds
            let aliveTime = 
                if node.Validity.InvalidAt.IsNone then
                    (DateTime.UtcNow - node.Validity.ValidFrom).TotalSeconds
                else
                    0.0 // It's currently dead
            
            if lifetime = 0.0 then 1.0 else min 1.0 (aliveTime / lifetime)
        | None -> 0.0

    /// Find all facts mentioning an entity within a time period.
    let findFactsInPeriod (graph: TemporalKnowledgeGraph.TemporalGraph) (entity: TarsEntity) (start: DateTime) (finish: DateTime) =
        let entityId = TarsEntity.getId entity
        graph.GetAllEdges()
        |> List.filter (fun e ->
            let isRelevant = 
                TarsEntity.getId (TarsFact.source e.Fact) = entityId ||
                (TarsFact.target e.Fact |> Option.map TarsEntity.getId |> Option.defaultValue "" = entityId)
            
            let overlaps = 
                let vStart = e.Validity.ValidFrom
                let vEnd = e.Validity.InvalidAt |> Option.defaultValue DateTime.MaxValue
                max vStart start < min vEnd finish
            
            isRelevant && overlaps)
        |> List.map (fun e -> e.Fact)
