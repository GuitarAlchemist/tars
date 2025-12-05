namespace Tars.Core

open System

/// Temporal graph implementation for Graphiti
module TemporalKnowledgeGraph =

    /// Enhanced node with temporal tracking
    type TemporalNode = {
        Entity: TarsEntity
        Validity: TemporalValidity
        CommunityId: string option
    }

    /// Enhanced edge with temporal tracking
    type TemporalEdge = {
        Fact: TarsFact
        Id: Guid
        Validity: TemporalValidity
        SupersededBy: Guid option
    }

    /// Graph event for temporal replay
    type GraphEvent =
        | NodeAdded of TemporalNode
        | EdgeAdded of TemporalEdge
        | EdgeInvalidated of edgeId: Guid * invalidAt: DateTime * reason: string
        | EdgeSuperseded of oldEdgeId: Guid * newEdgeId: Guid * at: DateTime

    /// In-memory temporal graph
    type TemporalGraph() =
        let mutable nodes = Map.empty<string, TemporalNode> // Keyed by Canonical ID
        let mutable edges = Map.empty<Guid, TemporalEdge>
        let mutable edgesBySource = Map.empty<string, Guid list>
        
        /// Get canonical ID for an entity
        let getEntityId (entity: TarsEntity) = TarsEntity.getId entity

        /// Add a node if it doesn't exist
        member this.AddNode(entity: TarsEntity) =
            let id = getEntityId entity
            if not (nodes.ContainsKey id) then
                let node = {
                    Entity = entity
                    Validity = TemporalValidityOps.now()
                    CommunityId = None
                }
                nodes <- nodes |> Map.add id node
                // In a real event-sourced system, we'd emit an event here
            id

        /// Add a fact (edge) and handle invalidation logic
        member this.AddFact(fact: TarsFact) =
            let sourceId = this.AddNode(TarsFact.source fact)
            let targetId = 
                match TarsFact.target fact with
                | Some t -> Some (this.AddNode t)
                | None -> None
            
            let edgeId = Guid.NewGuid()
            let newEdge = {
                Fact = fact
                Id = edgeId
                Validity = TemporalValidityOps.now()
                SupersededBy = None
            }
            
            edges <- edges |> Map.add edgeId newEdge
            
            let currentSourceEdges = edgesBySource |> Map.tryFind sourceId |> Option.defaultValue []
            edgesBySource <- edgesBySource |> Map.add sourceId (edgeId :: currentSourceEdges)

            // Handle invalidation logic
            this.ProcessInvalidation(newEdge)
            
            edgeId

        /// Process invalidation rules based on new edge
        member private this.ProcessInvalidation(newEdge: TemporalEdge) =
            match newEdge.Fact with
            | EvolvedFrom(source, target, _) ->
                // If Source evolved from Target, Target is effectively superseded by Source
                // We should invalidate edges where Target is the *Subject* (Source) if they are stateful?
                // Or maybe we just mark the Target entity itself as superseded?
                // For now, let's look for specific previous versions of this fact.
                
                // Common pattern: AgentBelief updated.
                // New belief EvolvedFrom Old belief.
                // We should mark the Old belief node as invalid? 
                // Or just facts originating from it?
                
                // Let's invalidate the "Target" entity's validity if it's an EvolvedFrom relationship
                let targetId = getEntityId target
                match nodes.TryFind targetId with
                | Some targetNode ->
                    let updatedValidity = TemporalValidityOps.invalidate targetNode.Validity
                    nodes <- nodes |> Map.add targetId { targetNode with Validity = updatedValidity }
                | None -> ()
                
            | Contradicts(source, target, _) ->
                // Explicit contradiction.
                // If Source contradicts Target, and Source is newer/higher confidence, maybe invalidate Target?
                // This is risky to do automatically without more logic.
                // For Phase 2.5, we'll focus on EvolvedFrom.
                ()
                
            | _ -> ()

        /// Get a node by ID
        member this.GetNode(id: string) =
            nodes |> Map.tryFind id

        /// Get valid edges at a specific time
        member this.GetSnapshot(at: DateTime) =
            edges.Values
            |> Seq.filter (fun e -> 
                // Edge must be valid
                let edgeValid = TemporalValidityOps.isValidAt at e.Validity
                
                // Source node must be valid
                let sourceValid = 
                    match nodes.TryFind (getEntityId (TarsFact.source e.Fact)) with
                    | Some n -> TemporalValidityOps.isValidAt at n.Validity
                    | None -> false
                    
                // Target node (if any) must be valid
                // Exception: For EvolvedFrom, the target is expected to be invalid (superseded), so we ignore its validity.
                let targetValid = 
                    match e.Fact with
                    | EvolvedFrom _ -> true
                    | _ ->
                        match TarsFact.target e.Fact with
                        | Some t -> 
                            match nodes.TryFind (getEntityId t) with
                            | Some n -> TemporalValidityOps.isValidAt at n.Validity
                            | None -> false
                        | None -> true // No target, so valid if source is valid
                    
                edgeValid && sourceValid && targetValid
            )
            |> Seq.map (fun e -> e.Fact)
            |> Seq.toList

        /// Get all current valid facts
        member this.GetCurrentFacts() =
            this.GetSnapshot(DateTime.UtcNow)
