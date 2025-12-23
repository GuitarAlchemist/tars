namespace Tars.Core

// BeliefGraph - Graph-based storage for agent beliefs and principles
// Phase 2.5.2 of the TARS v2 Roadmap

open System

/// Edge type representing relationships between beliefs
type BeliefRelation =
    | SupportsBy of confidence: float
    | ContradictedBy of evidence: string
    | RefinedFrom
    | GeneralizedFrom
    | AppliesTo of context: string

/// Edge in the belief graph
type BeliefEdge =
    { SourceId: Guid
      TargetId: Guid
      Relation: BeliefRelation
      CreatedAt: DateTime }

/// The belief graph structure
type BeliefGraph =
    { Beliefs: Map<Guid, Belief>
      Edges: BeliefEdge list
      CreatedAt: DateTime
      LastModified: DateTime }

module BeliefGraph =

    /// Create an empty belief graph
    let empty () =
        { Beliefs = Map.empty
          Edges = []
          CreatedAt = DateTime.UtcNow
          LastModified = DateTime.UtcNow }

    /// Add a belief to the graph
    let addBelief (belief: Belief) (graph: BeliefGraph) =
        { graph with
            Beliefs = graph.Beliefs |> Map.add belief.Id belief
            LastModified = DateTime.UtcNow }

    /// Remove a belief from the graph
    let removeBelief (beliefId: Guid) (graph: BeliefGraph) =
        { graph with
            Beliefs = graph.Beliefs |> Map.remove beliefId
            Edges =
                graph.Edges
                |> List.filter (fun e -> e.SourceId <> beliefId && e.TargetId <> beliefId)
            LastModified = DateTime.UtcNow }

    /// Add an edge between beliefs
    let addEdge (sourceId: Guid) (targetId: Guid) (relation: BeliefRelation) (graph: BeliefGraph) =
        let edge =
            { SourceId = sourceId
              TargetId = targetId
              Relation = relation
              CreatedAt = DateTime.UtcNow }

        { graph with
            Edges = edge :: graph.Edges
            LastModified = DateTime.UtcNow }

    /// Get a belief by ID
    let tryGetBelief (id: Guid) (graph: BeliefGraph) = graph.Beliefs |> Map.tryFind id

    /// Get all beliefs with a specific status
    let getBeliefsByStatus (status: EpistemicStatus) (graph: BeliefGraph) =
        graph.Beliefs
        |> Map.values
        |> Seq.filter (fun b -> b.Status = status)
        |> Seq.toList

    /// Get all beliefs in a context
    let getBeliefsByContext (context: string) (graph: BeliefGraph) =
        graph.Beliefs
        |> Map.values
        |> Seq.filter (fun b -> b.Context.Contains(context, StringComparison.OrdinalIgnoreCase))
        |> Seq.toList

    /// Get beliefs that support a given belief
    let getSupportingBeliefs (beliefId: Guid) (graph: BeliefGraph) =
        graph.Edges
        |> List.filter (fun e ->
            e.TargetId = beliefId
            && match e.Relation with
               | SupportsBy _ -> true
               | _ -> false)
        |> List.choose (fun e -> graph.Beliefs |> Map.tryFind e.SourceId)

    /// Get beliefs that contradict a given belief
    let getContradictingBeliefs (beliefId: Guid) (graph: BeliefGraph) =
        graph.Edges
        |> List.filter (fun e ->
            e.TargetId = beliefId
            && match e.Relation with
               | ContradictedBy _ -> true
               | _ -> false)
        |> List.choose (fun e -> graph.Beliefs |> Map.tryFind e.SourceId)

    /// Update belief status
    let updateStatus (beliefId: Guid) (newStatus: EpistemicStatus) (graph: BeliefGraph) =
        match graph.Beliefs |> Map.tryFind beliefId with
        | Some belief ->
            let updated =
                { belief with
                    Status = newStatus
                    LastVerified = DateTime.UtcNow }

            { graph with
                Beliefs = graph.Beliefs |> Map.add beliefId updated
                LastModified = DateTime.UtcNow }
        | None -> graph

    /// Update belief confidence
    let updateConfidence (beliefId: Guid) (newConfidence: float) (graph: BeliefGraph) =
        match graph.Beliefs |> Map.tryFind beliefId with
        | Some belief ->
            let updated =
                { belief with
                    Confidence = newConfidence
                    LastVerified = DateTime.UtcNow }

            { graph with
                Beliefs = graph.Beliefs |> Map.add beliefId updated
                LastModified = DateTime.UtcNow }
        | None -> graph

    /// Get count of beliefs by status
    let countByStatus (graph: BeliefGraph) =
        graph.Beliefs |> Map.values |> Seq.countBy (fun b -> b.Status) |> Map.ofSeq

    /// Get high-confidence principles
    let getPrinciples (minConfidence: float) (graph: BeliefGraph) =
        graph.Beliefs
        |> Map.values
        |> Seq.filter (fun b -> b.Status = UniversalPrinciple && b.Confidence >= minConfidence)
        |> Seq.toList

    /// Find contradictions in the graph
    let findContradictions (graph: BeliefGraph) =
        graph.Edges
        |> List.filter (fun e ->
            match e.Relation with
            | ContradictedBy _ -> true
            | _ -> false)
        |> List.choose (fun e ->
            match graph.Beliefs |> Map.tryFind e.SourceId, graph.Beliefs |> Map.tryFind e.TargetId with
            | Some src, Some tgt -> Some(src, tgt, e.Relation)
            | _ -> None)
