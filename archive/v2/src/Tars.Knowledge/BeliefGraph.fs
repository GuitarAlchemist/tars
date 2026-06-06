/// TARS Belief Graph - In-memory graph view of beliefs
/// This is a materialized view for fast traversal, not the source of truth
namespace Tars.Knowledge

open System
open System.Collections.Generic

/// In-memory belief graph for fast queries
/// The Postgres ledger is the source of truth; this is a view
type BeliefGraph() =
    let beliefs = Dictionary<BeliefId, Belief>()
    let bySubject = Dictionary<EntityId, HashSet<BeliefId>>()
    let byObject = Dictionary<EntityId, HashSet<BeliefId>>()
    let byPredicate = Dictionary<RelationType, HashSet<BeliefId>>()
    let contradictions = HashSet<BeliefId * BeliefId>()

    /// Add a belief to the graph
    member this.Add(belief: Belief) =
        beliefs.[belief.Id] <- belief

        // Index by subject
        if not (bySubject.ContainsKey(belief.Subject)) then
            bySubject.[belief.Subject] <- HashSet<BeliefId>()

        bySubject.[belief.Subject].Add(belief.Id) |> ignore

        // Index by object
        if not (byObject.ContainsKey(belief.Object)) then
            byObject.[belief.Object] <- HashSet<BeliefId>()

        byObject.[belief.Object].Add(belief.Id) |> ignore

        // Index by predicate
        if not (byPredicate.ContainsKey(belief.Predicate)) then
            byPredicate.[belief.Predicate] <- HashSet<BeliefId>()

        byPredicate.[belief.Predicate].Add(belief.Id) |> ignore

    /// Get a belief by ID
    member this.Get(id: BeliefId) : Belief option =
        match beliefs.TryGetValue(id) with
        | true, b -> Some b
        | false, _ -> None

    /// Get all beliefs about a subject
    member this.GetBySubject(subject: EntityId) : Belief seq =
        match bySubject.TryGetValue(subject) with
        | true, ids -> ids |> Seq.choose (fun id -> this.Get(id))
        | false, _ -> Seq.empty

    /// Get all beliefs with a given object
    member this.GetByObject(obj: EntityId) : Belief seq =
        match byObject.TryGetValue(obj) with
        | true, ids -> ids |> Seq.choose (fun id -> this.Get(id))
        | false, _ -> Seq.empty

    /// Get all beliefs of a relation type
    member this.GetByPredicate(predicate: RelationType) : Belief seq =
        match byPredicate.TryGetValue(predicate) with
        | true, ids -> ids |> Seq.choose (fun id -> this.Get(id))
        | false, _ -> Seq.empty

    /// Find beliefs matching a pattern (None = wildcard)
    member this.Query(?subject: EntityId, ?predicate: RelationType, ?obj: EntityId) : Belief seq =
        let allBeliefs = beliefs.Values |> Seq.filter (fun b -> b.IsValid)

        allBeliefs
        |> Seq.filter (fun b ->
            let matchSubject =
                subject |> Option.map (fun s -> b.Subject = s) |> Option.defaultValue true

            let matchPredicate =
                predicate |> Option.map (fun p -> b.Predicate = p) |> Option.defaultValue true

            let matchObject =
                obj |> Option.map (fun o -> b.Object = o) |> Option.defaultValue true

            matchSubject && matchPredicate && matchObject)

    /// Mark two beliefs as contradicting
    member this.MarkContradiction(id1: BeliefId, id2: BeliefId) =
        // Store in canonical order
        let pair = if id1.Value < id2.Value then (id1, id2) else (id2, id1)
        contradictions.Add(pair) |> ignore

    /// Get all contradictions
    member this.GetContradictions() : (Belief * Belief) seq =
        contradictions
        |> Seq.choose (fun (id1, id2) ->
            match this.Get(id1), this.Get(id2) with
            | Some b1, Some b2 -> Some(b1, b2)
            | _ -> None)

    /// Find path between two entities (BFS up to maxHops)
    member this.FindPath(from: EntityId, to': EntityId, maxHops: int) : Belief list option =
        let visited = HashSet<EntityId>()
        let queue = Queue<EntityId * Belief list>()
        queue.Enqueue((from, []))

        let rec search () =
            if queue.Count = 0 then
                None
            else
                let (current, path) = queue.Dequeue()

                if current = to' then
                    Some(List.rev path)
                elif path.Length >= maxHops then
                    search ()
                elif visited.Contains(current) then
                    search ()
                else
                    visited.Add(current) |> ignore

                    // Get outgoing beliefs (current is subject)
                    for belief in this.GetBySubject(current) do
                        if belief.IsValid && not (visited.Contains(belief.Object)) then
                            queue.Enqueue((belief.Object, belief :: path))

                    // Get incoming beliefs (current is object)
                    for belief in this.GetByObject(current) do
                        if belief.IsValid && not (visited.Contains(belief.Subject)) then
                            queue.Enqueue((belief.Subject, belief :: path))

                    search ()

        search ()

    /// Get neighborhood around an entity
    member this.GetNeighborhood(entity: EntityId, depth: int) : Belief seq =
        let visited = HashSet<BeliefId>()
        let result = ResizeArray<Belief>()
        let queue = Queue<EntityId * int>()
        queue.Enqueue((entity, 0))

        while queue.Count > 0 do
            let (current, d) = queue.Dequeue()

            if d < depth then
                // Get outgoing
                for belief in this.GetBySubject(current) do
                    if belief.IsValid && not (visited.Contains(belief.Id)) then
                        visited.Add(belief.Id) |> ignore
                        result.Add(belief)
                        queue.Enqueue((belief.Object, d + 1))

                // Get incoming
                for belief in this.GetByObject(current) do
                    if belief.IsValid && not (visited.Contains(belief.Id)) then
                        visited.Add(belief.Id) |> ignore
                        result.Add(belief)
                        queue.Enqueue((belief.Subject, d + 1))

        result :> seq<_>

    /// Invalidate a belief
    member this.Invalidate(id: BeliefId) =
        match beliefs.TryGetValue(id) with
        | true, belief ->
            let invalidated =
                { belief with
                    InvalidAt = Some DateTime.UtcNow }

            beliefs.[id] <- invalidated
        | false, _ -> ()

    /// Count valid beliefs
    member this.Count = beliefs.Values |> Seq.filter (fun b -> b.IsValid) |> Seq.length

    /// Count total beliefs (including invalidated)
    member this.TotalCount = beliefs.Count

    /// Get statistics
    member this.Stats() =
        let valid = beliefs.Values |> Seq.filter (fun b -> b.IsValid) |> Seq.toList

        let byType =
            valid
            |> List.groupBy (fun b -> b.Predicate)
            |> List.map (fun (p, bs) -> (p, bs.Length))

        let avgConfidence =
            if valid.IsEmpty then
                0.0
            else
                valid |> List.averageBy (fun b -> b.Confidence)

        {| ValidBeliefs = valid.Length
           TotalBeliefs = beliefs.Count
           Contradictions = contradictions.Count
           ByPredicate = byType
           AverageConfidence = avgConfidence
           UniqueSubjects = bySubject.Count
           UniqueObjects = byObject.Count |}

    /// Clear all data
    member this.Clear() =
        beliefs.Clear()
        bySubject.Clear()
        byObject.Clear()
        byPredicate.Clear()
        contradictions.Clear()
