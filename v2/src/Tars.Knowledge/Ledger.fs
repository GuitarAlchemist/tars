/// TARS Knowledge Ledger - Event-sourced belief storage
/// "Evolution is logged, not forgotten"
namespace Tars.Knowledge

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

/// Interface for persistent ledger storage
type ILedgerStorage =
    abstract member Append: entry: BeliefEventEntry -> Task<Result<unit, string>>
    abstract member GetEvents: since: DateTime option -> Task<BeliefEventEntry list>
    abstract member GetEventsByBelief: beliefId: BeliefId -> Task<BeliefEventEntry list>
    abstract member GetSnapshot: unit -> Task<Belief list>

/// Interface for evidence and ingestion storage (Phase 9)
type IEvidenceStorage =
    abstract member SaveCandidate: candidate: EvidenceCandidate -> Task<Result<unit, string>>
    abstract member SaveProposal: proposal: ProposedAssertion * evidenceId: Guid option -> Task<Result<unit, string>>
    abstract member GetPendingCandidates: unit -> Task<EvidenceCandidate list>
    abstract member GetProposalsByEvidence: evidenceId: Guid -> Task<ProposedAssertion list>

/// In-memory ledger storage (for development/testing)
/// Thread-safe with proper locking for both reads and writes
type InMemoryLedgerStorage() =
    let events = ResizeArray<BeliefEventEntry>()
    let syncLock = obj ()

    // In-memory storage for evidence candidates and proposals
    let candidates = Dictionary<Guid, EvidenceCandidate>()
    let proposals = Dictionary<Guid, ProposedAssertion>()
    let proposalByEvidence = Dictionary<Guid, ResizeArray<Guid>>()
    let candidateLock = obj ()
    let proposalLock = obj ()

    interface ILedgerStorage with
        member _.Append(entry) =
            task {
                lock syncLock (fun () -> events.Add(entry))
                return Ok()
            }

        member _.GetEvents(since) =
            task {
                let snapshot = lock syncLock (fun () -> events |> Seq.toList)

                return
                    snapshot
                    |> List.filter (fun e ->
                        match since with
                        | Some d -> e.Timestamp > d
                        | None -> true)
            }

        member _.GetEventsByBelief(beliefId) =
            task {
                let snapshot = lock syncLock (fun () -> events |> Seq.toList)

                return
                    snapshot
                    |> List.filter (fun e ->
                        match e.Event with
                        | Assert b -> b.Id = beliefId
                        | Retract(id, _, _) -> id = beliefId
                        | Weaken(id, _, _) -> id = beliefId
                        | Strengthen(id, _, _) -> id = beliefId
                        | Link(s, t, _) -> s = beliefId || t = beliefId
                        | Contradict(b1, b2, _) -> b1 = beliefId || b2 = beliefId
                        | SchemaEvolve(_, affected) -> affected |> List.contains beliefId)
            }

        member _.GetSnapshot() =
            task {
                let snapshot = lock syncLock (fun () -> events |> Seq.toList)

                // Replay events to build current state
                let beliefs = Dictionary<BeliefId, Belief>()

                for entry in snapshot do
                    match entry.Event with
                    | Assert belief -> beliefs.[belief.Id] <- belief
                    | Retract(id, _, _) ->
                        match beliefs.TryGetValue(id) with
                        | true, b ->
                            beliefs.[id] <-
                                { b with
                                    InvalidAt = Some entry.Timestamp }
                        | false, _ -> ()
                    | Weaken(id, newConf, _) ->
                        match beliefs.TryGetValue(id) with
                        | true, b -> beliefs.[id] <- { b with Confidence = newConf }
                        | false, _ -> ()
                    | Strengthen(id, newConf, _) ->
                        match beliefs.TryGetValue(id) with
                        | true, b -> beliefs.[id] <- { b with Confidence = newConf }
                        | false, _ -> ()
                    | _ -> ()

                return beliefs.Values |> Seq.toList
            }

    interface IEvidenceStorage with
        member _.SaveCandidate(candidate) =
            task {
                lock candidateLock (fun () -> candidates.[candidate.Id] <- candidate)
                return Ok()
            }

        member _.SaveProposal(proposal, evidenceId) =
            task {
                lock proposalLock (fun () ->
                    proposals.[proposal.Id] <- proposal
                    match evidenceId with
                    | Some id ->
                        if not (proposalByEvidence.ContainsKey(id)) then
                            proposalByEvidence.[id] <- ResizeArray<Guid>()
                        proposalByEvidence.[id].Add(proposal.Id)
                    | None -> ())
                return Ok()
            }

        member _.GetPendingCandidates() =
            task {
                return
                    lock candidateLock (fun () ->
                        candidates.Values
                        |> Seq.filter (fun c -> c.Status = EvidenceStatus.Pending)
                        |> Seq.toList)
            }

        member _.GetProposalsByEvidence(evidenceId) =
            task {
                return
                    lock proposalLock (fun () ->
                        match proposalByEvidence.TryGetValue(evidenceId) with
                        | true, ids ->
                            ids
                            |> Seq.choose (fun id ->
                                match proposals.TryGetValue(id) with
                                | true, proposal -> Some proposal
                                | _ -> None)
                            |> Seq.toList
                        | _ -> [])
            }


/// The Knowledge Ledger - append-only event log for beliefs
/// "Symbols are earned, not assumed"
type KnowledgeLedger(storage: ILedgerStorage) =
    let graph = BeliefGraph()
    let mutable lastSync = DateTime.MinValue
    let graphLock = obj ()

    member private this.UpdateGraphConfidence(beliefId: BeliefId, newConfidence: float) =
        lock graphLock (fun () ->
            match graph.Get(beliefId) with
            | Some belief ->
                let updated =
                    { belief with
                        Confidence = newConfidence }

                graph.Add(updated)
            | None -> ())

    /// Initialize by loading existing beliefs
    member this.Initialize() =
        task {
            let! beliefs = storage.GetSnapshot()

            lock graphLock (fun () ->
                for belief in beliefs do
                    graph.Add(belief))

            lastSync <- DateTime.UtcNow
        }

    /// Assert a new belief (with full provenance)
    member this.Assert(belief: Belief, agentId: AgentId, ?runId: RunId) =
        task {
            let entry = BeliefEventEntry.Create(Assert belief, agentId, ?runId = runId)
            let! result = storage.Append(entry)

            match result with
            | Ok() ->
                lock graphLock (fun () -> graph.Add(belief))
                return Ok belief.Id
            | Error e -> return Error e
        }

    /// Create and assert a belief from components
    member this.AssertTriple
        (subject: string, predicate: RelationType, obj: string, provenance: Provenance, agentId: AgentId, ?runId: RunId)
        =
        task {
            let belief = Belief.create subject predicate obj provenance
            return! this.Assert(belief, agentId, ?runId = runId)
        }

    /// Retract a belief (mark as invalid)
    member this.Retract(beliefId: BeliefId, reason: string, agentId: AgentId, ?runId: RunId) =
        task {
            let entry =
                BeliefEventEntry.Create(Retract(beliefId, reason, agentId), agentId, ?runId = runId)

            let! result = storage.Append(entry)

            match result with
            | Ok() ->
                lock graphLock (fun () -> graph.Invalidate(beliefId))
                return Ok()
            | Error e -> return Error e
        }

    /// Weaken a belief's confidence
    member this.Weaken(beliefId: BeliefId, newConfidence: float, reason: string, agentId: AgentId) =
        task {
            let entry =
                BeliefEventEntry.Create(Weaken(beliefId, newConfidence, reason), agentId)

            let! result = storage.Append(entry)

            match result with
            | Ok() ->
                this.UpdateGraphConfidence(beliefId, newConfidence)
                return Ok()
            | Error e -> return Error e
        }

    /// Strengthen a belief's confidence
    member this.Strengthen(beliefId: BeliefId, newConfidence: float, reason: string, agentId: AgentId) =
        task {
            let entry =
                BeliefEventEntry.Create(Strengthen(beliefId, newConfidence, reason), agentId)

            let! result = storage.Append(entry)

            match result with
            | Ok() ->
                this.UpdateGraphConfidence(beliefId, newConfidence)
                return Ok()
            | Error e -> return Error e
        }

    /// Mark two beliefs as contradicting
    member this.MarkContradiction(belief1: BeliefId, belief2: BeliefId, explanation: string, agentId: AgentId) =
        task {
            let entry =
                BeliefEventEntry.Create(Contradict(belief1, belief2, explanation), agentId)

            let! result = storage.Append(entry)

            match result with
            | Ok() ->
                lock graphLock (fun () -> graph.MarkContradiction(belief1, belief2))
                return Ok()
            | Error e -> return Error e
        }

    /// Get a belief by ID
    member this.Get(beliefId: BeliefId) : Belief option =
        lock graphLock (fun () -> graph.Get(beliefId))

    /// Query beliefs by pattern
    member this.Query(?subject: string, ?predicate: RelationType, ?obj: string) : Belief seq =
        lock graphLock (fun () ->
            graph.Query(
                ?subject = (subject |> Option.map EntityId),
                ?predicate = predicate,
                ?obj = (obj |> Option.map EntityId)
            )
            |> Seq.toList)
        :> seq<_>

    /// Get neighborhood around an entity
    member this.GetNeighborhood(entity: string, depth: int) : Belief seq =
        lock graphLock (fun () -> graph.GetNeighborhood(EntityId entity, depth) |> Seq.toList) :> seq<_>

    /// Get all contradictions
    member this.GetContradictions() : (Belief * Belief) seq =
        lock graphLock (fun () -> graph.GetContradictions() |> Seq.toList) :> seq<_>

    /// Find path between entities
    member this.FindPath(from: string, to': string, maxHops: int) : Belief list option =
        lock graphLock (fun () -> graph.FindPath(EntityId from, EntityId to', maxHops))

    /// Get full event history for a belief
    member this.GetHistory(beliefId: BeliefId) = storage.GetEventsByBelief(beliefId)


    /// Get statistics
    member this.Stats() =
        lock graphLock (fun () -> graph.Stats())

    /// Get the in-memory graph for direct querying
    member this.Graph = graph

    /// Get the underlying storage
    member this.Storage = storage

/// Factory for creating ledgers
module KnowledgeLedger =
    /// Create an in-memory ledger (for testing)
    let createInMemory () =
        KnowledgeLedger(InMemoryLedgerStorage())

    /// Initialize and return a ledger
    let initialize (ledger: KnowledgeLedger) =
        task {
            do! ledger.Initialize()
            return ledger
        }
