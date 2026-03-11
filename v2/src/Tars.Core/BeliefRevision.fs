namespace Tars.Core

open System

// =============================================================================
// PHASE 15.2: BELIEF REVISION ENGINE
// =============================================================================
//
// Applies symbolic reflections to update the belief store.
// Implements conflict resolution, belief merging, and cascade updates.
// Reference: docs/3_Roadmap/2_Phases/phase_15_symbolic_reflection.md

module BeliefRevision =

    /// Error that can occur during belief revision
    type RevisionError =
        | BeliefNotFound of beliefId: Guid
        | ContradictionCreated of newBelief: string * conflictsWith: string list
        | InvalidConfidence of value: float
        | CascadeLoop of beliefIds: Guid list
        | InsufficientEvidence of required: int * provided: int
        | JustificationMissing of updateId: Guid
        | ProofInvalid of reason: string

    /// Events emitted during revision for auditing
    type RevisionEvent =
        | BeliefAdded of belief: ReflectionBelief * evidenceCount: int
        | BeliefRevoked of beliefId: Guid * reason: string
        | ConfidenceAdjusted of beliefId: Guid * oldValue: float * newValue: float
        | ContradictionResolved of conflictId: Guid * strategy: string
        | BeliefsMerged of sourceIds: Guid list * resultId: Guid
        | BeliefSplit of originalId: Guid * resultIds: Guid list
        | EvidenceAdded of beliefId: Guid * evidenceId: Guid
        | EvidenceRemoved of beliefId: Guid * evidenceId: Guid
        | CascadeUpdate of triggeredBy: Guid * affectedCount: int

    /// Result of a belief revision operation
    type RevisionResult<'T> = Result<'T * RevisionEvent list, RevisionError>

    /// Simple in-memory belief store for reflection beliefs
    type ReflectionBeliefStore =
        { Beliefs: Map<Guid, ReflectionBelief>
          Relations: Map<Guid, Guid list> // beliefId -> dependent beliefIds
          RevisionHistory: RevisionEvent list }

    module ReflectionBeliefStore =
        let empty =
            { Beliefs = Map.empty
              Relations = Map.empty
              RevisionHistory = [] }

        let add (belief: ReflectionBelief) (store: ReflectionBeliefStore) =
            { store with
                Beliefs = Map.add belief.Id belief store.Beliefs }

        let remove beliefId store =
            { store with
                Beliefs = Map.remove beliefId store.Beliefs }

        let get beliefId (store: ReflectionBeliefStore) = Map.tryFind beliefId store.Beliefs

        let update beliefId (f: ReflectionBelief -> ReflectionBelief) store =
            match Map.tryFind beliefId store.Beliefs with
            | Some belief ->
                { store with
                    Beliefs = Map.add beliefId (f belief) store.Beliefs }
            | None -> store

        let addRelation fromId toId (store: ReflectionBeliefStore) =
            let existing = Map.tryFind fromId store.Relations |> Option.defaultValue []

            { store with
                Relations = Map.add fromId (toId :: existing) store.Relations }

        let getDependents beliefId (store: ReflectionBeliefStore) =
            Map.tryFind beliefId store.Relations |> Option.defaultValue []

        let recordEvent event (store: ReflectionBeliefStore) =
            { store with
                RevisionHistory = event :: store.RevisionHistory }

        let count (store: ReflectionBeliefStore) = store.Beliefs.Count

    // =========================================================================
    // Core Revision Functions
    // =========================================================================

    /// Check for contradictions between a new belief and existing beliefs
    let findContradictions (newBelief: ReflectionBelief) (store: ReflectionBeliefStore) : ReflectionBelief list =
        let statement = newBelief.Statement.ToLowerInvariant()

        store.Beliefs
        |> Map.values
        |> Seq.filter (fun existing ->
            let existingStatement = existing.Statement.ToLowerInvariant()

            (statement.Contains("not")
             && existingStatement.Contains(statement.Replace("not ", "")))
            || (existingStatement.Contains("not")
                && statement.Contains(existingStatement.Replace("not ", ""))))
        |> Seq.toList

    /// Apply an AddBelief update
    let applyAddBelief
        (belief: ReflectionBelief)
        (evidence: ReflectionEvidence list)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        let contradictions = findContradictions belief store

        if not contradictions.IsEmpty then
            FSharp.Core.Error(ContradictionCreated(belief.Statement, contradictions |> List.map (fun b -> b.Statement)))
        else
            let beliefWithEvidence =
                { belief with
                    Evidence = evidence
                    LastUpdated = DateTimeOffset.UtcNow }

            let event = BeliefAdded(beliefWithEvidence, evidence.Length)

            let updatedStore =
                store
                |> ReflectionBeliefStore.add beliefWithEvidence
                |> ReflectionBeliefStore.recordEvent event

            FSharp.Core.Ok(updatedStore, [ event ])

    /// Apply a RevokeBelief update
    let applyRevokeBelief
        (beliefId: Guid)
        (reason: string)
        (replacement: ReflectionBelief option)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        match ReflectionBeliefStore.get beliefId store with
        | None -> FSharp.Core.Error(BeliefNotFound beliefId)
        | Some _ ->
            let events = [ BeliefRevoked(beliefId, reason) ]
            let dependents = ReflectionBeliefStore.getDependents beliefId store

            let cascadeEvents =
                dependents
                |> List.choose (fun depId ->
                    ReflectionBeliefStore.get depId store
                    |> Option.map (fun dep -> ConfidenceAdjusted(depId, dep.Confidence, dep.Confidence * 0.8)))

            let mutable updatedStore = ReflectionBeliefStore.remove beliefId store

            for depId in dependents do
                updatedStore <-
                    ReflectionBeliefStore.update
                        depId
                        (fun b ->
                            { b with
                                Confidence = b.Confidence * 0.8 })
                        updatedStore

            match replacement with
            | Some rep -> updatedStore <- ReflectionBeliefStore.add rep updatedStore
            | None -> ()

            for event in events @ cascadeEvents do
                updatedStore <- ReflectionBeliefStore.recordEvent event updatedStore

            if not dependents.IsEmpty then
                updatedStore <-
                    ReflectionBeliefStore.recordEvent (CascadeUpdate(beliefId, dependents.Length)) updatedStore

            let allEvents =
                events
                @ cascadeEvents
                @ (if dependents.IsEmpty then
                       []
                   else
                       [ CascadeUpdate(beliefId, dependents.Length) ])

            FSharp.Core.Ok(updatedStore, allEvents)

    /// Apply an AdjustConfidence update
    let applyAdjustConfidence
        (beliefId: Guid)
        (newConfidence: float)
        (reason: string)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        if newConfidence < 0.0 || newConfidence > 1.0 then
            FSharp.Core.Error(InvalidConfidence newConfidence)
        else
            match ReflectionBeliefStore.get beliefId store with
            | None -> FSharp.Core.Error(BeliefNotFound beliefId)
            | Some belief ->
                let event = ConfidenceAdjusted(beliefId, belief.Confidence, newConfidence)

                let updatedBelief =
                    { belief with
                        Confidence = newConfidence
                        LastUpdated = DateTimeOffset.UtcNow }

                let updatedStore =
                    store
                    |> ReflectionBeliefStore.add updatedBelief
                    |> ReflectionBeliefStore.recordEvent event

                if newConfidence < 0.1 then
                    applyRevokeBelief beliefId "Confidence dropped below threshold" None updatedStore
                else
                    FSharp.Core.Ok(updatedStore, [ event ])

    /// Apply a ResolveContradiction update
    let applyResolveContradiction
        (resolution: ReflectionConflictResolution)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        let strategyName = $"%A{resolution.Strategy}"
        let event = ContradictionResolved(resolution.ConflictId, strategyName)
        let mutable updatedStore = store |> ReflectionBeliefStore.recordEvent event

        match resolution.Strategy with
        | HighestConfidenceWins ->
            for loser in
                resolution.ConflictingBeliefs
                |> List.filter (fun id -> Some id <> resolution.Winner) do
                updatedStore <- ReflectionBeliefStore.remove loser updatedStore
        | MergeCompatible ->
            for id in resolution.ConflictingBeliefs do
                updatedStore <- ReflectionBeliefStore.remove id updatedStore

            match resolution.MergedBelief with
            | Some merged -> updatedStore <- ReflectionBeliefStore.add merged updatedStore
            | None -> ()
        | _ -> ()

        FSharp.Core.Ok(updatedStore, [ event ])

    /// Apply a MergeBeliefs update
    let applyMergeBeliefs
        (sourceIds: Guid list)
        (merged: ReflectionBelief)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        let allEvidence =
            sourceIds
            |> List.choose (fun id -> ReflectionBeliefStore.get id store)
            |> List.collect (fun b -> b.Evidence)

        let mergedWithEvidence =
            { merged with
                Evidence = allEvidence
                LastUpdated = DateTimeOffset.UtcNow }

        let event = BeliefsMerged(sourceIds, merged.Id)

        let mutable updatedStore = store

        for id in sourceIds do
            updatedStore <- ReflectionBeliefStore.remove id updatedStore

        updatedStore <- ReflectionBeliefStore.add mergedWithEvidence updatedStore
        updatedStore <- ReflectionBeliefStore.recordEvent event updatedStore

        FSharp.Core.Ok(updatedStore, [ event ])

    /// Apply a SplitBelief update
    let applySplitBelief
        (originalId: Guid)
        (refined: ReflectionBelief list)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        match ReflectionBeliefStore.get originalId store with
        | None -> FSharp.Core.Error(BeliefNotFound originalId)
        | Some original ->
            let refinedIds = refined |> List.map (fun b -> b.Id)
            let event = BeliefSplit(originalId, refinedIds)

            let mutable updatedStore = ReflectionBeliefStore.remove originalId store

            for belief in refined do
                let withEvidence =
                    { belief with
                        Evidence = original.Evidence }

                updatedStore <- ReflectionBeliefStore.add withEvidence updatedStore

            updatedStore <- ReflectionBeliefStore.recordEvent event updatedStore

            FSharp.Core.Ok(updatedStore, [ event ])

    /// Apply a single belief update to the store
    let applyUpdate
        (update: ReflectionBeliefUpdate)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        match update with
        | AddBelief(belief, evidence) -> applyAddBelief belief evidence store
        | RevokeBelief(beliefId, reason, replacement) -> applyRevokeBelief beliefId reason replacement store
        | AdjustConfidence(beliefId, _, newConfidence, reason) ->
            applyAdjustConfidence beliefId newConfidence reason store
        | ResolveContradiction resolution -> applyResolveContradiction resolution store
        | MergeBeliefs(sourceIds, merged) -> applyMergeBeliefs sourceIds merged store
        | SplitBelief(originalId, refined) -> applySplitBelief originalId refined store
        | AddEvidence(beliefId, evidence) ->
            match ReflectionBeliefStore.get beliefId store with
            | None -> FSharp.Core.Error(BeliefNotFound beliefId)
            | Some belief ->
                let updated =
                    { belief with
                        Evidence = evidence :: belief.Evidence
                        LastUpdated = DateTimeOffset.UtcNow }

                let event = EvidenceAdded(beliefId, evidence.Id)

                FSharp.Core.Ok(
                    store
                    |> ReflectionBeliefStore.add updated
                    |> ReflectionBeliefStore.recordEvent event,
                    [ event ]
                )
        | RemoveEvidence(beliefId, evidenceId, _) ->
            match ReflectionBeliefStore.get beliefId store with
            | None -> FSharp.Core.Error(BeliefNotFound beliefId)
            | Some belief ->
                let updated =
                    { belief with
                        Evidence = belief.Evidence |> List.filter (fun e -> e.Id <> evidenceId)
                        LastUpdated = DateTimeOffset.UtcNow }

                let event = EvidenceRemoved(beliefId, evidenceId)

                FSharp.Core.Ok(
                    store
                    |> ReflectionBeliefStore.add updated
                    |> ReflectionBeliefStore.recordEvent event,
                    [ event ]
                )

    // =========================================================================
    // High-Level Functions
    // =========================================================================

    /// Apply all belief updates from a reflection
    let applyReflection
        (reflection: SymbolicReflection)
        (store: ReflectionBeliefStore)
        : RevisionResult<ReflectionBeliefStore> =
        let rec applyAll updates currentStore accEvents =
            match updates with
            | [] -> FSharp.Core.Ok(currentStore, accEvents)
            | update :: rest ->
                match applyUpdate update currentStore with
                | FSharp.Core.Error e -> FSharp.Core.Error e
                | FSharp.Core.Ok(newStore, newEvents) -> applyAll rest newStore (accEvents @ newEvents)

        applyAll reflection.BeliefUpdates store []

    /// Validate a reflection before applying it
    let validateReflection (reflection: SymbolicReflection) (store: ReflectionBeliefStore) : Result<unit, string list> =
        let errors = ResizeArray<string>()

        for update in reflection.BeliefUpdates do
            match update with
            | RevokeBelief(beliefId, _, _) ->
                if ReflectionBeliefStore.get beliefId store |> Option.isNone then
                    errors.Add $"Belief %A{beliefId} to revoke does not exist"
            | AdjustConfidence(beliefId, _, newConf, _) ->
                if ReflectionBeliefStore.get beliefId store |> Option.isNone then
                    errors.Add $"Belief %A{beliefId} to adjust does not exist"

                if newConf < 0.0 || newConf > 1.0 then
                    errors.Add $"Invalid confidence value: %f{newConf}"
            | _ -> ()

        for update in reflection.BeliefUpdates do
            let hasJustification =
                reflection.Justifications
                |> List.exists (fun j ->
                    match j.Update, update with
                    | AddBelief(b1, _), AddBelief(b2, _) -> b1.Id = b2.Id
                    | RevokeBelief(id1, _, _), RevokeBelief(id2, _, _) -> id1 = id2
                    | AdjustConfidence(id1, _, _, _), AdjustConfidence(id2, _, _, _) -> id1 = id2
                    | _ -> false)

            if not hasJustification then
                errors.Add(sprintf "Update has no justification")

        if errors.Count = 0 then
            FSharp.Core.Ok()
        else
            FSharp.Core.Error(errors |> Seq.toList)

    /// Get revision history for auditing
    let getRevisionHistory (store: ReflectionBeliefStore) (limit: int option) =
        match limit with
        | Some n -> store.RevisionHistory |> List.truncate n
        | None -> store.RevisionHistory

    /// Find beliefs that might need revision based on age or low confidence
    let findStaleBeliefs (store: ReflectionBeliefStore) (maxAge: TimeSpan) (minConfidence: float) =
        let now = DateTimeOffset.UtcNow

        store.Beliefs
        |> Map.values
        |> Seq.filter (fun b -> (now - b.LastUpdated) > maxAge || b.Confidence < minConfidence)
        |> Seq.toList
