module Tars.Evolution.PromotionPipeline

/// The 7-step CompoundCore loop:
///   Inspect → Extract → Classify → Propose → Validate → Persist → Govern
///
/// Each completed reasoning task feeds into this pipeline. Patterns that
/// recur across tasks are promoted up the staircase, gated by the Grammar Governor.

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

// ─────────────────────────────────────────────────────────────────────
// State: injectable promotion store (recurrence + lineage + weights)
// ─────────────────────────────────────────────────────────────────────

/// Process-global disk-backed store rooted at ~/.tars/promotion. This is the
/// default for production callers; tests inject an `InMemoryPromotionStore`
/// to stay hermetic. The global mutable ConcurrentDictionary state that used
/// to live here — and leaked across tests — now lives inside the store.
let defaultStore : IPromotionStore = DiskPromotionStore() :> IPromotionStore

// ─────────────────────────────────────────────────────────────────────
// Step 1: INSPECT — Analyze completed work artifacts
// ─────────────────────────────────────────────────────────────────────

type TraceArtifact = {
    TaskId: string
    PatternName: string
    PatternTemplate: string
    Context: string
    Score: float
    Timestamp: DateTime
    /// The lower-level code this pattern was abstracted from (for round-trip validation)
    RollbackExpansion: string option
}

let inspect (artifacts: TraceArtifact list) : TraceArtifact list =
    // Filter to valid, scoreable artifacts
    artifacts |> List.filter (fun a -> a.Score > 0.0 && a.PatternName <> "")

// ─────────────────────────────────────────────────────────────────────
// Step 2: EXTRACT — Pull recurring structural patterns
// ─────────────────────────────────────────────────────────────────────

let extractInto (store: IPromotionStore) (artifacts: TraceArtifact list) : RecurrenceRecord list =
    artifacts
    |> List.groupBy (fun a -> a.PatternName)
    |> List.map (fun (name, group) ->
        let existing =
            match store.TryGetRecurrence name with
            | Some r -> r
            | None ->
                { PatternId = Guid.NewGuid().ToString("N").[..7]
                  PatternName = name
                  FirstSeen = DateTime.UtcNow
                  LastSeen = DateTime.UtcNow
                  OccurrenceCount = 0
                  TaskIds = []
                  Contexts = []
                  CurrentLevel = Implementation
                  PromotionHistory = [ (Implementation, DateTime.UtcNow) ]
                  AverageScore = 0.0 }

        let newTaskIds = group |> List.map (fun a -> a.TaskId)
        let newContexts = group |> List.map (fun a -> a.Context)
        let allScores = existing.AverageScore :: (group |> List.map (fun a -> a.Score))
        let avgScore = allScores |> List.average

        let updated =
            { existing with
                LastSeen = DateTime.UtcNow
                OccurrenceCount = existing.OccurrenceCount + group.Length
                TaskIds = (existing.TaskIds @ newTaskIds) |> List.distinct
                Contexts = (existing.Contexts @ newContexts) |> List.distinct
                AverageScore = avgScore }

        store.UpsertRecurrence updated
        updated)

/// Extract using the process-global default store (backward-compatible entry point).
let extract (artifacts: TraceArtifact list) : RecurrenceRecord list =
    extractInto defaultStore artifacts

// ─────────────────────────────────────────────────────────────────────
// Step 3: CLASSIFY — Determine if promotion is warranted
// ─────────────────────────────────────────────────────────────────────

let classify (minOccurrences: int) (record: RecurrenceRecord) : PromotionCandidate option =
    // Only consider patterns that meet minimum occurrence threshold
    if record.OccurrenceCount < minOccurrences then
        None
    else
        match PromotionLevel.next record.CurrentLevel with
        | None -> None  // Already at highest level
        | Some nextLevel ->
            Some {
                Record = record
                ProposedLevel = nextLevel
                Criteria = PromotionCriteria.empty
                Evidence = record.TaskIds |> List.map (fun id -> $"Used in task %s{id}")
                PatternTemplate = ""
                RollbackExpansion = None
            }

/// Classify with probabilistic ranking: candidates are sorted by weighted
/// probability so higher-weight patterns are promoted first.
let classifyWeighted
    (minOccurrences: int)
    (weights: WeightedGrammar.WeightedRule list)
    (records: RecurrenceRecord list)
    : PromotionCandidate list =
    records
    |> List.choose (classify minOccurrences)
    |> List.sortByDescending (fun c ->
        weights
        |> List.tryFind (fun w -> w.PatternId = c.Record.PatternId)
        |> Option.map (fun w -> w.Weight)
        |> Option.defaultValue 0.0)

// ─────────────────────────────────────────────────────────────────────
// Step 4: PROPOSE — Set the proposed level and template
// ─────────────────────────────────────────────────────────────────────

let propose (template: string) (rollback: string option) (candidate: PromotionCandidate) : PromotionCandidate =
    { candidate with
        PatternTemplate = template
        RollbackExpansion = rollback }

// ─────────────────────────────────────────────────────────────────────
// Step 5: VALIDATE — Check promotion criteria
// ─────────────────────────────────────────────────────────────────────

/// Deterministic criteria that can be checked without LLM
let validateDeterministic (existing: RecurrenceRecord list) (candidate: PromotionCandidate) : PromotionCriteria =
    let r = candidate.Record
    { MinOccurrences = r.OccurrenceCount >= 3
      RemovesComplexity = candidate.PatternTemplate.Length > 0
      MoreReadable = true  // Default true, LLM can override
      StableSemantics = r.Contexts |> List.distinct |> List.length <= r.OccurrenceCount
      AutoValidatable = candidate.RollbackExpansion.IsSome
      NoOverlap = not (GrammarGovernor.checkOverlap candidate existing)
      ComposesCleanly = true  // Default true, LLM can override
      ImprovesPlanning = r.AverageScore > 0.6 }

/// Full validation with optional LLM assessment for subjective criteria
let validate
    (existing: RecurrenceRecord list)
    (llmAssessment: PromotionCriteria option)
    (candidate: PromotionCandidate)
    : PromotionCandidate =
    let deterministic = validateDeterministic existing candidate
    let criteria =
        match llmAssessment with
        | Some llm ->
            // Merge: LLM overrides subjective criteria, keep deterministic for objective ones
            { MinOccurrences = deterministic.MinOccurrences  // Always deterministic
              RemovesComplexity = llm.RemovesComplexity
              MoreReadable = llm.MoreReadable
              StableSemantics = deterministic.StableSemantics  // Always deterministic
              AutoValidatable = deterministic.AutoValidatable  // Always deterministic
              NoOverlap = deterministic.NoOverlap  // Always deterministic
              ComposesCleanly = llm.ComposesCleanly
              ImprovesPlanning = llm.ImprovesPlanning }
        | None -> deterministic
    { candidate with Criteria = criteria }

// ─────────────────────────────────────────────────────────────────────
// Step 6: PERSIST — Store the decision and update lineage
// ─────────────────────────────────────────────────────────────────────

let persistInto (store: IPromotionStore) (candidate: PromotionCandidate) (decision: GovernanceDecision) : LineageRecord =
    let lineage = {
        Id = Guid.NewGuid().ToString("N").[..7]
        PatternId = candidate.Record.PatternId
        FromLevel = candidate.Record.CurrentLevel
        ToLevel = candidate.ProposedLevel
        PromotedAt = DateTime.UtcNow
        Criteria = candidate.Criteria
        Decision = decision
        RollbackExpansion = candidate.RollbackExpansion
        DerivedFrom = candidate.Record.TaskIds |> List.tryHead
        PromotedBy = "grammar_governor"
        Confidence = float (PromotionCriteria.score candidate.Criteria) / 8.0
    }

    store.AddLineage lineage

    // If approved, update the recurrence record's level
    match decision with
    | Approve _ ->
        let updated =
            { candidate.Record with
                CurrentLevel = candidate.ProposedLevel
                PromotionHistory =
                    candidate.Record.PromotionHistory
                    @ [ (candidate.ProposedLevel, DateTime.UtcNow) ] }
        store.UpsertRecurrence updated
    | _ -> ()

    // Persist to disk after each decision
    store.Flush ()

    lineage

/// Persist using the process-global default store (backward-compatible entry point).
let persist (candidate: PromotionCandidate) (decision: GovernanceDecision) : LineageRecord =
    persistInto defaultStore candidate decision

// ─────────────────────────────────────────────────────────────────────
// Step 7: GOVERN — Grammar Governor makes final decision
// ─────────────────────────────────────────────────────────────────────

let govern (existing: RecurrenceRecord list) (candidate: PromotionCandidate) : GovernanceDecision =
    GrammarGovernor.evaluate existing candidate

// ─────────────────────────────────────────────────────────────────────
// FULL PIPELINE — Run all 7 steps
// ─────────────────────────────────────────────────────────────────────

type PipelineResult = {
    Candidate: PromotionCandidate
    Decision: GovernanceDecision
    Lineage: LineageRecord
    AuditReport: string
    RoundtripValidation: RoundtripValidation.RoundtripResult option
}

/// Run the full 7-step promotion pipeline on a batch of trace artifacts against
/// a caller-supplied store (inject `defaultStore` for the process-global disk
/// store, or an `InMemoryPromotionStore` for hermetic tests).
/// Uses probabilistic weights to rank candidates and updates weights from outcomes.
let run (store: IPromotionStore) (minOccurrences: int) (artifacts: TraceArtifact list) : PipelineResult list =
    let existing = store.GetRecurrence ()

    // Load probabilistic weights (empty list on first run)
    let mutable weights = store.LoadWeights ()

    // Steps 1-2: Inspect and Extract
    let inspected = artifacts |> inspect
    let records = inspected |> extractInto store

    // Build a lookup of rollback expansions by pattern name
    let rollbackByPattern =
        inspected
        |> List.choose (fun a ->
            a.RollbackExpansion |> Option.map (fun rb -> a.PatternName, rb))
        |> List.distinctBy fst
        |> Map.ofList

    // Ensure all records have weight entries
    let missingRecords =
        records
        |> List.filter (fun r ->
            not (weights |> List.exists (fun w -> w.PatternId = r.PatternId)))
    if not missingRecords.IsEmpty then
        let newWeights =
            missingRecords
            |> List.map (fun r -> (r, GrammarGovernor.score (PromotionCriteria.empty)))
            |> WeightedGrammar.fromRecurrenceRecords WeightedGrammar.defaultConfig
        weights <- weights @ newWeights

    // Step 3: Classify with weighted ranking (higher-weight candidates first)
    let candidates = classifyWeighted minOccurrences weights records

    // Steps 4-7: propose → validate → govern → persist for each candidate
    let results =
        candidates
        |> List.map (fun candidate ->
            let rollback = rollbackByPattern |> Map.tryFind candidate.Record.PatternName
            let candidate =
                candidate
                |> propose candidate.Record.PatternName rollback
                |> validate existing None

            // ONE verdict from the gate: the Grammar Governor's decision plus
            // any round-trip override, decided in a single place. The pipeline
            // never re-inspects SemanticMatch to second-guess the decision.
            let verdict = PromotionGate.decide RoundtripValidation.quickValidate existing candidate
            let decision = verdict.Decision
            let roundtripResult = verdict.Roundtrip

            // Update weight from governance outcome (Bayesian update)
            let success = match decision with Approve _ -> true | _ -> false
            weights <-
                weights |> List.map (fun w ->
                    if w.PatternId = candidate.Record.PatternId then
                        WeightedGrammar.updateWeight WeightedGrammar.defaultConfig w success
                    else w)

            let lineage = persistInto store candidate decision
            let report =
                let govReport = GrammarGovernor.auditReport candidate decision
                match roundtripResult with
                | Some rt -> govReport + "\n" + RoundtripValidation.auditReport rt
                | None -> govReport

            { Candidate = candidate
              Decision = decision
              Lineage = lineage
              AuditReport = report
              RoundtripValidation = roundtripResult })

    // Persist updated weights
    store.SaveWeights weights

    results

/// Get all recurrence records (for inspection/debugging)
let getRecurrenceRecords () : RecurrenceRecord list =
    defaultStore.GetRecurrence ()

/// Get all lineage records
let getLineageRecords () : LineageRecord list =
    defaultStore.GetLineage ()

// ── Root-aware store access (hermetic, parallel-safe) ───────────────
// The functions above read the process-global in-memory store (lazily
// loaded once from ~/.tars). These read/write a caller-supplied
// promotion directory directly, so callers (notably tests) can operate
// on an isolated store without racing the shared one.

/// Read recurrence records straight from a specific promotion directory.
let recurrenceRecordsFrom (promotionDir: string) : RecurrenceRecord list =
    let store = DiskPromotionStore(promotionDir) :> IPromotionStore
    store.GetRecurrence()

/// Read lineage records straight from a specific promotion directory.
let lineageRecordsFrom (promotionDir: string) : LineageRecord list =
    let store = DiskPromotionStore(promotionDir) :> IPromotionStore
    store.GetLineage()

/// Persist recurrence + lineage stores to a specific promotion directory.
let saveStoresTo
    (promotionDir: string)
    (recurrence: RecurrenceRecord list)
    (lineage: LineageRecord list)
    : unit =
    let store = DiskPromotionStore(promotionDir) :> IPromotionStore
    for r in recurrence do
        store.UpsertRecurrence r
    for l in lineage do
        store.AddLineage l
    store.Flush()

/// Get patterns at a specific promotion level
let getPatternsAtLevel (level: PromotionLevel) : RecurrenceRecord list =
    defaultStore.GetRecurrence ()
    |> List.filter (fun r -> r.CurrentLevel = level)
