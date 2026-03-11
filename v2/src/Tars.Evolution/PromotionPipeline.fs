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
// State: persistent recurrence and lineage stores
// ─────────────────────────────────────────────────────────────────────

let private recurrenceStore =
    System.Collections.Concurrent.ConcurrentDictionary<string, RecurrenceRecord>()

let private lineageStore =
    System.Collections.Concurrent.ConcurrentDictionary<string, LineageRecord>()

let private jsonOptions =
    let o = JsonSerializerOptions(JsonSerializerDefaults.General)
    o.Converters.Add(JsonFSharpConverter())
    o.WriteIndented <- true
    o

let private getPromotionDir () =
    let dir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".tars", "promotion")
    if not (Directory.Exists dir) then
        Directory.CreateDirectory dir |> ignore
    dir

/// Save recurrence and lineage stores to disk.
let save () =
    try
        let dir = getPromotionDir ()
        let recurrencePath = Path.Combine(dir, "recurrence.json")
        let lineagePath = Path.Combine(dir, "lineage.json")

        let recurrenceData = recurrenceStore.Values |> Seq.toList
        let lineageData = lineageStore.Values |> Seq.toList

        File.WriteAllText(recurrencePath, JsonSerializer.Serialize(recurrenceData, jsonOptions))
        File.WriteAllText(lineagePath, JsonSerializer.Serialize(lineageData, jsonOptions))
        Ok (recurrenceData.Length, lineageData.Length)
    with ex ->
        Error $"Failed to save promotion state: {ex.Message}"

/// Load recurrence and lineage stores from disk.
let load () =
    try
        let dir = getPromotionDir ()
        let recurrencePath = Path.Combine(dir, "recurrence.json")
        let lineagePath = Path.Combine(dir, "lineage.json")

        if File.Exists recurrencePath then
            let json = File.ReadAllText(recurrencePath)
            let records = JsonSerializer.Deserialize<RecurrenceRecord list>(json, jsonOptions)
            for r in records do
                recurrenceStore.[r.PatternName] <- r

        if File.Exists lineagePath then
            let json = File.ReadAllText(lineagePath)
            let records = JsonSerializer.Deserialize<LineageRecord list>(json, jsonOptions)
            for r in records do
                lineageStore.[r.Id] <- r

        Ok (recurrenceStore.Count, lineageStore.Count)
    with ex ->
        Error $"Failed to load promotion state: {ex.Message}"

/// Initialize: load from disk on first use.
let private initialized =
    lazy (
        match load () with
        | Ok (r, l) ->
            if r > 0 || l > 0 then
                Console.Error.WriteLine($"[Promotion] Loaded {r} recurrence records, {l} lineage records from disk")
        | Error err ->
            Console.Error.WriteLine($"[Promotion] {err}")
    )

let private ensureLoaded () = initialized.Force()

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

let extract (artifacts: TraceArtifact list) : RecurrenceRecord list =
    ensureLoaded ()
    artifacts
    |> List.groupBy (fun a -> a.PatternName)
    |> List.map (fun (name, group) ->
        let existing =
            match recurrenceStore.TryGetValue(name) with
            | true, r -> r
            | false, _ ->
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

        recurrenceStore.[name] <- updated
        updated)

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

let persist (candidate: PromotionCandidate) (decision: GovernanceDecision) : LineageRecord =
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

    lineageStore.[lineage.Id] <- lineage

    // If approved, update the recurrence record's level
    match decision with
    | Approve _ ->
        let updated =
            { candidate.Record with
                CurrentLevel = candidate.ProposedLevel
                PromotionHistory =
                    candidate.Record.PromotionHistory
                    @ [ (candidate.ProposedLevel, DateTime.UtcNow) ] }
        recurrenceStore.[candidate.Record.PatternName] <- updated
    | _ -> ()

    // Persist to disk after each decision
    save () |> ignore

    lineage

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

/// Run the full 7-step promotion pipeline on a batch of trace artifacts
let run (minOccurrences: int) (artifacts: TraceArtifact list) : PipelineResult list =
    ensureLoaded ()
    let existing = recurrenceStore.Values |> Seq.toList

    // Steps 1-2: Inspect and Extract
    let inspected = artifacts |> inspect
    let records = inspected |> extract

    // Build a lookup of rollback expansions by pattern name
    // (take the first non-None rollback for each pattern)
    let rollbackByPattern =
        inspected
        |> List.choose (fun a ->
            a.RollbackExpansion |> Option.map (fun rb -> a.PatternName, rb))
        |> List.distinctBy fst
        |> Map.ofList

    // Steps 3-7: For each record, classify → propose → validate → govern → persist
    records
    |> List.choose (fun record ->
        classify minOccurrences record
        |> Option.map (fun candidate ->
            let rollback = rollbackByPattern |> Map.tryFind record.PatternName
            let candidate =
                candidate
                |> propose record.PatternName rollback
                |> validate existing None

            let rawDecision = govern existing candidate

            // Round-trip validation: if approved, verify the abstraction
            // can expand and re-abstract without semantic loss
            let roundtripResult, decision =
                match rawDecision with
                | Approve _ ->
                    let rtResult = RoundtripValidation.quickValidate candidate
                    if rtResult.Passed then
                        Some rtResult, rawDecision
                    else
                        let reason =
                            sprintf "Round-trip validation failed (semantic match: %.2f). %s"
                                rtResult.SemanticMatch
                                (rtResult.Issues |> String.concat "; ")
                        Some rtResult, Reject reason
                | _ ->
                    None, rawDecision

            let lineage = persist candidate decision
            let report =
                let govReport = GrammarGovernor.auditReport candidate decision
                match roundtripResult with
                | Some rt -> govReport + "\n" + RoundtripValidation.auditReport rt
                | None -> govReport

            { Candidate = candidate
              Decision = decision
              Lineage = lineage
              AuditReport = report
              RoundtripValidation = roundtripResult }))

/// Get all recurrence records (for inspection/debugging)
let getRecurrenceRecords () : RecurrenceRecord list =
    ensureLoaded ()
    recurrenceStore.Values |> Seq.toList

/// Get all lineage records
let getLineageRecords () : LineageRecord list =
    ensureLoaded ()
    lineageStore.Values |> Seq.toList

/// Get patterns at a specific promotion level
let getPatternsAtLevel (level: PromotionLevel) : RecurrenceRecord list =
    ensureLoaded ()
    recurrenceStore.Values
    |> Seq.filter (fun r -> r.CurrentLevel = level)
    |> Seq.toList
