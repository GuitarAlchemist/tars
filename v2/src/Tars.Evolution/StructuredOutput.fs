module Tars.Evolution.StructuredOutput

/// Strict output schemas for the promotion pipeline.
/// Every pipeline output is structured JSON — no prose, no "chat vapor."
/// This enables headless automation: CI pipelines, nightly audits, PR summaries.

open System
open System.Text.Json
open System.Text.Json.Serialization

// ─────────────────────────────────────────────────────────────────────
// Output DTOs — strict schemas for every pipeline stage
// ─────────────────────────────────────────────────────────────────────

/// Structured output for a single recurrence observation
[<CLIMutable>]
type RecurrenceOutput = {
    [<JsonPropertyName("pattern_id")>] PatternId: string
    [<JsonPropertyName("pattern_name")>] PatternName: string
    [<JsonPropertyName("occurrence_count")>] OccurrenceCount: int
    [<JsonPropertyName("current_level")>] CurrentLevel: string
    [<JsonPropertyName("avg_score")>] AvgScore: float
    [<JsonPropertyName("first_seen")>] FirstSeen: string
    [<JsonPropertyName("last_seen")>] LastSeen: string
    [<JsonPropertyName("task_ids")>] TaskIds: string list
}

/// Structured output for a governance decision
[<CLIMutable>]
type GovernanceOutput = {
    [<JsonPropertyName("pattern_id")>] PatternId: string
    [<JsonPropertyName("pattern_name")>] PatternName: string
    [<JsonPropertyName("from_level")>] FromLevel: string
    [<JsonPropertyName("to_level")>] ToLevel: string
    [<JsonPropertyName("decision")>] Decision: string  // "approve" | "reject" | "defer"
    [<JsonPropertyName("reason")>] Reason: string
    [<JsonPropertyName("criteria_score")>] CriteriaScore: int
    [<JsonPropertyName("criteria_met")>] CriteriaMet: string list
    [<JsonPropertyName("criteria_missed")>] CriteriaMissed: string list
    [<JsonPropertyName("has_rollback")>] HasRollback: bool
    [<JsonPropertyName("confidence")>] Confidence: float
    [<JsonPropertyName("timestamp")>] Timestamp: string
}

/// Structured output for a complete pipeline run
[<CLIMutable>]
type PipelineRunOutput = {
    [<JsonPropertyName("run_id")>] RunId: string
    [<JsonPropertyName("timestamp")>] Timestamp: string
    [<JsonPropertyName("artifacts_processed")>] ArtifactsProcessed: int
    [<JsonPropertyName("patterns_observed")>] PatternsObserved: int
    [<JsonPropertyName("candidates_evaluated")>] CandidatesEvaluated: int
    [<JsonPropertyName("approved")>] Approved: GovernanceOutput list
    [<JsonPropertyName("rejected")>] Rejected: GovernanceOutput list
    [<JsonPropertyName("deferred")>] Deferred: GovernanceOutput list
    [<JsonPropertyName("library_size")>] LibrarySize: int
    [<JsonPropertyName("avg_fitness")>] AvgFitness: float
}

/// Structured output for a retroaction cycle
[<CLIMutable>]
type CycleOutput = {
    [<JsonPropertyName("cycle_id")>] CycleId: string
    [<JsonPropertyName("problem_title")>] ProblemTitle: string
    [<JsonPropertyName("success")>] Success: bool
    [<JsonPropertyName("score")>] Score: float
    [<JsonPropertyName("validation_passed")>] ValidationPassed: bool
    [<JsonPropertyName("pattern_used")>] PatternUsed: string option
    [<JsonPropertyName("pattern_compiled")>] PatternCompiled: bool
    [<JsonPropertyName("improvements_found")>] ImprovementsFound: int
    [<JsonPropertyName("promotions")>] Promotions: GovernanceOutput list
    [<JsonPropertyName("library_size")>] LibrarySize: int
    [<JsonPropertyName("curriculum_mastery")>] CurriculumMastery: float
    [<JsonPropertyName("timestamp")>] Timestamp: string
}

// ─────────────────────────────────────────────────────────────────────
// Converters: internal types → strict output DTOs
// ─────────────────────────────────────────────────────────────────────

let private jsonOptions =
    let o = JsonSerializerOptions(JsonSerializerDefaults.General)
    o.Converters.Add(JsonFSharpConverter())
    o.WriteIndented <- true
    o

let private criteriaLabels (c: PromotionCriteria) =
    let all =
        [ (c.MinOccurrences, "min_occurrences")
          (c.RemovesComplexity, "removes_complexity")
          (c.MoreReadable, "more_readable")
          (c.StableSemantics, "stable_semantics")
          (c.AutoValidatable, "auto_validatable")
          (c.NoOverlap, "no_overlap")
          (c.ComposesCleanly, "composes_cleanly")
          (c.ImprovesPlanning, "improves_planning") ]
    let met = all |> List.filter fst |> List.map snd
    let missed = all |> List.filter (fst >> not) |> List.map snd
    (met, missed)

/// Convert a RecurrenceRecord to structured output
let fromRecurrence (r: RecurrenceRecord) : RecurrenceOutput =
    { PatternId = r.PatternId
      PatternName = r.PatternName
      OccurrenceCount = r.OccurrenceCount
      CurrentLevel = PromotionLevel.label r.CurrentLevel
      AvgScore = Math.Round(r.AverageScore, 3)
      FirstSeen = r.FirstSeen.ToString("o")
      LastSeen = r.LastSeen.ToString("o")
      TaskIds = r.TaskIds }

/// Convert a PipelineResult to structured governance output
let fromPipelineResult (r: PromotionPipeline.PipelineResult) : GovernanceOutput =
    let decisionStr, reason =
        match r.Decision with
        | Approve reason -> "approve", reason
        | Reject reason -> "reject", reason
        | Defer reason -> "defer", reason
    let met, missed = criteriaLabels r.Candidate.Criteria
    { PatternId = r.Candidate.Record.PatternId
      PatternName = r.Candidate.Record.PatternName
      FromLevel = PromotionLevel.label r.Candidate.Record.CurrentLevel
      ToLevel = PromotionLevel.label r.Candidate.ProposedLevel
      Decision = decisionStr
      Reason = reason
      CriteriaScore = PromotionCriteria.score r.Candidate.Criteria
      CriteriaMet = met
      CriteriaMissed = missed
      HasRollback = r.Candidate.RollbackExpansion.IsSome
      Confidence = r.Lineage.Confidence
      Timestamp = r.Lineage.PromotedAt.ToString("o") }

/// Convert a batch of pipeline results to a structured run output
let fromPipelineRun (results: PromotionPipeline.PipelineResult list) (artifactCount: int) : PipelineRunOutput =
    let outputs = results |> List.map fromPipelineResult
    let records = PromotionPipeline.getRecurrenceRecords ()
    let avgFitness =
        if records.IsEmpty then 0.0
        else records |> List.averageBy (fun r -> r.AverageScore) |> fun x -> Math.Round(x, 3)

    { RunId = Guid.NewGuid().ToString("N").[..7]
      Timestamp = DateTime.UtcNow.ToString("o")
      ArtifactsProcessed = artifactCount
      PatternsObserved = records.Length
      CandidatesEvaluated = results.Length
      Approved = outputs |> List.filter (fun o -> o.Decision = "approve")
      Rejected = outputs |> List.filter (fun o -> o.Decision = "reject")
      Deferred = outputs |> List.filter (fun o -> o.Decision = "defer")
      LibrarySize = records |> List.filter (fun r -> r.CurrentLevel <> Implementation) |> List.length
      AvgFitness = avgFitness }

// ─────────────────────────────────────────────────────────────────────
// JSON Serialization
// ─────────────────────────────────────────────────────────────────────

/// Serialize any output DTO to strict JSON
let toJson<'T> (output: 'T) : string =
    JsonSerializer.Serialize(output, jsonOptions)

/// Serialize a pipeline run to JSON
let pipelineRunToJson (results: PromotionPipeline.PipelineResult list) (artifactCount: int) : string =
    fromPipelineRun results artifactCount |> toJson

/// Serialize a single governance decision to JSON
let governanceToJson (result: PromotionPipeline.PipelineResult) : string =
    fromPipelineResult result |> toJson

/// Build a CycleOutput from individual fields (avoids compile-order dependency on RetroactionLoop)
let buildCycleOutput
    (cycleId: string)
    (problemTitle: string)
    (success: bool)
    (score: float)
    (validationPassed: bool)
    (patternUsed: string option)
    (patternCompiled: bool)
    (improvementsFound: int)
    (promotionResults: PromotionPipeline.PipelineResult list)
    (librarySize: int)
    (masteryScore: float)
    : CycleOutput =
    { CycleId = cycleId
      ProblemTitle = problemTitle
      Success = success
      Score = score
      ValidationPassed = validationPassed
      PatternUsed = patternUsed
      PatternCompiled = patternCompiled
      ImprovementsFound = improvementsFound
      Promotions = promotionResults |> List.map fromPipelineResult
      LibrarySize = librarySize
      CurriculumMastery = Math.Round(masteryScore, 3)
      Timestamp = DateTime.UtcNow.ToString("o") }

/// Serialize a CycleOutput to JSON
let cycleToJson (output: CycleOutput) : string =
    toJson output
