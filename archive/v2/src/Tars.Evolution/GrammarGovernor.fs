module Tars.Evolution.GrammarGovernor

/// The Grammar Governor gates pattern promotions using the 8 criteria.
/// It is a strict, deterministic evaluator — the "anti-goblin valve" that
/// prevents the language from becoming a haunted mansion of half-baked abstractions.
///
/// Approval thresholds:
///   Approve: >= 6 criteria met AND MinOccurrences is true
///   Reject:  < 4 criteria met OR MinOccurrences is false
///   Defer:   4-5 criteria met (needs more evidence)

open System

/// Score a candidate's criteria (0-8)
let score (criteria: PromotionCriteria) : int =
    PromotionCriteria.score criteria

/// Check for overlap with existing patterns at the proposed level
let checkOverlap (candidate: PromotionCandidate) (existing: RecurrenceRecord list) : bool =
    existing
    |> List.exists (fun r ->
        r.PatternId <> candidate.Record.PatternId
        && r.CurrentLevel = candidate.ProposedLevel
        && r.PatternName = candidate.Record.PatternName)

/// Evaluate a promotion candidate — the core governance decision
let evaluate (existing: RecurrenceRecord list) (candidate: PromotionCandidate) : GovernanceDecision =
    let criteria = candidate.Criteria
    let s = score criteria

    // Hard requirement: must have minimum occurrences
    if not criteria.MinOccurrences then
        Reject $"Pattern '%s{candidate.Record.PatternName}' has only %d{candidate.Record.OccurrenceCount} occurrences. Need >= 3 real tasks."

    // Hard requirement: no overlap with existing constructs
    elif checkOverlap candidate existing then
        Reject $"Pattern '%s{candidate.Record.PatternName}' overlaps with an existing construct at level %s{PromotionLevel.label candidate.ProposedLevel}."

    // Must have a rollback path for DslClause and above
    elif PromotionLevel.rank candidate.ProposedLevel >= PromotionLevel.rank DslClause
         && candidate.RollbackExpansion.IsNone then
        Reject $"Promotion to %s{PromotionLevel.label candidate.ProposedLevel} requires a rollback expansion path."

    // Score-based decision
    elif s >= 6 then
        Approve $"Pattern '%s{candidate.Record.PatternName}' meets %d{s}/8 criteria. Approved for promotion to %s{PromotionLevel.label candidate.ProposedLevel}."

    elif s < 4 then
        Reject $"Pattern '%s{candidate.Record.PatternName}' meets only %d{s}/8 criteria. Insufficient for promotion."

    else
        Defer $"Pattern '%s{candidate.Record.PatternName}' meets %d{s}/8 criteria. Need more evidence before promoting to %s{PromotionLevel.label candidate.ProposedLevel}."

/// Generate a human-readable audit report
let auditReport (candidate: PromotionCandidate) (decision: GovernanceDecision) : string =
    let c = candidate.Criteria
    let check b label = if b then $"  [✓] %s{label}" else $"  [✗] %s{label}"
    let decisionStr =
        match decision with
        | Approve reason -> $"APPROVED: %s{reason}"
        | Reject reason -> $"REJECTED: %s{reason}"
        | Defer reason -> $"DEFERRED: %s{reason}"

    [ "═══════════════════════════════════════════════"
      $"  GRAMMAR GOVERNOR AUDIT REPORT"
      "═══════════════════════════════════════════════"
      $"  Pattern:    %s{candidate.Record.PatternName}"
      $"  Pattern ID: %s{candidate.Record.PatternId}"
      $"  Current:    %s{PromotionLevel.label candidate.Record.CurrentLevel}"
      $"  Proposed:   %s{PromotionLevel.label candidate.ProposedLevel}"
      $"  Occurrences: %d{candidate.Record.OccurrenceCount}"
      $"  Avg Score:   %.2f{candidate.Record.AverageScore}"
      $"  Evidence:    %d{candidate.Evidence.Length} items"
      "───────────────────────────────────────────────"
      $"  CRITERIA ({score c}/8):"
      check c.MinOccurrences "Minimum occurrences (>= 3 tasks)"
      check c.RemovesComplexity "Removes incidental complexity"
      check c.MoreReadable "More readable than expanded form"
      check c.StableSemantics "Stable semantics across uses"
      check c.AutoValidatable "Can be validated automatically"
      check c.NoOverlap "No overlap with existing constructs"
      check c.ComposesCleanly "Composes cleanly with existing types"
      check c.ImprovesPlanning "Improves AI planning (not just typing)"
      "───────────────────────────────────────────────"
      $"  DECISION: %s{decisionStr}"
      match candidate.RollbackExpansion with
      | Some r -> $"  ROLLBACK:  %s{r.[..min 80 (r.Length - 1)]}..."
      | None -> "  ROLLBACK:  None specified"
      "═══════════════════════════════════════════════" ]
    |> String.concat "\n"
