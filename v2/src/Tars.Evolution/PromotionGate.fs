module Tars.Evolution.PromotionGate

/// The single decision point for a promotion candidate.
///
/// Previously the pipeline asked the Grammar Governor for a decision and then
/// SEPARATELY re-inspected `RoundtripValidation.SemanticMatch` to override an
/// Approve into a Reject — a verdict leak that produced two competing
/// decisions outside the gate. This module folds both concerns into ONE
/// verdict: the governor evaluates the criteria, and if it approves, the
/// injected round-trip validator has the final say. Callers consume
/// `Verdict.Decision` and never re-inspect the round-trip afterward.

open System

/// A round-trip validator: given a candidate the governor approved, produce a
/// structural round-trip result. Injected so the gate — not the pipeline —
/// owns the semantic-match decision.
type RoundtripValidator = PromotionCandidate -> RoundtripValidation.RoundtripResult

/// The single governance verdict: the final decision plus the round-trip
/// evidence that produced it (present only when the governor approved and the
/// candidate was therefore round-trip-checked).
type Verdict = {
    Decision: GovernanceDecision
    Roundtrip: RoundtripValidation.RoundtripResult option
}

/// Decide a candidate's fate in one place. The Grammar Governor evaluates the
/// criteria; on approval the injected validator gates the promotion on
/// round-trip semantic fidelity. Non-approvals pass through untouched.
let decide
    (validate: RoundtripValidator)
    (existing: RecurrenceRecord list)
    (candidate: PromotionCandidate)
    : Verdict =
    match GrammarGovernor.evaluate existing candidate with
    | Approve _ as approved ->
        let rt = validate candidate
        if rt.Passed then
            { Decision = approved; Roundtrip = Some rt }
        else
            let reason =
                sprintf "Round-trip validation failed (semantic match: %.2f). %s"
                    rt.SemanticMatch
                    (rt.Issues |> String.concat "; ")
            { Decision = Reject reason; Roundtrip = Some rt }
    | other ->
        { Decision = other; Roundtrip = None }
