module Tars.Evolution.RoundtripValidation

/// Round-trip validation for the promotion pipeline.
/// After the Grammar Governor approves a promotion, this module checks
/// that the new abstraction has a plausible lower-level expansion.
///
/// The gate is quickValidate: a structural heuristic (expansion length and
/// identifier coverage), not a proof of semantic equivalence.
/// A pattern WITHOUT a RollbackExpansion automatically fails.

open System

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

type RoundtripResult = {
    PatternId: string
    OriginalTemplate: string       // The promoted abstraction
    ExpandedCode: string           // The lower-level expansion
    ReconstructedTemplate: string  // Re-abstracted from expanded code
    SemanticMatch: float           // 0.0-1.0 how similar original and reconstructed are
    Passed: bool                   // SemanticMatch >= threshold
    Issues: string list            // Any detected semantic losses
}

module RoundtripResult =
    let empty patternId = {
        PatternId = patternId
        OriginalTemplate = ""
        ExpandedCode = ""
        ReconstructedTemplate = ""
        SemanticMatch = 0.0
        Passed = false
        Issues = []
    }

// ─────────────────────────────────────────────────────────────────────
// Helpers: structural comparison
// ─────────────────────────────────────────────────────────────────────

/// Extract identifiers (alphanumeric tokens) from a string.
let private extractIdentifiers (text: string) : Set<string> =
    text.Split([| ' '; '\n'; '\r'; '\t'; '('; ')'; '{'; '}'; '['; ']'; ';'; ','; ':'; '.'; '|'; '>' ; '<'; '='; '+'; '-'; '*'; '/'; '"'; '\'' |],
               StringSplitOptions.RemoveEmptyEntries)
    |> Array.filter (fun t -> t.Length > 2) // skip noise tokens
    |> Set.ofArray

/// Simple re-abstraction: extract the "shape" of the expanded code by
/// removing literal values and collapsing whitespace. This is a
/// heuristic stand-in for a proper re-abstraction pass.
let private reabstract (expandedCode: string) : string =
    expandedCode
    |> fun s -> System.Text.RegularExpressions.Regex.Replace(s, "\"[^\"]*\"", "\"...\"")
    |> fun s -> System.Text.RegularExpressions.Regex.Replace(s, @"\b\d+\b", "N")
    |> fun s -> System.Text.RegularExpressions.Regex.Replace(s, @"\s+", " ")
    |> fun s -> s.Trim()

// ─────────────────────────────────────────────────────────────────────
// quickValidate: pure structural, no LLM
// ─────────────────────────────────────────────────────────────────────

let private defaultThreshold = 0.5

/// Fast, deterministic validation without LLM.
/// Checks structural properties only:
///   - RollbackExpansion exists
///   - Expansion is longer than template (more verbose = lower level)
///   - Key identifiers from template appear in expansion
let quickValidate (candidate: PromotionCandidate) : RoundtripResult =
    let patternId = candidate.Record.PatternId

    match candidate.RollbackExpansion with
    | None ->
        { RoundtripResult.empty patternId with
            OriginalTemplate = candidate.PatternTemplate
            Issues = [ "No RollbackExpansion — automatic failure" ] }
    | Some expansion ->
        let mutable issues = []
        let mutable score = 1.0

        // Check 1: Expansion should be longer than template (lower-level = more verbose)
        if expansion.Length <= candidate.PatternTemplate.Length then
            issues <- "Expansion is not longer than template — may not be a true lower-level form" :: issues
            score <- score - 0.3

        // Check 2: Key identifiers from template appear in expansion
        let templateIds = extractIdentifiers candidate.PatternTemplate
        let expansionIds = extractIdentifiers expansion
        let overlap = Set.intersect templateIds expansionIds
        let coverage =
            if Set.isEmpty templateIds then 1.0
            else float (Set.count overlap) / float (Set.count templateIds)

        if coverage < 0.3 then
            issues <- sprintf "Only %.0f%% of template identifiers found in expansion" (coverage * 100.0) :: issues
            score <- score - 0.4
        elif coverage < 0.6 then
            issues <- sprintf "Only %.0f%% of template identifiers found in expansion" (coverage * 100.0) :: issues
            score <- score - 0.2

        // Check 3: Expansion is non-trivial (not just whitespace)
        if expansion.Trim().Length < 10 then
            issues <- "Expansion is trivially short" :: issues
            score <- score - 0.3

        let finalScore = max 0.0 (min 1.0 score)

        { PatternId = patternId
          OriginalTemplate = candidate.PatternTemplate
          ExpandedCode = expansion
          ReconstructedTemplate = reabstract expansion
          SemanticMatch = finalScore
          Passed = finalScore >= defaultThreshold
          Issues = List.rev issues }

// ─────────────────────────────────────────────────────────────────────
// Audit report
// ─────────────────────────────────────────────────────────────────────

/// Generate a human-readable audit report for a round-trip validation result.
let auditReport (result: RoundtripResult) : string =
    let status = if result.Passed then "PASSED" else "FAILED"
    let issueLines =
        if result.Issues.IsEmpty then [ "  (none)" ]
        else result.Issues |> List.map (fun i -> sprintf "  - %s" i)

    [ "-----------------------------------------------"
      sprintf "  ROUND-TRIP VALIDATION: %s" status
      "-----------------------------------------------"
      sprintf "  Pattern ID:      %s" result.PatternId
      sprintf "  Semantic Match:  %.2f" result.SemanticMatch
      sprintf "  Template length: %d chars" result.OriginalTemplate.Length
      sprintf "  Expanded length: %d chars" result.ExpandedCode.Length
      "  Issues:"
      yield! issueLines
      "-----------------------------------------------" ]
    |> String.concat "\n"
