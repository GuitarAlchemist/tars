module Tars.Evolution.RoundtripValidation

/// Round-trip validation for the promotion pipeline.
/// After the Grammar Governor approves a promotion, this module proves
/// the new abstraction can be expanded back to lower-level code and
/// re-abstracted without semantic loss.
///
/// The round-trip: template -> expand -> re-abstract -> compare
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

/// Jaccard similarity between two sets of identifiers.
let private jaccardSimilarity (a: Set<string>) (b: Set<string>) : float =
    if Set.isEmpty a && Set.isEmpty b then 1.0
    else
        let intersection = Set.intersect a b |> Set.count |> float
        let union = Set.union a b |> Set.count |> float
        if union = 0.0 then 0.0 else intersection / union

/// Check which structural elements from the original are missing in the reconstructed version.
let private findMissingElements (original: string) (reconstructed: string) : string list =
    let origIds = extractIdentifiers original
    let reconIds = extractIdentifiers reconstructed
    let missing = Set.difference origIds reconIds
    missing
    |> Set.toList
    |> List.map (fun id -> sprintf "Missing identifier: '%s'" id)

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
// Core: validate (full round-trip)
// ─────────────────────────────────────────────────────────────────────

let private defaultThreshold = 0.5

/// Validate a promotion candidate through the full round-trip:
///   1. Expand  — use RollbackExpansion to get lower-level code
///   2. Reconstruct — re-derive the abstraction template from expanded code
///   3. Compare — measure semantic similarity
///   4. Check for semantic loss
let validate (threshold: float) (candidate: PromotionCandidate) : RoundtripResult =
    let patternId = candidate.Record.PatternId

    match candidate.RollbackExpansion with
    | None ->
        { RoundtripResult.empty patternId with
            OriginalTemplate = candidate.PatternTemplate
            Issues = [ "No RollbackExpansion provided — round-trip validation impossible" ] }
    | Some expansion ->
        // Step 1: The expansion IS the lower-level code
        let expandedCode = expansion

        // Step 2: Re-abstract from the expanded code
        let reconstructed = reabstract expandedCode

        // Step 3: Compare original template and reconstructed
        let origIds = extractIdentifiers candidate.PatternTemplate
        let reconIds = extractIdentifiers reconstructed
        let similarity = jaccardSimilarity origIds reconIds

        // Step 4: Check for semantic loss
        let missingElements = findMissingElements candidate.PatternTemplate reconstructed

        let issues =
            [ if similarity < threshold then
                  yield sprintf "Semantic similarity %.2f is below threshold %.2f" similarity threshold
              yield! missingElements ]

        { PatternId = patternId
          OriginalTemplate = candidate.PatternTemplate
          ExpandedCode = expandedCode
          ReconstructedTemplate = reconstructed
          SemanticMatch = similarity
          Passed = similarity >= threshold
          Issues = issues }

// ─────────────────────────────────────────────────────────────────────
// quickValidate: pure structural, no LLM
// ─────────────────────────────────────────────────────────────────────

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
// validateWithLlm: LLM-assessed semantic equivalence
// ─────────────────────────────────────────────────────────────────────

open Tars.Llm

/// Use an LLM to assess whether the expansion is semantically equivalent
/// to the original template. Returns a RoundtripResult with LLM-assessed
/// SemanticMatch score.
let validateWithLlm (llm: ILlmService) (candidate: PromotionCandidate) : Async<RoundtripResult> =
    async {
        let patternId = candidate.Record.PatternId

        match candidate.RollbackExpansion with
        | None ->
            return
                { RoundtripResult.empty patternId with
                    OriginalTemplate = candidate.PatternTemplate
                    Issues = [ "No RollbackExpansion — automatic failure" ] }
        | Some expansion ->
            let prompt =
                $"""You are a semantic equivalence judge for code abstractions.

TASK: Determine whether an abstraction template and its lower-level expansion
are semantically equivalent — meaning the expansion faithfully implements
everything the template describes, with no loss of meaning.

ORIGINAL TEMPLATE (higher-level abstraction):
{candidate.PatternTemplate}

EXPANDED CODE (lower-level implementation):
{expansion}

Evaluate on these criteria:
1. Does the expansion implement all behaviors described in the template?
2. Are there any semantic elements in the template that are lost in the expansion?
3. Could the expansion be re-abstracted back to the original template without ambiguity?

Respond with EXACTLY this format (no other text):
SCORE: <float 0.0-1.0>
ISSUES: <comma-separated list of issues, or "none">"""

            let req : LlmRequest =
                { ModelHint = Some "reasoning"
                  Model = None
                  SystemPrompt = Some "You are a precise semantic equivalence evaluator. Follow the output format exactly."
                  MaxTokens = Some 500
                  Temperature = Some 0.1
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                let! resp = llm.CompleteAsync req |> Async.AwaitTask
                let text = resp.Text.Trim()

                // Parse SCORE line
                let scoreLine =
                    text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.tryFind (fun l -> l.TrimStart().StartsWith("SCORE:", StringComparison.OrdinalIgnoreCase))

                let llmScore =
                    match scoreLine with
                    | Some line ->
                        let parts = line.Split(':')
                        if parts.Length >= 2 then
                            match Double.TryParse(parts.[1].Trim()) with
                            | true, v -> min 1.0 (max 0.0 v)
                            | _ -> 0.5
                        else 0.5
                    | None -> 0.5

                // Parse ISSUES line
                let issuesLine =
                    text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.tryFind (fun l -> l.TrimStart().StartsWith("ISSUES:", StringComparison.OrdinalIgnoreCase))

                let llmIssues =
                    match issuesLine with
                    | Some line ->
                        let content = line.Substring(line.IndexOf(':') + 1).Trim()
                        if content.Equals("none", StringComparison.OrdinalIgnoreCase) then []
                        else
                            content.Split(',')
                            |> Array.map (fun s -> s.Trim())
                            |> Array.filter (fun s -> s.Length > 0)
                            |> Array.toList
                    | None -> []

                return
                    { PatternId = patternId
                      OriginalTemplate = candidate.PatternTemplate
                      ExpandedCode = expansion
                      ReconstructedTemplate = reabstract expansion
                      SemanticMatch = llmScore
                      Passed = llmScore >= defaultThreshold
                      Issues = llmIssues }

            with ex ->
                // On LLM failure, fall back to structural validation
                let structural = quickValidate candidate
                return
                    { structural with
                        Issues = (sprintf "LLM validation failed (%s), fell back to structural" ex.Message) :: structural.Issues }
    }

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
