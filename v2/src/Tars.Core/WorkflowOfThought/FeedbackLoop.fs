namespace Tars.Core.WorkflowOfThought

open System

// =============================================================================
// PHASE 17.3: FEEDBACK LOOPS
// =============================================================================
// Implements GoT-style feedback loops for iterative refinement:
// - Score aggregation across evidence nodes
// - Hypothesis re-evaluation when new evidence arrives
// - Contradiction detection and resolution
// - Confidence tracking and propagation
// =============================================================================

/// Source of evidence for a claim
type EvidenceSource =
    | ToolContribution of toolName: string * stepId: string
    | ReasonerThought of stepId: string
    | ExternalFact of url: string
    | LogicalInference of rule: string
    | SymbolicMemoryRef of entryId: Guid

/// A single piece of evidence with provenance
type Evidence =
    { Id: string
      Source: EvidenceSource
      Content: string
      Confidence: float // 0.0-1.0
      Weight: float // 0.0-1.0: How much it influences the score
      IsContradiction: bool
      ParentIds: string list // Graph of Thoughts: Preceding thoughts
      Timestamp: DateTime }

/// Score for a hypothesis or claim with full provenance (Phase 17.4)
type HypothesisScore =
    { Id: string
      Label: string
      EvidenceCoverage: float // 0.0-1.0: How much evidence supports this
      TemporalFit: float // 0.0-1.0: How well timing aligns
      MechanismPlausibility: float // 0.0-1.0: How mechanically plausible
      Contradictions: int // Number of contradicting pieces of evidence
      ImpactScore: float // 0.0-1.0: Potential impact if true
      OverallScore: float // Weighted aggregate
      Reason: string
      EvidenceIds: string list // Provenance links
      ConflictingEvidenceIds: string list
      Volume: int } // GoT: Total preceding thoughts

/// Result of a scoring operation
type ScoringResult =
    { Hypothesis: string
      Before: HypothesisScore option
      After: HypothesisScore
      Delta: float
      Evidence: Evidence list }

/// Feedback loop state with provenance tracking
type FeedbackState =
    { RunId: Guid
      Hypotheses: Map<string, HypothesisScore>
      EvidenceLibrary: Map<string, Evidence>
      EvidenceChain: string list // Sequence of evidence IDs
      Iterations: int
      LastUpdated: DateTime }

module FeedbackLoop =

    /// Create initial feedback state
    let create (runId: Guid) : FeedbackState =
        { RunId = runId
          Hypotheses = Map.empty
          EvidenceLibrary = Map.empty
          EvidenceChain = []
          Iterations = 0
          LastUpdated = DateTime.UtcNow }

    /// Calculate overall score from components
    let private calculateOverall
        (coverage: float)
        (temporal: float)
        (mechanism: float)
        (contradictions: int)
        (impact: float)
        : float =
        // Weights based on GoT paper: evidence and mechanism are most important
        let weights =
            {| EvidenceCoverage = 0.25
               TemporalFit = 0.15
               Mechanism = 0.30
               Contradictions = 0.15
               Impact = 0.15 |}

        let contradictionPenalty =
            if contradictions = 0 then 1.0
            elif contradictions = 1 then 0.85
            elif contradictions = 2 then 0.65
            else 0.4

        let raw =
            (coverage * weights.EvidenceCoverage)
            + (temporal * weights.TemporalFit)
            + (mechanism * weights.Mechanism)
            + (contradictionPenalty * weights.Contradictions)
            + (impact * weights.Impact)

        // Normalize to 0.0-1.0
        min 1.0 (max 0.0 raw)

    /// Compute the Volume (transitive ancestor count) of a set of evidence
    let private computeVolume (evidenceIds: string list) (library: Map<string, Evidence>) : int =
        let mutable visited = Set.empty

        let rec visit (id: string) =
            if not (visited.Contains id) then
                visited <- visited.Add id

                match library.TryFind id with
                | Some ev -> ev.ParentIds |> List.iter visit
                | None -> ()

        evidenceIds |> List.iter visit
        visited.Count

    /// Create a hypothesis score with provenance tracking
    let score
        (id: string)
        (label: string)
        (coverage: float)
        (temporal: float)
        (mechanism: float)
        (contradictions: int)
        (impact: float)
        (reason: string)
        (evidenceIds: string list)
        (conflictingIds: string list)
        (volume: int)
        : HypothesisScore =
        { Id = id
          Label = label
          EvidenceCoverage = coverage
          TemporalFit = temporal
          MechanismPlausibility = mechanism
          Contradictions = contradictions
          ImpactScore = impact
          OverallScore = calculateOverall coverage temporal mechanism contradictions impact
          Reason = reason
          EvidenceIds = evidenceIds
          ConflictingEvidenceIds = conflictingIds
          Volume = volume }

    /// Register a hypothesis for tracking
    let registerHypothesis (hypothesis: HypothesisScore) (state: FeedbackState) : FeedbackState =
        { state with
            Hypotheses = state.Hypotheses.Add(hypothesis.Id, hypothesis)
            LastUpdated = DateTime.UtcNow }

    /// Add rich evidence to the library and chain
    let addEvidence (evidence: Evidence) (state: FeedbackState) : FeedbackState =
        { state with
            EvidenceLibrary = state.EvidenceLibrary.Add(evidence.Id, evidence)
            EvidenceChain = evidence.Id :: state.EvidenceChain }

    /// Update a hypothesis with new rich evidence
    let updateHypothesis
        (hypothesisId: string)
        (evidence: Evidence)
        (state: FeedbackState)
        : FeedbackState * ScoringResult option =

        match state.Hypotheses.TryFind(hypothesisId) with
        | None -> state, None
        | Some existing ->
            let newContradictions =
                if evidence.IsContradiction then
                    existing.Contradictions + 1
                else
                    existing.Contradictions

            let evidenceIds =
                if not evidence.IsContradiction then
                    evidence.Id :: existing.EvidenceIds
                else
                    existing.EvidenceIds

            let conflictingIds =
                if evidence.IsContradiction then
                    evidence.Id :: existing.ConflictingEvidenceIds
                else
                    existing.ConflictingEvidenceIds

            // Confidence-weighted coverage update
            let contribution =
                if evidence.IsContradiction then
                    -evidence.Weight * 0.1
                else
                    evidence.Weight * 0.1

            let newCoverage = min 1.0 (max 0.0 (existing.EvidenceCoverage + contribution))

            let updated =
                { existing with
                    EvidenceCoverage = newCoverage
                    Contradictions = newContradictions
                    EvidenceIds = evidenceIds
                    ConflictingEvidenceIds = conflictingIds
                    Volume = computeVolume (evidenceIds @ conflictingIds) state.EvidenceLibrary
                    OverallScore =
                        calculateOverall
                            newCoverage
                            existing.TemporalFit
                            existing.MechanismPlausibility
                            newContradictions
                            existing.ImpactScore
                    Reason = $"Updated with evidence: {evidence.Id}" }

            let result =
                { Hypothesis = hypothesisId
                  Before = Some existing
                  After = updated
                  Delta = updated.OverallScore - existing.OverallScore
                  Evidence = [ evidence ] }

            let newState =
                { state with
                    Hypotheses = state.Hypotheses.Add(hypothesisId, updated)
                    EvidenceLibrary = state.EvidenceLibrary.Add(evidence.Id, evidence)
                    EvidenceChain = evidence.Id :: state.EvidenceChain
                    Iterations = state.Iterations + 1
                    LastUpdated = DateTime.UtcNow }

            newState, Some result

    /// Get the top-K hypotheses by score
    let topK (k: int) (state: FeedbackState) : HypothesisScore list =
        state.Hypotheses
        |> Map.toList
        |> List.map snd
        |> List.sortByDescending (fun h -> h.OverallScore)
        |> List.truncate k

    /// Find contradictions between hypotheses
    let findContradictions (state: FeedbackState) : (HypothesisScore * HypothesisScore) list =
        let hypotheses = state.Hypotheses |> Map.toList |> List.map snd

        // Simple heuristic: hypotheses that are both high-scoring and have contradicting evidence
        hypotheses
        |> List.filter (fun h -> h.Contradictions > 0 && h.OverallScore > 0.5)
        |> List.collect (fun h1 ->
            hypotheses
            |> List.filter (fun h2 -> h2.Id <> h1.Id && h2.OverallScore > 0.5)
            |> List.map (fun h2 -> h1, h2))

    /// Prune hypotheses below threshold
    let prune (threshold: float) (state: FeedbackState) : FeedbackState * string list =
        let pruned, kept =
            state.Hypotheses |> Map.partition (fun _ h -> h.OverallScore < threshold)

        let prunedIds = pruned |> Map.toList |> List.map fst

        { state with Hypotheses = kept }, prunedIds

    /// Check if convergence has been reached
    let hasConverged (minScore: float) (maxIterations: int) (state: FeedbackState) : bool =
        if state.Iterations >= maxIterations then
            true
        else
            let topHypothesis = topK 1 state |> List.tryHead

            match topHypothesis with
            | Some h when h.OverallScore >= minScore -> true
            | _ -> false

    /// Get full evidence trail for a hypothesis
    let getEvidenceTrail (hypothesisId: string) (state: FeedbackState) : Evidence list =
        match state.Hypotheses.TryFind hypothesisId with
        | None -> []
        | Some h ->
            (h.EvidenceIds @ h.ConflictingEvidenceIds)
            |> List.choose (fun id -> state.EvidenceLibrary.TryFind id)
            |> List.sortBy (fun e -> e.Timestamp)

    /// Summarize the evidence for a hypothesis into a string
    let summarizeEvidence (hypothesisId: string) (state: FeedbackState) : string =
        let trail = getEvidenceTrail hypothesisId state

        if trail.IsEmpty then
            "No evidence found."
        else
            trail
            |> List.map (fun e ->
                let sign = if e.IsContradiction then "[-] " else "[+] "
                $"{sign}{e.Content} (Source: {e.Source})")
            |> String.concat "\n"

    /// Aggregate evidence from multiple hypotheses/steps
    let aggregateEvidence (ids: string list) (state: FeedbackState) : string =
        ids
        |> List.map (fun id ->
            let hStr =
                match state.Hypotheses.TryFind id with
                | Some h -> $"Hypothesis '{h.Label}':\n{summarizeEvidence id state}"
                | None ->
                    match state.EvidenceLibrary.TryFind id with
                    | Some e -> $"Evidence '{id}': {e.Content}"
                    | None -> $"Wait-node/Step '{id}' result."

            hStr)
        |> String.concat "\n\n"

    /// Mark a hypothesis as a dead-end (Backtrack)
    let invalidateHypothesis (hypothesisId: string) (reason: string) (state: FeedbackState) : FeedbackState =
        match state.Hypotheses.TryFind hypothesisId with
        | None -> state
        | Some existing ->
            let updated =
                { existing with
                    OverallScore = 0.05
                    Reason = $"INVALIDATED: {reason}" }

            { state with
                Hypotheses = state.Hypotheses.Add(hypothesisId, updated)
                Iterations = state.Iterations + 1 }

    /// Check if consensus has been reached according to a protocol
    let checkConsensus (protocol: ConsensusProtocol) (state: FeedbackState) : bool * string =
        let top = topK 1 state |> List.tryHead

        match top with
        | None -> false, "No hypotheses yet."
        | Some h ->
            match protocol with
            | MajorityVote ->
                let supporting = h.EvidenceIds.Length
                let total = supporting + h.ConflictingEvidenceIds.Length

                if total = 0 then
                    false, "No evidence gathered."
                else
                    let reached = float supporting / float total > 0.5

                    reached,
                    if reached then
                        $"Majority reached ({supporting}/{total})."
                    else
                        $"Majority not reached ({supporting}/{total})."

            | Unanimous ->
                let conflicts = h.ConflictingEvidenceIds.Length
                let reached = conflicts = 0 && h.EvidenceIds.Length > 0

                reached,
                if reached then
                    "Unanimous agreement (zero conflicts)."
                else
                    $"Disagreements exist ({conflicts} conflicts)."

            | ThresholdScore minScore ->
                let reached = h.OverallScore >= minScore

                reached,
                if reached then
                    $"Score {h.OverallScore:P0} meets threshold."
                else
                    $"Score {h.OverallScore:P0} below threshold."

            | WeightedAverage ->
                // Basic weighted average: use supporting evidence weights vs conflicting weights
                let trail = getEvidenceTrail h.Id state

                let supportingWeight =
                    trail
                    |> List.filter (fun e -> not e.IsContradiction)
                    |> List.sumBy (fun e -> e.Weight)

                let conflictingWeight =
                    trail
                    |> List.filter (fun e -> e.IsContradiction)
                    |> List.sumBy (fun e -> e.Weight)

                let totalWeight = supportingWeight + conflictingWeight

                if totalWeight = 0.0 then
                    false, "No weighted evidence."
                else
                    let reached = supportingWeight > conflictingWeight

                    reached,
                    if reached then
                        $"Weighted consensus reached ({supportingWeight:F1} vs {conflictingWeight:F1})."
                    else
                        $"Weighted disagreement ({conflictingWeight:F1} vs {supportingWeight:F1})."

            | SuperiorHierarchy ->
                // Heuristic: Is there evidence from a 'Verifier' or high-confidence source that is NOT contradicted?
                let trail = getEvidenceTrail h.Id state

                let verifierEvidence =
                    trail
                    |> List.filter (fun e ->
                        match e.Source with
                        | ReasonerThought s when s.Contains "verifier" || s.Contains "qa" -> true
                        | _ -> false)

                let highConfVerifier =
                    verifierEvidence
                    |> List.filter (fun e -> not e.IsContradiction && e.Confidence > 0.9)

                let reached = not highConfVerifier.IsEmpty

                reached,
                if reached then
                    "Superior hierarchy (Verifier) has confirmed."
                else
                    "No final confirmation from superior role."

    /// Get summary of current state with provenance breakdown
    let summary (state: FeedbackState) : string =
        let topHypotheses = topK 3 state

        let hypothesesStr =
            topHypotheses
            |> List.map (fun h ->
                let supporting = h.EvidenceIds.Length
                let conflicts = h.ConflictingEvidenceIds.Length
                $"  - {h.Label}: {h.OverallScore:P0} (Evidence: +{supporting}/-{conflicts})")
            |> String.concat "\n"

        let sourceBreakdown =
            state.EvidenceLibrary
            |> Map.toList
            |> List.map (snd >> (fun e -> e.Source))
            |> List.groupBy (fun s ->
                match s with
                | ToolContribution _ -> "Tool"
                | ReasonerThought _ -> "Thought"
                | ExternalFact _ -> "External"
                | LogicalInference _ -> "Logic"
                | SymbolicMemoryRef _ -> "Memory")
            |> List.map (fun (name, group) -> $"{name}: {group.Length}")
            |> String.concat ", "

        $"""Feedback Loop State:
  Run: {state.RunId}
  Iterations: {state.Iterations}
  Hypotheses tracked: {state.Hypotheses.Count}
  Evidence Sources: {sourceBreakdown}
Top Hypotheses:
{hypothesesStr}
"""

    /// Find evidence by source ID (e.g. step ID)
    let findEvidenceBySourceId (sourceId: string) (state: FeedbackState) : Evidence option =
        state.EvidenceLibrary
        |> Map.tryPick (fun _ ev ->
            match ev.Source with
            | ReasonerThought stepId when stepId = sourceId -> Some ev
            | ToolContribution(_, stepId) when stepId = sourceId -> Some ev
            | _ -> None)
