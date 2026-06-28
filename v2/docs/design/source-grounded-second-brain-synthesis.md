# Source-Grounded Synthesis for TARS Second Brain

## Goal
Design the source-grounded synthesis and evidence-bundle promotion flow for the TARS second-brain harness. The core objective is to prevent the promotion of plausible, unsupported summaries into durable memory. Knowledge is only promoted when grounded in explicit sources, shaped by grammar contracts, scored by IX, and accepted through rigorous promotion rules (Seldon/Demerzel/human).

## Core Principles
1. **Source Set is Bounded:** Synthesis operates over a specific, finite set of artifacts.
2. **Explicit Citations:** Every claim must cite specific source artifacts to be valid.
3. **Claims ≠ Evidence:** Synthesized claims must be separated from the raw evidence supporting them.
4. **Contradictions & Staleness:** Conflicting information or outdated sources must be proactively surfaced.
5. **Explicit Promotion:** Promoting findings to durable memory is a deliberate, gated process, not automatic.

## Candidate Flow

```text
SourceSet
  -> semantic synthesis
  -> candidate ResearchFinding
  -> EvidenceBundle
  -> grammar/schema validation
  -> IX scoring / drift / contradiction checks
  -> Seldon learning assessment
  -> MemoryUpdateCandidate
  -> accepted MemoryItem or escalation
```

### Flow Breakdown

1. **SourceSet & Semantic Synthesis**: A bounded `SourceSet` (e.g. recent logs, traces, issues) is fed into the LLM context. The LLM performs a semantic synthesis to generate a candidate `ResearchFinding`.
2. **EvidenceBundle Creation**: The synthesis is structured into an `EvidenceBundle`, separating claims from evidence. Every `GroundedClaim` is linked back to a `SourceReference`.
3. **Validation & Assessment**:
   - **Grammar/Schema Validation:** Ensures the bundle aligns with established `.trsx` and JSON schemas.
   - **IX Scoring:** IX analyzes the claims for logical drift and checks against known contradictions.
   - **Seldon Assessment:** Seldon evaluates the "learning value" of the claim against the existing knowledge graph.
4. **Memory Promotion**: The evaluated bundle becomes a `MemoryUpdateCandidate`. The decision logic outputs a `MemoryPromotionDecision` resulting in an accepted `MemoryItem` or an escalation for human review.

## Promotion Rules (Seldon / Demerzel / Human)

- **Seldon (Teacher/Assessor):** Validates if a claim is a novel, source-grounded learning. If Seldon determines the claim is missing a citation, it is automatically rejected.
- **Demerzel (Governance):** Checks the claim against existing policies and cross-repo health metrics. If the claim implies a policy violation, it is escalated.
- **IX (Scoring):** Assesses the confidence and robustness of the claim based on the provided evidence. Low-scoring claims are rejected or flagged as `ReviewRequired`.
- **Human Escalation:** Any `MemoryUpdateCandidate` that flags contradictions, staleness, or falls below a high-confidence threshold but above absolute rejection must be reviewed by a human before becoming a durable `MemoryItem`.

## Contradiction and Staleness Handling

- **Staleness:** A `StalenessWarning` is generated when a cited source is older than a specified decay threshold or when a newer version of the same artifact exists. Stale evidence reduces the IX score.
- **Contradiction:** If a new `GroundedClaim` directly opposes an existing durable `MemoryItem`, a `ContradictionCandidate` is raised. The system does not overwrite the existing memory automatically; it escalates the conflict for human resolution, preserving both sides of the contradiction in the escalation context.

## Ecosystem Boundaries

- **TARS:** Drives the core semantic extraction and execution flow via `Tars.Evolution`. TARS orchestrates the pipeline but defers judgment.
- **IX:** Provides the MCTS-driven logical scoring, evaluating the strength of evidence and detecting drift.
- **Seldon:** Acts as the learning assessor, ensuring knowledge fits the pedagogical and governance frameworks before integration.
