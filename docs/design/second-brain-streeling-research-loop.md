# Design: Second-Brain Streeling Research Loop

- **Status:** Draft
- **Area:** Runtime / Second Brain
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## 1. Goal

Connect the TARS V2 second-brain harness to Streeling University so the second brain can conduct bounded research over the ecosystem's institutional learning corpus.

The second brain should not only store memory. It should be able to ask research questions, query Streeling, synthesize evidence, propose memory updates, and create follow-up work — all with grammar validation, provenance, and governance gates.

## 2. Boundaries and Philosophy

To preserve the GuitarAlchemist ecosystem's architectural integrity:
- **No unbounded web research:** The second brain evaluates the *internal* Streeling corpus, preventing hallucinations and preserving focus.
- **Streeling is an index, not the source of truth:** Streeling aggregates pointers to raw learnings. TARS must not duplicate the duckDB catalog or Registrar logic.
- **No direct memory mutation:** Research Findings produce `MemoryUpdateCandidate`s. Governance (Demerzel or human) must approve the injection.
- **Auditable loop:** Every synthesis requires an explicit `EvidenceBundle` tracing back to the Registrar's pointers.

## 3. The Research Loop Workflows

The bounded, local-first research loop consists of the following steps:

1. **Trigger:** A `ResearchQuestion` is submitted to the Second Brain (via user prompt or background IX trigger).
2. **Planning:** TARS generates a `ResearchPlan` outlining the Streeling Registrar query strategy.
3. **Query:** TARS executes a `StreelingQuery` against the local DuckDB catalog (`state/streeling/catalog.jsonl`).
4. **Retrieval:** The Registrar returns pointers. TARS fetches the raw source learning artifacts.
5. **Synthesis:** TARS analyzes the sources to produce a grammar-validated `ResearchFinding`.
6. **Evaluation:** The IX scorecard grades the finding for relevance and quality.
7. **Resolution:**
   - **Success:** TARS emits a `MemoryUpdateCandidate`.
   - **Gap Identified:** If Streeling lacks the data, TARS emits a `LearningGap` and proposes a `ResearchFollowUpIssue`.
   - **Closure:** The original question is marked resolved and the trace is stored for replay.

## 4. Proposed Contracts

The entities in the loop are defined via JSON-first contracts to ensure type safety and boundary enforcement.

### ResearchQuestion
The trigger for the loop.
- `id` (UUID)
- `question` (String): The core query.
- `context` (String): Why is this being asked?
- `constraints` (Array<String>): Explicit bounds (e.g., "only consider learnings from 2024").

### ResearchPlan
The strategy.
- `question_id` (UUID)
- `queries` (Array<StreelingQuery>)
- `expected_outcomes` (String)

### StreelingQuery
The interface to the Registrar.
- `sql_dialect` (DuckDB)
- `target_departments` (Array<String>)
- `keywords` (Array<String>)

### ResearchFinding
The synthesized result.
- `id` (UUID)
- `question_id` (UUID)
- `claims` (Array<SemanticClaim>)
- `evidence` (EvidenceBundle)
- `scorecard` (IXScorecard)

### EvidenceBundle
Provenance metadata.
- `registrar_pointers` (Array<URI>)
- `source_hashes` (Array<String>)
- `staleness_marker` (ISO8601)

### LearningGap
Emitted when the Registrar returns no relevant hits or the synthesis fails to answer the question.
- `question_id` (UUID)
- `missing_concepts` (Array<String>)
- `proposed_issue` (ResearchFollowUpIssue)

### MemoryUpdateCandidate
A proposed patch to the Second Brain state.
- `finding_id` (UUID)
- `action` (Enum: Insert, Update, Deprecate)
- `target_node` (String)
- `payload` (JSON)

### ResearchFollowUpIssue
A Demerzel-gated issue draft for human/agent review.
- `title` (String)
- `body` (Markdown)
- `labels` (Array<String>)
- `risk_level` (Enum: Low, Medium, High)

## 5. Governance and Safety Rules

- **Halt Marker Compliance:** The loop must respect `governance/state/afk-halt.json`. If halted, no queries are run.
- **Freshness Gate:** The loop must verify the Streeling catalog's last-updated timestamp. "Green-but-dead" catalogs trigger an abort.
- **Approval Flow:** High-risk `ResearchFollowUpIssue`s must be labeled `requires-human-approval`.
- **Cost Controls:** The entire loop must execute within the `free-local` tier budget (e.g., Ollama-local). Cloud-API fallbacks are strictly prohibited for background research unless explicitly authorized.