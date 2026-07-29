# Contract: Research and Thesis Digests

- **Status:** Draft
- **Area:** Runtime / Technology Watch
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## Purpose
This contract defines the structure for research paper and thesis digests consumed by the TARS "technology-watch" loop. These digests enable the system to decide whether a piece of work is relevant to the GuitarAlchemist harness, identify actionable claims, and propose improvements via GitHub issues.

## Entities

### ResearchDigest
The primary container for a research paper's analysis.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID/String | Unique identifier for the digest. |
| `title` | String | Title of the research paper. |
| `authors` | Array<String> | List of authors. |
| `source_uri` | URI | Link to the original paper (e.g., arXiv, ACM). |
| `publication_date` | ISO8601 | Date of publication. |
| `source_type` | Enum | `paper`, `preprint`, `technical-report`. |
| `research_area` | String | e.g., "MCTS", "LLM Reasoning", "Formal Methods". |
| `summary` | String | High-level executive summary. |
| `key_claims` | Array<ResearchClaim> | Actionable claims made by the paper. |
| `methods` | Array<MethodCandidate> | Proposed algorithms or methodologies. |
| `datasets` | Array<DatasetCandidate> | Data used for training or evaluation. |
| `benchmarks` | Array<BenchmarkCandidate> | Standardized tests or results. |
| `limitations` | String | Known weaknesses or constraints. |
| `applicability_to_tars` | Score (0-1) | Relevance to the TARS agent framework. |
| `applicability_to_ix` | Score (0-1) | Relevance to the IX skill engine. |
| `applicability_to_demerzel` | Score (0-1) | Relevance to governance/policies. |
| `applicability_to_ga` | Score (0-1) | Relevance to music theory/applications. |
| `confidence` | Score (0-1) | The digester's confidence in this analysis. |
| `novelty_score` | Score (0-1) | Estimated novelty of the work. |
| `implementation_risk` | Enum | `low`, `medium`, `high`. |
| `suggested_followups` | Array<HarnessImprovementProposal> | Concrete steps for integration. |
| `provenance` | EvidenceBundle | Metadata about the digestion process. |

### ThesisDigest
Extends `ResearchDigest` with thesis-specific fields.

| Field | Type | Description |
| :--- | :--- | :--- |
| `institution` | String | University or organization. |
| `degree` | String | e.g., PhD, MSc. |
| `advisor` | String | Primary supervisor. |

### ResearchClaim
A specific, testable assertion found in the research.

| Field | Type | Description |
| :--- | :--- | :--- |
| `claim` | String | The assertion. |
| `evidence_score` | Score (0-1) | Strength of evidence provided in the source. |
| `actionable` | Boolean | Can this be implemented in the harness? |

### MethodCandidate
| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | String | Name of the method. |
| `description` | String | Brief explanation. |
| `complexity` | Enum | `low`, `medium`, `high`. |

### HarnessImprovementProposal
A proposal to improve the GuitarAlchemist ecosystem based on the digest.

| Field | Type | Description |
| :--- | :--- | :--- |
| `target_component` | Enum | `tars`, `ix`, `demerzel`, `ga`. |
| `description` | String | What should be changed or added. |
| `rationale` | String | Why this improvement is justified by the research. |
| `issue_draft` | IssueDraftCandidate | A pre-filled draft for a GitHub issue. |

### EvidenceBundle
| Field | Type | Description |
| :--- | :--- | :--- |
| `digester_id` | String | ID of the agent/model that created the digest. |
| `digestion_timestamp` | ISO8601 | When the digest was created. |
| `source_hash` | String | SHA-256 hash of the source document. |

### IssueDraftCandidate
| Field | Type | Description |
| :--- | :--- | :--- |
| `title` | String | Proposed title. |
| `body` | String | Markdown body with tasks and labels. |
| `labels` | Array<String> | e.g., `feature`, `research`. |

## Implementation Notes
- All scores are normalized between `0.0` and `1.0`.
- Dates must follow RFC3339/ISO8601.
- `provenance` is mandatory for accountability in the TARS evolution pipeline.
