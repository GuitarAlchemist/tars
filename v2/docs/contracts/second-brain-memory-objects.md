# TARS V2 Second-Brain Memory Object Contracts

## Overview

This document defines the core contracts for memory objects in the TARS V2 second-brain harness. The second brain is designed to move beyond "vague blobs" of text, ensuring every piece of information has clear provenance, confidence, and structure.

## Design Principles

1.  **JSON-First**: Easy to store as JSONL or Parquet.
2.  **Symbolic Grounding**: Symbols are earned through evidence and provenance.
3.  **Vector-Agnostic**: Compatible with vector indexing, but vectors are not the source of truth.
4.  **Privacy-Aware**: Native support for privacy levels from the start.
5.  **Staleness-Aware**: Built-in mechanisms to track how "fresh" a memory is.

---

## Base Memory Contract

All memory objects must implement the base contract fields.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Unique identifier for the memory item. |
| `kind` | String | The type of memory (e.g., `SemanticClaim`, `DecisionRecord`). |
| `created_at` | ISO8601 | When the record was first created. |
| `observed_at` | ISO8601 | When the underlying fact was last observed. |
| `extracted_by` | String | Agent ID or process that extracted/created this memory. |
| `confidence` | Float | 0.0 to 1.0 score of how certain we are about this memory. |
| `staleness` | Object | Metadata about the expiration/decay of this memory. |
| `privacy_level` | Enum | `public`, `internal`, `private`, `secret`. |
| `provenance` | Object | Detailed origin tracking (see Provenance Model). |
| `semantic_summary` | String | Human-readable (and LLM-readable) summary of the memory. |
| `structured_payload` | Object | Kind-specific structured data. |
| `schema_id` | String | Optional reference to a JSON schema for the payload. |
| `grammar_id` | String | Optional reference to a TARS Grammar for validation. |
| `embedding_ref` | String | Optional reference to an external vector embedding. |
| `ix_score_ref` | String | Optional reference to an IX engine scoring output. |
| `links` | Array | List of related `id`s and the nature of the relationship. |

### Staleness Model

```json
"staleness": {
  "ttl_seconds": 86400,
  "expires_at": "2025-12-25T12:00:00Z",
  "last_verified_at": "2025-12-24T12:00:00Z",
  "verification_count": 5
}
```

### Privacy Levels

-   **public**: Shared across all instances/users.
-   **internal**: Shared within the specific GuitarAlchemist organization/workspace.
-   **private**: Specific to a single agent instance or user.
-   **secret**: Encrypted at rest, never leaves the secure vault (credentials, keys).

---

## Provenance Model

Provenance is non-negotiable. Every memory must answer: "Who? When? From what?"

| Field | Description |
| :--- | :--- |
| `source_uri` | Primary URL or URI of the source material. |
| `source_repo` | Git repository where this was found (if applicable). |
| `source_path` | File path within the repository. |
| `source_issue` | Issue number/ID related to this extraction. |
| `evidence_chain` | Array of `id`s of `EvidenceRecord`s that support this item. |
| `content_hash` | Hash of the raw source content to detect changes. |

---

## Memory Kinds

### 1. MemoryItem (Generic)
A catch-all for general observations that don't yet fit a stricter schema.

### 2. SemanticClaim
A subject-predicate-object triple, aligned with the `Tars.Knowledge.Belief` system.
- **Payload**: `{ "subject": "id", "predicate": "is_a", "object": "id" }`

### 3. DecisionRecord
Records why a specific choice was made during execution or design.
- **Payload**: `{ "context": "...", "alternatives": [], "rationale": "..." }`

### 4. AssumptionRecord
Implicit or explicit assumptions made by an agent that need validation.
- **Payload**: `{ "assumption": "...", "risks": "...", "validation_plan": "..." }`

### 5. TaskRecord
High-level summary of a completed or planned task.
- **Payload**: `{ "goal": "...", "status": "completed", "outcome": "..." }`

### 6. CapabilityRecord
A mapping of what an agent or tool can do.
- **Payload**: `{ "capability": "...", "constraints": "...", "success_rate": 0.95 }`

### 7. ClosureRunMemory
Episodic memory from a specific execution "closure".
- **Payload**: `{ "run_id": "...", "steps_taken": [], "final_state": "..." }`

### 8. TraceSummary
Compressed summary of an execution trace.
- **Payload**: `{ "trace_id": "...", "bottlenecks": [], "key_events": [] }`

### 9. EvidenceRecord
Raw or semi-structured data chunks used to ground other memories.
- **Payload**: `{ "raw_text": "...", "snippets": [] }`

### 10. ReplayRecord
A deterministic sequence of inputs/outputs to reproduce a behavior.
- **Payload**: `{ "steps": [] }`

### 11. HumanNote
Direct input from a human user.
- **Payload**: `{ "note": "...", "author": "..." }`

---

## Integration with IX

The `ix_score_ref` allows TARS to query the IX engine for Monte Carlo Tree Search (MCTS) or other scoring evaluations of a memory's utility or truthfulness without making the memory dependent on IX's runtime.

## Storage Strategy

- **JSONL**: Recommended for append-only activity logs.
- **Parquet**: Recommended for periodic snapshots and high-performance analytical queries (DuckDB/Polars).
- **Git**: Certain `DecisionRecord`s or `CapabilityRecord`s should be promoted to the repository's `knowledge/` directory.
