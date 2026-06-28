# TARS V2 Second-Brain Memory Objects Contract

## Goal
Define the core memory object contracts and provenance model for the TARS V2 second-brain harness. The system requires structured memory objects; vague blobs are insufficient. Each memory item must include provenance, confidence, staleness, source links, semantic type, and an optional grammar-validated contract.

## Design Constraints
- **JSON-First:** Easy to store as JSONL/Parquet.
- **Future-Proof:** Compatible with future vector indexing without requiring vectors as the source of truth.
- **IX Compatibility:** Compatible with IX scoring outputs, but does not mandate IX to be used.
- **Privacy & Safety:** Supports privacy levels, staleness, and provenance out of the box. No secrets or raw private logs are stored.
- **Simplicity:** Does not require a cloud database and avoids a complex, over-engineered ontology for the initial loop.

---

## Memory Objects

Below are the candidate memory objects for the TARS second-brain ecosystem:

- `MemoryItem`: The base wrapper for any memory artifact.
- `SemanticClaim`: A verifiable claim or fact extracted from a source.
- `DecisionRecord`: An immutable record of a decision made by an agent or human.
- `AssumptionRecord`: Documented assumptions, often tied to a `DecisionRecord`.
- `TaskRecord`: High-level tracking of action items or task execution.
- `CapabilityRecord`: Asserted or demonstrated capabilities of an agent or system.
- `ClosureRunMemory`: A snapshot of memory captured at the conclusion of a significant run.
- `TraceSummary`: Distilled context from a raw execution trace.
- `EvidenceRecord`: Supporting material validating a semantic claim.
- `ReplayRecord`: Information required to deterministically replay a specific action or chain.
- `HumanNote`: Manually authored context injected into the system.

---

## Shared Contract Fields

Most memory objects implement a subset of these core fields:

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier (e.g., UUID or content hash). |
| `kind` | `string` | Type of the object (e.g., `"DecisionRecord"`). |
| `source_uri` | `string` | Canonical URI to the primary source. |
| `source_repo` | `string` | GitHub or local repository identifier. |
| `source_path` | `string` | File path within the repository. |
| `source_issue` | `string` | Issue/PR number or link. |
| `created_at` | `timestamp` | Time of creation of this memory artifact. |
| `observed_at` | `timestamp` | Time the agent extracted or noticed the underlying fact. |
| `extracted_by` | `string` | Agent or tool that created this record. |
| `confidence` | `float` | Subjective or statistical confidence (0.0 to 1.0). |
| `staleness` | `object` | Expiration strategy and status (see below). |
| `privacy_level` | `string` | Classification (e.g., `"public"`, `"internal"`, `"confidential"`). |
| `provenance` | `object` | Lineage describing how this artifact was synthesized. |
| `semantic_summary`| `string` | Human-readable explanation of the memory. |
| `structured_payload`| `object` | The strongly-typed data of the memory. |
| `schema_id` | `string` | Optional JSON schema identifier for the payload. |
| `grammar_id` | `string` | Optional EBNF grammar identifier for validation. |
| `embedding_ref` | `string` | External reference to an embedding. |
| `ix_score_ref` | `string` | External reference to an IX score evaluation. |
| `links` | `array` | Relationships to other memory item IDs. |

---

## Provenance Model

The provenance model requires that every synthetic or derived fact maintains a line of sight to its origin. Unbounded knowledge ingestion is rejected.

```json
"provenance": {
  "derived_from": ["id-1", "id-2"],
  "extraction_method": "Tars.Evolution.Pipeline",
  "prompt_hash": "a1b2c3d4",
  "grounding_sources": [
    "https://github.com/GuitarAlchemist/tars/issues/90"
  ]
}
```

## Staleness Mechanism

Information degrades over time. Staleness attributes allow the system to decay the value or retrieve-ability of a memory.

```json
"staleness": {
  "expires_at": "2026-12-31T23:59:59Z",
  "review_required": true,
  "decay_factor_per_day": 0.01
}
```

---

## Privacy Notes

The TARS agent operates over mixed-visibility environments.
- **`privacy_level`** must be strictly enforced. Valid enum values are typically `public`, `internal`, and `confidential`.
- **Secrets:** Under no circumstances should secrets (API keys, PII, raw private logs) be stored in `structured_payload` or `semantic_summary`. The agent must strip secrets before creating memory objects.
- Filtering happens at the RAG layer based on this attribute.

---

## Cost Notes

Vector search and LLM extraction are expensive.
- **No vector truth:** Vectors are pointers, not truth. They can be deleted and regenerated.
- **JSON storage:** Local JSON/JSONL/Parquet reduces cloud storage costs to zero.
- RAG systems should prioritize graph relationships (`links`) and `source_repo`/`source_path` indexing to minimize expensive global vector searches.
