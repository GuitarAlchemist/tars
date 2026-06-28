# Second-Brain MCP and A2A Exposure Design

- **Status:** Draft
- **Area:** Runtime / Memory
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## Purpose
This document defines how TARS safely exposes its second-brain memory through the Model Context Protocol (MCP) and Agent-to-Agent (A2A) artifacts. The design prioritizes privacy, provenance, and safety to prevent data leakage and unbounded context growth while making the harness's knowledge usable by external clients and autonomous sub-agents.

## 1. Candidate MCP Surfaces
The MCP server will expose explicit read-only boundaries by default. The following resources, prompts, and tools are designed to surface targeted context without overwhelming the client.

### Resources
- `resources/list_second_brain_collections`: Returns metadata about available knowledge scopes (e.g., `traces`, `patterns`, `issues`, `docs`).
- `resources/read_memory_item/{id}`: Returns a specific memory object by its unique identifier (e.g., UUID or Hash).

### Prompts
- `prompts/get_context_assembly`: A system-level prompt that instructs the model on how to interpret and cite memory items when reasoning. It ensures the model treats memories as "context candidates" rather than absolute ground truth.

### Tools
- `tools/search_memory`: Searches the second brain based on a query string and a mandatory `scope` parameter to limit the blast radius. Returns a bounded array of results.
- `tools/explain_memory_provenance`: Given a memory ID, returns its lineage, creation timestamp, author (digester_id), and original source hash.

## 2. Candidate A2A Artifacts
A2A (Agent-to-Agent) artifacts are serialized structures shared between TARS and external components (like IX, ga, Demerzel). They must be highly structured and strictly scoped.

- **MemoryBrief**: A short, high-density summary of a specific topic, intended to fit within strict token budgets.
- **ContextPack**: A bundled collection of relevant memory items, explicitly bounded by relevance score and scope, typically generated as a response to an agent's `tools/search_memory` call.
- **DecisionBrief**: A record of a past decision, including the rationale, alternatives considered, and outcome.
- **TraceSummary**: A condensed version of a full execution trace (e.g., from GA), highlighting key actions and results without the verbosity of raw logs.
- **ReplayReport**: An artifact containing the state necessary to reproduce or simulate a past execution.
- **IXScorecard**: A performance or metric scorecard specifically formatted for consumption by the IX skill engine.

## 3. Consent and Scope Requirements
To enforce safety and privacy across the ecosystem, the following constraints must govern all memory exposures:

### Scope Limitation
- Memory is *never* exposed globally or fully.
- All retrieval operations (`tools/search_memory`, A2A artifact generation) must require an explicit scope (e.g., `scope="repository-docs"` or `scope="ga-traces"`).

### Provenance and Staleness
- Every exposed memory item must include its provenance (source hash, digester ID) and a staleness marker (e.g., `last_verified_date`, `is_stale` flag).
- Seldon (the teaching persona) and other consumers must use this metadata to weigh the reliability of the information.

### Redaction and Privacy
- Private data and secrets must be redacted or entirely omitted before transmission.
- A "Privacy Level" or classification label (e.g., `public`, `internal`, `restricted`) should be enforced at the memory extraction layer.

### Mutation Approval
- The first pass is strictly **read-only**.
- Any future tool-driven memory mutations (e.g., `tools/update_memory`) must require an explicit, signed contract and human-in-the-loop (or Demerzel tribunal) approval.

### Contextual Candor
- Memory search results are explicitly labeled as "context candidates." Agents must be instructed not to treat retrieved memory as unassailable truth but as historical context subject to verification.

## 4. Implementation Guidelines
- **Format:** All A2A artifacts must be JSON-first.
- **F# Compatibility:** Design data structures to map cleanly to F# immutable records and discriminated unions.
- **Errors:** Invalid scopes or unauthorized access should return standard MCP error codes mapped to F# `Result.Error`.
