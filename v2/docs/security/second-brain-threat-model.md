# Second-Brain Threat Model

- **Status:** Draft
- **Area:** Security / Memory
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## Purpose
This document outlines the threat model for exposing the TARS second brain via the Model Context Protocol (MCP) and A2A interfaces. It identifies potential attack vectors, privacy risks, and details the necessary mitigations to ensure safe, read-only defaults.

## 1. Threat Scenarios

### T1: Unbounded Queries (Denial of Service / Context Overflow)
- **Description:** A client or compromised agent requests a massive slice of memory (e.g., querying for all traces without a filter), either maliciously or accidentally. This can exhaust context windows, cause expensive API usage (if passed to an LLM), or lead to system OOM (Out Of Memory).
- **Mitigation:**
  - All memory extraction and search endpoints must enforce hard pagination limits (e.g., `limit=10`).
  - Search tools must require an explicit, narrow `scope` parameter.
  - Implement token-estimation before dispatching A2A artifacts (e.g., `ContextPack`).

### T2: Secret Exfiltration / Privacy Leakage
- **Description:** Sensitive information (API keys, user data, proprietary prompts) inadvertently ingested into the second brain is exposed via MCP reads or searches.
- **Mitigation:**
  - All memory items must have a data classification level. By default, MCP read operations filter out anything not explicitly marked `public` or `safe`.
  - A pre-ingestion redaction layer should strip known secret patterns.
  - The first pass implementation of MCP tools is explicitly **read-only**, meaning attackers cannot use the tool surface to *insert* new extraction payloads.

### T3: Prompt Injection via Contaminated Memory
- **Description:** An attacker places malicious text in an external system (e.g., a GitHub issue) that gets ingested into the second brain. Later, when an agent retrieves that memory via `tools/search_memory`, the text acts as a prompt injection payload against the retrieving agent.
- **Mitigation:**
  - Memory retrieved via MCP is treated strictly as **data**, not instructions.
  - The `prompts/get_context_assembly` prompt must explicitly instruct the LLM to sandbox the retrieved content and not treat it as executable commands.
  - Strict provenance tracking (`source_hash`, `digester_id`) ensures that if a contamination occurs, the source can be blacklisted.

### T4: Unauthorized Mutation
- **Description:** An agent or connected MCP client attempts to delete, overwrite, or corrupt memory records without authorization.
- **Mitigation:**
  - The exposed MCP interface for the second brain is structurally **read-only**.
  - Any future write operations must pass through Demerzel's governance framework (e.g., requiring tribunal approval or an explicit signed A2A contract).
  - TARS grammar validation must reject unapproved mutation commands at the parser level.

### T5: Hallucination Amplification (Stale or False Context)
- **Description:** Agents act confidently on retrieved memory that is outdated, contradicted by newer data, or was poorly generated initially.
- **Mitigation:**
  - Mandatory staleness markers (`last_verified_date`) and confidence scores must accompany every retrieved memory object.
  - Agents must use `tools/explain_memory_provenance` when a critical decision relies on a piece of memory, allowing them to assess its validity.

## 2. Safe Defaults Summary
To adhere to the GuitarAlchemist ecosystem's security posture, the following safe defaults are applied unconditionally:

1. **Deny-by-Default Visibility:** Memory items without explicit clearance metadata are hidden from external tools.
2. **Read-Only Surface:** No mutation tools are exposed in the initial design.
3. **Mandatory Scoping:** A `scope` parameter is universally required for search/list operations.
4. **Token Caps:** Hard limits on the size of returned `ContextPack` and `search_memory` arrays.
5. **Provenance Transparency:** No memory is ever returned without its origin ID and timestamp attached.