# [TARS-V2] Implement Context Pruning for Memory Objects

**Repo:** `GuitarAlchemist/tars`
**Parent Epic:** GuitarAlchemist/tars#105
**Related Issues:** GuitarAlchemist/tars#106

## Problem Statement
The second-brain memory context grows linearly, leading to context window exhaustion during long agent sessions. As agents process more tasks, irrelevant memory items persist in the context, degrading performance and increasing costs.

## Goal
Implement an LRU-based pruning strategy for `MemoryItem` objects before injecting them into the LLM context.

## Non-Goals
- Do not implement vector similarity search in this issue.
- Do not delete items from the durable memory store (only prune from the active context).

## Acceptance Criteria
- [ ] A `PruneContext` function is added to the `MemoryStore` module.
- [ ] The function correctly identifies and removes the oldest items that exceed a specified token limit threshold.
- [ ] Tests verify that the most recently accessed/modified items are preserved.

## Test Plan
- Write unit tests in `Tars.Tests/MemoryStoreTests.fs`.
- Test pruning with an exact token limit match.
- Test pruning where the removal of one item drops the total below the threshold.
- Test behavior with an empty memory store.

## Governance & Policy
**Risk Policy:** medium
**Budget Policy:**
- **Tier:** free-local
- **Max Cost USD:** 0
- **Max Runner Minutes:** 15
**AFK Readiness:** candidate

## Stop Conditions
- If the token counting library cannot be integrated without adding new external Nuget dependencies (violating zero-dependency goals), halt and request review.

## Evidence Bundle
**Digester ID:** `tars-cortex`
**Digestion Timestamp:** `2024-06-26T10:00:00Z`
**Source Hash:** `sha256:7b2266696e64696e67223a226761702d323032342d303530227d`
