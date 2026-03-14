# ADR-007: Knowledge System of Record

**Date:** December 22, 2025  
**Status:** Accepted  
**Context:** Codex review identified multiple overlapping knowledge graph implementations

---

## Context

The codebase has multiple knowledge storage systems:

1. **Tars.Knowledge (NEW)** - Event-sourced belief ledger (Postgres-backed)
2. **Tars.Core.TemporalKnowledgeGraph** - Graphiti-style temporal graph
3. **Tars.Core.BeliefGraph** - Legacy in-memory graph
4. **Tars.Tools.GraphTools** - In-memory demo graph (for dev/testing)
5. **Neo4j** - External graph database

This creates confusion about the "system of record" and sync direction.

---

## Decision

### System of Record Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KNOWLEDGE SYSTEM OF RECORD                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. Postgres Ledger (TRUTH)                                                   │
│     ├── Source of truth for ALL beliefs                                       │
│     ├── Event-sourced (append-only)                                           │
│     ├── Full provenance                                                       │
│     └── Tables: knowledge_ledger, evidence_store, plans                       │
│                                                                               │
│  2. Neo4j/Graphiti (MATERIALIZED VIEW)                                        │
│     ├── Synced FROM Postgres                                                  │
│     ├── Optimized for traversal/path queries                                  │
│     ├── May lag slightly behind ledger                                        │
│     └── Can be rebuilt from ledger at any time                                │
│                                                                               │
│  3. In-Memory Graphs (CACHE)                                                  │
│     ├── Tars.Knowledge.BeliefGraph - Ledger cache                             │
│     ├── Tars.Core.TemporalKnowledgeGraph - Deprecated (Phase 10 removal)      │
│     └── GraphTools in-memory - Dev/demo only                                  │
│                                                                               │
│  SYNC DIRECTION: Ledger → Neo4j → In-Memory                                   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sync Strategy

1. **Writes** always go through `KnowledgeLedger.Assert()` / `Retract()` etc.
2. **Postgres** receives the event first (source of truth)
3. **In-memory BeliefGraph** is updated synchronously for fast reads
4. **Neo4j** is updated asynchronously (eventual consistency)
5. **Recovery**: Full rebuild from Postgres event replay

---

## Consequences

### Positive

- Single source of truth (Postgres)
- Full event history for audit/versioning
- Neo4j optimized for graph queries without compromising consistency
- In-memory cache for sub-millisecond reads

### Negative

- Eventual consistency between Postgres and Neo4j (acceptable tradeoff)
- Must maintain sync logic

### Migration Path

1. **Phase 9.1**: Postgres ledger tables + KnowledgeLedger service
2. **Phase 9.2**: Neo4j sync worker (FROM ledger TO graph)
3. **Phase 10**: Deprecate TemporalKnowledgeGraph, use KnowledgeLedger.BeliefGraph

---

## Addressing Codex Findings

| Finding | Resolution |
|---------|------------|
| Multiple overlapping graphs | This ADR defines single SoR |
| GraphTools in-memory | Documented as dev/demo only |
| Sync direction unclear | Ledger → Neo4j → Cache |

---

## Tool Access Policy (Addressing HIGH Finding)

### Current State

Tools can read arbitrary filesystem paths and mutate graph data without gates.

### Resolution

1. **Filesystem Access**: Tools should be limited to workspace roots
2. **Graph Mutations**: All mutations MUST go through ledger ingestion pipeline
3. **GraphTools**: Renamed to use "dev" prefix, documented as non-production

### Implementation Plan

- [ ] Add `ToolPolicy` type with allowed paths
- [ ] Enforce policy in `ToolRegistry` before execution
- [ ] Require provenance for all graph mutations
- [ ] Add approval gate for external data ingestion

---

## References

- [Architectural Vision](../1_Vision/architectural_vision.md)
- [Phase 9 Roadmap](phase9_symbolic_knowledge.md)
- Codex Review (December 22, 2025)
