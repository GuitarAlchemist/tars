# Phase 11: Cognitive Grounding & Production Intelligence

**Status**: 🔜 Planned (January 2025)  
**Goal**: Transform TARS from "task-level cognition" (L2) to "grounded, self-verifying, ledger-guided" cognition (L3).

---

## Overview

Phase 11 addresses critical gaps in TARS's cognitive architecture that prevent it from being truly production-ready and self-improving.

**Core Insight**:
> TARS is at "task-level cognition with reflective scaffolding" — it can reason, reflect, and log beliefs, but it isn't grounded, self-verifying, or ledger-guided in behavior yet.

---

## Core Problems Addressed

| Problem | Current State | Target State |
|---------|---------------|--------------|
| Self-referential evaluation | LLM judges LLM output | External verification (tests, static checks) |
| Ledger is write-only | Knowledge stored but never consulted | Belief-aware prompts, contradiction gating |
| Timeout is ceremonial | `TaskDefinition.Timeout` unused | Real CTS enforcement with cancellation |
| Memory drift | 3+ parallel stores with no reconciliation | Unified consolidation layer |
| Evidence proposals orphaned | Proposals stored but never promoted/invalidated | Automated verification pipeline |

---

## 11.1 External Verification Gateway

**Priority**: Critical  
**Files**: `Tars.Evolution/Evaluation.fs`, `Tars.Evolution/Verification.fs` (new)

| Task | Status | Description |
|------|--------|-------------|
| `IVerifier` interface | 🔜 | `VerifyAsync(artifact: string) -> VerificationResult` |
| `TestRunner` verifier | 🔜 | Runs `dotnet test`, captures results |
| `StaticAnalyzer` verifier | 🔜 | Runs analyzers, linters |
| `ExecutionVerifier` | 🔜 | Sandboxed execution with assertions |
| Gate evolution on verifiers | 🔜 | Success requires external verification |
| CLI `tars verify <artifact>` | 🔜 | User interface |

---

## 11.2 Ledger Read-Back Integration

**Priority**: Critical  
**Files**: `Tars.Evolution/Engine.fs`, `Tars.Cortex/LedgerAwarePrompting.fs` (new)

| Task | Status | Description |
|------|--------|-------------|
| Query ledger before execution | 🔜 | Relevant beliefs lookup |
| Inject belief context | 🔜 | "Known facts: [...]" in prompts |
| Contradiction gating | 🔜 | Refuse actions violating beliefs |
| `LedgerContext.GetRelevantBeliefs` | 🔜 | Topic-based belief retrieval |
| Belief influence on curriculum | 🔜 | Guide task generation |

---

## 11.3 Real Timeout Enforcement

**Priority**: High  
**Files**: `Tars.Evolution/Protocol.fs`, `Tars.Evolution/Engine.fs`

| Task | Status | Description |
|------|--------|-------------|
| Wire `Timeout` to CTS | 🔜 | CancellationTokenSource |
| Pass CTS to all calls | 🔜 | LLM and tool calls |
| Graceful shutdown | 🔜 | Save partial state on timeout |
| Timeout metrics | 🔜 | Track timeout occurrences |
| Remove loop-count fallbacks | 🔜 | When timeout specified |

---

## 11.4 Memory Consolidation Layer

**Priority**: High  
**Files**: `Tars.Knowledge/Consolidation.fs` (new)

| Task | Status | Description |
|------|--------|-------------|
| Audit parallel stores | 🔜 | VectorStore, Graph, Ledger, SemanticMemory |
| Canonical knowledge format | 🔜 | Unified representation |
| `ConsolidationService` | 🔜 | Reconcile stores periodically |
| Decay/forgetting | 🔜 | Low-confidence belief pruning |
| CLI `tars memory consolidate` | 🔜 | User interface |

---

## 11.5 Evidence Lifecycle Automation

**Priority**: Medium  
**Files**: `Tars.Knowledge/Ledger.fs`, `Tars.Knowledge/EvidenceVerifier.fs` (new)

| Task | Status | Description |
|------|--------|-------------|
| In-memory evidence storage | 🔜 | Full implementation |
| Background verifier service | 🔜 | Process pending proposals |
| Auto-promote verified | 🔜 | Proposals → Beliefs |
| Auto-invalidate contradicted | 🔜 | Retract conflicting |
| CLI `tars know verify` | 🔜 | User interface |

---

## 11.6 Code Cleanup

**Priority**: Low  
**Files**: Various

| Task | Status | Description |
|------|--------|-------------|
| Wire reviewer agent | 🔜 | Into evolution loop OR remove |
| Connect `KnowledgeBase` | 🔜 | To runtime OR remove initialization |
| Complete `PatternRecognition.fs` | 🔜 | OR remove mock code |
| Audit dead code | 🔜 | Find other unused paths |

---

## Success Criteria

- [ ] `tars evolve` tasks pass ONLY if external verifiers pass (not just LLM approval)
- [ ] `tars evolve` prompts include relevant beliefs from ledger
- [ ] `TaskDefinition.Timeout` actually cancels after specified duration
- [ ] `tars memory status` shows unified view across all stores
- [ ] Evidence proposals auto-verify/invalidate within 1 hour

---

## Dependencies

- Phase 9: Symbolic Knowledge (for ledger integration)
- Phase 7: Production Hardening (for metrics/logging)

---

*Phase 11 planned: January 2025*
