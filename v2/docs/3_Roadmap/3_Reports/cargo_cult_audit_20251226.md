# TARS v2: Comprehensive Codebase Audit (BS, Mock, & Cargo Cult Detection)

**Date:** December 26, 2025
**Auditor:** TARS Autonomous Agent
**Scope:** `src/` directory (Core, Cortex, Metascript, Tools, Interface)

---

## 1. Executive Summary

This audit identifies areas where the codebase uses "architectural astronauting," placeholder logic, or simplified "cargo cult" algorithms that do not meet the claimed design intent. While some mocks are necessary for testing, several have leaked into production or serve as brittle bridges for missing features.

**Overall Verdict**: The system is conceptually mature but contains significant "operational scaffolding" that must be replaced with robust implementations to achieve Phase 17+ goals.

---

## 2. Architectural Astronauting (Empty Shells)

These modules define complex types but provide nearly zero functional implementation.

### 2.1 Pattern Recognition (`src/Tars.Core/PatternRecognition.fs`)
- **Claim**: "Detect patterns and anomalies in the graph."
- **Reality**: `TagEntity` uses "mock logic for now" (Line 21). `DetectAnomalies` only checks for isolated nodes.
- **Impact**: Knowledge graph enrichment is currently decorative.

### 2.2 Cognitive Analyzer (`src/Tars.Cortex/CognitiveAnalyzer.fs`)
- **Claim**: "Analyzes the system's cognitive state... Eigenvalue... Entropy."
- **Reality**: `Eigenvalue` is a hardcoded penalty calculation. `GroundingFidelity` is a "placeholder for now" (Line 90) that just counts capabilities and divides by 100.
- **Impact**: The "Self-Correction" based on entropy is currently a pseudo-random trigger.

### 2.3 Output Guard Analyzer (`src/Tars.Core/OutputGuardAnalyzerStub.fs`)
- **Claim**: "Stub analyzer that can be swapped with real LLM-based cargo-cult analysis."
- **Reality**: Returns `None` by default. 
- **Impact**: Provides a false sense of security; no actual guardrails are active beyond simple regex.

---

## 3. Production Mocks & Stubs

Code that belongs in `tests/` but is currently in `src/`.

### 3.1 Vector Store Stubs (`src/Tars.Cortex/VectorStore.fs` & `PostgresVectorStore.fs`)
- **Problem**: `PruneAsync` and `UpdateMetadataAsync` are empty stubs marked "Skipping for MVP."
- **Impact**: Memory decay (Phases 11 & 15) is impossible without these implementations.

### 3.2 Demo Models (`src/Tars.Interface.Cli/Commands/MacroDemo.fs` & `RagDemo.fs`)
- **Problem**: `MockLlm` and `StubLlm` are implemented inside command files to avoid API calls.
- **Impact**: Brittle demo code mixed with CLI logic.

### 3.3 Agent Stubs (`src/Tars.Interface.Cli/Commands/AgentHelpers.fs` & `Agent.fs`)
- **Problem**: `mockRegistry` and `mockExecutor` are used to bootstrap agent contexts.
- **Impact**: Agents in the CLI are isolated and cannot actually collaborate or look up each other in a real registry.

---

## 4. Cargo Cult Algorithms

Simplified implementations that borrow names from complex algorithms but lack the core logic.

### 4.1 "BM25" Scoring (`src/Tars.Metascript/Engine.fs`)
- **Claim**: "BM25-like keyword scoring."
- **Reality**: Lacks Inverse Document Frequency (IDF) because there is no corpus statistics tracking (Line 155). It's essentially just term frequency (TF).
- **Impact**: Retrieval quality will degrade as the memory store grows.

### 4.2 "Cross-Encoder Reranking" (`src/Tars.Metascript/Engine.fs`)
- **Claim**: "Rerank using cross-encoder."
- **Reality**: Sends documents to an LLM with a prompt "Rate relevance 0-10" (Line 645).
- **Impact**: This is extremely slow and expensive compared to a real cross-encoder, and highly susceptible to prompt injection or model bias.

### 4.3 "Multi-Hop Retrieval" (`src/Tars.Metascript/Engine.fs`)
- **Claim**: "Perform multi-hop retrieval using knowledge graph."
- **Reality**: Simple BFS search with manual weights (0.8, 0.7, 0.0) based on hardcoded relation types (Line 380).
- **Impact**: Knowledge discovery is limited to shallow, pre-defined links.

---

## 5. Critical Gaps (The TODO Debt)

| File | Line | TODO Description |
|------|------|------------------|
| `Metascript/Engine.fs` | 1060 | **Evaluate Condition logic**: Workflows cannot branch based on variables yet. |
| `Metascript/IrCompiler.fs`| 26 | **Control flow mapping**: The IR compiler skips flow logic. |
| `Core/AgentWorkflow.fs` | 443 | **Measure semantic distance**: The `stabilize` (Inductor) combinator is a NoOp. |
| `Cortex/CapabilityStore.fs`| 136 | **Metrics tracking**: Performance-based routing is not implemented. |
| `Symbolic/Invariants.fs` | 83 | **Contradiction detection**: Real logic is missing. |

---

## 6. Recommendations

1.  **Consolidate Mocks**: Move all `Mock*` and `Stub*` implementations from `src/` to a new shared test library `tests/Tars.Mocks`.
2.  **Implement or Prune**: If a module is "architectural astronauting" (like the current `PatternRecognition`), either implement a minimal *real* logic or remove the file until needed.
3.  **Unify RAG Logic**: Move the "Cargo Cult" algorithms from `Metascript/Engine.fs` into `Tars.Cortex` and implement real versions (e.g., actual BM25 with a local stats store).
4.  **Close the Loop**: Prioritize the **Condition Evaluation** in `Metascript/Engine.fs` as it blocks real autonomous workflows.
5.  **Ground the Metrics**: Replace hardcoded metrics in `CognitiveAnalyzer` with actual data from `Metrics.fs` and `ToolLedger.fs`.

---
*End of Audit Report*
