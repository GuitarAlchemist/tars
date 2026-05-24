# Cargo Cult Code Analysis & Removal Log

This document tracks "cargo cult" code—legacy artifacts, over-engineered features, or unused patterns—that have been identified in TARS v1/v2 and removed or flagged for exclusion to ensure a lean, efficient architecture.

## 1. Removed Components

### Core Kernel & Infrastructure

* **`src/Tars.Core/Kernel.fs` (Mini-Kernel)**
  * **Description:** A lightweight, in-memory agent registry and kernel implementation.
  * **Reason for Removal:** Redundant with the dedicated `Tars.Kernel` project. It created confusion about the source of truth for agent management.
  * **Replacement:** `Tars.Kernel` (specifically `AgentRegistry`, `AgentFactory`, `EventBus`).

### Grammar & Patterns

* **`src/Tars.Core/GrammarTypes.fs` (16-Tier Evolution)**
  * **Description:** A complex discriminated union defining 16 specific tiers of AI evolution (e.g., "Tier12_SymbolicGrounding") and associated evolution records.
  * **Reason for Removal:** "Visionary" over-engineering. The current roadmap focuses on practical, iterative improvement (Tiers 1-3), making a rigid 16-tier structure unnecessary bloat.
  * **Replacement:** Simplified, dynamic evolution metrics in `Tars.Evolution`.

* **`src/Tars.Core/GrammarPipeline.fs`**
  * **Description:** A complex pipeline for grammar distillation, pattern detection, and storage.
  * **Reason for Removal:** Unused and overly complex for the current needs.
  * **Replacement:** On-demand grammar generation in `Tars.Graph` (e.g., `GrammarDistill.fs` - though this is also under review).

* **`src/Tars.Core/Patterns.fs` & `ErrorPatterns.fs`**
  * **Description:** Hardcoded implementations of specific agentic patterns (Chain of Thought, ReAct) and error taxonomies.
  * **Reason for Removal:** Agent behaviors should be emergent or defined in `Metascripts` (FLUX), not hardcoded in F# types.
  * **Replacement:** `Tars.Graph` (Graphiti) and Metascripts.

* **`src/Tars.Evolution/TierEvolution.fs`**
  * **Description:** Logic specifically tied to the 16-tier grammar evolution system.
  * **Reason for Removal:** Dependent on the removed `GrammarTypes` and represented a legacy approach to evolution.
  * **Replacement:** `Tars.Evolution/Engine.fs` (General-purpose evolutionary loop).

### Domain Types

* **`GraphNode` & `GraphEdge` in `Tars.Core/Domain.fs`**
  * **Description:** Duplicate definitions of graph elements.
  * **Reason for Removal:** Redundant with `Tars.Graph/Domain.fs`.
  * **Replacement:** `Tars.Graph` definitions.

## 2. Flagged for Review / Potential Cargo Cult

### `src/Tars.Core/GrammarDistill.fs`

* **Status:** **Kept (for now)**
* **Description:** A regex-based JSON validator and prompt hint generator.
* **Critique:** Naive implementation. Regex is insufficient for robust JSON validation.
* **Recommendation:** Replace with a proper schema validation library (e.g., `JsonSchema.Net`) or LLM-native structured output constraints.

### `src/Tars.Core/Functional.fs`

* **Status:** **Kept (Minimal)**
* **Description:** Functional programming utilities (`AsyncResult`, `Reader`, `Writer`).
* **Critique:** Often leads to "functional purity" for its own sake.
* **Action:** Kept only the essential `AsyncResult` builder used extensively. Other unused combinators should be pruned if found.

## 3. Principles for Avoiding Cargo Cult in v2

1. **YAGNI (You Aren't Gonna Need It):** Do not implement features for "Tier 10" when we are at "Tier 1".
2. **No Duplicate Abstractions:** If a concept exists in a specialized project (e.g., `Tars.Graph`), do not duplicate it in `Tars.Core`.
3. **Emergence over Hardcoding:** Avoid hardcoding complex behaviors (like specific reasoning patterns) that should ideally be learned or scripted dynamically.
4. **Single Source of Truth:** `Tars.Kernel` is the core. `Tars.Core` is for shared domain types and pure functions.
