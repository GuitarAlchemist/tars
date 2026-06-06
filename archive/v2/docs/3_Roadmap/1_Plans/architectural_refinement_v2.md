# TARS v2 Refinement Plan (Architecture Critique Integration)

**Source:** Architecture Review (ChatGPT)
**Date:** December 25, 2025
**Status:** In Progress

## Overview

Refining TARS v2 to be more measurable, explicit, and evolution-oriented based on external architectural critique.

## 1. Explicit Execution Loop
**Critique:** "Control loop is implicit. Agents run freely."
**Action:** Define `RunCycleContext`.
- [ ] Define `RunCycleContext` (TraceId, StartTime, Budget, Goal).
- [ ] Ensure all agents accept `RunCycleContext` as input.

## 2. Diagnostics as First-Class Citizen
**Critique:** "Logging != Measuring cognition. If it can't be graphed, it's not real."
**Action:** Structured `Diagnostics`.
- [ ] Add `PuzzleResult` with `Metrics` (Entropy, Latency, TokenEfficiency).
- [ ] Create `DiagnosticsAgent` (Runs post-reasoning).
- [ ] Persist `diagnostics.trsx` (structured run data).

## 3. Memory Decay & Promotion
**Critique:** "Accumulating everything is biologically wrong."
**Action:** Implement Memory Lifecycle.
- [ ] Add `MemoryScoringAgent`.
- [ ] Implement `Prune()` for `VectorStore` based on access frequency (Half-life).

## 4. Agent Spawning Costs
**Critique:** "Agents proliferate without friction."
**Action:** Budget & Fitness.
- [ ] Add `SpawningBudget` to `RunCycleContext`.
- [ ] Track `FitnessScore` per agent.

## 5. Convergent DSLs
**Critique:** "DSLs risk fragmentation."
**Action:** Core Intermediate Representation (IR).
- [ ] Define "Core TARS IR" (Belief, Goal, Constraint, Evidence).
- [ ] Compile `.trsx` and `.tars` to this IR.

## 6. Operational Reflection
**Critique:** "Reflection that only produces text is therapy."
**Action:** Code-level consequences.
- [ ] `OperationalReflection` can disable agents or adjust weights.
- [ ] Reflection produces `StateUpdate` events, not just logs.

## 7. Reduced LLM Dependence
**Critique:** "LLMs should generate ideas, not arbitrate truth."
**Action:** Symbolic Validation.
- [ ] Use F# types for strict validation.
- [ ] Use deterministic rules for memory pruning.

## 8. Define Success per Run
**Critique:** "A run shouldn't just 'finish'."
**Action:** Explicit Success Criteria.
- [ ] Add `SuccessCriteria` to `TaskDefinition`.
- [ ] Return `RunResult` (Success/Partial/Failure + Confidence).

## 9. Philosophical Tension
**Critique:** "TARS reasons but wants nothing."
**Action:** Competing Goals.
- [ ] Introduce `BaseDrives` (e.g., Accuracy vs. Speed).
