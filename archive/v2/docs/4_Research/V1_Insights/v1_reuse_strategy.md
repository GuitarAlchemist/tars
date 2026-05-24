# TARS V1 Reuse Strategy for V2

**Date:** December 20, 2025  
**Status:** Approved for Implementation  
**Source:** [v1_component_reusability_analysis.md](./v1_component_reusability_analysis.md), [v1_tars_artifacts_analysis.md](./v1_tars_artifacts_analysis.md)

---

## 🎯 Objective
Leverage the high-value assets and battle-tested logic from TARS v1 to accelerate v2 development while maintaining the new architecture's clean separation and pragmatic philosophy.

---

## 🛠️ Phase 1: Immediate Porting (Epic 3 & 5)

### 1.1 AgenticTraceCapture (Observability)
- **Source:** `v1/src/TarsEngine.FSharp.Core/Tracing/AgenticTraceCapture.fs`
- **Destination:** `Tars.Observability.Tracing` (New project)
- **Actions:**
    - Extract event types (`AgentEvent`, `InterAgentCommunication`, etc.).
    - Port session management and JSON serialization logic.
    - Adapt to v2's `SemanticMessage` and `AgentId` models.
    - **Goal:** Comprehensive structured trace capture for every run.

### 1.2 Agent Registry Bootstrapping
- **Source:** `v1/.tars/agents/tars_agent_organization.yaml`
- **Destination:** `v2/config/agents/initial_registry.yaml`
- **Actions:**
    - Copy the organizational blueprint to v2.
    - Implement a YAML loader in `Tars.Kernel.Registry` to bootstrap agents from this file.
    - Ensure agents are created as data-driven definitions rather than hardcoded functions.

---

## 🧠 Phase 2: Cortex Enhancement (Epic 2)

### 2.1 Grammar Migration
- **Source:** `v1/.tars/grammars/*.tars`
- **Destination:** `v2/resources/grammars/`
- **Actions:**
    - Migrate core EBNF grammars (Wolfram, MiniQuery, RFCs).
    - Update `Tars.Cortex.Grammar` to support loading these definitions.
    - Integrate `GrammarConstraint` into the `ICognitiveProvider.AskAsync` interface.

### 2.2 Cognitive Provider Refinement
- **Source:** `v1/src/TARS.AI.Inference/TarsInferenceEngine.fs` logic
- **Destination:** `Tars.Llm` & `Tars.Cortex.Inference`
- **Actions:**
    - Enhance `ICognitiveProvider` with structured output support (via grammars).
    - Port robust error handling and model fallback logic.

## ✅ Phase 3: Metascript & FLUX Full Port (Epic 5) - COMPLETE

### 3.1 Unified Metascript Engine (Full Port)
- **Status:** ✅ **COMPLETED**
- **Actions:**
    - **Parser Port:** Ported the full `.tars`/`.trsx` block-based parser.
    - **Deterministic Logic:** Implemented `FSharpBlockHandler` using F# Interactive (FSI).
    - **Variable System:** Ported rich variable system with interpolation.
    - **Literate Programming:** Enabled seamless mixing of LLM prompts and raw code.

### 3.2 Grammar-Metascript Bridge
- **Status:** ✅ **COMPLETED**
- **Actions:**
    - **Grammar-Aware Blocks:** Added support for `grammar="Name"` parameter.
    - **On-the-fly Validation:** Integrated `Tars.Cortex.Grammar` validation.
    - **Type-Safe Outputs:** Automated mapping of LLM outputs to variables.

### 3.3 FLUX Multi-Engine Execution
- **Status:** ✅ **COMPLETED** (Core Handlers)
- **Actions:**
    - **Executor Registry:** Implemented extensible block handler registry.
    - **Inter-Engine Data Flow:** Enabled data flow between F#, Query, and Shell blocks.

---

## 🚀 Phase 4: Evolution Loop (Epic 6 & 8)

### 3.1 Self-Improvement Logic (Ouroboros)
- **Source:** `v1/.tars/tars-self-improvement-cycle.trsx` logic
- **Destination:** `Tars.Evolution.Engine`
- **Actions:**
    - Extract `analyzeCurrentIntelligence` and `designIntelligenceEnhancements` logic.
    - Implement as F# functions in the `Tars.Evolution` project.
    - Integrate these into the `EvolutionState` transition loop.

---

## 📋 Summary of Reusable Artifacts

| Artifact | Type | Value | Priority |
|----------|------|-------|----------|
| `AgenticTraceCapture.fs` | Code | Observability Foundation | **P0** |
| `tars_agent_organization.yaml` | Data | Ready-made Org Structure | **P0** |
| `.tars` / `.trsx` Engine | Code | Rich Workflow Capability | **P1** |
| `*.tars` (Grammars) | Resource | Structured Reasoning | **P1** |
| `tars-self-improvement-cycle.trsx` | Logic | Evolution Blueprint | **P1** |

---

## 🚫 What NOT to Reuse (Explicitly Excluded)
- **CUDA/GPU Kernels:** Defer to v3.
- **Advanced Math (Sedenions):** Defer to v3.
- **Fractal Grammars:** Defer to v3.
- **Janus Cosmology Logic:** Domain-specific, keep as reference only.

---

## 🔗 Next Steps
1. Add these porting tasks to `docs/3_Roadmap/1_Plans/implementation_plan.md`.
2. Create the `Tars.Observability.Tracing` project.
3. Begin Agent Registry migration.
