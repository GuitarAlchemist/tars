# TARS v2 Architecture Robustness Audit

**Date:** 2025-12-03
**Source:** [Construire un agent IA robuste : les leçons de Guillaume Laforge](https://www.blogdumoderateur.com/construire-agent-ia-robuste-lecons-guillaume-laforge/)

## Executive Summary

This document evaluates the TARS v2 architecture against the robustness patterns identified by Guillaume Laforge (Google Cloud). While TARS v2 aligns well with orchestration and agent-to-agent communication patterns, a critical gap exists in how tools are exposed to agents.

## Compliance Matrix

| Pattern | Status | Analysis |
| :--- | :--- | :--- |
| **1. The Orchestrator** | ✅ **Compliant** | TARS uses `GraphRuntime` and `EvolutionEngine` to decompose tasks. The separation of `Planning` (Curriculum Agent) and `Coding` (Executor Agent) aligns with the recommendation to specialize sub-tasks. |
| **2. Rethink Tools** | ❌ **Non-Compliant** | **Critical Gap.** TARS currently exposes low-level OS primitives (`run_command`, `read_file`, `web_fetch`) via `StandardTools.fs`. The recommendation is to expose high-level "business capabilities" (e.g., `SummarizeDocument`, `RefactorCode`) to reduce agent confusion and hallucinations. |
| **3. MCP Standardization** | 🔄 **In Progress** | Adoption of the Model Context Protocol (MCP) is underway via the Auggie integration. |
| **4. Agent-2-Agent (A2A)** | ✅ **Compliant** | The `SemanticMessage` structure (Performative, Intent, Ontology) provides a robust protocol for inter-agent communication, mirroring Google's A2A approach. |
| **5. No "Chatbot Mandate"** | ✅ **Compliant** | TARS is designed as an autonomous task engine (`evolve`, `run`), not primarily as a conversational bot. |
| **6. Evaluation & Grounding** | ⚠️ **Partial** | While `EpistemicGovernor` tracks belief status (`Hypothesis` vs `VerifiedFact`), the system lacks a rigorous "Golden Response" benchmark set for domain-specific validation. |

## Action Plan

To address the critical gap in **Pattern 2**, we will implement a "Semantic Tooling" layer.

### 1. Deprecate Low-Level Tools

Direct access to `run_command` (shell) and raw `read_file` will be restricted for high-level agents.

### 2. Implement Semantic Capabilities

We will create new, intent-driven tools:

* **Filesystem Capabilities**:
  * `ExploreProject`: Returns a summarized tree view of the project structure, filtering out noise (obj, bin, .git).
  * `AnalyzeFile`: Returns the content of a file *with* context (line numbers, symbols) and optional summarization for large files.
  * `SearchCodebase`: Semantic search (RAG) or regex search over the codebase.

* **Coding Capabilities**:
  * `ApplyPatch`: A structured tool to apply changes to a file (replacing `sed`/`echo` shell commands).
  * `RunTests`: A dedicated tool to run tests and return structured results (Pass/Fail, Error Log) rather than raw stdout.

### 3. Integration

These new tools will be registered in `Tars.Tools` and assigned to the `Executor` agent in the Evolution Engine.
