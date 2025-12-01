# Leveraging Google Code Wiki for TARS v2

**Date:** November 22, 2025
**Status:** Proposed Integration Strategy

---

## Executive Summary

Google Code Wiki (<https://codewiki.google/>) is an AI-powered platform that automatically generates and maintains documentation, diagrams, and knowledge graphs from codebases.

For TARS v2, Code Wiki represents the **perfect "Read-Only Memory" companion**. While TARS focuses on *writing* and *evolving* code, Code Wiki focuses on *understanding* and *explaining* it.

By integrating Code Wiki, we solve one of the hardest problems in agentic coding: **Context Window Saturation**. Instead of stuffing raw code files into the context, TARS can query Code Wiki for high-level summaries, architecture diagrams, and specific answers, drastically reducing token usage while increasing reasoning accuracy.

---

## The "Ouroboros" Loop: Code & Documentation Symbiosis

The core value proposition is closing the loop between creation and understanding:

1. **TARS (The Architect)**: Writes code, refactors systems, and implements features.
2. **Code Wiki (The Scribe)**: Automatically observes changes, updates documentation, redraws diagrams, and indexes knowledge.
3. **TARS (The Reader)**: Queries Code Wiki to understand the current state of the system before making new changes.

This prevents "Agent Drift" where an agent's internal mental model diverges from the actual codebase reality.

---

## Integration Architecture

### 1. The `CodeWikiSkill` (Tool Layer)

We should implement a specific skill in `Tars.Skills` that interfaces with Code Wiki.

**Capabilities:**

- `GetModuleSummary(path)`: Returns the high-level explanation of a folder/module.
- `GetArchitectureDiagram()`: Retrieves the Mermaid/PlantUML representation of the system structure.
- `AskCodebase(question)`: Proxies a natural language question to Code Wiki's Gemini-powered chat (e.g., "How does the EventBus handle backpressure?").

**Why this is better than raw RAG:**

- **Pre-Synthesized**: Code Wiki has already done the heavy lifting of summarizing code into English.
- **Graph-Aware**: It understands relationships (imports, calls) better than simple vector search.
- **Visual**: It provides diagrams which TARS v2's multi-modal capabilities (future) or text-based diagram parsers can use.

### 2. Context Loading Strategy (Memory Layer)

When an agent spins up for a task (e.g., "Refactor the Vector Store"), it typically needs to read 50+ files to understand the context.

**With Code Wiki:**

1. **Agent Init**: Agent requests `CodeWiki.GetModuleSummary("src/Tars.Memory.Vector")`.
2. **Context Injection**: TARS loads the *English documentation* and *Class Diagrams* into the System Prompt.
3. **Efficiency**: 50 files (100k tokens) becomes 1 documentation page (2k tokens).
4. **Deep Dive**: The agent only reads raw code files when it identifies exactly which file needs modification.

### 3. Architecture Verification (Governance Layer)

TARS v2 has strict architectural rules (e.g., "Kernel must not depend on Skills").

- **Current Way**: Static analysis tools (Roslyn analyzers).
- **Code Wiki Way**:
    1. TARS asks Code Wiki: "Generate a dependency graph for Tars.Kernel".
    2. TARS analyzes the graph text.
    3. If `Tars.Kernel -> Tars.Skills` edge exists, TARS detects the violation *conceptually* before even running a build.

---

## Specific Use Cases for TARS v2

### Use Case A: "The Archaeologist" (Legacy Migration)

We are currently migrating v1 components to v2.

- **Problem**: v1 code is complex, undocumented, and uses esoteric math (Hyperbolic embeddings).
- **Solution**: Point Code Wiki at the `v1/` directory. Let it generate the "Rosetta Stone". TARS v2 agents read the Code Wiki explanation of `HyperComplexGeometricDSL.fs` to understand *what it does* mathematically, rather than trying to reverse-engineer the F# implementation details line-by-line.

### Use Case B: "The Librarian" (Self-Documentation)

TARS creates new agents and skills dynamically.

- **Problem**: Who documents the code the AI wrote?
- **Solution**: TARS doesn't need to write documentation comments. It just commits the code. Code Wiki auto-generates the docs. TARS then *verifies* the docs: "Does this wiki page accurately reflect what I just wrote?" If not, TARS refactors the code to be clearer (Documentation-Driven Refactoring).

---

## Implementation Plan

### Phase 1: Manual Proxy (Now)

Since Code Wiki is in preview/internal:

1. User manually visits `https://codewiki.google/` (or public equivalent) for the TARS repo.
2. User copies key summaries (Architecture, Core Modules) into `v2/docs/knowledge_base/`.
3. TARS uses these markdown files as a "Simulated Code Wiki".

### Phase 2: CLI Integration (Future)

Once Google releases the Gemini CLI extension for Code Wiki:

1. Build `Tars.Skills.CodeWiki`.
2. Implement `ask_codewiki` tool for agents.
3. Wire into the `Tars.Cortex` context loading pipeline.

---

## Recommendation

**Adopt Code Wiki concepts immediately**, even if we simulate the tool initially.

1. **Structure our docs** to match Code Wiki's format (Summary, Diagram, API).
2. **Train agents** to look for high-level docs *first* before reading code.
3. **Prepare the repo** by ensuring clean directory structures that Code Wiki parsers love.
