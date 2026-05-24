# Memory Architecture Analysis: TARS v2

**Date:** November 26, 2025
**Status:** Draft

## 1. Overview

To support the "superintelligence" goals of TARS v2, we need a sophisticated memory architecture that goes beyond a simple vector store. We propose a multi-tiered memory system based on **Persistence** (Duration) and **Scope** (Visibility).

## 2. Memory Tiers (Duration)

### 2.1 Short-Term Memory (Working Memory)

* **Purpose:** Immediate context for the current task or conversation.
* **Implementation:** In-memory context window of the LLM + ephemeral vector store (e.g., `InMemoryVectorStore`).
* **Volatility:** High. Cleared after task completion or agent termination.
* **Analogy:** RAM / CPU Cache.

### 2.2 Mid-Term Memory (Episodic Memory)

* **Purpose:** Context relevant to a specific project, session, or ongoing workflow.
* **Implementation:** Local vector store (e.g., ChromaDB collection per project) or file-based persistence.
* **Volatility:** Medium. Persists across agent restarts but scoped to a specific context.
* **Analogy:** Project files / Browser local storage.

### 2.3 Long-Term Memory (Semantic/Procedural Memory)

* **Purpose:** Global knowledge, learned skills, best practices, and "self" identity.
* **Implementation:** Centralized Vector Database (ChromaDB/Qdrant) + Knowledge Graph (Graphiti).
* **Volatility:** Low. Permanent and evolving.
* **Analogy:** Hard Drive / Cloud Storage / Human Long-term memory.

## 3. Memory Scopes (Visibility)

### 3.1 Agent Scope (Private)

* **Visibility:** Only the specific agent instance.
* **Use Case:** Scratchpad, intermediate reasoning steps, private thoughts.

### 3.2 Swarm Scope (Shared)

* **Visibility:** All agents within a specific swarm/group working on a common goal.
* **Use Case:** Shared context, message bus history, collaborative planning.

### 3.3 Global Scope (Public)

* **Visibility:** All agents in the system.
* **Use Case:** Core knowledge base, tool definitions, system prompts, "Constitution".

## 4. Proposed Architecture

We will define a `IMemoryStore` interface that can be composed with different backends and scopes.

```fsharp
type MemoryScope =
    | Private of AgentId
    | Shared of SwarmId
    | Global

type MemoryTier =
    | ShortTerm  // In-Memory
    | MidTerm    // File/Local DB
    | LongTerm   // Central DB

type IMemoryInterface =
    abstract member StoreAsync: scope: MemoryScope -> tier: MemoryTier -> content: string -> Task<unit>
    abstract member RetrieveAsync: scope: MemoryScope -> query: string -> Task<string list>
```

## 5. Next Steps

1. **Refine Interfaces:** Update `Tars.Core` with these new definitions.
2. **Implement Backends:**
    * `InMemoryVectorStore` (Done - for ShortTerm).
    * `ChromaVectorStore` (External - for Mid/LongTerm).
    * `GraphStore` (Future - for Structured LongTerm).
3. **Integration:** Update `IAgent` to have access to this tiered memory system.
